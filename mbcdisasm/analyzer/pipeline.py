"""High level orchestration for pipeline analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..instruction import InstructionWord
from ..knowledge import KnowledgeBase
from .instruction_profile import (
    InstructionKind,
    InstructionProfile,
    dominant_kind,
)
from .diagnostics import DiagnosticBuilder
from .dfa import DeterministicAutomaton
from .heuristics import HeuristicEngine, HeuristicReport
from .patterns import PatternMatch, PatternRegistry, default_patterns
from .signatures import SignatureDetector
from .stats import StatisticsBuilder
from .report import PipelineBlock, PipelineReport, build_block
from .stack import StackEvent, StackSummary, StackTracker
from .block_features import BlockFeatures


@dataclass
class AnalyzerSettings:
    """Configuration knobs for :class:`PipelineAnalyzer`."""

    max_window: int = 6
    min_confidence: float = 0.35
    stack_change_bias: float = 0.05


class PipelineAnalyzer:
    """Analyse instruction streams and recover high level pipeline blocks."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        *,
        settings: Optional[AnalyzerSettings] = None,
        patterns: Optional[PatternRegistry] = None,
    ) -> None:
        self.knowledge = knowledge
        self.settings = settings or AnalyzerSettings()
        self.registry = patterns or default_patterns()
        self.heuristics = HeuristicEngine()
        self.diagnostics = DiagnosticBuilder()
        self.statistics_builder = StatisticsBuilder()
        self.automaton = DeterministicAutomaton(self.registry)
        self.signatures = SignatureDetector()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def analyse_segment(self, instructions: Sequence[InstructionWord]) -> PipelineReport:
        profiles = self._profile_instructions(instructions)
        if not profiles:
            return PipelineReport.empty()
        events = self._compute_events(profiles)
        blocks = self._segment_into_blocks(profiles, events)
        warnings = self._generate_warnings(blocks)
        statistics = self.statistics_builder.collect(blocks)
        return PipelineReport(blocks=tuple(blocks), warnings=tuple(warnings), statistics=statistics)

    analyze_segment = analyse_segment  # alias for US spelling

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _profile_instructions(self, instructions: Sequence[InstructionWord]) -> Tuple[InstructionProfile, ...]:
        return tuple(InstructionProfile.from_word(word, self.knowledge) for word in instructions)

    def _compute_events(self, profiles: Sequence[InstructionProfile]) -> Tuple[StackEvent, ...]:
        tracker = StackTracker()
        events = [tracker.process(profile) for profile in profiles]
        return tuple(events)

    def _segment_into_blocks(
        self,
        profiles: Sequence[InstructionProfile],
        events: Sequence[StackEvent],
    ) -> List[PipelineBlock]:
        blocks: List[PipelineBlock] = []
        idx = 0
        window = self.settings.max_window
        total = len(profiles)
        while idx < total:
            best_match: Optional[PatternMatch] = None
            best_span = 1
            for size in range(2, window + 1):
                end = idx + size
                if end > total:
                    break
                slice_events = events[idx:end]
                match = self.automaton.best_match(slice_events)
                if match is None:
                    continue
                if best_match is None or match.score > best_match.score:
                    best_match = match
                    best_span = size
            span = best_span
            if best_match is None:
                span = max(1, min(window, total - idx))
            block_profiles = profiles[idx : idx + span]
            stack_summary = StackTracker().process_block(block_profiles)
            previous = profiles[idx - 1] if idx > 0 else None
            following = profiles[idx + span] if idx + span < total else None
            block_features = BlockFeatures.from_profiles(block_profiles)
            heuristic_report = self.heuristics.analyse(
                block_profiles,
                stack_summary,
                previous=previous,
                following=following,
            )
            category, confidence, notes = self._classify_block(
                block_profiles,
                block_features,
                stack_summary,
                best_match,
                heuristic_report,
            )
            notes.append(heuristic_report.describe())
            block = build_block(block_profiles, stack_summary, best_match, category, confidence, notes)
            blocks.append(block)
            idx += span
        return blocks

    def _classify_block(
        self,
        profiles: Sequence[InstructionProfile],
        features: BlockFeatures,
        stack: StackSummary,
        match: Optional[PatternMatch],
        heuristics: HeuristicReport,
    ) -> Tuple[str, float, List[str]]:
        notes: List[str] = []
        if match is not None:
            category = match.pattern.category
            confidence = min(1.0, match.score)
            if stack.uncertain:
                confidence *= 0.85
                notes.append("uncertain stack delta")
            return category, confidence, notes

        signature = self.signatures.detect(profiles, stack)
        if signature is not None:
            notes.append(f"signature={signature.name}")
            notes.extend(signature.notes)
            confidence = max(self.settings.min_confidence, signature.confidence)
            if stack.uncertain:
                confidence *= 0.85
                notes.append("uncertain stack delta")
            return signature.category, confidence, notes

        dominant = dominant_kind(profiles)
        feature_map = heuristics.feature_map()
        notes.extend(features.summarise())

        category = "literal"
        confidence = self.settings.min_confidence

        literal_ratio = features.literal_ratio()
        call_ratio = features.call_ratio()
        return_ratio = features.return_ratio()
        branch_ratio = features.branch_ratio()
        compute_ratio = features.compute_ratio()
        indirect_ratio = features.indirect_ratio()

        call_score = call_ratio
        if "call_helper" in feature_map:
            call_score += feature_map["call_helper"].score

        has_explicit_call = features.call_like > 0 or dominant in {
            InstructionKind.CALL,
            InstructionKind.TAILCALL,
        }
        call_signal = has_explicit_call or (
            call_score >= 0.35 and (literal_ratio < 0.45 or stack.change <= 0)
        )

        if call_signal:
            category = "call"
            confidence = max(confidence, min(1.0, 0.45 + call_score * 0.4 + heuristics.confidence * 0.2))
        else:
            return_score = return_ratio
            if "return_sequence" in feature_map:
                return_score += feature_map["return_sequence"].score
            if "stack_teardown" in feature_map:
                return_score += feature_map["stack_teardown"].score
            if return_score >= 0.2 or dominant in {
                InstructionKind.RETURN,
                InstructionKind.TERMINATOR,
                InstructionKind.STACK_TEARDOWN,
            }:
                category = "return"
                confidence = max(confidence, min(1.0, 0.45 + return_score * 0.35 + heuristics.confidence * 0.2))
            else:
                if branch_ratio >= 0.2 or dominant is InstructionKind.TEST:
                    category = "test"
                    confidence = max(confidence, min(1.0, 0.4 + branch_ratio * 0.3 + heuristics.confidence * 0.1))
                elif indirect_ratio >= 0.2 or "indirect_pattern" in feature_map:
                    indirect_score = indirect_ratio
                    if "indirect_pattern" in feature_map:
                        indirect_score += feature_map["indirect_pattern"].score
                    category = "indirect"
                    confidence = max(confidence, min(1.0, 0.45 + indirect_score * 0.35 + heuristics.confidence * 0.1))
                elif compute_ratio >= 0.25 or dominant in {
                    InstructionKind.ARITHMETIC,
                    InstructionKind.REDUCE,
                }:
                    category = "compute"
                    confidence = max(confidence, min(1.0, 0.4 + compute_ratio * 0.4 + heuristics.confidence * 0.1))
                else:
                    literal_score = literal_ratio
                    if dominant in {InstructionKind.LITERAL, InstructionKind.ASCII_CHUNK, InstructionKind.PUSH}:
                        literal_score = max(literal_score, 0.45)
                    if literal_score < 0.2 and features.meta_like > features.literal_like and stack.change < 0:
                        category = "compute"
                        confidence = max(confidence, 0.4 + features.meta_ratio() * 0.3)
                    else:
                        category = "literal"
                        confidence = max(confidence, min(1.0, 0.45 + literal_score * 0.4 + heuristics.confidence * 0.1))

        if stack.change > 0 and category == "compute":
            notes.append("positive stack change in compute block")
        if stack.change < 0 and category == "literal":
            notes.append("literal block reduced stack")

        if stack.uncertain:
            confidence *= 0.85
            notes.append("uncertain stack delta")

        confidence = max(confidence, self.settings.min_confidence)
        confidence = min(1.0, confidence)
        return category, confidence, notes

    def _generate_warnings(self, blocks: Sequence[PipelineBlock]) -> List[str]:
        report = self.diagnostics.evaluate(blocks)
        return [entry.describe() for entry in report.filter("warning")]
