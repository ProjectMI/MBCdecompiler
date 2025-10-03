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
from .signatures import SignatureDetector, is_literal_marker
from .stats import StatisticsBuilder
from .report import PipelineBlock, PipelineReport, build_block
from .stack import StackEvent, StackSummary, StackTracker


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
        normalised_events, spans = self._normalise_events(events)
        blocks = self._segment_into_blocks(profiles, normalised_events, spans)
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
        normalised: Sequence[StackEvent],
        spans: Sequence[Tuple[int, int]],
    ) -> List[PipelineBlock]:
        blocks: List[PipelineBlock] = []
        idx = 0
        window = self.settings.max_window
        total = len(normalised)
        while idx < total:
            best_match: Optional[PatternMatch] = None
            best_span = 1
            for size in range(2, window + 1):
                end = idx + size
                if end > total:
                    break
                slice_events = normalised[idx:end]
                match = self.automaton.best_match(slice_events)
                if match is None:
                    continue
                if best_match is None or match.score > best_match.score:
                    best_match = match
                    best_span = size
            span = best_span
            if best_match is None:
                span = max(1, min(window, total - idx))
            span_start, span_end = spans[idx][0], spans[idx + span - 1][1]
            block_profiles = profiles[span_start:span_end]
            stack_summary = StackTracker().process_block(block_profiles)
            previous = profiles[span_start - 1] if span_start > 0 else None
            following = profiles[span_end] if span_end < len(profiles) else None
            heuristic_report = self.heuristics.analyse(
                block_profiles,
                stack_summary,
                previous=previous,
                following=following,
            )
            category, confidence, notes = self._classify_block(
                block_profiles,
                stack_summary,
                best_match,
                heuristic_report,
            )
            notes.append(heuristic_report.describe())
            block = build_block(block_profiles, stack_summary, best_match, category, confidence, notes)
            blocks.append(block)
            idx += span
        return blocks

    def _normalise_events(
        self, events: Sequence[StackEvent]
    ) -> Tuple[Tuple[StackEvent, ...], Tuple[Tuple[int, int], ...]]:
        """Collapse literal marker runs into pseudo events for DFA matching."""

        compressed: List[StackEvent] = []
        spans: List[Tuple[int, int]] = []
        idx = 0
        total = len(events)

        while idx < total:
            event = events[idx]
            start_idx = idx
            if self._is_marker_event(event):
                while idx < total and self._is_marker_event(events[idx]):
                    idx += 1
                grouped = events[start_idx:idx]
                compressed.append(self._collapse_group(grouped))
                spans.append((start_idx, idx))
                continue

            compressed.append(event)
            spans.append((idx, idx + 1))
            idx += 1

        return tuple(compressed), tuple(spans)

    @staticmethod
    def _collapse_group(group: Sequence[StackEvent]) -> StackEvent:
        """Return a synthetic stack event that represents ``group``."""

        first = group[0]
        last = group[-1]
        delta = sum(event.delta for event in group)
        minimum = min(event.minimum for event in group)
        maximum = max(event.maximum for event in group)
        confidence = min(event.confidence for event in group)
        uncertain = any(event.uncertain for event in group)
        return StackEvent(
            profile=first.profile,
            delta=delta,
            minimum=minimum,
            maximum=maximum,
            confidence=confidence,
            depth_before=first.depth_before,
            depth_after=last.depth_after,
            uncertain=uncertain,
        )

    @staticmethod
    def _is_marker_event(event: StackEvent) -> bool:
        profile = event.profile
        if profile.kind is not InstructionKind.LITERAL:
            return False
        if not is_literal_marker(profile):
            return False
        return True

    def _classify_block(
        self,
        profiles: Sequence[InstructionProfile],
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
        category = "unknown"
        confidence = self.settings.min_confidence

        if dominant in {InstructionKind.LITERAL, InstructionKind.ASCII_CHUNK, InstructionKind.PUSH}:
            category = "literal"
            confidence = 0.55
        elif dominant in {InstructionKind.REDUCE, InstructionKind.ARITHMETIC}:
            category = "compute"
            confidence = 0.5
        elif dominant in {InstructionKind.STACK_TEARDOWN, InstructionKind.RETURN, InstructionKind.TERMINATOR}:
            category = "return"
            confidence = 0.6
        elif dominant in {InstructionKind.CALL, InstructionKind.TAILCALL}:
            category = "call"
            confidence = 0.6
        elif dominant is InstructionKind.TEST:
            category = "test"
            confidence = 0.5
        elif dominant in {InstructionKind.INDIRECT, InstructionKind.TABLE_LOOKUP}:
            category = "indirect"
            confidence = 0.5

        feature_map = heuristics.feature_map()

        if "indirect_pattern" in feature_map:
            category = "indirect"
            confidence = max(confidence, 0.6)

        if "call_helper" in feature_map:
            category = "call"
            confidence = max(confidence, 0.65)

        if "return_sequence" in feature_map or "stack_teardown" in feature_map:
            category = "return"
            confidence = max(confidence, 0.6)

        if stack.change > 0 and category == "compute":
            notes.append("positive stack change in compute block")
        if stack.change < 0 and category == "literal":
            notes.append("literal block reduced stack")
        if stack.uncertain:
            confidence *= 0.85
            notes.append("uncertain stack delta")
        return category, confidence, notes

    def _generate_warnings(self, blocks: Sequence[PipelineBlock]) -> List[str]:
        report = self.diagnostics.evaluate(blocks)
        return [entry.describe() for entry in report.filter("warning")]
