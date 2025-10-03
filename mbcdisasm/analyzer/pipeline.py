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
        return tuple(self._compress_marker_runs(events))

    def _segment_into_blocks(
        self,
        profiles: Sequence[InstructionProfile],
        events: Sequence[StackEvent],
    ) -> List[PipelineBlock]:
        blocks: List[PipelineBlock] = []
        idx_event = 0
        idx_profile = 0
        window = self.settings.max_window
        total_events = len(events)
        total_profiles = len(profiles)
        while idx_event < total_events and idx_profile < total_profiles:
            best_match: Optional[PatternMatch] = None
            best_span_events = 1
            best_span_profiles = events[idx_event].span if events else 1
            for size in range(2, window + 1):
                end_event = idx_event + size
                if end_event > total_events:
                    break
                slice_events = events[idx_event:end_event]
                match = self.automaton.best_match(slice_events)
                if match is None:
                    continue
                span_profiles = sum(event.span for event in slice_events)
                if idx_profile + span_profiles > total_profiles:
                    continue
                if best_match is None or match.score > best_match.score:
                    best_match = match
                    best_span_events = size
                    best_span_profiles = span_profiles
            span_events = best_span_events
            span_profiles = best_span_profiles
            if best_match is None:
                span_events = 1
                span_profiles = events[idx_event].span if events else 1
            block_profiles = profiles[idx_profile : idx_profile + span_profiles]
            stack_summary = StackTracker().process_block(block_profiles)
            previous = profiles[idx_profile - 1] if idx_profile > 0 else None
            following = (
                profiles[idx_profile + span_profiles]
                if idx_profile + span_profiles < total_profiles
                else None
            )
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
            idx_event += span_events
            idx_profile += span_profiles
        return blocks

    def _compress_marker_runs(self, events: Sequence[StackEvent]) -> List[StackEvent]:
        if not events:
            return []

        compressed: List[StackEvent] = []
        run: List[StackEvent] = []

        def flush_run() -> None:
            if not run:
                return
            compressed.append(self._merge_marker_run(run))
            run.clear()

        for event in events:
            if is_literal_marker(event.profile):
                run.append(event)
                continue
            flush_run()
            compressed.append(event)

        flush_run()
        return compressed

    @staticmethod
    def _merge_marker_run(run: Sequence[StackEvent]) -> StackEvent:
        first = run[0]
        span = sum(event.span for event in run)
        delta = sum(event.delta for event in run)
        minimum = min(event.minimum for event in run)
        maximum = max(event.maximum for event in run)
        confidence = min(event.confidence for event in run)
        depth_before = first.depth_before
        depth_after = run[-1].depth_after
        uncertain = any(event.uncertain for event in run)
        return StackEvent(
            profile=first.profile,
            delta=delta,
            minimum=minimum,
            maximum=maximum,
            confidence=confidence,
            depth_before=depth_before,
            depth_after=depth_after,
            uncertain=uncertain,
            marker_run=True,
            span=span,
        )

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
