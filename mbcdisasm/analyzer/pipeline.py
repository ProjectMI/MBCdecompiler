"""High level orchestration for pipeline analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
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


@dataclass(frozen=True)
class AggregatedEvent:
    """Represents a stack event that may cover several instructions."""

    event: StackEvent
    start: int
    span: int


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
        return tracker.process_sequence(profiles)

    def _segment_into_blocks(
        self,
        profiles: Sequence[InstructionProfile],
        events: Sequence[StackEvent],
    ) -> List[PipelineBlock]:
        blocks: List[PipelineBlock] = []
        clustered = self._cluster_marker_events(events)
        total_profiles = len(profiles)
        total_clusters = len(clustered)
        cluster_index = 0
        profile_index = 0
        hard_window = self.settings.max_window
        extended_window = min(9, hard_window + 3)

        while cluster_index < total_clusters and profile_index < total_profiles:
            start_profile = clustered[cluster_index].start
            profile_index = start_profile
            best_match: Optional[PatternMatch] = None
            best_clusters = 1

            remaining_instructions = total_profiles - start_profile
            search_window = min(remaining_instructions, extended_window)

            instruction_span = 0
            for span_clusters in range(1, total_clusters - cluster_index + 1):
                instruction_span += clustered[cluster_index + span_clusters - 1].span
                if instruction_span > search_window:
                    break
                slice_events = [item.event for item in clustered[cluster_index : cluster_index + span_clusters]]
                match = self.automaton.best_match(slice_events)
                if match is None:
                    continue
                if best_match is None or match.score > best_match.score:
                    best_match = match
                    best_clusters = span_clusters

            if best_match is None:
                soft_target = search_window
                if remaining_instructions >= 4:
                    soft_target = max(4, soft_target)
                soft_target = max(clustered[cluster_index].span, soft_target)
                instruction_span = 0
                span_clusters = 0
                while cluster_index + span_clusters < total_clusters and instruction_span < soft_target:
                    instruction_span += clustered[cluster_index + span_clusters].span
                    span_clusters += 1
                best_clusters = max(1, span_clusters)

            clusters_to_consume = best_clusters
            span_instructions = sum(
                clustered[cluster_index + offset].span
                for offset in range(clusters_to_consume)
            )
            block_profiles = profiles[start_profile : start_profile + span_instructions]
            stack_summary = StackTracker().process_block(block_profiles)
            previous = profiles[start_profile - 1] if start_profile > 0 else None
            following_index = start_profile + span_instructions
            following = profiles[following_index] if following_index < total_profiles else None
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
            profile_index += span_instructions
            cluster_index += clusters_to_consume

        return blocks

    def _cluster_marker_events(self, events: Sequence[StackEvent]) -> Tuple[AggregatedEvent, ...]:
        clustered: List[AggregatedEvent] = []
        index = 0
        total = len(events)
        while index < total:
            event = events[index]
            if event.profile.is_literal_marker():
                start = index
                cluster_events: List[StackEvent] = [event]
                index += 1
                while index < total and events[index].profile.is_literal_marker():
                    cluster_events.append(events[index])
                    index += 1
                aggregated = self._merge_marker_cluster(cluster_events)
                clustered.append(AggregatedEvent(event=aggregated, start=start, span=len(cluster_events)))
            else:
                clustered.append(AggregatedEvent(event=event, start=index, span=1))
                index += 1
        return tuple(clustered)

    def _merge_marker_cluster(self, events: Sequence[StackEvent]) -> StackEvent:
        first = events[0]
        last = events[-1]
        delta = sum(event.delta for event in events)
        minimum = min(event.minimum for event in events)
        maximum = max(event.maximum for event in events)
        confidence = min(event.confidence for event in events)
        popped = tuple(chain.from_iterable(event.popped_types for event in events))
        pushed = tuple(chain.from_iterable(event.pushed_types for event in events))
        uncertain = any(event.uncertain for event in events)
        return StackEvent(
            profile=first.profile,
            delta=delta,
            minimum=minimum,
            maximum=maximum,
            confidence=confidence,
            depth_before=first.depth_before,
            depth_after=last.depth_after,
            kind=first.kind,
            popped_types=popped,
            pushed_types=pushed,
            uncertain=uncertain,
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
        elif dominant in {
            InstructionKind.INDIRECT,
            InstructionKind.INDIRECT_LOAD,
            InstructionKind.INDIRECT_STORE,
            InstructionKind.TABLE_LOOKUP,
        }:
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
