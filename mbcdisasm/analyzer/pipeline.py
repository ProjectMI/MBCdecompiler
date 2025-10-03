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


@dataclass(frozen=True)
class _EventCluster:
    """Aggregate representing one or more stack events."""

    start: int
    end: int
    event: StackEvent


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
        clusters = self._cluster_stack_events(events)
        blocks: List[PipelineBlock] = []
        idx = 0
        soft_window = max(self.settings.max_window, 9)
        total_clusters = len(clusters)
        total_profiles = len(profiles)
        while idx < total_clusters:
            best_match: Optional[PatternMatch] = None
            best_span = 1
            max_window = min(total_clusters - idx, soft_window)
            for size in range(2, max_window + 1):
                end = idx + size
                slice_events = tuple(cluster.event for cluster in clusters[idx:end])
                match = self.automaton.best_match(slice_events)
                if match is None:
                    continue
                if best_match is None or match.score > best_match.score:
                    best_match = match
                    best_span = size
            span = best_span
            if best_match is None:
                span = self._fallback_span(clusters, idx, max_window)
            cluster_start = clusters[idx].start
            cluster_end = clusters[idx + span - 1].end if idx + span - 1 < total_clusters else total_profiles
            block_profiles = profiles[cluster_start:cluster_end]
            stack_summary = StackTracker().process_block(block_profiles)
            previous = profiles[cluster_start - 1] if cluster_start > 0 else None
            following = profiles[cluster_end] if cluster_end < total_profiles else None
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

    def _fallback_span(self, clusters: Sequence[_EventCluster], start: int, limit: int) -> int:
        """Return the number of clusters to consume when no pattern matched."""

        if limit <= 1:
            return limit

        max_index = min(start + limit, len(clusters))
        eligible_span: Optional[int] = None
        best_span: int = 1
        block_start = clusters[start].start
        for offset in range(start, max_index):
            span = offset - start + 1
            block_end = clusters[offset].end
            length = block_end - block_start
            if length > 9:
                break
            best_span = span
            if length >= 4:
                eligible_span = span
        if eligible_span is not None:
            return eligible_span
        return best_span

    def _cluster_stack_events(self, events: Sequence[StackEvent]) -> Tuple[_EventCluster, ...]:
        """Collapse consecutive literal markers into a single matching unit."""

        clusters: List[_EventCluster] = []
        idx = 0
        total = len(events)
        while idx < total:
            event = events[idx]
            if event.profile.is_literal_marker():
                start = idx
                end = idx + 1
                while end < total and events[end].profile.is_literal_marker():
                    end += 1
                merged = self._merge_marker_events(events[start:end])
                clusters.append(_EventCluster(start=start, end=end, event=merged))
                idx = end
            else:
                clusters.append(_EventCluster(start=idx, end=idx + 1, event=event))
                idx += 1
        return tuple(clusters)

    def _merge_marker_events(self, events: Sequence[StackEvent]) -> StackEvent:
        """Return a synthetic stack event representing a marker cluster."""

        if len(events) == 1:
            return events[0]

        first = events[0]
        last = events[-1]
        delta = sum(event.delta for event in events)
        minimum = min(event.minimum for event in events)
        maximum = max(event.maximum for event in events)
        confidence = min(event.confidence for event in events)
        uncertain = any(event.uncertain for event in events)
        merged = StackEvent(
            profile=first.profile,
            delta=delta,
            minimum=minimum,
            maximum=maximum,
            confidence=confidence,
            depth_before=first.depth_before,
            depth_after=last.depth_after,
            kind=first.kind,
            popped_types=first.popped_types,
            pushed_types=last.pushed_types,
            uncertain=uncertain,
        )
        setattr(merged, "cluster_start", first.profile.word.offset)
        setattr(merged, "cluster_end", last.profile.word.offset)
        return merged

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
