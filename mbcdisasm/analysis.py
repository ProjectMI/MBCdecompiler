"""Statistical analysis helpers for opcode discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .instruction import InstructionWord, WORD_SIZE, read_instructions
from .knowledge import (
    KnowledgeBase,
    OpcodeProfile,
    ProfileAssessment,
    StackObservation,
)
from .mbc import MbcContainer, Segment
from .stack_model import StackDeltaEstimate, StackDeltaModeler
from .stack_seed import seed_stack_modeler_from_knowledge


@dataclass
class SegmentReport:
    index: int
    start: int
    length: int
    classification: str
    instruction_count: int
    issues: List[str] = field(default_factory=list)


@dataclass
class EmulationQualityMetrics:
    """Aggregate visibility into stack emulation confidence."""

    total_instructions: int = 0
    uncertain_instructions: int = 0
    segment_unknown_ratios: Dict[int, float] = field(default_factory=dict)

    def record_segment(self, index: int, total: int, uncertain: int) -> None:
        if total <= 0:
            return
        self.total_instructions += total
        self.uncertain_instructions += uncertain
        self.segment_unknown_ratios[index] = uncertain / total

    @property
    def unknown_ratio(self) -> Optional[float]:
        if self.total_instructions == 0:
            return None
        return self.uncertain_instructions / self.total_instructions

    @property
    def determined_instructions(self) -> int:
        return self.total_instructions - self.uncertain_instructions

    def to_json(self) -> dict:
        payload: dict = {
            "total_instructions": self.total_instructions,
            "uncertain_instructions": self.uncertain_instructions,
            "segment_unknown_ratios": {
                str(index): ratio for index, ratio in sorted(self.segment_unknown_ratios.items())
            },
        }
        ratio = self.unknown_ratio
        if ratio is not None:
            payload["unknown_ratio"] = ratio
        payload["determined_instructions"] = self.determined_instructions
        return payload


@dataclass
class AnalysisResult:
    segment_reports: List[SegmentReport]
    opcode_profiles: Dict[str, OpcodeProfile]
    total_instructions: int
    total_segments: int
    issues: List[str]
    profile_assessments: List[ProfileAssessment]
    stack_observations: Dict[str, StackObservation]
    emulation_quality: EmulationQualityMetrics

    def iter_profiles(self) -> Iterable[OpcodeProfile]:
        return self.opcode_profiles.values()

    def most_common(self, limit: int = 10) -> List[tuple[str, int]]:
        return sorted(
            ((key, profile.count) for key, profile in self.opcode_profiles.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

    def opcode_mode_matrix(self) -> Dict[str, List[str]]:
        matrix: Dict[str, set[str]] = {}
        for key in self.opcode_profiles:
            if ":" not in key:
                continue
            opcode, mode = key.split(":", 1)
            bucket = matrix.setdefault(opcode, set())
            bucket.add(mode)
        return {opcode: sorted(modes) for opcode, modes in sorted(matrix.items())}

    def confidence_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for assessment in self.profile_assessments:
            counts[assessment.status] = counts.get(assessment.status, 0) + 1
        order = {
            "conflict": 0,
            "partial": 1,
            "confirmed": 2,
            "unknown": 3,
            "undetermined": 4,
        }
        return dict(
            sorted(counts.items(), key=lambda item: (order.get(item[0], 99), item[0]))
        )

    def conflicting_profiles(self) -> List[ProfileAssessment]:
        return [a for a in self.profile_assessments if a.status == "conflict"]

    def unknown_stack_profiles(self, threshold: float = 0.6) -> List[StackObservation]:
        return [
            obs
            for obs in self.stack_observations.values()
            if obs.unknown_ratio is not None
            and obs.unknown_ratio >= threshold
            and obs.total_samples > 0
        ]


class Analyzer:
    """Traverse containers to gather opcode statistics."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        *,
        stack_modeler: Optional[StackDeltaModeler] = None,
    ) -> None:
        self.knowledge = knowledge
        self.stack_modeler = stack_modeler or StackDeltaModeler()
        self._seeded_keys: set[str] = set()

    def analyze(self, container: MbcContainer) -> AnalysisResult:
        self._seeded_keys.clear()
        opcode_profiles: Dict[str, OpcodeProfile] = {}
        segment_reports: List[SegmentReport] = []
        aggregated_issue_counts: Dict[str, int] = {}
        total_instructions = 0
        assessments: List[ProfileAssessment] = []
        stack_observations: Dict[str, StackObservation] = {}
        emulation_quality = EmulationQualityMetrics()

        def record_issue(issue: str) -> None:
            normalized = issue
            if issue.startswith("trailing_bytes="):
                normalized = "trailing_bytes"
            aggregated_issue_counts[normalized] = (
                aggregated_issue_counts.get(normalized, 0) + 1
            )

        for segment in container.segments():
            if not segment.data:
                issue = "empty"
                segment_reports.append(
                    SegmentReport(
                        index=segment.index,
                        start=segment.start,
                        length=segment.length,
                        classification=segment.classification,
                        instruction_count=0,
                        issues=[issue],
                    )
                )
                record_issue(issue)
                continue

            if segment.classification != "code":
                segment_reports.append(
                    SegmentReport(
                        segment.index,
                        segment.start,
                        segment.length,
                        segment.classification,
                        0,
                        [],
                    )
                )
                continue

            instructions, remainder = read_instructions(segment.data, segment.start)
            self._seed_stack_modeler(instructions)
            estimates = self.stack_modeler.model_segment(instructions)

            report = SegmentReport(
                segment.index,
                segment.start,
                segment.length,
                segment.classification,
                len(instructions),
                [],
            )
            if remainder:
                trailing_issue = f"trailing_bytes={remainder}"
                report.issues.append(trailing_issue)
                record_issue(trailing_issue)
            segment_reports.append(report)
            total_instructions += len(instructions)

            uncertain_instructions = 0
            for index, current in enumerate(instructions):
                previous: Optional[InstructionWord]
                if index == 0:
                    previous = None
                else:
                    previous = instructions[index - 1]
                estimate = estimates[index] if index < len(estimates) else StackDeltaEstimate(
                    key=current.label(), delta=None, confidence=0.0
                )
                if not self.stack_modeler.is_confident(estimate):
                    uncertain_instructions += 1
                self._update_profiles(
                    opcode_profiles,
                    previous,
                    current,
                    estimate,
                )
            emulation_quality.record_segment(
                segment.index,
                len(instructions),
                uncertain_instructions,
            )

        for profile in opcode_profiles.values():
            assessments.append(self.knowledge.assess_profile(profile))
            stack_observations[profile.key] = StackObservation.from_profile(profile)

        issue_labels = {
            "empty": "empty segments",
            "trailing_bytes": "segments with trailing bytes",
        }

        issues: List[str] = []
        for key in sorted(aggregated_issue_counts):
            count = aggregated_issue_counts[key]
            label = issue_labels.get(key, key)
            if key == "trailing_bytes":
                # Still surface these counts, but keep the messaging concise.
                formatted = f"{label} ({count} segment{'s' if count != 1 else ''})"
            elif count == 1:
                formatted = label
            else:
                formatted = f"{label} ({count} segments)"
            issues.append(formatted)

        return AnalysisResult(
            segment_reports=segment_reports,
            opcode_profiles=opcode_profiles,
            total_instructions=total_instructions,
            total_segments=len(segment_reports),
            issues=issues,
            profile_assessments=assessments,
            stack_observations=stack_observations,
            emulation_quality=emulation_quality,
        )

    def clone_stack_modeler(self) -> StackDeltaModeler:
        """Return a copy of the internal stack modeler state."""

        return self.stack_modeler.clone()

    def _update_profiles(
        self,
        profiles: Dict[str, OpcodeProfile],
        previous: Optional[InstructionWord],
        current: InstructionWord,
        estimate: StackDeltaEstimate,
    ) -> None:
        key = current.label()
        profile = profiles.get(key)
        if profile is None:
            profile = OpcodeProfile(key)
            profiles[key] = profile
        profile.count += 1

        operand = current.operand
        operand_format: Optional[str] = None
        flow_hint = self.knowledge.flow_target_hint(key)
        if flow_hint == "relative":
            signed = operand if operand < 0x8000 else operand - 0x10000
            operand = signed
            operand_format = "relative_word"
        operand_type = categorize_operand(operand, format_hint=operand_format)
        profile.operand_types[operand_type] += 1

        if self.stack_modeler.is_confident(estimate) and estimate.delta is not None:
            stack_key: object = float(estimate.delta)
        else:
            stack_key = "unknown"
        profile.stack_deltas[stack_key] += 1

        if previous is not None:
            prev_key = previous.label()
            profile.preceding[prev_key] += 1
            prev_profile = profiles.get(prev_key)
            if prev_profile is None:
                prev_profile = OpcodeProfile(prev_key)
                profiles[prev_key] = prev_profile
            prev_profile.following[key] += 1

    def _seed_stack_modeler(self, instructions: Sequence[InstructionWord]) -> None:
        seed_stack_modeler_from_knowledge(
            self.stack_modeler,
            self.knowledge,
            instructions,
            seeded_keys=self._seeded_keys,
        )


def categorize_operand(value: int, *, format_hint: Optional[str] = None) -> str:
    if value == 0:
        category = "zero"
    else:
        magnitude = abs(value)
        if magnitude < 0x10:
            category = "small"
        elif magnitude < 0x100:
            category = "tiny"
        elif magnitude < 0x1000:
            category = "medium"
        else:
            category = "large"
    if format_hint:
        return f"{format_hint}:{category}"
    return category
