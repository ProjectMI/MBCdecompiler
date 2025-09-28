"""Extremely lightweight stack emulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .instruction import InstructionWord, read_instructions
from .knowledge import KnowledgeBase
from .manual_semantics import AnnotatedInstruction, ManualSemanticAnalyzer
from .mbc import Segment
from .stack_model import StackDeltaEstimate, StackDeltaModeler
from .stack_seed import seed_stack_modeler_from_knowledge


@dataclass
class InstructionTrace:
    instruction: AnnotatedInstruction
    stack_before: float
    stack_after: float
    warnings: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        warn = f" ({', '.join(self.warnings)})" if self.warnings else ""
        word = self.instruction.word
        semantics = self.instruction.semantics
        return (
            f"{word.offset:08X}: {word.raw:08X} {semantics.manual_name}"
            f" stack {self.stack_before:6.1f} -> {self.stack_after:6.1f}" + warn
        )


@dataclass
class EmulationReport:
    segment_index: int
    start: int
    traces: List[InstructionTrace]
    remainder: int
    final_stack: float
    min_stack: float
    max_stack: float
    total_instructions: int
    unknown_delta_instructions: int
    unknown_delta_ratio: float
    is_reliable: bool

    def to_text(self) -> str:
        lines = [
            f"segment {self.segment_index} stack emulation (start=0x{self.start:06X})",
            "  "
            + " ".join(
                filter(
                    None,
                    [
                        f"final={self.final_stack:.1f}",
                        f"min={self.min_stack:.1f}",
                        f"max={self.max_stack:.1f}",
                        (
                            f"unknown-delta={self.unknown_delta_instructions}/"
                            f"{self.total_instructions}"
                            f" ({self.unknown_delta_ratio * 100:.1f}%)"
                            if self.total_instructions
                            else None
                        ),
                        (
                            "reliability=ok"
                            if self.is_reliable
                            else "reliability=low"
                        ),
                    ],
                )
            ),
        ]
        for trace in self.traces:
            lines.append("  " + trace.to_text())
        if self.remainder:
            lines.append(f"  trailing bytes ignored: {self.remainder}")
        return "\n".join(lines) + "\n"


class Emulator:
    """Simulate stack deltas using the knowledge base statistics."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        initial_stack: float = 0.0,
        *,
        unknown_ratio_threshold: float = 0.25,
        stack_modeler: Optional[StackDeltaModeler] = None,
        model_confidence_threshold: Optional[float] = None,
        semantic_analyzer: Optional[ManualSemanticAnalyzer] = None,
    ) -> None:
        self.knowledge = knowledge
        self.initial_stack = initial_stack
        self.unknown_ratio_threshold = unknown_ratio_threshold
        self.stack_modeler = stack_modeler or StackDeltaModeler()
        if model_confidence_threshold is None:
            model_confidence_threshold = self.stack_modeler.confidence_threshold
        self.model_confidence_threshold = model_confidence_threshold
        self._seeded_keys: set[str] = set()
        self._semantic_analyzer = semantic_analyzer or ManualSemanticAnalyzer(knowledge)

    def simulate_segment(
        self,
        segment: Segment,
        *,
        max_instructions: Optional[int] = None,
    ) -> EmulationReport:
        raw_instructions, remainder = read_instructions(segment.data, segment.start)
        if max_instructions is not None:
            raw_instructions = raw_instructions[:max_instructions]

        annotated = [
            self._semantic_analyzer.describe_word(instr) for instr in raw_instructions
        ]
        instructions = [item.word for item in annotated]

        seed_stack_modeler_from_knowledge(
            self.stack_modeler,
            self.knowledge,
            instructions,
            seeded_keys=self._seeded_keys,
        )

        stack = float(self.initial_stack)
        traces: List[InstructionTrace] = []
        min_stack = stack
        max_stack = stack
        total_instructions = 0
        unknown_delta_instructions = 0

        self._seed_stack_modeler(instructions)
        estimates = self.stack_modeler.model_segment(instructions)

        for index, annotated_instr in enumerate(annotated):
            instr = annotated_instr.word
            semantics = annotated_instr.semantics
            estimate = estimates[index] if index < len(estimates) else StackDeltaEstimate(
                key=instr.label(), delta=None, confidence=0.0
            )
            warnings: List[str] = []
            total_instructions += 1
            if estimate.delta is None:
                manual_delta = semantics.stack_effect.delta
                if manual_delta is not None:
                    delta_value = float(manual_delta)
                    warnings.append("manual-delta")
                else:
                    delta_value = 0.0
                    warnings.append("unknown-delta")
                    unknown_delta_instructions += 1
            else:
                delta_value = float(estimate.delta)
                if estimate.confidence < self.model_confidence_threshold:
                    warnings.append("model-low-confidence")
                    unknown_delta_instructions += 1
            stack_after = stack + delta_value
            if stack_after < 0:
                warnings.append("underflow")
            traces.append(
                InstructionTrace(
                    instruction=annotated_instr,
                    stack_before=stack,
                    stack_after=stack_after,
                    warnings=warnings,
                )
            )
            stack = stack_after
            min_stack = min(min_stack, stack)
            max_stack = max(max_stack, stack)

        if total_instructions:
            unknown_ratio = unknown_delta_instructions / total_instructions
        else:
            unknown_ratio = 0.0
        is_reliable = unknown_ratio <= self.unknown_ratio_threshold

        return EmulationReport(
            segment_index=segment.index,
            start=segment.start,
            traces=traces,
            remainder=remainder,
            final_stack=stack,
            min_stack=min_stack,
            max_stack=max_stack,
            total_instructions=total_instructions,
            unknown_delta_instructions=unknown_delta_instructions,
            unknown_delta_ratio=unknown_ratio,
            is_reliable=is_reliable,
        )

    def simulate_container(
        self,
        segments: Iterable[Segment],
        *,
        max_instructions: Optional[int] = None,
    ) -> List[EmulationReport]:
        reports: List[EmulationReport] = []
        for segment in segments:
            reports.append(
                self.simulate_segment(segment, max_instructions=max_instructions)
            )
        return reports

    def _seed_stack_modeler(self, instructions: Sequence[InstructionWord]) -> None:
        for instr in instructions:
            key = instr.label()
            if key in self._seeded_keys:
                continue
            if self.stack_modeler.known_delta(key) is not None:
                self._seeded_keys.add(key)
                continue
            estimate = self.knowledge.estimate_stack_delta(key)
            if estimate is None:
                continue
            self.stack_modeler.seed_known_delta(key, float(estimate))
            self._seeded_keys.add(key)


def render_reports(reports: Sequence[EmulationReport]) -> str:
    lines = []
    for report in reports:
        lines.append(report.to_text().rstrip())
    return "\n\n".join(lines) + "\n"


def write_emulation_reports(
    reports: Sequence[EmulationReport], path: Path, *, encoding: str = "utf-8"
) -> None:
    """Persist the rendered emulation reports to ``path``."""

    path.write_text(render_reports(reports), encoding)
