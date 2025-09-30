"""Reporting structures for pipeline analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from .instruction_profile import InstructionKind, InstructionProfile, dominant_kind
from .patterns import PatternMatch
from .stack import StackSummary
from .stats import PipelineStatistics


@dataclass
class PipelineBlock:
    """Description of a single pipeline block."""

    profiles: Tuple[InstructionProfile, ...]
    stack: StackSummary
    kind: InstructionKind
    category: str
    pattern: Optional[PatternMatch] = None
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def start_offset(self) -> int:
        return self.profiles[0].word.offset if self.profiles else 0

    @property
    def end_offset(self) -> int:
        return self.profiles[-1].word.offset if self.profiles else 0

    def describe(self) -> str:
        pattern = self.pattern.pattern.name if self.pattern else "?"
        return (
            f"[{self.start_offset:08X}-{self.end_offset:08X}] "
            f"kind={self.kind.name} cat={self.category} "
            f"stack={self.stack.describe()} pattern={pattern} conf={self.confidence:.2f}"
        )

    def histogram(self) -> dict[InstructionKind, int]:
        counts: dict[InstructionKind, int] = {}
        for profile in self.profiles:
            counts[profile.kind] = counts.get(profile.kind, 0) + 1
        return counts

    def add_note(self, message: str) -> None:
        self.notes.append(message)

    def as_dict(self) -> dict[str, object]:
        return {
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "kind": self.kind.name,
            "category": self.category,
            "stack_change": self.stack.change,
            "stack_min": self.stack.minimum,
            "stack_max": self.stack.maximum,
            "confidence": self.confidence,
            "pattern": self.pattern.pattern.name if self.pattern else None,
            "notes": list(self.notes),
        }


@dataclass
class PipelineReport:
    """Container for all blocks extracted from a segment."""

    blocks: Tuple[PipelineBlock, ...]
    warnings: Tuple[str, ...] = tuple()
    statistics: Optional[PipelineStatistics] = None

    def describe(self) -> str:
        lines = ["Pipeline analysis report:"]
        for block in self.blocks:
            lines.append("  " + block.describe())
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append("  - " + warning)
        if self.statistics:
            lines.append("Statistics: " + self.statistics.describe())
        return "\n".join(lines)

    def filter_by_category(self, category: str) -> Tuple[PipelineBlock, ...]:
        return tuple(block for block in self.blocks if block.category == category)

    def total_stack_change(self) -> int:
        return sum(block.stack.change for block in self.blocks)

    @classmethod
    def empty(cls) -> "PipelineReport":
        return cls(blocks=tuple(), warnings=tuple(), statistics=None)


def build_block(
    profiles: Sequence[InstructionProfile],
    stack: StackSummary,
    pattern: Optional[PatternMatch],
    category: str,
    confidence: float,
    notes: Optional[Iterable[str]] = None,
) -> PipelineBlock:
    dominant = dominant_kind(profiles)
    block = PipelineBlock(
        profiles=tuple(profiles),
        stack=stack,
        kind=dominant,
        category=category,
        pattern=pattern,
        confidence=confidence,
    )
    for note in notes or []:
        block.add_note(note)
    return block
