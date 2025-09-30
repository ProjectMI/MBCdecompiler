"""Context building utilities for the pipeline analyser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from .instruction_profile import InstructionProfile
from .stack import StackSummary, StackTracker


@dataclass(frozen=True)
class InstructionWindow:
    """A contiguous window of instruction profiles."""

    profiles: Tuple[InstructionProfile, ...]

    @property
    def start_offset(self) -> int:
        return self.profiles[0].word.offset if self.profiles else 0

    @property
    def end_offset(self) -> int:
        return self.profiles[-1].word.offset if self.profiles else 0

    def stack_summary(self) -> StackSummary:
        tracker = StackTracker()
        return tracker.run(self.profiles)

    def __iter__(self) -> Iterator[InstructionProfile]:
        return iter(self.profiles)

    def __len__(self) -> int:
        return len(self.profiles)


@dataclass
class ControlBoundary:
    """Represents a control flow boundary in the instruction stream."""

    ordinal: int
    position: int
    profile: InstructionProfile

    def describe(self) -> str:
        return f"boundary#{self.ordinal} {self.profile.describe()}"


@dataclass
class LinearBlock:
    """A linear region between two control flow boundaries."""

    start: ControlBoundary
    end: ControlBoundary
    window: InstructionWindow
    stack_summary: StackSummary

    def describe(self) -> str:
        return (
            f"block {self.start.ordinal}->{self.end.ordinal} "
            f"[{self.window.start_offset:08X}-{self.window.end_offset:08X}] "
            f"{self.stack_summary.describe()}"
        )

    def profiles(self) -> Tuple[InstructionProfile, ...]:
        return self.window.profiles


class SegmentContext:
    """Builds :class:`LinearBlock` objects from instruction profiles."""

    def __init__(self, profiles: Sequence[InstructionProfile]) -> None:
        self.profiles = tuple(profiles)
        self.boundaries: List[ControlBoundary] = []
        self.blocks: List[LinearBlock] = []
        self._build()

    def _build(self) -> None:
        boundary_indices = [idx for idx, profile in enumerate(self.profiles) if profile.is_control()]
        if not boundary_indices:
            return

        self.boundaries = [
            ControlBoundary(ordinal=i, position=idx, profile=self.profiles[idx])
            for i, idx in enumerate(boundary_indices)
        ]

        for left, right in zip(self.boundaries, self.boundaries[1:]):
            window = InstructionWindow(self.profiles[left.position + 1 : right.position])
            summary = window.stack_summary()
            self.blocks.append(LinearBlock(start=left, end=right, window=window, stack_summary=summary))

    def iter_blocks(self) -> Iterable[LinearBlock]:
        return iter(self.blocks)

    def summary(self) -> "ContextSummary":
        lengths = [len(block.window.profiles) for block in self.blocks]
        if not lengths:
            return ContextSummary(block_count=0, average_length=0.0, max_length=0)
        average = sum(lengths) / len(lengths)
        maximum = max(lengths)
        return ContextSummary(block_count=len(lengths), average_length=average, max_length=maximum)


@dataclass
class ContextSummary:
    """Small utility structure summarising a segment."""

    block_count: int
    average_length: float
    max_length: int

    def describe(self) -> str:
        return f"blocks={self.block_count} avg={self.average_length:.2f} max={self.max_length}"

    def is_empty(self) -> bool:
        return self.block_count == 0
