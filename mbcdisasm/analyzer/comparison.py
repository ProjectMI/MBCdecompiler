"""Utilities for comparing pipeline analysis reports.

Reverse engineering sessions often involve iterating on the analyser heuristics
and checking how the classification changes across several binaries.  The helper
functions defined in this module compute structural diffs between two
:class:`~mbcdisasm.analyzer.report.PipelineReport` objects and highlight drifts in
category assignments, stack deltas and block boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import List, Mapping, Sequence, Tuple

from .instruction_profile import InstructionKind
from .report import PipelineBlock, PipelineReport

__all__ = ["BlockDiff", "CategoryDelta", "ReportDiff", "ReportComparator"]


@dataclass(frozen=True)
class CategoryDelta:
    """Report category count differences between two analyses."""

    category: str
    old_count: int
    new_count: int

    def delta(self) -> int:
        return self.new_count - self.old_count

    def describe(self) -> str:
        return f"{self.category}:{self.old_count}->{self.new_count} (Δ={self.delta():+d})"


@dataclass(frozen=True)
class BlockDiff:
    """Describe how a block changed between two analysis runs."""

    index: int
    old_category: str
    new_category: str
    old_confidence: float
    new_confidence: float
    old_stack: int
    new_stack: int
    message: str

    def describe(self) -> str:
        return (
            f"#{self.index} {self.old_category}->{self.new_category} "
            f"conf {self.old_confidence:.2f}->{self.new_confidence:.2f} "
            f"stack {self.old_stack:+d}->{self.new_stack:+d} : {self.message}"
        )


@dataclass(frozen=True)
class ReportDiff:
    """Summary of differences between two reports."""

    category_changes: Tuple[BlockDiff, ...]
    category_stats: Tuple[CategoryDelta, ...]
    size_delta: int
    stack_delta: int

    def describe(self) -> str:
        lines = [
            f"blocksΔ={self.size_delta:+d}",
            f"stackΔ={self.stack_delta:+d}",
        ]
        if self.category_stats:
            cat_summary = ", ".join(delta.describe() for delta in self.category_stats)
            lines.append(cat_summary)
        for diff in self.category_changes:
            lines.append(diff.describe())
        return " | ".join(lines)


class ReportComparator:
    """Compare two pipeline reports and highlight differences."""

    def compare(self, old: PipelineReport, new: PipelineReport) -> ReportDiff:
        old_blocks = old.blocks
        new_blocks = new.blocks
        size_delta = len(new_blocks) - len(old_blocks)
        stack_delta = (new.total_stack_change() if new else 0) - (old.total_stack_change() if old else 0)

        changes: List[BlockDiff] = []
        for index, pair in enumerate(zip_longest(old_blocks, new_blocks)):
            old_block, new_block = pair
            if old_block is None or new_block is None:
                continue
            if old_block.category == new_block.category and abs(old_block.confidence - new_block.confidence) < 1e-3:
                continue
            message = self._build_message(old_block, new_block)
            changes.append(
                BlockDiff(
                    index=index,
                    old_category=old_block.category,
                    new_category=new_block.category,
                    old_confidence=old_block.confidence,
                    new_confidence=new_block.confidence,
                    old_stack=old_block.stack.change,
                    new_stack=new_block.stack.change,
                    message=message,
                )
            )

        category_stats = self._category_deltas(old_blocks, new_blocks)

        return ReportDiff(
            category_changes=tuple(changes),
            category_stats=tuple(category_stats),
            size_delta=size_delta,
            stack_delta=stack_delta,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_message(self, old: PipelineBlock, new: PipelineBlock) -> str:
        parts = []
        if old.kind != new.kind:
            parts.append(f"kind {old.kind.name}->{new.kind.name}")
        if old.stack.change != new.stack.change:
            parts.append(f"stack {old.stack.change:+d}->{new.stack.change:+d}")
        literal_delta = self._literal_ratio(new) - self._literal_ratio(old)
        if abs(literal_delta) > 0.05:
            parts.append(f"literal_ratioΔ={literal_delta:+.2f}")
        return ", ".join(parts) if parts else "classification drift"

    def _literal_ratio(self, block: PipelineBlock) -> float:
        if not block.profiles:
            return 0.0
        literal_like = 0
        for profile in block.profiles:
            if profile.kind in {
                InstructionKind.LITERAL,
                InstructionKind.MARKER,
                InstructionKind.PUSH,
                InstructionKind.ASCII_CHUNK,
            }:
                literal_like += 1
        return literal_like / len(block.profiles)

    def _category_deltas(
        self, old_blocks: Sequence[PipelineBlock], new_blocks: Sequence[PipelineBlock]
    ) -> List[CategoryDelta]:
        old_histogram = self._category_histogram(old_blocks)
        new_histogram = self._category_histogram(new_blocks)
        categories = sorted(set(old_histogram) | set(new_histogram))
        deltas: List[CategoryDelta] = []
        for category in categories:
            deltas.append(
                CategoryDelta(
                    category=category,
                    old_count=old_histogram.get(category, 0),
                    new_count=new_histogram.get(category, 0),
                )
            )
        return deltas

    def _category_histogram(self, blocks: Sequence[PipelineBlock]) -> Mapping[str, int]:
        histogram: dict[str, int] = {}
        for block in blocks:
            histogram[block.category] = histogram.get(block.category, 0) + 1
        return histogram


def describe_diff(old: PipelineReport, new: PipelineReport) -> str:
    """Return a textual summary describing how ``new`` differs from ``old``."""

    comparator = ReportComparator()
    diff = comparator.compare(old, new)
    return diff.describe()
