"""Aggregated statistics for pipeline analysis runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence

from .instruction_profile import InstructionKind

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from .report import PipelineBlock


@dataclass
class CategoryStats:
    """Aggregated information about a single block category."""

    count: int = 0
    stack_delta: int = 0
    stack_abs: int = 0

    def record(self, block: "PipelineBlock") -> None:
        self.count += 1
        self.stack_delta += block.stack.change
        self.stack_abs += abs(block.stack.change)

    def average_delta(self) -> float:
        if not self.count:
            return 0.0
        return self.stack_delta / self.count

    def describe(self) -> str:
        return f"count={self.count} delta={self.stack_delta:+d}"


@dataclass
class KindStats:
    """Aggregated statistics keyed by :class:`InstructionKind`."""

    counts: Dict[InstructionKind, int] = field(default_factory=dict)

    def record(self, kind: InstructionKind, amount: int = 1) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + amount

    def most_common(self) -> Optional[InstructionKind]:
        if not self.counts:
            return None
        return max(self.counts.items(), key=lambda item: item[1])[0]

    def describe(self) -> str:
        parts = [f"{kind.name}:{count}" for kind, count in sorted(self.counts.items(), key=lambda item: item[1], reverse=True)]
        return "{" + ", ".join(parts) + "}"


@dataclass
class PipelineStatistics:
    """Aggregated view of a full pipeline analysis run."""

    block_count: int
    instruction_count: int
    total_stack_delta: int
    categories: Mapping[str, CategoryStats]
    kinds: KindStats

    def category_ratio(self, category: str) -> float:
        stats = self.categories.get(category)
        if not stats or not self.block_count:
            return 0.0
        return stats.count / self.block_count

    def dominant_category(self) -> Optional[str]:
        if not self.categories:
            return None
        return max(self.categories.items(), key=lambda item: item[1].count)[0]

    def recognised_ratio(self) -> float:
        """Return the fraction of blocks assigned a non-unknown category."""

        if not self.block_count:
            return 0.0
        unknown = self.categories.get("unknown")
        unknown_count = unknown.count if unknown else 0
        recognised = self.block_count - unknown_count
        return recognised / self.block_count

    def describe(self) -> str:
        pieces = [f"blocks={self.block_count}", f"instr={self.instruction_count}", f"stackΔ={self.total_stack_delta:+d}"]
        for category, stats in self.categories.items():
            pieces.append(f"{category}:{stats.count}")
        pieces.append("kinds=" + self.kinds.describe())
        return " ".join(pieces)


class StatisticsBuilder:
    """Compute :class:`PipelineStatistics` from pipeline blocks."""

    def collect(self, blocks: Sequence["PipelineBlock"]) -> PipelineStatistics:
        categories: Dict[str, CategoryStats] = {}
        kind_stats = KindStats()
        instruction_count = 0
        total_delta = 0

        for block in blocks:
            instruction_count += len(block.profiles)
            total_delta += block.stack.change
            categories.setdefault(block.category, CategoryStats()).record(block)
            for profile in block.profiles:
                kind_stats.record(profile.kind)

        return PipelineStatistics(
            block_count=len(blocks),
            instruction_count=instruction_count,
            total_stack_delta=total_delta,
            categories=categories,
            kinds=kind_stats,
        )
