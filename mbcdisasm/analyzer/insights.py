"""Generate human readable summaries from pipeline reports.

The high level analyser exposes a detailed :class:`~mbcdisasm.analyzer.report.PipelineReport`
object but in practice engineers often need a condensed overview when scanning
multiple binaries.  The helpers defined here provide textual summaries and table
renderers that highlight the dominant categories, stack usage and refinement
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .report import PipelineBlock, PipelineReport
from .block_refiner import RefinementSummary

__all__ = ["CategoryRow", "ReportTable", "build_report_table", "format_report_table"]


@dataclass(frozen=True)
class CategoryRow:
    """Representation of a single row in the report summary table."""

    category: str
    count: int
    stack_delta: int
    average_confidence: float

    def describe(self) -> str:
        return (
            f"{self.category:<10} count={self.count:<3d} stackΔ={self.stack_delta:+4d} "
            f"conf={self.average_confidence:.2f}"
        )


@dataclass
class ReportTable:
    """Structured representation of a pipeline report summary."""

    rows: List[CategoryRow]
    total_blocks: int
    total_stack_delta: int
    refinement: RefinementSummary | None = None

    def describe(self) -> str:
        lines = [f"blocks={self.total_blocks} stackΔ={self.total_stack_delta:+d}"]
        for row in self.rows:
            lines.append("  " + row.describe())
        if self.refinement:
            lines.append("  refinement " + self.refinement.describe())
        return "\n".join(lines)


def build_report_table(report: PipelineReport) -> ReportTable:
    """Create a :class:`ReportTable` from ``report``."""

    rows: List[CategoryRow] = []
    histogram = _category_histogram(report.blocks)
    stack_totals = _category_stack_delta(report.blocks)
    confidence_totals = _category_confidence(report.blocks)

    for category in sorted(histogram):
        count = histogram[category]
        stack_delta = stack_totals.get(category, 0)
        confidence = 0.0
        if count:
            confidence = confidence_totals.get(category, 0.0) / count
        rows.append(
            CategoryRow(
                category=category,
                count=count,
                stack_delta=stack_delta,
                average_confidence=confidence,
            )
        )

    return ReportTable(
        rows=rows,
        total_blocks=len(report.blocks),
        total_stack_delta=report.total_stack_change(),
        refinement=report.refinement,
    )


def format_report_table(table: ReportTable) -> str:
    """Format ``table`` as a human readable multi-line string."""

    return table.describe()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _category_histogram(blocks: Sequence[PipelineBlock]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for block in blocks:
        histogram[block.category] = histogram.get(block.category, 0) + 1
    return histogram


def _category_stack_delta(blocks: Sequence[PipelineBlock]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for block in blocks:
        totals[block.category] = totals.get(block.category, 0) + block.stack.change
    return totals


def _category_confidence(blocks: Sequence[PipelineBlock]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for block in blocks:
        totals[block.category] = totals.get(block.category, 0.0) + block.confidence
    return totals
