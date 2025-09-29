"""High level Lua AST post-processing helpers.

This module focuses on structural clean-ups that make the reconstructed Lua
source easier to inspect.  The initial translation stage in
``mbcdisasm.highlevel`` deliberately keeps a close mapping to the VM stack so
that the code is simple and deterministic.  While helpful for debugging the
translation itself, the raw output ends up littered with extremely repetitive
literal assignments.  A single bytecode sequence that pushes dozens (or even
hundreds) of literal operands therefore produces a wall of ``local literal_X =``
statements which obscures the actual control-flow logic surrounding the run.

The optimiser implemented here performs two main tasks:

``LiteralRunCompactor``
    Scans a block of Lua statements and replaces consecutive literal
    assignments with a single multi-assignment.  The transformation preserves
    evaluation order and reuse of stack slots while drastically shrinking the
    textual footprint.  Optionally a short explanatory comment is injected to
    document the size and composition of the run.  This retains the semantic
    intent of the literal pushes while making it obvious where large constant
    tables originate from.

``LuaStatementOptimizer``
    Convenience facade that wires the individual passes together and exposes a
    stable interface to the reconstruction pipeline.  The optimiser returns the
    rewritten statements alongside rich metadata describing the literal runs it
    collapsed.  Callers can surface this information in summaries or attach
    additional diagnostics without having to duplicate the grouping logic.

The implementation is intentionally verbose: the optimiser doubles as a
catalogue of heuristics that future passes can extend.  Detailed docstrings and
type annotations make the behaviour explicit which in turn simplifies writing
unit tests that exercise edge cases (comments between literals, mixed literal
kinds, non-local assignments, …).
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .lua_ast import Assignment, CommentStatement, LiteralExpr, LuaStatement, MultiAssignment, NameExpr
from .lua_literals import LuaLiteral


# ---------------------------------------------------------------------------
# utility data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiteralAssignment:
    """Describe a single literal assignment encountered during optimisation.

    The optimiser manipulates :class:`~mbcdisasm.lua_ast.Assignment` nodes
    directly.  Keeping a small data carrier around avoids repeatedly poking at
    the AST during grouping and ensures we preserve identity of the expression
    objects (important because other statements may refer to the exact
    :class:`~mbcdisasm.lua_ast.NameExpr` instance).
    """

    statement: Assignment
    prefix: str
    index: Optional[int]
    literal: LuaLiteral

    @property
    def target(self) -> NameExpr:
        return self.statement.targets[0]  # pragma: no cover - enforced by caller

    @property
    def kind(self) -> str:
        return self.literal.kind


@dataclass
class LiteralRunSummary:
    """Human readable summary of a collapsed literal run.

    The reconstruction pipeline renders function summaries before the actual
    Lua body.  Collapsing literal runs changes the layout of the function in a
    non-trivial way, therefore we capture enough context to explain the
    optimisation to the reader:

    * ``prefix`` records the symbolic prefix of the generated locals
      (``literal``/``string``/etc.).
    * ``first`` and ``last`` denote the boundaries of the run which makes it
      straightforward to map back to the original VM stack slots.
    * ``count`` stores how many individual literal pushes the run represents.
    * ``kind_breakdown`` tallies the literal kinds present in the sequence; it
      is emitted in a stable order to keep diffs readable.
    """

    prefix: str
    first: str
    last: str
    count: int
    kind_breakdown: List[Tuple[str, int]] = field(default_factory=list)
    contiguous: bool = True
    value_sample: List[str] = field(default_factory=list)
    numeric_range: Optional[Tuple[int, int]] = None
    string_preview: Optional[str] = None
    byte_size: int = 0

    def to_comment(self) -> str:
        """Return a single line comment describing the run.

        Comments are intentionally short – the detailed numbers surface in the
        function metadata block.  The intent is to provide just enough context
        next to the multi-assignment so that readers do not have to scroll back
        to the top of the file while following control flow.
        """

        details = ", ".join(f"{kind}={count}" for kind, count in self.kind_breakdown)
        pieces = [f"literal run ({self.count} values"]
        if details:
            pieces[-1] += f"; {details}"
        if not self.contiguous:
            pieces[-1] += "; gaps present"
        sample = self.format_sample(limit=3)
        if sample:
            pieces[-1] += f"; sample: {sample}"
        if self.byte_size:
            pieces[-1] += f"; {self.byte_size} bytes"
        pieces[-1] += ")"
        return pieces[0]

    def to_metadata_line(self) -> str:
        """Render the run in a bullet-list friendly format."""

        details = ", ".join(f"{kind}={count}" for kind, count in self.kind_breakdown)
        suffix = ""
        if details:
            suffix = f"; {details}"
        extras: List[str] = []
        if not self.contiguous:
            extras.append("gaps present")
        sample = self.format_sample()
        if sample:
            extras.append(f"sample: {sample}")
        range_text = self.format_range()
        if range_text:
            extras.append(f"range: {range_text}")
        preview = self.format_preview()
        if preview:
            extras.append(f"preview: {preview}")
        if self.byte_size:
            extras.append(f"bytes: {self.byte_size}")
        if extras:
            if suffix:
                suffix += "; " + "; ".join(extras)
            else:
                suffix = "; " + "; ".join(extras)
        elif suffix:
            pass
        return f"- {self.prefix}: {self.first}..{self.last} ({self.count} values{suffix})"

    def format_sample(self, limit: int = 6) -> Optional[str]:
        if not self.value_sample:
            return None
        head = self.value_sample[:limit]
        text = ", ".join(head)
        if len(self.value_sample) > limit:
            text += ", …"
        return text

    def format_range(self) -> Optional[str]:
        if not self.numeric_range:
            return None
        low, high = self.numeric_range
        if low == high:
            return str(low)
        return f"{low}..{high}"

    def format_preview(self, limit: int = 64) -> Optional[str]:
        if not self.string_preview:
            return None
        if len(self.string_preview) <= limit:
            return self.string_preview
        return self.string_preview[: limit - 1] + "…"

    def additional_metadata_lines(self, indent: str = "  ") -> List[str]:
        lines: List[str] = []
        range_text = self.format_range()
        preview_text = self.format_preview()
        sample_text = self.format_sample()
        if sample_text:
            lines.append(f"{indent}sample: {sample_text}")
        if range_text:
            lines.append(f"{indent}numeric range: {range_text}")
        if preview_text:
            lines.append(f"{indent}string preview: {preview_text}")
        if self.byte_size:
            lines.append(f"{indent}estimated bytes: {self.byte_size}")
        if not self.contiguous:
            lines.append(f"{indent}note: non-contiguous indices")
        return lines

    def to_dict(self) -> Dict[str, object]:
        return {
            "prefix": self.prefix,
            "first": self.first,
            "last": self.last,
            "count": self.count,
            "kinds": {kind: count for kind, count in self.kind_breakdown},
            "contiguous": self.contiguous,
            "sample": list(self.value_sample),
            "numeric_range": self.numeric_range,
            "string_preview": self.string_preview,
            "byte_size": self.byte_size,
        }


# ---------------------------------------------------------------------------
# literal run compactor
# ---------------------------------------------------------------------------


class LiteralRunCompactor:
    """Collapse consecutive literal assignments into multi-assignments.

    Parameters
    ----------
    min_run_length:
        Minimum number of consecutive literal assignments required before a run
        is compacted.  Shorter runs are left untouched to avoid producing
        multi-assignments for just two locals where the readability improvement
        is debatable.
    comment_threshold:
        Emit a descriptive comment when the run size meets or exceeds this
        value.  Large literal blobs are frequently the result of string or
        table initialisers encoded in bytecode; providing a succinct summary
        next to the multi-assignment helps readers orient themselves quickly.
    """

    def __init__(self, *, min_run_length: int = 3, comment_threshold: int = 5) -> None:
        self._min_run_length = max(2, min_run_length)
        self._comment_threshold = max(0, comment_threshold)

    # ------------------------------------------------------------------
    def compact(
        self, statements: Sequence[LuaStatement]
    ) -> Tuple[List[LuaStatement], List[LiteralRunSummary]]:
        """Return optimised statements and metadata about the collapsed runs."""

        result: List[LuaStatement] = []
        summaries: List[LiteralRunSummary] = []
        buffer: List[LiteralAssignment] = []

        for statement in statements:
            literal = self._extract_literal(statement)
            if literal is None:
                result.extend(self._flush(buffer, summaries))
                buffer.clear()
                result.append(statement)
                continue

            assignment, literal_value = literal
            prefix, index = self._split_name(assignment.targets[0].name)
            if buffer and buffer[-1].prefix != prefix:
                result.extend(self._flush(buffer, summaries))
                buffer.clear()

            buffer.append(
                LiteralAssignment(
                    statement=assignment,
                    prefix=prefix,
                    index=index,
                    literal=literal_value,
                )
            )

        result.extend(self._flush(buffer, summaries))
        return result, summaries

    # ------------------------------------------------------------------
    def _extract_literal(
        self, statement: LuaStatement
    ) -> Optional[Tuple[Assignment, LuaLiteral]]:
        if not isinstance(statement, Assignment):
            return None
        if not statement.is_local:
            return None
        if len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        if not isinstance(target, NameExpr):
            return None
        value = statement.value
        if not isinstance(value, LiteralExpr):
            return None
        literal = value.literal
        if literal is None:
            return None
        return statement, literal

    # ------------------------------------------------------------------
    @staticmethod
    def _split_name(name: str) -> Tuple[str, Optional[int]]:
        if "_" not in name:
            return name, None
        prefix, _, suffix = name.partition("_")
        try:
            return prefix, int(suffix)
        except ValueError:  # pragma: no cover - defensive
            return prefix, None

    # ------------------------------------------------------------------
    def _flush(
        self,
        buffer: List[LiteralAssignment],
        summaries: List[LiteralRunSummary],
    ) -> List[LuaStatement]:
        if len(buffer) < self._min_run_length:
            return [entry.statement for entry in buffer]

        first = buffer[0]
        last = buffer[-1]
        targets = [entry.target for entry in buffer]
        values = [entry.statement.value for entry in buffer]

        multi = MultiAssignment(targets, values, is_local=buffer[0].statement.is_local)
        emitted: List[LuaStatement] = []

        summary = self._summarise(buffer)
        summaries.append(summary)
        if len(buffer) >= self._comment_threshold and self._comment_threshold > 0:
            emitted.append(CommentStatement(summary.to_comment()))
        emitted.append(multi)
        return emitted

    # ------------------------------------------------------------------
    def _summarise(self, buffer: Sequence[LiteralAssignment]) -> LiteralRunSummary:
        prefix = buffer[0].prefix
        first_name = buffer[0].target.name
        last_name = buffer[-1].target.name
        breakdown = _count_literal_kinds(entry.literal for entry in buffer)
        contiguous = _is_contiguous(entry.index for entry in buffer)
        sample, numeric_range, preview, byte_size = _collect_summary_stats(entry.literal for entry in buffer)
        return LiteralRunSummary(
            prefix=prefix,
            first=first_name,
            last=last_name,
            count=len(buffer),
            kind_breakdown=breakdown,
            contiguous=contiguous,
            value_sample=sample,
            numeric_range=numeric_range,
            string_preview=preview,
            byte_size=byte_size,
        )


# ---------------------------------------------------------------------------
# orchestration facade
# ---------------------------------------------------------------------------


def _count_literal_kinds(literals: Iterable[LuaLiteral]) -> List[Tuple[str, int]]:
    counts: dict[str, int] = {}
    for literal in literals:
        counts[literal.kind] = counts.get(literal.kind, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0])


def _is_contiguous(indices: Iterable[Optional[int]]) -> bool:
    materialised = [index for index in indices if index is not None]
    if not materialised:
        return True
    sorted_indices = sorted(materialised)
    start = sorted_indices[0]
    for offset, value in enumerate(sorted_indices):
        if value != start + offset:
            return False
    return True


def _collect_summary_stats(
    literals: Iterable[LuaLiteral],
    *,
    sample_limit: int = 16,
    preview_limit: int = 128,
) -> Tuple[List[str], Optional[Tuple[int, int]], Optional[str], int]:
    sample: List[str] = []
    numeric_values: List[int] = []
    string_parts: List[str] = []
    total_string_length = 0
    byte_size = 0
    for literal in literals:
        if len(sample) < sample_limit:
            sample.append(literal.text)
        if literal.kind == "number" and isinstance(literal.value, int):
            numeric_values.append(literal.value)
            byte_size += 2
        if literal.kind == "string":
            piece = str(literal.value)
            if piece:
                byte_size += len(piece)
                remaining = preview_limit - total_string_length
                if remaining <= 0:
                    continue
                string_parts.append(piece[:remaining])
                total_string_length += min(len(piece), remaining)
        if literal.kind not in {"number", "string"}:
            byte_size += 2
    numeric_range: Optional[Tuple[int, int]] = None
    if numeric_values:
        numeric_range = (min(numeric_values), max(numeric_values))
    string_preview = "".join(string_parts) or None
    return sample, numeric_range, string_preview, byte_size


@dataclass
class PrefixStatistics:
    prefix: str
    run_count: int = 0
    byte_count: int = 0

    def register(self, byte_size: int) -> None:
        self.run_count += 1
        self.byte_count += byte_size

    def merge(self, other: "PrefixStatistics") -> None:
        if self.prefix != other.prefix:
            raise ValueError("cannot merge prefixes with different labels")
        self.run_count += other.run_count
        self.byte_count += other.byte_count

    def summary(self) -> str:
        return f"{self.prefix}: {self.run_count} runs; {self.byte_count} bytes"


class LiteralRunRegistry:
    """Collect literal run summaries for module-wide reporting."""

    def __init__(self) -> None:
        self._runs: List[LiteralRunSummary] = []
        self._prefix_totals: Dict[str, PrefixStatistics] = {}

    def clear(self) -> None:
        self._runs.clear()
        self._prefix_totals.clear()

    def register(self, summary: LiteralRunSummary) -> None:
        self._runs.append(summary)
        stats = self._prefix_totals.get(summary.prefix)
        if stats is None:
            stats = PrefixStatistics(prefix=summary.prefix)
            self._prefix_totals[summary.prefix] = stats
        stats.register(summary.byte_size)

    def register_many(self, summaries: Iterable[LiteralRunSummary]) -> None:
        for summary in summaries:
            self.register(summary)

    def total_runs(self) -> int:
        return len(self._runs)

    def total_bytes(self) -> int:
        return sum(summary.byte_size for summary in self._runs)

    def prefixes_by_bytes(self) -> List[PrefixStatistics]:
        totals = list(self._prefix_totals.values())
        totals.sort(key=lambda item: item.byte_count, reverse=True)
        return totals

    def prefix_statistics(self, prefix: str) -> PrefixStatistics:
        stats = self._prefix_totals.get(prefix)
        if stats is None:
            return PrefixStatistics(prefix=prefix)
        return PrefixStatistics(prefix=stats.prefix, run_count=stats.run_count, byte_count=stats.byte_count)

    def runs_for_prefix(self, prefix: str, *, limit: Optional[int] = None) -> List[LiteralRunSummary]:
        matches = [run for run in self._runs if run.prefix == prefix]
        if limit is not None:
            return matches[:limit]
        return matches

    def longest_runs(self, limit: int = 5) -> List[LiteralRunSummary]:
        runs = sorted(self._runs, key=lambda item: item.count, reverse=True)
        return runs[:limit]

    def to_dict(self, *, prefix_limit: int = 10, run_limit: int = 50) -> Dict[str, object]:
        return {
            "total_runs": self.total_runs(),
            "total_bytes": self.total_bytes(),
            "prefix_totals": [
                {
                    "prefix": stats.prefix,
                    "runs": stats.run_count,
                    "bytes": stats.byte_count,
                }
                for stats in self.prefixes_by_bytes()[:prefix_limit]
            ],
            "runs": [summary.to_dict() for summary in self._runs[:run_limit]],
        }

    def summary_lines(self, prefix: str = "literal registry overview", limit: int = 5) -> List[str]:
        if not self._runs:
            return []
        lines = [f"- {prefix}:"]
        totals = self.prefixes_by_bytes()
        for stats in totals[:limit]:
            lines.append(f"  - {stats.summary()}")
        if len(totals) > limit:
            remaining = len(totals) - limit
            lines.append(f"  - ... ({remaining} additional prefixes)")
        longest = self.longest_runs(limit)
        if longest:
            lines.append("  - largest runs:")
            for entry in longest:
                details = entry.to_metadata_line().lstrip("- ")
                lines.append(f"    * {details}")
        return lines


class LuaStatementOptimizer:
    """Apply a collection of clean-up passes to Lua statements.

    The class currently only runs :class:`LiteralRunCompactor` but is structured
    so that additional passes can be slotted in easily.  Each pass returns both
    rewritten statements and metadata which we merge together.  Downstream
    components receive a coherent view of the transformed code without having
    to understand individual pass implementations.
    """

    def __init__(self) -> None:
        self._literal_compactor = LiteralRunCompactor()

    def optimise(
        self, statements: Sequence[LuaStatement]
    ) -> Tuple[List[LuaStatement], List[LiteralRunSummary]]:
        rewritten, summaries = self._literal_compactor.compact(statements)
        return rewritten, summaries


class LiteralRunReporter:
    """Render :class:`LiteralRunRegistry` contents as text or JSON."""

    def __init__(self, registry: LiteralRunRegistry) -> None:
        self._registry = registry

    def as_text(self, *, limit: int = 5) -> str:
        lines = ["literal runs report"]
        lines.append(f"total runs: {self._registry.total_runs()}")
        lines.append(f"total bytes: {self._registry.total_bytes()}")
        prefix_lines = self._registry.summary_lines(limit=limit)
        if prefix_lines:
            lines.extend(prefix_lines)
        else:
            lines.append("- no literal runs recorded")
        return "\n".join(lines)

    def as_json(self, *, limit: int = 20) -> str:
        payload = {
            "total_runs": self._registry.total_runs(),
            "total_bytes": self._registry.total_bytes(),
            "prefix_totals": [
                {
                    "prefix": stats.prefix,
                    "runs": stats.run_count,
                    "bytes": stats.byte_count,
                }
                for stats in self._registry.prefixes_by_bytes()[:limit]
            ],
            "largest_runs": [summary.to_dict() for summary in self._registry.longest_runs(limit)],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def as_markdown_table(self, *, limit: int = 5) -> str:
        totals = self._registry.prefixes_by_bytes()[:limit]
        if not totals:
            return "| prefix | runs | bytes |\n| --- | --- | --- |\n"
        rows = ["| prefix | runs | bytes |", "| --- | --- | --- |"]
        for stats in totals:
            rows.append(f"| {stats.prefix} | {stats.run_count} | {stats.byte_count} |")
        return "\n".join(rows)


class LiteralRunAnalyzer:
    """Compute aggregate statistics across literal runs."""

    def __init__(self, registry: LiteralRunRegistry) -> None:
        self._registry = registry

    def run_length_histogram(self) -> Dict[int, int]:
        histogram: Dict[int, int] = {}
        for run in self._registry._runs:
            histogram[run.count] = histogram.get(run.count, 0) + 1
        return dict(sorted(histogram.items()))

    def byte_statistics(self) -> Dict[str, float]:
        if not self._registry._runs:
            return {"min": 0.0, "max": 0.0, "average": 0.0}
        sizes = [run.byte_size for run in self._registry._runs]
        total = sum(sizes)
        minimum = min(sizes)
        maximum = max(sizes)
        average = total / len(sizes)
        return {"min": float(minimum), "max": float(maximum), "average": float(average)}

    def kind_distribution(self) -> Dict[str, int]:
        distribution: Dict[str, int] = {}
        for run in self._registry._runs:
            for kind, count in run.kind_breakdown:
                distribution[kind] = distribution.get(kind, 0) + count
        return distribution

    def prefix_summary(self, prefix: str) -> Dict[str, float]:
        runs = self._registry.runs_for_prefix(prefix)
        if not runs:
            return {"count": 0.0, "bytes": 0.0, "average_length": 0.0}
        total_bytes = sum(run.byte_size for run in runs)
        average_length = sum(run.count for run in runs) / len(runs)
        return {
            "count": float(len(runs)),
            "bytes": float(total_bytes),
            "average_length": float(average_length),
        }

    def byte_percentiles(self, percentiles: Sequence[float] = (0.25, 0.5, 0.75)) -> Dict[str, float]:
        if not self._registry._runs:
            return {f"p{int(p*100)}": 0.0 for p in percentiles}
        values = sorted(run.byte_size for run in self._registry._runs)
        results: Dict[str, float] = {}
        for percentile in percentiles:
            index = percentile * (len(values) - 1)
            lower = int(index)
            upper = min(lower + 1, len(values) - 1)
            fraction = index - lower
            estimate = values[lower] * (1 - fraction) + values[upper] * fraction
            results[f"p{int(percentile * 100)}"] = float(estimate)
        return results

    def describe(self) -> str:
        lines = ["literal run analysis"]
        histogram = self.run_length_histogram()
        if histogram:
            lines.append("length histogram:")
            for length, count in histogram.items():
                lines.append(f"  len={length}: {count}")
        stats = self.byte_statistics()
        lines.append(
            f"byte statistics: min={stats['min']:.0f}, max={stats['max']:.0f}, avg={stats['average']:.2f}"
        )
        percentiles = self.byte_percentiles()
        if percentiles:
            formatted = ", ".join(f"{name}={value:.0f}" for name, value in sorted(percentiles.items()))
            lines.append(f"byte percentiles: {formatted}")
        distribution = self.kind_distribution()
        if distribution:
            lines.append("literal kinds:")
            for kind, count in sorted(distribution.items()):
                lines.append(f"  {kind}: {count}")
        return "\n".join(lines)


def merge_registries(registries: Sequence[LiteralRunRegistry]) -> LiteralRunRegistry:
    """Return a registry containing runs from all ``registries``."""

    merged = LiteralRunRegistry()
    for registry in registries:
        merged.register_many(registry._runs)
    return merged


__all__ = [
    "LiteralRunSummary",
    "LiteralRunCompactor",
    "LuaStatementOptimizer",
    "LiteralRunRegistry",
    "LiteralRunReporter",
    "LiteralRunAnalyzer",
    "merge_registries",
]

