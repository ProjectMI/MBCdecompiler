"""Helpers for tracking literal push sequences during reconstruction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .lua_ast import LuaExpression, LiteralExpr, NameExpr
from .lua_literals import LuaLiteral, escape_lua_string


@dataclass(frozen=True)
class LiteralDescriptor:
    """Describes a single literal value encountered while decoding."""

    kind: str
    text: str
    expression: LuaExpression

    def is_string(self) -> bool:
        return self.kind == "string"

    def is_numeric(self) -> bool:
        return self.kind == "number"


@dataclass(frozen=True)
class LiteralRun:
    """Collection of consecutive literal pushes."""

    kind: str
    descriptors: Tuple[LiteralDescriptor, ...]
    offsets: Tuple[int, ...]
    block_start: int

    def __post_init__(self) -> None:
        if not self.descriptors:
            raise ValueError("literal run must contain at least one descriptor")
        if len(self.descriptors) != len(self.offsets):
            raise ValueError("descriptors/offsets mismatch")

    def length(self) -> int:
        return len(self.descriptors)

    def start_offset(self) -> int:
        return self.offsets[0]

    def end_offset(self) -> int:
        return self.offsets[-1]

    def combined_string(self) -> Optional[str]:
        if self.kind != "string":
            return None
        return "".join(descriptor.text for descriptor in self.descriptors)

    def numeric_values(self) -> Sequence[str]:
        if self.kind != "number":
            return ()
        return [descriptor.text for descriptor in self.descriptors]

    def render_preview(self, limit: int = 64) -> str:
        if self.kind == "string":
            combined = self.combined_string() or ""
            escaped = escape_lua_string(combined)
            if len(escaped) <= limit:
                return escaped
            return escaped[: limit - 3] + "..."
        if self.kind == "number":
            values = self.numeric_values()
            if len(values) <= 4:
                return ", ".join(values)
            return ", ".join(values[:4]) + " ..."
        return ", ".join(descriptor.text for descriptor in self.descriptors)

    def describe(self) -> str:
        preview = self.render_preview()
        return (
            f"kind={self.kind} count={self.length()}"
            f" start=0x{self.start_offset():06X} preview={preview}"
        )


class LiteralRunTracker:
    """Track literal push runs while translating blocks."""

    def __init__(self) -> None:
        self._runs: List[LiteralRun] = []
        self._buffer: List[LiteralDescriptor] = []
        self._offsets: List[int] = []
        self._current_kind: Optional[str] = None
        self._current_block: int = 0

    # ------------------------------------------------------------------
    def start_block(self, block_start: int) -> None:
        if self._buffer:
            self._flush()
        self._current_block = block_start

    def observe(self, offset: int, expression: LuaExpression) -> None:
        descriptor = _describe_expression(expression)
        if descriptor is None:
            self.break_sequence()
            return
        kind = descriptor.kind
        if self._buffer and self._current_kind != kind:
            self._flush()
        self._buffer.append(descriptor)
        self._offsets.append(offset)
        self._current_kind = kind

    def break_sequence(self) -> None:
        self._flush()

    def finalize(self) -> None:
        self._flush()

    # ------------------------------------------------------------------
    def runs(self) -> Sequence[LiteralRun]:
        return tuple(self._runs)

    def extend(self, runs: Iterable[LiteralRun]) -> None:
        self._runs.extend(runs)

    # ------------------------------------------------------------------
    def _flush(self) -> None:
        if not self._buffer:
            return
        run = LiteralRun(
            kind=self._current_kind or "unknown",
            descriptors=tuple(self._buffer),
            offsets=tuple(self._offsets),
            block_start=self._current_block,
        )
        self._runs.append(run)
        self._buffer.clear()
        self._offsets.clear()
        self._current_kind = None


def _describe_expression(expression: LuaExpression) -> Optional[LiteralDescriptor]:
    if isinstance(expression, LiteralExpr):
        literal = _resolve_literal(expression)
        if literal is not None:
            kind = literal.kind
            text = str(literal.value) if literal.kind == "string" else literal.render()
            return LiteralDescriptor(kind=kind, text=text, expression=expression)
        if isinstance(expression.value, str):
            return LiteralDescriptor(kind="string", text=expression.value, expression=expression)
    if isinstance(expression, NameExpr):
        return LiteralDescriptor(kind="name", text=expression.name, expression=expression)
    return None


def _resolve_literal(expression: LiteralExpr) -> Optional[LuaLiteral]:
    literal = expression.literal
    if literal is not None:
        return literal
    value = expression.value
    if isinstance(value, LuaLiteral):
        return value
    return None


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class StringLiteralStats:
    """Aggregate statistics derived from string literal runs."""

    run_count: int
    total_length: int
    average_length: float
    longest_run: Optional[LiteralRun]
    top_runs: Tuple[LiteralRun, ...]
    token_frequency: Dict[str, int]
    char_frequency: Dict[str, int]
    run_length_histogram: Dict[int, int]
    token_length_histogram: Dict[int, int]
    token_length_average: float

    def summary_lines(self, *, limit: int = 5) -> List[str]:
        if self.run_count == 0:
            return []
        lines = [
            f"string runs: count={self.run_count} avg_len={self.average_length:.2f} total_len={self.total_length}",
            f"unique tokens={len(self.token_frequency)} unique chars={len(self.char_frequency)}",
            f"run lengths: {self._format_histogram(self.run_length_histogram, limit)}",
            f"token lengths: avg={self.token_length_average:.2f} {self._format_histogram(self.token_length_histogram, limit)}",
        ]
        if self.longest_run is not None:
            preview = self.longest_run.render_preview(48)
            lines.append(
                "longest string: "
                f"len={len(self.longest_run.combined_string() or '')} "
                f"start=0x{self.longest_run.start_offset():06X} preview={preview}"
            )
        if self.token_frequency:
            most_common = sorted(
                self.token_frequency.items(), key=lambda item: (-item[1], item[0])
            )[:limit]
            token_line = ", ".join(f"{token}:{count}" for token, count in most_common)
            lines.append(f"top tokens: {token_line}")
        if self.char_frequency:
            most_common_chars = sorted(
                self.char_frequency.items(), key=lambda item: (-item[1], item[0])
            )[:limit]
            char_line = ", ".join(
                f"{repr(ch)[1:-1]}:{count}" for ch, count in most_common_chars
            )
            lines.append(f"top chars: {char_line}")
        if self.top_runs:
            preview_line = "; ".join(
                f"0x{run.start_offset():06X}:{run.render_preview(32)}"
                for run in self.top_runs[:limit]
            )
            lines.append(f"notable runs: {preview_line}")
        return lines

    @staticmethod
    def _format_histogram(histogram: Dict[int, int], limit: int) -> str:
        if not histogram:
            return "<empty>"
        entries = sorted(histogram.items())[:limit]
        return ", ".join(f"{length}:{count}" for length, count in entries)


@dataclass(frozen=True)
class NumericLiteralStats:
    """Aggregate statistics derived from numeric literal runs."""

    run_count: int
    total_values: int
    average_value: Optional[float]
    min_value: Optional[int]
    max_value: Optional[int]
    top_runs: Tuple[LiteralRun, ...]
    value_histogram: Dict[int, int]
    run_length_histogram: Dict[int, int]

    def summary_lines(self, *, limit: int = 5) -> List[str]:
        if self.run_count == 0:
            return []
        lines = [
            f"numeric runs: count={self.run_count} values={self.total_values}",
        ]
        if self.min_value is not None and self.max_value is not None:
            avg = f"{self.average_value:.2f}" if self.average_value is not None else "n/a"
            lines.append(
                f"range: min={self.min_value} max={self.max_value} average={avg}"
            )
        if self.value_histogram:
            common = sorted(
                self.value_histogram.items(), key=lambda item: (-item[1], item[0])
            )[:limit]
            hist = ", ".join(f"{value}:{count}" for value, count in common)
            lines.append(f"top values: {hist}")
        if self.top_runs:
            run_line = "; ".join(
                f"0x{run.start_offset():06X}[{run.length()}]"
                for run in self.top_runs[:limit]
            )
            lines.append(f"notable runs: {run_line}")
        if self.run_length_histogram:
            histogram = ", ".join(
                f"{length}:{count}"
                for length, count in sorted(self.run_length_histogram.items())[:limit]
            )
            lines.append(f"run lengths: {histogram}")
        return lines


@dataclass(frozen=True)
class LiteralStatistics:
    """Combined statistics for all literal runs within a function."""

    total_runs: int
    kind_counts: Dict[str, int]
    string_stats: Optional[StringLiteralStats]
    numeric_stats: Optional[NumericLiteralStats]

    def summary_lines(self) -> List[str]:
        lines = [
            "literal statistics:",
            "- run kinds: "
            + ", ".join(
                f"{kind}={count}" for kind, count in sorted(self.kind_counts.items())
            ),
        ]
        if self.string_stats:
            lines.extend(f"- {line}" for line in self.string_stats.summary_lines())
        if self.numeric_stats:
            lines.extend(f"- {line}" for line in self.numeric_stats.summary_lines())
        return lines


def compute_literal_statistics(runs: Sequence[LiteralRun]) -> LiteralStatistics:
    """Compute aggregate literal statistics for the provided runs."""

    kind_counts: Dict[str, int] = defaultdict(int)
    for run in runs:
        kind_counts[run.kind] += 1

    string_runs = [run for run in runs if run.kind == "string"]
    numeric_runs = [run for run in runs if run.kind == "number"]

    string_stats = _aggregate_string_runs(string_runs)
    numeric_stats = _aggregate_numeric_runs(numeric_runs)

    return LiteralStatistics(
        total_runs=len(runs),
        kind_counts=dict(kind_counts),
        string_stats=string_stats,
        numeric_stats=numeric_stats,
    )


def _aggregate_string_runs(runs: Sequence[LiteralRun]) -> Optional[StringLiteralStats]:
    if not runs:
        return None
    total_length = 0
    char_frequency: Counter[str] = Counter()
    token_frequency: Counter[str] = Counter()
    run_length_histogram: Counter[int] = Counter()
    token_length_histogram: Counter[int] = Counter()
    for run in runs:
        text = run.combined_string() or ""
        total_length += len(text)
        char_frequency.update(text)
        tokens = list(_tokenise_string(text))
        token_frequency.update(tokens)
        run_length_histogram[len(text)] += 1
        for token in tokens:
            token_length_histogram[len(token)] += 1
    average = total_length / len(runs)
    token_length_average = (
        sum(length * count for length, count in token_length_histogram.items()) / sum(token_length_histogram.values())
        if token_length_histogram
        else 0.0
    )
    longest = max(runs, key=lambda run: len(run.combined_string() or ""))
    top_runs = tuple(sorted(runs, key=lambda run: (-len(run.combined_string() or ""), run.start_offset())))
    return StringLiteralStats(
        run_count=len(runs),
        total_length=total_length,
        average_length=average,
        longest_run=longest,
        top_runs=top_runs[:8],
        token_frequency=dict(token_frequency),
        char_frequency=dict(char_frequency),
        run_length_histogram=dict(run_length_histogram),
        token_length_histogram=dict(token_length_histogram),
        token_length_average=token_length_average,
    )


def _aggregate_numeric_runs(runs: Sequence[LiteralRun]) -> Optional[NumericLiteralStats]:
    if not runs:
        return None
    all_values: List[int] = []
    value_histogram: Counter[int] = Counter()
    for run in runs:
        for value in _numeric_values(run):
            all_values.append(value)
            value_histogram[value] += 1
    run_length_histogram: Counter[int] = Counter()
    for run in runs:
        run_length_histogram[run.length()] += 1
    if all_values:
        min_value = min(all_values)
        max_value = max(all_values)
        average = sum(all_values) / len(all_values)
    else:
        min_value = None
        max_value = None
        average = None
    top_runs = tuple(sorted(runs, key=lambda run: (-run.length(), run.start_offset())))
    return NumericLiteralStats(
        run_count=len(runs),
        total_values=len(all_values),
        average_value=average,
        min_value=min_value,
        max_value=max_value,
        top_runs=top_runs[:8],
        value_histogram=dict(value_histogram),
        run_length_histogram=dict(run_length_histogram),
    )


def _tokenise_string(text: str) -> Iterator[str]:
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if token:
            yield token.lower()


def _numeric_values(run: LiteralRun) -> Iterator[int]:
    for raw in run.numeric_values():
        try:
            yield int(raw, 0)
        except ValueError:
            continue


def literal_statistics_to_dict(stats: LiteralStatistics) -> Dict[str, object]:
    """Convert :class:`LiteralStatistics` into a serialisable dictionary."""

    result: Dict[str, object] = {
        "total_runs": stats.total_runs,
        "kind_counts": dict(stats.kind_counts),
    }
    if stats.string_stats is not None:
        string_stats = stats.string_stats
        result["strings"] = {
            "run_count": string_stats.run_count,
            "total_length": string_stats.total_length,
            "average_length": string_stats.average_length,
            "token_length_average": string_stats.token_length_average,
            "run_length_histogram": dict(string_stats.run_length_histogram),
            "token_length_histogram": dict(string_stats.token_length_histogram),
            "token_frequency": dict(string_stats.token_frequency),
            "char_frequency": dict(string_stats.char_frequency),
        }
    if stats.numeric_stats is not None:
        numeric_stats = stats.numeric_stats
        result["numbers"] = {
            "run_count": numeric_stats.run_count,
            "total_values": numeric_stats.total_values,
            "average_value": numeric_stats.average_value,
            "min_value": numeric_stats.min_value,
            "max_value": numeric_stats.max_value,
            "value_histogram": dict(numeric_stats.value_histogram),
            "run_length_histogram": dict(numeric_stats.run_length_histogram),
        }
    return result


def literal_statistics_to_json(stats: LiteralStatistics, *, indent: int = 2) -> str:
    """Serialise :class:`LiteralStatistics` to JSON."""

    import json

    payload = literal_statistics_to_dict(stats)
    return json.dumps(payload, indent=indent)


class LiteralRunCatalogue:
    """Index literal runs by block and kind for efficient querying."""

    def __init__(self, runs: Sequence[LiteralRun]) -> None:
        self._runs: Tuple[LiteralRun, ...] = tuple(runs)
        self._by_kind: Dict[str, List[LiteralRun]] = defaultdict(list)
        self._by_block: Dict[int, List[LiteralRun]] = defaultdict(list)
        for run in runs:
            self._by_kind[run.kind].append(run)
            self._by_block[run.block_start].append(run)
        for kind in self._by_kind:
            self._by_kind[kind].sort(key=lambda run: (run.start_offset(), -run.length()))
        for block in self._by_block:
            self._by_block[block].sort(key=lambda run: (run.start_offset(), -run.length()))

    def runs(self) -> Tuple[LiteralRun, ...]:
        return self._runs

    def kinds(self) -> List[str]:
        return sorted(self._by_kind)

    def runs_for_kind(self, kind: str) -> Tuple[LiteralRun, ...]:
        return tuple(self._by_kind.get(kind, ()))

    def runs_in_block(self, block_start: int) -> Tuple[LiteralRun, ...]:
        return tuple(self._by_block.get(block_start, ()))

    def longest_runs(self, *, kind: Optional[str] = None, limit: int = 5) -> Tuple[LiteralRun, ...]:
        if kind is None:
            runs = self._runs
        else:
            runs = self._by_kind.get(kind, ())
        sorted_runs = sorted(runs, key=lambda run: (-run.length(), run.start_offset()))
        return tuple(sorted_runs[:limit])

    def filter_by_min_length(
        self, length: int, *, kind: Optional[str] = None
    ) -> Tuple[LiteralRun, ...]:
        if kind is None:
            candidates = self._runs
        else:
            candidates = self._by_kind.get(kind, ())
        return tuple(run for run in candidates if run.length() >= length)

    def render_table(self, *, limit: int = 10) -> str:
        """Render a simple table summarising literal runs."""

        header = "kind    start     count  preview"
        rows = [header, "-" * len(header)]
        for run in list(self._runs)[:limit]:
            rows.append(
                f"{run.kind:<7} 0x{run.start_offset():06X} {run.length():>6}  {run.render_preview(40)}"
            )
        remaining = len(self._runs) - limit
        if remaining > 0:
            rows.append(f"... {remaining} additional runs omitted")
        return "\n".join(rows)

    def search(self, text: str, *, kind: Optional[str] = None) -> Tuple[LiteralRun, ...]:
        """Return runs whose preview contains ``text``."""

        if not text:
            return ()
        haystack = self._runs if kind is None else self._by_kind.get(kind, ())
        needle = text.lower()
        matches = [
            run
            for run in haystack
            if needle in run.render_preview(128).lower()
        ]
        return tuple(sorted(matches, key=lambda run: (run.start_offset(), -run.length())))

    def runs_with_min_tokens(self, token_count: int) -> Tuple[LiteralRun, ...]:
        """Return string runs containing at least ``token_count`` distinct tokens."""

        if token_count <= 0:
            return self._runs
        candidates: List[LiteralRun] = []
        for run in self._runs:
            if run.kind != "string":
                continue
            tokens = set(_tokenise_string(run.combined_string() or ""))
            if len(tokens) >= token_count:
                candidates.append(run)
        return tuple(sorted(candidates, key=lambda run: (-run.length(), run.start_offset())))


@dataclass(frozen=True)
class LiteralRunBlockSummary:
    """Aggregated literal information for a single IR block."""

    block_start: int
    total_runs: int
    kind_counts: Dict[str, int]
    longest_run: Optional[LiteralRun]
    preview_runs: Tuple[LiteralRun, ...]
    token_frequency: Dict[str, int]
    numeric_frequency: Dict[int, int]

    def summary_lines(self, *, limit: int = 4) -> List[str]:
        prefix = f"block 0x{self.block_start:06X}:"
        if self.total_runs == 0:
            return [f"{prefix} no literal runs"]
        lines = [
            f"{prefix} runs={self.total_runs} "
            + ", ".join(
                f"{kind}={count}" for kind, count in sorted(self.kind_counts.items())
            )
        ]
        if self.longest_run is not None:
            preview = self.longest_run.render_preview(48)
            lines.append(
                "  longest: "
                f"kind={self.longest_run.kind} count={self.longest_run.length()} "
                f"start=0x{self.longest_run.start_offset():06X} preview={preview}"
            )
        if self.token_frequency:
            lines.append(f"  tokens: {_format_frequency(self.token_frequency, limit)}")
        if self.numeric_frequency:
            lines.append(f"  numbers: {_format_frequency(self.numeric_frequency, limit)}")
        if self.preview_runs:
            preview_line = "; ".join(
                f"0x{run.start_offset():06X}[{run.length()}]"
                for run in self.preview_runs[:limit]
            )
            lines.append(f"  notable runs: {preview_line}")
        return lines


@dataclass(frozen=True)
class LiteralRunReport:
    """Combined literal overview for a single function or module."""

    runs: Tuple[LiteralRun, ...]
    block_summaries: Tuple[LiteralRunBlockSummary, ...]
    token_frequency: Dict[str, int]
    numeric_frequency: Dict[int, int]
    longest_runs: Tuple[LiteralRun, ...]

    def total_runs(self) -> int:
        return len(self.runs)

    def top_tokens(self, limit: int = 5) -> List[Tuple[str, int]]:
        return [(str(key), count) for key, count in _top_items(self.token_frequency, limit)]

    def top_numbers(self, limit: int = 5) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        for key, count in _top_items(self.numeric_frequency, limit):
            if isinstance(key, int):
                result.append((key, count))
            else:
                try:
                    result.append((int(str(key), 0), count))
                except ValueError:
                    continue
        return result

    def longest_previews(self, limit: int = 5) -> List[str]:
        previews: List[str] = []
        for run in self.longest_runs[:limit]:
            previews.append(
                f"0x{run.start_offset():06X}[{run.length()}]={run.render_preview(32)}"
            )
        return previews

    def summary_lines(self, *, limit: int = 5) -> List[str]:
        if not self.runs:
            return ["literal run report:", "- no literal runs recorded"]
        lines = [
            "literal run report:",
            f"- runs: {len(self.runs)} blocks={len(self.block_summaries)}",
        ]
        if self.token_frequency:
            lines.append(
                "- top tokens: " + _format_frequency(self.token_frequency, limit)
            )
        if self.numeric_frequency:
            lines.append(
                "- top numbers: " + _format_frequency(self.numeric_frequency, limit)
            )
        if self.longest_runs:
            preview = "; ".join(
                f"0x{run.start_offset():06X}:{run.render_preview(32)}"
                for run in self.longest_runs[:limit]
            )
            lines.append(f"- longest runs: {preview}")
        return lines

    def block_lines(self, *, limit: int = 3) -> List[str]:
        lines: List[str] = []
        for summary in self.block_summaries[:limit]:
            lines.extend(summary.summary_lines(limit=limit))
        remaining = len(self.block_summaries) - limit
        if remaining > 0:
            lines.append(f"... {remaining} additional blocks omitted")
        return lines


def build_literal_run_report(runs: Sequence[LiteralRun]) -> LiteralRunReport:
    catalogue = LiteralRunCatalogue(runs)
    block_summaries = _summarise_blocks(catalogue)
    token_frequency: Counter[str] = Counter()
    numeric_frequency: Counter[int] = Counter()
    for run in runs:
        if run.kind == "string":
            token_frequency.update(_tokenise_string(run.combined_string() or ""))
        if run.kind == "number":
            numeric_frequency.update(_numeric_values(run))
    longest_runs = catalogue.longest_runs(limit=8)
    return LiteralRunReport(
        runs=tuple(runs),
        block_summaries=block_summaries,
        token_frequency=dict(token_frequency),
        numeric_frequency=dict(numeric_frequency),
        longest_runs=longest_runs,
    )


def literal_report_to_dict(report: LiteralRunReport) -> Dict[str, object]:
    return {
        "total_runs": report.total_runs(),
        "tokens": dict(report.token_frequency),
        "numbers": dict(report.numeric_frequency),
        "longest_runs": [
            {
                "kind": run.kind,
                "start": run.start_offset(),
                "length": run.length(),
                "preview": run.render_preview(64),
            }
            for run in report.longest_runs
        ],
        "blocks": [
            {
                "block_start": summary.block_start,
                "total_runs": summary.total_runs,
                "kind_counts": dict(summary.kind_counts),
                "tokens": dict(summary.token_frequency),
                "numbers": dict(summary.numeric_frequency),
            }
            for summary in report.block_summaries
        ],
    }


def _summarise_blocks(catalogue: LiteralRunCatalogue) -> Tuple[LiteralRunBlockSummary, ...]:
    summaries: List[LiteralRunBlockSummary] = []
    for block_start in sorted({run.block_start for run in catalogue.runs()}):
        runs = list(catalogue.runs_in_block(block_start))
        kind_counts: Counter[str] = Counter(run.kind for run in runs)
        token_frequency: Counter[str] = Counter()
        numeric_frequency: Counter[int] = Counter()
        for run in runs:
            if run.kind == "string":
                token_frequency.update(_tokenise_string(run.combined_string() or ""))
            if run.kind == "number":
                numeric_frequency.update(_numeric_values(run))
        longest_run = max(runs, key=lambda run: (run.length(), run.start_offset()), default=None)
        preview_runs = tuple(sorted(runs, key=lambda run: (-run.length(), run.start_offset())))
        summaries.append(
            LiteralRunBlockSummary(
                block_start=block_start,
                total_runs=len(runs),
                kind_counts=dict(kind_counts),
                longest_run=longest_run,
                preview_runs=preview_runs[:6],
                token_frequency=dict(token_frequency),
                numeric_frequency=dict(numeric_frequency),
            )
        )
    return tuple(summaries)


def _format_frequency(counter: Dict[object, int], limit: int) -> str:
    if not counter:
        return "<empty>"
    entries = _top_items(counter, limit)
    return ", ".join(f"{key}:{count}" for key, count in entries)


def _top_items(counter: Dict[object, int], limit: int) -> List[Tuple[object, int]]:
    if not counter:
        return []
    return sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[:limit]


__all__ = [
    "LiteralDescriptor",
    "LiteralRun",
    "LiteralRunTracker",
    "LiteralStatistics",
    "StringLiteralStats",
    "NumericLiteralStats",
    "compute_literal_statistics",
    "literal_statistics_to_dict",
    "literal_statistics_to_json",
    "LiteralRunCatalogue",
    "LiteralRunReport",
    "LiteralRunBlockSummary",
    "build_literal_run_report",
    "literal_report_to_dict",
]
