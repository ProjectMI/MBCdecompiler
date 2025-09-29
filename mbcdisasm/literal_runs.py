"""Utilities for analysing runs of literal instructions."""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ir import IRInstruction
from .lua_ast import LuaExpression, LiteralExpr, NameExpr
from .lua_literals import escape_lua_string
from .manual_semantics import InstructionSemantics


_PREVIEW_LIMIT = 5
_PREVIEW_WIDTH = 72
_HISTOGRAM_BUCKETS: Tuple[Tuple[int, Optional[int], str], ...] = (
    (1, 1, "single"),
    (2, 3, "small"),
    (4, 7, "medium"),
    (8, None, "large"),
)

_PRINTABLE_ASCII = {"\t", "\n", "\r"}.union(
    {chr(code) for code in range(32, 127)}
)
_POWER_OF_TWO_CACHE = {1 << bit for bit in range(0, 31)}


def _expression_label(expression: LuaExpression) -> str:
    if isinstance(expression, LiteralExpr):
        literal = expression.literal
        if literal is not None:
            if literal.kind == "string":
                return "string"
            if literal.kind == "number":
                return "number"
        return "literal"
    if isinstance(expression, NameExpr):
        return "enum"
    return "expression"


def _expression_render(expression: LuaExpression) -> str:
    try:
        return expression.render()
    except Exception:  # pragma: no cover - defensive
        return repr(expression)


def _truncate_preview(text: str, width: int = _PREVIEW_WIDTH) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return "..."
    return text[: width - 3] + "..."


def _string_payload(expression: LuaExpression) -> Optional[str]:
    if isinstance(expression, LiteralExpr):
        literal = expression.literal
        if literal is not None and literal.kind == "string":
            return str(literal.value)
    return None


def _numeric_value(expression: LuaExpression) -> Optional[int]:
    if isinstance(expression, LiteralExpr):
        literal = expression.literal
        if literal is not None and literal.kind == "number":
            value = literal.value
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    rendered = _expression_render(expression)
    try:
        if rendered.startswith("0x") or rendered.startswith("-0x"):
            return int(rendered, 16)
        return int(rendered)
    except ValueError:
        return None


def _enum_name(expression: LuaExpression) -> Optional[str]:
    if isinstance(expression, NameExpr):
        return expression.render()
    return None


def _is_printable_string(text: str) -> bool:
    return bool(text) and all(ch in _PRINTABLE_ASCII for ch in text)


def _format_metadata_items(items: Sequence[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    return tuple((str(key), str(value)) for key, value in items)


def _detect_numeric_patterns(values: Sequence[int]) -> Tuple[List[LiteralRunPattern], List[LiteralRunNote]]:
    patterns: List[LiteralRunPattern] = []
    notes: List[LiteralRunNote] = []
    if not values:
        return patterns, notes

    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        value = unique_values[0]
        patterns.append(
            LiteralRunPattern(
                "number",
                "constant",
                "Constant numeric literal",
                metadata=_format_metadata_items((("value", value),)),
            )
        )
    if len(unique_values) > 1:
        diffs = [b - a for a, b in zip(values, values[1:])]
        if diffs and all(diff == diffs[0] for diff in diffs):
            step = diffs[0]
            if step != 0:
                patterns.append(
                    LiteralRunPattern(
                        "number",
                        "progression",
                        "Arithmetic progression",
                        metadata=_format_metadata_items(
                            (
                                ("step", step),
                                ("start", values[0]),
                                ("length", len(values)),
                            )
                        ),
                    )
                )
        sorted_unique = sorted(unique_values)
        contiguous = (
            len(sorted_unique) > 1
            and sorted_unique[-1] - sorted_unique[0] + 1 == len(sorted_unique)
        )
        if contiguous:
            patterns.append(
                LiteralRunPattern(
                    "number",
                    "contiguous",
                    "Contiguous numeric range",
                    metadata=_format_metadata_items(
                        (
                            ("start", sorted_unique[0]),
                            ("end", sorted_unique[-1]),
                            ("unique", len(sorted_unique)),
                        )
                    ),
                )
            )
        if diffs and all(diff > 0 for diff in diffs):
            patterns.append(
                LiteralRunPattern(
                    "number",
                    "increasing",
                    "Monotonically increasing sequence",
                )
            )
        elif diffs and all(diff < 0 for diff in diffs):
            patterns.append(
                LiteralRunPattern(
                    "number",
                    "decreasing",
                    "Monotonically decreasing sequence",
                )
            )
    if all(value in _POWER_OF_TWO_CACHE for value in values if value > 0):
        patterns.append(
            LiteralRunPattern(
                "number",
                "bitmask",
                "Potential power-of-two bitmask",
            )
        )
    if all(0 <= value <= 255 for value in values):
        notes.append(LiteralRunNote("values fit within byte range"))
    if len(values) >= 6 and len(unique_values) <= 2:
        notes.append(
            LiteralRunNote(
                "repeating numeric pattern", category="warning"
            )
        )
    return patterns, notes


def _detect_string_patterns(
    payload: Optional[str], fragments: Sequence[str]
) -> Tuple[List[LiteralRunPattern], List[LiteralRunNote]]:
    patterns: List[LiteralRunPattern] = []
    notes: List[LiteralRunNote] = []
    if not payload:
        return patterns, notes
    if _is_printable_string(payload):
        patterns.append(
            LiteralRunPattern("string", "printable", "Printable ASCII payload")
        )
    else:
        notes.append(
            LiteralRunNote("contains non-printable characters", category="warning")
        )
    if "\n" in payload or "\r" in payload:
        notes.append(LiteralRunNote("multiline string payload"))
    if payload.strip() != payload:
        notes.append(LiteralRunNote("leading or trailing whitespace preserved"))
    if payload.isidentifier():
        patterns.append(
            LiteralRunPattern("string", "identifier", "Identifier-like payload")
        )
    if fragments:
        lengths = [len(fragment) for fragment in fragments]
        if lengths and all(length == lengths[0] for length in lengths):
            patterns.append(
                LiteralRunPattern(
                    "string",
                    "uniform-chunk",
                    "Uniform fragment sizes",
                    metadata=_format_metadata_items((("length", lengths[0]),)),
                )
            )
        if any(len(fragment) == 1 for fragment in fragments) and len(fragments) > 8:
            notes.append(LiteralRunNote("single character chunking", category="warning"))
    return patterns, notes


def _detect_enum_patterns(names: Sequence[str]) -> Tuple[List[LiteralRunPattern], List[LiteralRunNote]]:
    patterns: List[LiteralRunPattern] = []
    notes: List[LiteralRunNote] = []
    if not names:
        return patterns, notes
    namespaces = {name.split(".")[0] for name in names if "." in name}
    if len(namespaces) == 1:
        namespace = next(iter(namespaces))
        patterns.append(
            LiteralRunPattern(
                "enum",
                "namespace",
                "Single enum namespace",
                metadata=_format_metadata_items((("namespace", namespace),)),
            )
        )
    if len(set(names)) == len(names):
        notes.append(LiteralRunNote("enum values are unique"))
    else:
        notes.append(LiteralRunNote("repeated enum values", category="warning"))
    return patterns, notes


def _detect_mixed_patterns(
    classifications: Sequence[str],
) -> Tuple[List[LiteralRunPattern], List[LiteralRunNote]]:
    counter: Counter[str] = Counter(classifications)
    if not counter:
        return [], []
    patterns: List[LiteralRunPattern] = []
    notes: List[LiteralRunNote] = []
    most_common = counter.most_common()
    if most_common:
        label, count = most_common[0]
        patterns.append(
            LiteralRunPattern(
                "mixed",
                "dominant",
                "Dominant literal classification",
                metadata=_format_metadata_items((("label", label), ("count", count))),
            )
        )
    if len(most_common) > 1:
        spread = ", ".join(f"{label}={count}" for label, count in most_common)
        notes.append(LiteralRunNote(f"distribution: {spread}"))
    return patterns, notes


def _general_run_notes(count: int, unique_values: int) -> List[LiteralRunNote]:
    notes: List[LiteralRunNote] = []
    if count >= 16:
        notes.append(LiteralRunNote("very large literal run", category="warning"))
    elif count >= 8:
        notes.append(LiteralRunNote("large literal run"))
    if unique_values == 1 and count > 1:
        notes.append(LiteralRunNote("repeated single literal", category="warning"))
    return notes


def _deduplicate_patterns(patterns: Sequence[LiteralRunPattern]) -> List[LiteralRunPattern]:
    result: List[LiteralRunPattern] = []
    seen: set[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = set()
    for pattern in patterns:
        key = (pattern.kind, pattern.label, pattern.metadata)
        if key in seen:
            continue
        seen.add(key)
        result.append(pattern)
    return result


def _deduplicate_notes(notes: Sequence[LiteralRunNote]) -> List[LiteralRunNote]:
    result: List[LiteralRunNote] = []
    seen: set[Tuple[str, str]] = set()
    for note in notes:
        key = (note.category, note.text)
        if key in seen:
            continue
        seen.add(key)
        result.append(note)
    return result


def classify_expression(expression: LuaExpression) -> str:
    """Return a coarse label describing the literal expression."""

    label = _expression_label(expression)
    if label == "literal":
        rendered = _expression_render(expression)
        if rendered.startswith("0x") or rendered.lstrip("-").isdigit():
            return "number"
    return label


@dataclass
class LiteralRunEntry:
    """Single literal instruction encountered during reconstruction."""

    instruction: IRInstruction
    semantics: InstructionSemantics
    expression: LuaExpression
    prefix: Optional[str]


@dataclass(frozen=True)
class LiteralRunPattern:
    """Describes a structural pattern detected within a literal run."""

    kind: str
    label: str
    description: str
    confidence: float = 1.0
    metadata: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def render_tag(self) -> str:
        tag = f"{self.kind}:{self.label}"
        if self.metadata:
            extras = ",".join(f"{key}={value}" for key, value in self.metadata)
            tag = f"{tag}[{extras}]"
        if self.confidence < 1.0:
            tag = f"{tag}@{self.confidence:.0%}"
        return tag

    def render_detail(self) -> str:
        if not self.metadata:
            return self.description
        extras = ", ".join(f"{key}={value}" for key, value in self.metadata)
        return f"{self.description} ({extras})"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "metadata": {key: value for key, value in self.metadata},
        }


@dataclass(frozen=True)
class LiteralRunNote:
    """Free-form annotation attached to a literal run."""

    text: str
    category: str = "info"

    def render(self) -> str:
        return f"{self.category}: {self.text}" if self.category else self.text

    def to_dict(self) -> dict:
        return {"text": self.text, "category": self.category}


@dataclass
class LiteralRunSummary:
    """Describes a consolidated run of literal instructions."""

    kind: str
    count: int
    start_offset: int
    end_offset: int
    preview: str
    comment: str
    unique_values: int
    string_payload: Optional[str] = None
    string_length: Optional[int] = None
    enum_hits: int = 0
    number_hits: int = 0
    string_hits: int = 0
    expression_hits: int = 0
    numeric_min: Optional[int] = None
    numeric_max: Optional[int] = None
    numeric_span: Optional[int] = None
    enum_values: Tuple[str, ...] = field(default_factory=tuple)
    classifications: Tuple[str, ...] = field(default_factory=tuple)
    patterns: Tuple[LiteralRunPattern, ...] = field(default_factory=tuple)
    notes: Tuple[LiteralRunNote, ...] = field(default_factory=tuple)

    def metadata_line(self) -> str:
        details = [
            f"0x{self.start_offset:06X}-0x{self.end_offset:06X}",
            f"{self.kind}",
            f"count={self.count}",
            f"unique={self.unique_values}",
        ]
        if self.kind == "string" and self.string_length is not None:
            details.append(f"len={self.string_length}")
        if self.kind == "number" and self.numeric_min is not None and self.numeric_max is not None:
            span = self.numeric_span if self.numeric_span is not None else self.numeric_max - self.numeric_min
            details.append(f"range={self.numeric_min}..{self.numeric_max} (span={span})")
        if self.patterns:
            tags = ", ".join(pattern.render_tag() for pattern in self.patterns)
            details.append(f"[{tags}]")
        return "- " + " ".join(details) + f": {self.preview}"

    def detail_lines(self) -> List[str]:
        lines: List[str] = []
        if self.kind == "string" and self.string_payload is not None:
            unique_chars = len(set(self.string_payload))
            lines.append(f"  - chunks: {self.count}")
            lines.append(f"  - unique chars: {unique_chars}")
        elif self.kind == "number" and self.numeric_min is not None and self.numeric_max is not None:
            span = self.numeric_span if self.numeric_span is not None else self.numeric_max - self.numeric_min
            lines.append(f"  - range: {self.numeric_min}..{self.numeric_max} (span={span})")
            lines.append(f"  - unique values: {self.unique_values}")
        elif self.kind == "enum" and self.enum_values:
            preview = ", ".join(self.enum_values[:_PREVIEW_LIMIT])
            if len(self.enum_values) > _PREVIEW_LIMIT:
                preview += ", ..."
            lines.append(f"  - values: {preview}")
        else:
            counts = []
            if self.string_hits:
                counts.append(f"string={self.string_hits}")
            if self.number_hits:
                counts.append(f"number={self.number_hits}")
            if self.enum_hits:
                counts.append(f"enum={self.enum_hits}")
            if self.expression_hits:
                counts.append(f"expression={self.expression_hits}")
            if counts:
                lines.append("  - classifications: " + ", ".join(counts))
        if self.patterns:
            for pattern in self.patterns:
                lines.append("  - pattern: " + pattern.render_detail())
        if self.notes:
            for note in self.notes:
                lines.append("  - note: " + note.render())
        return lines


@dataclass
class LiteralRunStatistics:
    """Aggregated statistics covering many literal runs."""

    total_runs: int = 0
    total_literals: int = 0
    string_runs: int = 0
    number_runs: int = 0
    enum_runs: int = 0
    mixed_runs: int = 0
    longest_run: int = 0
    longest_kind: Optional[str] = None
    total_string_literals: int = 0
    total_numeric_literals: int = 0
    total_enum_literals: int = 0
    max_numeric_span: int = 0
    max_string_length: int = 0
    pattern_counts: Counter[str] = field(default_factory=Counter)
    note_counts: Counter[str] = field(default_factory=Counter)

    def register(self, summary: LiteralRunSummary) -> None:
        self.total_runs += 1
        self.total_literals += summary.count
        if summary.count > self.longest_run:
            self.longest_run = summary.count
            self.longest_kind = summary.kind
        if summary.kind == "string":
            self.string_runs += 1
            if summary.string_length:
                self.max_string_length = max(self.max_string_length, summary.string_length)
        elif summary.kind == "number":
            self.number_runs += 1
        elif summary.kind == "enum":
            self.enum_runs += 1
        else:
            self.mixed_runs += 1
        self.total_string_literals += summary.string_hits
        self.total_numeric_literals += summary.number_hits
        self.total_enum_literals += summary.enum_hits
        if summary.numeric_span is not None:
            self.max_numeric_span = max(self.max_numeric_span, summary.numeric_span)
        for pattern in summary.patterns:
            self.pattern_counts[pattern.render_tag()] += 1
        for note in summary.notes:
            self.note_counts[note.render()] += 1

    def summary_lines(self) -> List[str]:
        if not self.total_runs:
            return []
        lines = [
            f"- literal runs: {self.total_runs} (values={self.total_literals})",
            f"- string runs: {self.string_runs}",
            f"- numeric runs: {self.number_runs}",
            f"- enum runs: {self.enum_runs}",
        ]
        if self.mixed_runs:
            lines.append(f"- mixed runs: {self.mixed_runs}")
        lines.append(
            "- literal value breakdown: "
            f"string={self.total_string_literals} "
            f"number={self.total_numeric_literals} "
            f"enum={self.total_enum_literals}"
        )
        if self.longest_run:
            lines.append(
                f"- longest run: {self.longest_run} ({self.longest_kind or 'unknown'})"
            )
        if self.max_numeric_span:
            lines.append(f"- max numeric span: {self.max_numeric_span}")
        if self.max_string_length:
            lines.append(f"- longest string payload: {self.max_string_length}")
        if self.total_runs:
            average = self.total_literals / self.total_runs
            lines.append(f"- average run length: {average:.2f}")
        if self.pattern_counts:
            top = ", ".join(
                f"{tag}={count}" for tag, count in self.pattern_counts.most_common(4)
            )
            lines.append(f"- dominant patterns: {top}")
        if self.note_counts:
            top_notes = ", ".join(
                f"{text} ({count})" for text, count in self.note_counts.most_common(3)
            )
            lines.append(f"- frequent notes: {top_notes}")
        return lines

    def to_dict(self) -> dict:
        return {
            "total_runs": self.total_runs,
            "total_literals": self.total_literals,
            "string_runs": self.string_runs,
            "number_runs": self.number_runs,
            "enum_runs": self.enum_runs,
            "mixed_runs": self.mixed_runs,
            "longest_run": self.longest_run,
            "longest_kind": self.longest_kind,
            "total_string_literals": self.total_string_literals,
            "total_numeric_literals": self.total_numeric_literals,
            "total_enum_literals": self.total_enum_literals,
            "max_numeric_span": self.max_numeric_span,
            "max_string_length": self.max_string_length,
            "pattern_counts": dict(self.pattern_counts),
            "note_counts": dict(self.note_counts),
            "average_run_length": (self.total_literals / self.total_runs)
            if self.total_runs
            else 0.0,
        }


def _compute_preview(entries: Sequence[LiteralRunEntry]) -> str:
    preview_values = [_expression_render(entry.expression) for entry in entries[:_PREVIEW_LIMIT]]
    preview = ", ".join(preview_values)
    if len(entries) > _PREVIEW_LIMIT:
        preview += ", ..."
    return _truncate_preview(preview)


def summarize_literal_run(entries: Sequence[LiteralRunEntry]) -> LiteralRunSummary:
    if not entries:
        raise ValueError("literal run cannot be empty")
    start = entries[0].instruction.offset
    end = entries[-1].instruction.offset
    classifications: List[str] = [classify_expression(entry.expression) for entry in entries]
    counts = {
        "string": sum(1 for label in classifications if label == "string"),
        "number": sum(1 for label in classifications if label == "number"),
        "enum": sum(1 for label in classifications if label == "enum"),
        "expression": sum(1 for label in classifications if label == "expression"),
    }
    numeric_values = [value for value in (_numeric_value(entry.expression) for entry in entries) if value is not None]
    enum_names = [name for name in (_enum_name(entry.expression) for entry in entries) if name]
    count = len(entries)
    if counts["string"] == count:
        kind = "string"
    elif counts["number"] == count:
        kind = "number"
    elif counts["enum"] == count:
        kind = "enum"
    else:
        kind = "mixed"
    preview = _compute_preview(entries)
    unique_values = len({preview for preview in (_expression_render(entry.expression) for entry in entries)})
    string_payload = None
    string_length = None
    string_fragments: List[str] = []
    comment = f"literal run ({count} values): {preview}"
    numeric_min = min(numeric_values) if numeric_values else None
    numeric_max = max(numeric_values) if numeric_values else None
    numeric_span = None
    if numeric_min is not None and numeric_max is not None:
        numeric_span = numeric_max - numeric_min
    patterns: List[LiteralRunPattern] = []
    notes: List[LiteralRunNote] = _general_run_notes(count, unique_values)
    if kind == "string":
        string_fragments = [
            _string_payload(entry.expression) or "" for entry in entries
        ]
        string_payload = "".join(string_fragments)
        string_length = len(string_payload)
        comment = (
            f"string literal sequence: {escape_lua_string(_truncate_preview(string_payload))}"
            f" (len={string_length})"
        )
        pattern_list, note_list = _detect_string_patterns(string_payload, string_fragments)
        patterns.extend(pattern_list)
        notes.extend(note_list)
    summary = LiteralRunSummary(
        kind=kind,
        count=count,
        start_offset=start,
        end_offset=end,
        preview=preview,
        comment=comment,
        unique_values=unique_values,
        string_payload=string_payload,
        string_length=string_length,
        enum_hits=counts["enum"],
        number_hits=counts["number"],
        string_hits=counts["string"],
        expression_hits=counts["expression"],
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        numeric_span=numeric_span,
        enum_values=tuple(sorted(set(enum_names))),
        classifications=tuple(classifications),
        patterns=tuple(_deduplicate_patterns(patterns)),
        notes=tuple(_deduplicate_notes(notes)),
    )
    if kind == "number" and numeric_values:
        pattern_list, note_list = _detect_numeric_patterns(numeric_values)
        summary = dataclasses.replace(
            summary,
            patterns=tuple(
                _deduplicate_patterns(list(summary.patterns) + pattern_list)
            ),
            notes=tuple(
                _deduplicate_notes(list(summary.notes) + note_list)
            ),
        )
    elif kind == "enum" and enum_names:
        pattern_list, note_list = _detect_enum_patterns(enum_names)
        summary = dataclasses.replace(
            summary,
            patterns=tuple(
                _deduplicate_patterns(list(summary.patterns) + pattern_list)
            ),
            notes=tuple(
                _deduplicate_notes(list(summary.notes) + note_list)
            ),
        )
    elif kind == "mixed":
        pattern_list, note_list = _detect_mixed_patterns(classifications)
        summary = dataclasses.replace(
            summary,
            patterns=tuple(
                _deduplicate_patterns(list(summary.patterns) + pattern_list)
            ),
            notes=tuple(
                _deduplicate_notes(list(summary.notes) + note_list)
            ),
        )
    return summary


def summarize_literal_runs(runs: Sequence[LiteralRunSummary], limit: int = 8) -> List[str]:
    if not runs:
        return []
    lines: List[str] = []
    for summary in runs[:limit]:
        lines.append(summary.metadata_line())
    remaining = len(runs) - limit
    if remaining > 0:
        lines.append(f"- ... ({remaining} additional literal runs)")
    return lines


def accumulate_statistics(runs: Iterable[LiteralRunSummary]) -> LiteralRunStatistics:
    stats = LiteralRunStatistics()
    for summary in runs:
        stats.register(summary)
    return stats


def literal_run_histogram(runs: Sequence[LiteralRunSummary]) -> List[str]:
    if not runs:
        return []
    counts = {label: 0 for _, _, label in _HISTOGRAM_BUCKETS}
    for summary in runs:
        for lower, upper, label in _HISTOGRAM_BUCKETS:
            if upper is None and summary.count >= lower:
                counts[label] += 1
                break
            if upper is not None and lower <= summary.count <= upper:
                counts[label] += 1
                break
    lines: List[str] = []
    for lower, upper, label in _HISTOGRAM_BUCKETS:
        total = counts[label]
        if upper is None:
            bounds = f">={lower}"
        elif lower == upper:
            bounds = f"={lower}"
        else:
            bounds = f"{lower}-{upper}"
        bar = "#" * min(total, 20)
        if bar:
            lines.append(f"- runs {label} ({bounds}): {total} {bar}")
        else:
            lines.append(f"- runs {label} ({bounds}): {total}")
    return lines


def literal_run_pattern_histogram(
    runs: Sequence[LiteralRunSummary], *, limit: int = 12
) -> List[str]:
    counter: Counter[str] = Counter()
    for summary in runs:
        for pattern in summary.patterns:
            counter[pattern.render_tag()] += 1
    if not counter:
        return []
    total = sum(counter.values())
    lines: List[str] = []
    for tag, count in counter.most_common(limit):
        percentage = (count / total) * 100 if total else 0.0
        lines.append(f"- pattern {tag}: {count} ({percentage:.1f}%)")
    remaining = len(counter) - limit
    if remaining > 0:
        lines.append(f"- ... {remaining} additional patterns")
    return lines


def group_literal_runs_by_pattern(
    runs: Sequence[LiteralRunSummary],
) -> Dict[str, List[LiteralRunSummary]]:
    grouped: Dict[str, List[LiteralRunSummary]] = {}
    for summary in runs:
        if not summary.patterns:
            grouped.setdefault("<none>", []).append(summary)
            continue
        for pattern in summary.patterns:
            grouped.setdefault(pattern.render_tag(), []).append(summary)
    return grouped


def literal_run_note_lines(
    runs: Sequence[LiteralRunSummary], *, limit: int = 6
) -> List[str]:
    counter: Counter[str] = Counter(
        note.render() for summary in runs for note in summary.notes
    )
    if not counter:
        return []
    lines: List[str] = []
    for text, count in counter.most_common(limit):
        lines.append(f"- note {text}: {count}")
    remaining = len(counter) - limit
    if remaining > 0:
        lines.append(f"- ... {remaining} additional notes")
    return lines


def serialize_literal_run(summary: LiteralRunSummary) -> dict:
    return {
        "kind": summary.kind,
        "count": summary.count,
        "start_offset": summary.start_offset,
        "end_offset": summary.end_offset,
        "unique_values": summary.unique_values,
        "string_length": summary.string_length,
        "numeric_min": summary.numeric_min,
        "numeric_max": summary.numeric_max,
        "numeric_span": summary.numeric_span,
        "enum_values": list(summary.enum_values),
        "comment": summary.comment,
        "detail": summary.detail_lines(),
        "patterns": [pattern.to_dict() for pattern in summary.patterns],
        "notes": [note.to_dict() for note in summary.notes],
    }


def serialize_literal_runs(runs: Sequence[LiteralRunSummary]) -> List[dict]:
    return [serialize_literal_run(run) for run in runs]


def group_literal_runs_by_kind(
    runs: Sequence[LiteralRunSummary],
) -> Dict[str, List[LiteralRunSummary]]:
    grouped: Dict[str, List[LiteralRunSummary]] = {}
    for run in runs:
        grouped.setdefault(run.kind, []).append(run)
    return grouped


def longest_literal_runs(
    runs: Sequence[LiteralRunSummary], limit: int = 5
) -> List[LiteralRunSummary]:
    if limit <= 0:
        return []
    return sorted(runs, key=lambda run: (run.count, run.start_offset), reverse=True)[
        :limit
    ]


def render_literal_runs_table(
    runs: Sequence[LiteralRunSummary], *, limit: int = 10, width: int = 80
) -> List[str]:
    if not runs or limit <= 0:
        return []
    header = f"{'start':<12}{'end':<12}{'kind':<8}{'count':<7}preview"
    lines = [header, "-" * min(width, len(header))]
    for summary in runs[:limit]:
        preview = _truncate_preview(summary.preview, width - 32)
        lines.append(
            f"0x{summary.start_offset:06X} 0x{summary.end_offset:06X}"
            f" {summary.kind:<8}{summary.count:<7}{preview}"
        )
    return lines


__all__ = [
    "LiteralRunEntry",
    "LiteralRunNote",
    "LiteralRunPattern",
    "LiteralRunStatistics",
    "LiteralRunSummary",
    "accumulate_statistics",
    "classify_expression",
    "group_literal_runs_by_pattern",
    "literal_run_histogram",
    "literal_run_note_lines",
    "literal_run_pattern_histogram",
    "group_literal_runs_by_kind",
    "longest_literal_runs",
    "render_literal_runs_table",
    "serialize_literal_run",
    "serialize_literal_runs",
    "summarize_literal_run",
    "summarize_literal_runs",
]
