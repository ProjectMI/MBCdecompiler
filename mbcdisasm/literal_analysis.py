"""Literal decoding helpers shared between the VM and high level renderers.

This module centralises the heuristics that decide how a raw VM operand should
be represented in Lua.  Historically the decision lived in
``vm_analysis._format_operand`` which only recognised small integers and the
most straight forward ASCII sequences.  The generated Lua was therefore
extremely noisy – values such as commas or new lines were rendered as ``0x2C00``
which makes the reconstructed scripts hard to read.  Some helpers already
describe textual data in their summary which further highlighted the mismatch
between the VM trace and the actual semantics.

The implementation below aims to keep the heuristics well documented and easy to
reason about.  Apart from the baseline features that shipped with the original
project the analyser now understands booleans, bit masks and exposes
introspection helpers that unit tests (and interactive debugging) can leverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, List, Optional, Sequence, Tuple

from .manual_semantics import InstructionSemantics

ASCII_PRINTABLE = frozenset(range(0x20, 0x7F))
CONTROL_ESCAPES = {
    0x00: "\\0",
    0x07: "\\a",
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0B: "\\v",
    0x0C: "\\f",
    0x0D: "\\r",
}
SPECIAL_ESCAPES = {
    0x22: "\\\"",
    0x27: "\\'",
    0x5C: "\\\\",
}


class LiteralCategory(Enum):
    """Describe the nature of a rendered literal."""

    ENUM = auto()
    STRING = auto()
    CONTROL = auto()
    BOOLEAN = auto()
    MASK = auto()
    DECIMAL = auto()
    HEX = auto()


@dataclass(frozen=True)
class LiteralValue:
    """Structured literal representation produced by :class:`LiteralAnalyzer`."""

    text: str
    category: LiteralCategory
    raw: int
    fragments: Tuple[str, ...] = ()
    reason: str | None = None
    confidence: float = 1.0
    details: Tuple[str, ...] = ()

    def render(self) -> str:
        """Return the Lua representation for the literal."""

        return self.text


@dataclass(frozen=True)
class LiteralContext:
    """Snapshot capturing the decision making inputs for a literal."""

    operand: int
    semantics: Optional[InstructionSemantics]
    prefer_string: bool
    prefer_boolean: bool
    prefer_mask: bool
    hint: Optional[str]
    enum_namespace: Optional[str]
    enum_key: Optional[Tuple[Tuple[int, str], ...]]

    def cache_key(self) -> Tuple[object, ...]:
        return (
            self.operand,
            self.hint,
            self.prefer_string,
            self.prefer_boolean,
            self.prefer_mask,
            self.enum_namespace,
            self.enum_key,
        )


@dataclass(frozen=True)
class LiteralDiagnostic:
    """Carries additional metadata explaining how a literal was formatted."""

    value: LiteralValue
    context: LiteralContext
    decisions: Tuple[str, ...]

    def to_dict(self) -> dict:
        """Return a serialisable representation useful for debugging."""

        return {
            "value": self.value.text,
            "category": self.value.category.name,
            "raw": self.value.raw,
            "reason": self.value.reason,
            "confidence": self.value.confidence,
            "details": list(self.value.details),
            "decisions": list(self.decisions),
            "hint": self.context.hint,
            "prefer_string": self.context.prefer_string,
            "prefer_boolean": self.context.prefer_boolean,
            "prefer_mask": self.context.prefer_mask,
        }


@dataclass
class LiteralStatistics:
    """Aggregated counters describing the literals observed in a dataset."""

    total: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_reason: dict[str, int] = field(default_factory=dict)

    def update(self, value: LiteralValue) -> None:
        """Record ``value`` into the statistics structure."""

        self.total += 1
        self.by_category[value.category.name] = (
            self.by_category.get(value.category.name, 0) + 1
        )
        if value.reason:
            self.by_reason[value.reason] = self.by_reason.get(value.reason, 0) + 1

    def merge(self, other: "LiteralStatistics") -> "LiteralStatistics":
        """Merge ``other`` into ``self`` returning the updated statistics."""

        self.total += other.total
        for category, count in other.by_category.items():
            self.by_category[category] = self.by_category.get(category, 0) + count
        for reason, count in other.by_reason.items():
            self.by_reason[reason] = self.by_reason.get(reason, 0) + count
        return self

    def most_common_category(self) -> Optional[str]:
        """Return the category with the highest observation count."""

        if not self.by_category:
            return None
        return max(self.by_category, key=self.by_category.get)

    def to_dict(self) -> dict:
        """Serialise the statistics into a simple dictionary."""

        return {
            "total": self.total,
            "categories": dict(self.by_category),
            "reasons": dict(self.by_reason),
        }


@dataclass(frozen=True)
class StringCandidate:
    """Description of a possible textual representation for an operand."""

    text: str
    fragments: Tuple[str, ...]
    contains_control: bool


class LiteralAnalyzer:
    """Decodes numeric operands into friendly Lua representations."""

    _STRING_KEYWORDS = (
        "string",
        "text",
        "char",
        "name",
        "message",
        "label",
        "glyph",
        "ascii",
        "title",
    )
    _BOOLEAN_KEYWORDS = (
        "bool",
        "boolean",
        "flag",
        "toggle",
        "enabled",
        "disabled",
    )
    _MASK_KEYWORDS = (
        "mask",
        "bitmask",
        "flags",
        "bits",
    )

    def __init__(self) -> None:
        self._cache: dict[Tuple[object, ...], LiteralValue] = {}

    # ------------------------------------------------------------------
    def analyse(
        self, operand: int, semantics: Optional[InstructionSemantics] = None
    ) -> LiteralValue:
        """Return a :class:`LiteralValue` describing ``operand``."""

        context = self._context_for(operand, semantics)
        cache_key = context.cache_key()
        if cache_key in self._cache:
            return self._cache[cache_key]
        value, _ = self._analyse_internal(context)
        self._cache[cache_key] = value
        return value

    def analyse_with_diagnostics(
        self, operand: int, semantics: Optional[InstructionSemantics] = None
    ) -> LiteralDiagnostic:
        """Analyse ``operand`` returning both the value and decision trail."""

        context = self._context_for(operand, semantics)
        value, decisions = self._analyse_internal(context)
        return LiteralDiagnostic(value=value, context=context, decisions=tuple(decisions))

    def analyse_sequence(
        self, operands: Sequence[int], semantics: Optional[InstructionSemantics] = None
    ) -> List[LiteralValue]:
        """Convenience helper that analyses a batch of operands."""

        return [self.analyse(operand, semantics) for operand in operands]

    def analyse_program_operands(
        self,
        operands: Sequence[int],
        semantics: Sequence[Optional[InstructionSemantics]],
    ) -> List[LiteralValue]:
        """Analyse a list of operands using matching semantics descriptors."""

        if len(operands) != len(semantics):
            raise ValueError("operands and semantics length mismatch")
        return [self.analyse(operand, semantic) for operand, semantic in zip(operands, semantics)]

    def clear_cache(self) -> None:
        """Drop any memoised literal decisions."""

        self._cache.clear()

    # ------------------------------------------------------------------
    def _context_for(
        self, operand: int, semantics: Optional[InstructionSemantics]
    ) -> LiteralContext:
        if semantics is None:
            return LiteralContext(
                operand=operand,
                semantics=None,
                prefer_string=False,
                prefer_boolean=False,
                prefer_mask=False,
                hint=None,
                enum_namespace=None,
                enum_key=None,
            )
        prefer_string = self._prefer_string(semantics)
        prefer_boolean = self._prefer_boolean(semantics)
        prefer_mask = self._prefer_mask(semantics)
        enum_key: Optional[Tuple[Tuple[int, str], ...]] = None
        if semantics.enum_values:
            enum_key = tuple(sorted(semantics.enum_values.items()))
        return LiteralContext(
            operand=operand,
            semantics=semantics,
            prefer_string=prefer_string,
            prefer_boolean=prefer_boolean,
            prefer_mask=prefer_mask,
            hint=semantics.operand_hint,
            enum_namespace=semantics.enum_namespace,
            enum_key=enum_key,
        )

    # ------------------------------------------------------------------
    def _analyse_internal(
        self, context: LiteralContext
    ) -> Tuple[LiteralValue, List[str]]:
        decisions: List[str] = []
        semantics = context.semantics
        operand = context.operand

        if semantics is not None and semantics.enum_values:
            enum_value = self._enum_literal(operand, semantics)
            if enum_value is not None:
                decisions.append("enum")
                return enum_value, decisions

        boolean_value = self._boolean_literal(context)
        if boolean_value is not None:
            decisions.append("boolean")
            return boolean_value, decisions

        string_value = self._string_literal(context)
        if string_value is not None:
            decisions.append("string")
            return string_value, decisions

        mask_value = self._mask_literal(context)
        if mask_value is not None:
            decisions.append("mask")
            return mask_value, decisions

        decisions.append("numeric")
        return self._numeric_literal(context), decisions

    # ------------------------------------------------------------------
    def _enum_literal(
        self, operand: int, semantics: InstructionSemantics
    ) -> Optional[LiteralValue]:
        label = semantics.enum_values.get(operand)
        if label is None:
            return None
        if semantics.enum_namespace:
            text = f"{semantics.enum_namespace}.{label}"
        else:
            text = label
        return LiteralValue(
            text=text,
            category=LiteralCategory.ENUM,
            raw=operand,
            fragments=(label,),
            reason="enum-annotation",
            details=("enum",),
        )

    # ------------------------------------------------------------------
    def _prefer_string(self, semantics: InstructionSemantics) -> bool:
        if semantics.has_tag("string"):
            return True
        sources = [semantics.summary or "", semantics.mnemonic or ""]
        if semantics.manual_name and semantics.manual_name != semantics.mnemonic:
            sources.append(semantics.manual_name)
        joined = " ".join(sources).lower()
        return any(keyword in joined for keyword in self._STRING_KEYWORDS)

    def _prefer_boolean(self, semantics: InstructionSemantics) -> bool:
        if semantics.has_tag("boolean") or semantics.has_tag("flag"):
            return True
        if semantics.operand_hint in {"boolean", "bool", "flag"}:
            return True
        joined = " ".join(
            filter(None, (semantics.summary, semantics.mnemonic, semantics.manual_name))
        ).lower()
        return any(keyword in joined for keyword in self._BOOLEAN_KEYWORDS)

    def _prefer_mask(self, semantics: InstructionSemantics) -> bool:
        if semantics.has_tag("mask") or semantics.has_tag("flags"):
            return True
        if semantics.operand_hint in {"mask", "flags"}:
            return True
        joined = " ".join(
            filter(None, (semantics.summary, semantics.mnemonic, semantics.manual_name))
        ).lower()
        return any(keyword in joined for keyword in self._MASK_KEYWORDS)

    # ------------------------------------------------------------------
    def _boolean_literal(self, context: LiteralContext) -> Optional[LiteralValue]:
        if not context.prefer_boolean:
            return None
        operand = context.operand
        if operand not in (0, 1):
            return None
        text = "true" if operand else "false"
        return LiteralValue(
            text=text,
            category=LiteralCategory.BOOLEAN,
            raw=operand,
            reason="boolean-hint",
            details=("boolean",),
        )

    def _mask_literal(self, context: LiteralContext) -> Optional[LiteralValue]:
        if not context.prefer_mask:
            return None
        operand = context.operand
        if operand == 0:
            return LiteralValue(
                text="0",
                category=LiteralCategory.DECIMAL,
                raw=operand,
                reason="mask-zero",
                details=("mask",),
            )
        if operand & (operand - 1) == 0:
            shift = operand.bit_length() - 1
            return LiteralValue(
                text=f"(1 << {shift})",
                category=LiteralCategory.MASK,
                raw=operand,
                reason="mask-single-bit",
                details=("mask", "single-bit"),
            )
        binary = f"0b{operand:016b}".rstrip("0")
        return LiteralValue(
            text=binary,
            category=LiteralCategory.MASK,
            raw=operand,
            reason="mask-multi-bit",
            details=("mask", "multi-bit"),
        )

    def _string_literal(self, context: LiteralContext) -> Optional[LiteralValue]:
        candidate = self._string_candidate(context.operand, context.prefer_string)
        if candidate is None:
            return None
        category = (
            LiteralCategory.CONTROL
            if candidate.contains_control
            else LiteralCategory.STRING
        )
        detail = "control" if candidate.contains_control else "string"
        return LiteralValue(
            text=candidate.text,
            category=category,
            raw=context.operand,
            fragments=candidate.fragments,
            reason="ascii-bytes",
            details=(detail,),
        )

    def _string_candidate(
        self, operand: int, prefer_string: bool
    ) -> Optional[StringCandidate]:
        raw = list(operand.to_bytes(2, "little", signed=False))
        while raw and raw[-1] == 0:
            raw.pop()
        while len(raw) > 1 and raw and raw[0] == 0:
            raw.pop(0)
        if not raw:
            return None
        meaningful = [byte for byte in raw if byte != 0]
        if not meaningful and not prefer_string:
            return None
        if not prefer_string and any(byte not in ASCII_PRINTABLE for byte in meaningful):
            return None
        fragments: List[str] = []
        contains_control = False
        for byte in raw:
            if byte == 0 and not prefer_string:
                continue
            escaped = self._escape_byte(byte)
            if escaped is None:
                return None
            if escaped in CONTROL_ESCAPES.values() or escaped.startswith("\\x"):
                contains_control = True
            fragments.append(escaped)
        if not fragments:
            return None
        text = '"' + "".join(fragments) + '"'
        return StringCandidate(text=text, fragments=tuple(fragments), contains_control=contains_control)

    def _numeric_literal(self, context: LiteralContext) -> LiteralValue:
        operand = context.operand
        semantics = context.semantics
        signed = operand if operand < 0x8000 else operand - 0x10000
        hint = context.hint
        prefer_hex = False
        if hint in {"large", "address", "pointer"}:
            prefer_hex = True
        elif hint in {"zero", "tiny"}:
            prefer_hex = False
        elif context.prefer_boolean:
            prefer_hex = False
        elif abs(signed) <= 9:
            prefer_hex = False
        elif abs(signed) <= 255 and hint not in {"large", "medium"}:
            prefer_hex = False
        else:
            prefer_hex = True
        if not prefer_hex:
            return LiteralValue(
                text=str(signed),
                category=LiteralCategory.DECIMAL,
                raw=operand,
                reason="signed-decimal",
                details=("numeric",),
            )
        return LiteralValue(
            text=f"0x{operand:04X}",
            category=LiteralCategory.HEX,
            raw=operand,
            reason="hex-fallback",
            details=("numeric", "hex"),
        )

    def _escape_byte(self, byte: int) -> Optional[str]:
        if byte in CONTROL_ESCAPES:
            return CONTROL_ESCAPES[byte]
        if byte in SPECIAL_ESCAPES:
            return SPECIAL_ESCAPES[byte]
        if byte in ASCII_PRINTABLE:
            return chr(byte)
        if byte:
            return f"\\x{byte:02X}"
        return "\\0"


_DEFAULT_ANALYZER = LiteralAnalyzer()


def render_literal(
    operand: int, semantics: Optional[InstructionSemantics] = None
) -> str:
    """Return the preferred Lua representation for ``operand``."""

    return _DEFAULT_ANALYZER.analyse(operand, semantics).render()


def render_literal_sequence(
    operands: Sequence[int], semantics: Optional[InstructionSemantics] = None
) -> List[str]:
    """Render a batch of operands returning their textual representations."""

    return [value.render() for value in _DEFAULT_ANALYZER.analyse_sequence(operands, semantics)]


def explain_literal(
    operand: int, semantics: Optional[InstructionSemantics] = None
) -> Tuple[str, LiteralCategory, Optional[str]]:
    """Expose the literal text, category and reasoning for tests."""

    diagnostic = _DEFAULT_ANALYZER.analyse_with_diagnostics(operand, semantics)
    return diagnostic.value.text, diagnostic.value.category, diagnostic.value.reason


def describe_literal(value: LiteralValue) -> str:
    """Return a human readable description highlighting the category."""

    base = f"{value.text} ({value.category.name.lower()})"
    if value.reason:
        base += f" reason={value.reason}"
    if value.details:
        base += f" details={','.join(value.details)}"
    return base


def compute_statistics(values: Iterable[LiteralValue]) -> LiteralStatistics:
    """Build :class:`LiteralStatistics` from ``values``."""

    stats = LiteralStatistics()
    for value in values:
        stats.update(value)
    return stats


def summarise_literals(values: Iterable[LiteralValue]) -> dict:
    """Aggregate statistics for a sequence of literal values."""

    return compute_statistics(values).to_dict()


def iter_literal_fragments(values: Iterable[LiteralValue]) -> Iterable[str]:
    """Yield individual string fragments embedded in ``values``."""

    for value in values:
        for fragment in value.fragments:
            yield fragment


def diagnostics_table(diagnostics: Sequence[LiteralDiagnostic]) -> List[str]:
    """Render diagnostics as a textual table suitable for debugging."""

    header = "category | text | reason | decisions"
    rows = [header, "-" * len(header)]
    for diagnostic in diagnostics:
        value = diagnostic.value
        decisions = ",".join(diagnostic.decisions) or "-"
        reason = value.reason or "-"
        rows.append(
            f"{value.category.name:<8} | {value.text:<12} | {reason:<16} | {decisions}"
        )
    return rows


def diagnostics_to_json(diagnostics: Sequence[LiteralDiagnostic]) -> str:
    """Serialise diagnostics into a JSON string."""

    payload = [diagnostic.to_dict() for diagnostic in diagnostics]
    return json.dumps(payload, indent=2)


def statistics_to_table(stats: LiteralStatistics) -> List[str]:
    """Render :class:`LiteralStatistics` as a textual table."""

    rows = ["category | count", "-" * 18]
    for category, count in sorted(stats.by_category.items()):
        rows.append(f"{category:<8} | {count:>5}")
    rows.append("")
    rows.append("reason | count")
    rows.append("-" * 15)
    for reason, count in sorted(stats.by_reason.items()):
        rows.append(f"{reason:<8} | {count:>5}")
    return rows


class LiteralReportBuilder:
    """High level helper that captures literals, diagnostics and statistics."""

    def __init__(self, analyzer: Optional[LiteralAnalyzer] = None) -> None:
        self._analyzer = analyzer or LiteralAnalyzer()
        self._values: List[LiteralValue] = []
        self._diagnostics: List[LiteralDiagnostic] = []

    def add_operand(
        self, operand: int, semantics: Optional[InstructionSemantics] = None
    ) -> LiteralValue:
        """Analyse ``operand`` and store both the value and diagnostic."""

        diagnostic = self._analyzer.analyse_with_diagnostics(operand, semantics)
        self._values.append(diagnostic.value)
        self._diagnostics.append(diagnostic)
        return diagnostic.value

    def add_value(
        self, value: LiteralValue, *, diagnostic: Optional[LiteralDiagnostic] = None
    ) -> None:
        """Insert an externally produced literal into the report."""

        self._values.append(value)
        if diagnostic is not None:
            self._diagnostics.append(diagnostic)

    def diagnostics(self) -> Sequence[LiteralDiagnostic]:
        """Return the collected diagnostics."""

        return list(self._diagnostics)

    def statistics(self) -> LiteralStatistics:
        """Return statistics computed from the current values."""

        return compute_statistics(self._values)

    def summary_lines(self) -> List[str]:
        """Render a human readable summary for the collected literals."""

        stats = self.statistics()
        lines = [f"total literals: {stats.total}"]
        for category, count in sorted(stats.by_category.items()):
            lines.append(f"  - {category.lower()}: {count}")
        if stats.by_reason:
            lines.append("reasons:")
            for reason, count in sorted(stats.by_reason.items()):
                lines.append(f"  - {reason}: {count}")
        return lines

    def to_json(self) -> str:
        """Serialise the report (values, diagnostics and statistics) into JSON."""

        payload = {
            "values": [value.render() for value in self._values],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self._diagnostics],
            "statistics": self.statistics().to_dict(),
        }
        return json.dumps(payload, indent=2)
