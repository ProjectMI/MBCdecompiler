"""Manual semantics analysis helpers built from manual annotations.

This module centralises all logic that interprets the rich manual annotations
bundled with the knowledge base.  It exposes high level descriptors that
capture common semantic features (stack behaviour, operand handling, tags) so
that downstream consumers such as the CFG renderer, IR builder and Lua
reconstruction layers can operate with consistent metadata.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .instruction import InstructionWord
from .knowledge import InstructionMetadata, KnowledgeBase


@dataclass(frozen=True)
class StackEffect:
    """Describes the approximated stack interaction for an instruction."""

    inputs: int
    outputs: int
    delta: Optional[float]
    source: str

    def as_tuple(self) -> Tuple[int, int, Optional[float]]:
        return self.inputs, self.outputs, self.delta


@dataclass(frozen=True)
class InstructionSemantics:
    """Normalised description for a specific opcode/mode combination."""

    key: str
    mnemonic: str
    manual_name: str
    summary: Optional[str]
    control_flow: Optional[str]
    stack_delta: Optional[float]
    stack_effect: StackEffect
    tags: Tuple[str, ...]
    comparison_operator: Optional[str]
    enum_values: Dict[int, str]
    enum_namespace: Optional[str]
    struct_context: Optional[str]
    stack_inputs: Optional[int]
    stack_outputs: Optional[int]
    uses_operand: bool
    operand_hint: Optional[str]
    vm_method: str
    vm_call_style: str

    def has_tag(self, tag: str) -> bool:
        return tag.lower() in self.tags


@dataclass(frozen=True)
class AnnotatedInstruction:
    """Instruction with baked-in manual semantics."""

    word: InstructionWord
    semantics: InstructionSemantics

    @property
    def offset(self) -> int:
        return self.word.offset

    @property
    def raw(self) -> int:
        return self.word.raw

    @property
    def opcode(self) -> int:
        return self.word.opcode

    @property
    def mode(self) -> int:
        return self.word.mode

    @property
    def operand(self) -> int:
        return self.word.operand

    def label(self) -> str:
        return self.word.label()


class ManualSemanticAnalyzer:
    """Inspect manual annotations and expose structured semantics."""

    _QUANTITY_KEYWORDS: Mapping[str, int] = {
        "zero": 0,
        "single": 1,
        "one": 1,
        "unity": 1,
        "solo": 1,
        "double": 2,
        "two": 2,
        "pair": 2,
        "both": 2,
        "twin": 2,
        "triple": 3,
        "three": 3,
        "trio": 3,
        "quad": 4,
        "quadruple": 4,
        "four": 4,
        "quartet": 4,
        "quint": 5,
        "quintet": 5,
        "five": 5,
        "penta": 5,
        "sextet": 6,
        "sextuple": 6,
        "six": 6,
        "hextet": 6,
        "septet": 7,
        "seven": 7,
        "hept": 7,
        "octet": 8,
        "eight": 8,
        "ennea": 9,
        "nine": 9,
        "deca": 10,
        "ten": 10,
    }

    _INPUT_KEYWORDS = {
        "consume",
        "consumes",
        "drop",
        "drops",
        "remove",
        "removes",
        "clear",
        "clears",
        "flush",
        "flushes",
        "teardown",
        "tears down",
        "tearing",
        "drain",
        "drains",
        "pop",
        "pops",
        "unwind",
        "cleanup",
        "cleanup",
        "destroy",
        "collapses",
        "collapse",
        "finalises",
        "finalises",
        "finalises",
        "finishes",
        "finish",
        "erase",
        "erases",
        "removing",
        "strip",
        "strips",
        "shrink",
        "shrinks",
        "reduce",
        "reduces",
        "reducing",
    }

    _OUTPUT_KEYWORDS = {
        "push",
        "pushes",
        "produce",
        "produces",
        "yield",
        "yields",
        "leave",
        "leaves",
        "emit",
        "emits",
        "expands",
        "expand",
        "inject",
        "injects",
        "replenish",
        "replenishes",
        "duplicate",
        "duplicates",
        "copy",
        "copies",
        "replicate",
        "replicates",
        "adds",
        "add",
        "restore",
        "restores",
        "returns",
        "returns",
        "return",
        "grow",
        "grows",
        "seed",
        "seeds",
        "loader",
        "load",
        "loads",
        "loaders",
        "fan",
        "fans",
        "spawn",
        "spawns",
        "refresh",
        "refreshes",
        "supply",
        "supplies",
    }

    _BINARY_KEYWORDS = {
        "binary",
        "pair",
        "double",
        "dual",
        "reduce",
        "combine",
        "merge",
        "both",
        "either",
        "fuse",
        "couple",
        "twinned",
    }

    _TERNARY_KEYWORDS = {
        "triple",
        "ternary",
        "triad",
        "trio",
        "triple",
        "third",
    }

    _UNARY_KEYWORDS = {
        "single",
        "unary",
        "negate",
        "invert",
        "flip",
        "toggle",
        "not",
        "solo",
    }

    _DUPLICATE_KEYWORDS = {
        "duplicate",
        "dup",
        "mirror",
        "clone",
        "copy",
        "repeat",
        "replicate",
        "fan",
        "broadcast",
    }

    _LITERAL_HINTS = {
        "literal",
        "const",
        "constant",
        "immediate",
        "immed",
        "value",
        "seed",
        "load",
        "push",
        "embed",
        "inline",
    }

    _STATEFUL_KEYWORDS = {
        "state",
        "counter",
        "cursor",
        "iterator",
        "context",
        "update",
        "store",
        "commit",
        "write",
        "persist",
        "cache",
        "buffer",
    }

    _SIDE_EFFECT_KEYWORDS = {
        "store",
        "write",
        "commit",
        "persist",
        "flush",
        "save",
        "record",
        "emit",
    }

    _COMPARISON_KEYWORDS = [
        ("not_equal", "~="),
        ("not_eq", "~="),
        ("ne", "~="),
        ("less_equal", "<="),
        ("le", "<="),
        ("greater_equal", ">="),
        ("ge", ">="),
        ("greater", ">"),
        ("gt", ">"),
        ("less", "<"),
        ("lt", "<"),
        ("equal", "=="),
        ("eq", "=="),
    ]

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge
        self._cache: Dict[str, InstructionSemantics] = {}
        self.enum_registry: Dict[str, Dict[int, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def describe_word(self, word: InstructionWord) -> AnnotatedInstruction:
        semantics = self.describe_key(word.label())
        return AnnotatedInstruction(word=word, semantics=semantics)

    def describe_key(self, key: str) -> InstructionSemantics:
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        metadata = self.knowledge.instruction_metadata(key)
        manual = self.knowledge.manual_annotation(key)
        semantics = self._build_semantics(key, metadata, manual)
        self._cache[key] = semantics
        return semantics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_semantics(
        self,
        key: str,
        metadata: InstructionMetadata,
        manual: Mapping[str, object],
    ) -> InstructionSemantics:
        mnemonic = metadata.mnemonic
        manual_name = _string_field(manual, "name") or mnemonic
        summary = _string_field(manual, "summary") or metadata.summary
        control_flow = _string_field(manual, "control_flow") or metadata.control_flow

        stack_delta = _coerce_optional_float(manual.get("stack_delta"))
        if stack_delta is None:
            stack_delta = metadata.stack_delta

        operand_hint = _string_field(manual, "operand_hint") or metadata.operand_hint
        uses_operand = manual.get("uses_operand")
        if isinstance(uses_operand, bool):
            operand_usage = uses_operand
        else:
            operand_usage = operand_hint != "none"

        tags = self._collect_tags(
            key,
            manual_name,
            summary,
            control_flow,
            manual,
            metadata=metadata,
            stack_delta=stack_delta,
            operand_hint=operand_hint,
            uses_operand=operand_usage,
        )
        comparison_operator = self._comparison_operator(manual_name, manual, metadata)
        struct_context = _string_field(manual, "struct") or _string_field(manual, "structure")

        enum_values = _mapping_field(manual, "enum_values") or _mapping_field(manual, "enums")
        enum_namespace = _string_field(manual, "enum_namespace") or _string_field(manual, "enum_type")
        parsed_enum: Dict[int, str] = {}
        if enum_values:
            for raw_key, raw_value in enum_values.items():
                numeric = _parse_int(raw_key)
                if numeric is None:
                    continue
                parsed_enum[numeric] = str(raw_value)
            namespace = enum_namespace or _sanitize_namespace(manual_name)
            registry = self.enum_registry.setdefault(namespace, {})
            registry.update(parsed_enum)
            enum_namespace = namespace

        stack_inputs = _coerce_optional_int(manual.get("stack_inputs"))
        stack_outputs = _coerce_optional_int(manual.get("stack_outputs"))

        stack_effect = self._infer_stack_effect(
            manual_name,
            summary,
            stack_delta,
            stack_inputs,
            stack_outputs,
            tags,
        )

        literal_tag = "literal" in tags

        vm_method = _sanitize_identifier(manual_name or mnemonic)
        vm_call_style = "literal" if literal_tag else "method"

        semantics = InstructionSemantics(
            key=key,
            mnemonic=mnemonic,
            manual_name=manual_name,
            summary=summary,
            control_flow=control_flow,
            stack_delta=stack_delta,
            stack_effect=stack_effect,
            tags=tuple(sorted(tags)),
            comparison_operator=comparison_operator,
            enum_values=parsed_enum,
            enum_namespace=enum_namespace,
            struct_context=struct_context,
            stack_inputs=stack_inputs,
            stack_outputs=stack_outputs,
            uses_operand=operand_usage,
            operand_hint=operand_hint,
            vm_method=vm_method,
            vm_call_style=vm_call_style,
        )
        return semantics

    def _collect_tags(
        self,
        key: str,
        manual_name: str,
        summary: Optional[str],
        control_flow: Optional[str],
        manual: Mapping[str, object],
        *,
        metadata: InstructionMetadata,
        stack_delta: Optional[float],
        operand_hint: Optional[str],
        uses_operand: bool,
    ) -> set[str]:
        tags: set[str] = set()
        raw_tags = manual.get("tags")
        if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes)):
            tags.update(str(tag).lower() for tag in raw_tags)
        category = _string_field(manual, "category")
        if category:
            tags.add(category.lower())

        text = f"{manual_name} {summary or ''}".lower()
        metadata_summary = (metadata.summary or "").lower()
        combined_text = f"{text} {metadata_summary}".strip()
        operand_hint_lower = (operand_hint or "").lower()

        if "literal" in text or "literal" in metadata_summary:
            tags.add("literal")
        if "compare" in combined_text or "cmp" in combined_text or "comparison" in combined_text:
            tags.add("comparison")
        if "branch" in combined_text or control_flow == "branch" or metadata.control_flow == "branch":
            tags.add("branch")
        if "jump" in combined_text or control_flow == "jump" or metadata.control_flow == "jump":
            tags.add("jump")
        if control_flow == "call" or metadata.control_flow == "call" or "call" in combined_text:
            tags.add("call")
        if control_flow == "return" or metadata.control_flow == "return" or "return" in combined_text:
            tags.add("return")
        if "loop" in combined_text:
            tags.add("loop")
        if "structure" in combined_text or "structured" in combined_text:
            tags.add("structure")
        if "dispatch" in combined_text:
            tags.add("dispatch")
        if "cleanup" in combined_text or "teardown" in combined_text:
            tags.add("cleanup")
        if "marker" in combined_text or "tag" in combined_text:
            tags.add("marker")
        if "load" in combined_text or "fetch" in combined_text:
            tags.add("load")
        if "store" in combined_text or "commit" in combined_text:
            tags.add("store")
        if any(keyword in combined_text for keyword in self._DUPLICATE_KEYWORDS):
            tags.add("duplicate")
        if "pop" in combined_text or "consume" in combined_text or "teardown" in combined_text:
            tags.add("pop")

        if any(keyword in combined_text for keyword in self._BINARY_KEYWORDS):
            tags.add("binary")
        if any(keyword in combined_text for keyword in self._TERNARY_KEYWORDS):
            tags.add("ternary")
        if any(keyword in combined_text for keyword in self._UNARY_KEYWORDS):
            tags.add("unary")

        if stack_delta is not None:
            if stack_delta > 0 and (
                key.startswith("00:")
                or uses_operand
                or operand_hint_lower in {"zero", "small", "medium", "large", "literal", "immediate"}
            ):
                tags.add("literal")
            if stack_delta < 0 and "cleanup" not in tags:
                tags.add("consume")

        if operand_hint_lower and operand_hint_lower not in {"none", "unknown"}:
            if any(hint in operand_hint_lower for hint in self._LITERAL_HINTS):
                tags.add("literal")

        if any(keyword in combined_text for keyword in self._STATEFUL_KEYWORDS):
            tags.add("stateful")
        if any(keyword in combined_text for keyword in self._SIDE_EFFECT_KEYWORDS):
            tags.add("side-effect")

        if key.endswith(":00") and metadata.control_flow is None and stack_delta:
            if stack_delta > 0:
                tags.add("literal")
        return tags

    def _comparison_operator(
        self,
        manual_name: str,
        manual: Mapping[str, object],
        metadata: InstructionMetadata,
    ) -> Optional[str]:
        explicit = _string_field(manual, "comparison_operator")
        if explicit:
            return explicit
        lower_name = manual_name.lower()
        for needle, operator in self._COMPARISON_KEYWORDS:
            if needle in lower_name:
                return operator
        summary = (metadata.summary or "").lower()
        for needle, operator in self._COMPARISON_KEYWORDS:
            if needle in summary:
                return operator
        return None

    def _infer_stack_effect(
        self,
        manual_name: str,
        summary: Optional[str],
        stack_delta: Optional[float],
        manual_inputs: Optional[int],
        manual_outputs: Optional[int],
        tags: Iterable[str],
    ) -> StackEffect:
        # Start with manual hints when present.
        if manual_inputs is not None and manual_outputs is not None:
            delta = float(manual_outputs - manual_inputs)
            if stack_delta is not None and not math.isclose(delta, stack_delta, abs_tol=0.5):
                delta = stack_delta
            return StackEffect(manual_inputs, manual_outputs, delta, source="manual-override")

        if stack_delta is not None and not math.isfinite(stack_delta):
            stack_delta = None

        text = f"{manual_name} {summary or ''}".lower()
        quantity = self._extract_quantity(text)

        tag_set = set(tag.lower() for tag in tags)

        has_output_hint = any(keyword in text for keyword in self._OUTPUT_KEYWORDS)
        has_input_hint = any(keyword in text for keyword in self._INPUT_KEYWORDS)
        inputs = manual_inputs
        outputs = manual_outputs

        if "unary" in tag_set and (inputs is None or inputs < 1):
            inputs = 1
        if "binary" in tag_set and (inputs is None or inputs < 2):
            inputs = 2
        if "ternary" in tag_set and (inputs is None or inputs < 3):
            inputs = 3
        if "duplicate" in tag_set:
            if inputs is None or inputs < 1:
                inputs = 1
            if outputs is None:
                outputs = inputs + 1
            else:
                outputs = max(outputs, inputs + 1)

        if inputs is None and has_input_hint and quantity is not None:
            inputs = max(quantity, 0)
        if outputs is None and has_output_hint and quantity is not None:
            outputs = max(quantity, 0)

        rounded_delta = None
        if stack_delta is not None:
            rounded = int(round(stack_delta))
            if math.isfinite(stack_delta) and abs(stack_delta - rounded) < 1e-3:
                rounded_delta = rounded

        if inputs is None and outputs is None and rounded_delta is not None:
            if rounded_delta > 0:
                outputs = rounded_delta
                inputs = 0
            elif rounded_delta < 0:
                inputs = abs(rounded_delta)
                outputs = 0
            else:
                inputs = 0
                outputs = 0

        if inputs is None:
            if rounded_delta is not None and outputs is not None:
                inputs = max(0, outputs - rounded_delta)
            elif has_input_hint and quantity is not None:
                inputs = max(0, quantity)
        if outputs is None:
            if rounded_delta is not None and inputs is not None:
                outputs = max(0, inputs + rounded_delta)
            elif has_output_hint and quantity is not None:
                outputs = max(0, quantity)

        if "consume" in tag_set and outputs is None:
            outputs = 0
        if "consume" in tag_set and outputs is not None and outputs > 0:
            outputs = 0

        if inputs is None:
            inputs = 0
        if outputs is None:
            outputs = 0

        if rounded_delta is None:
            rounded_delta = outputs - inputs
        else:
            delta_diff = rounded_delta - (outputs - inputs)
            if delta_diff > 0 and not has_output_hint:
                outputs += delta_diff
            elif delta_diff < 0 and not has_input_hint:
                inputs += abs(delta_diff)

        effect_source = "manual"
        if manual_inputs is None and manual_outputs is None:
            effect_source = "heuristic"
        if "literal" in tag_set and outputs == 0 and rounded_delta:
            outputs = max(outputs, rounded_delta)
        return StackEffect(inputs, outputs, float(rounded_delta), effect_source)

    def _extract_quantity(self, text: str) -> Optional[int]:
        matches = []
        for word, value in self._QUANTITY_KEYWORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", text):
                matches.append(value)
        if not matches:
            return None
        return max(matches)


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _string_field(data: Optional[Mapping[str, object]], key: str) -> Optional[str]:
    if not data:
        return None
    value = data.get(key)
    if value is None:
        return None
    return str(value)


def _mapping_field(
    data: Optional[Mapping[str, object]], key: str
) -> Optional[Mapping[str, object]]:
    if not data:
        return None
    value = data.get(key)
    if isinstance(value, Mapping):
        return value
    return None


def _coerce_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _coerce_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(str(value), 0)
        except ValueError:
            return None


def _parse_int(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def _sanitize_identifier(name: str) -> str:
    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in name]
    if not cleaned:
        return "op"
    identifier = "".join(cleaned)
    identifier = re.sub(r"_+", "_", identifier).strip("_")
    if not identifier:
        identifier = "op"
    if identifier[0].isdigit():
        identifier = f"op_{identifier}"
    return identifier


def _sanitize_namespace(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    if not cleaned:
        cleaned = "Enum"
    if cleaned[0].isdigit():
        cleaned = "N_" + cleaned
    return cleaned
