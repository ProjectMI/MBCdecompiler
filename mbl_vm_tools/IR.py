from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from mbl_vm_tools.parser import FunctionEntry, MBCModule
from mbl_vm_tools.tokenizer import Token, tokenize_stream


SEMANTIC_PRECEDENCE = {
    "return": 0,
    "syscall": 1,
    "call": 2,
    "branch": 3,
    "cmp": 4,
    "write_field": 5,
    "store": 6,
    "read_field": 7,
    "load": 8,
    "make_record": 9,
    "aggregate": 10,
    "const": 11,
    "data": 12,
    "opaque_const": 13,
    "opaque_load": 14,
    "opaque_store": 15,
    "opaque_call": 16,
    "opaque_branch": 17,
    "opaque_cmp": 18,
    "opaque_record": 19,
    "opaque_aggregate": 20,
    "opaque_data": 21,
    "opaque_op": 22,
    "unknown": 23,
}

CONST_KINDS = {"IMM", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32", "F32", "OPU16"}
REF_KINDS = {"REF", "REF16"}
RECORD_KINDS = {"REC41", "REC61", "REC62"}
CALL_KINDS = {"CALL66", "CALL63A", "CALL63B"}
BRANCH_KINDS = {"BR"}
RETURN_KINDS = {"PAIR72_23", "SIG_RETURN_TAIL"}
DATA_KINDS = {"PAD", "ASCII", "DWBLOB"}
AGGREGATE_KINDS = {"AGG", "AGG0"}
RAW_OP_KINDS = {"OP"}
UNKNOWN_KINDS = {"UNK"}
SIGNATURE_RULES: dict[str, tuple[str, list[str], float]] = {
    "SIG_GETWEAR_WRAPPER": ("read_field", ["aggregate", "load", "read_field"], 0.62),
    "SIG_GETP_WRAPPER": ("read_field", ["aggregate", "load", "read_field"], 0.66),
    "SIG_INPUTDONE_SHORT": ("cmp", ["load", "cmp"], 0.72),
    "SIG_AGG2_PARTIAL_HEAD": ("aggregate", ["aggregate"], 0.72),
    "SIG_AGG1_PARTIAL_HEAD": ("aggregate", ["aggregate"], 0.72),
    "SIG_USECLIENT_ALT_HEAD": ("opaque_call", ["const", "opaque_call"], 0.34),
    "SIG_PADDED_CHECKPUT": ("cmp", ["cmp"], 0.62),
    "SIG_USEOWNER_HEAD": ("opaque_call", ["opaque_call"], 0.34),
    "SIG_USECLIENT_HEAD": ("opaque_call", ["opaque_call"], 0.34),
    "SIG_UNIQUEGEN_HEAD": ("opaque_call", ["opaque_call"], 0.34),
    "SIG_USEOFF_HEAD": ("load", ["const", "load"], 0.70),
    "SIG_INPUTDONE_HEAD": ("cmp", ["load", "cmp"], 0.68),
    "SIG_CALL66_REFPAIR_HEAD": ("syscall", ["const", "load", "syscall"], 0.86),
    "SIG_CALL66_SMALLIMM": ("syscall", ["const", "syscall"], 0.86),
    "SIG_CONST_U32_TRAILER": ("const", ["const"], 0.94),
    "SIG_SLOT_CONST": ("read_field", ["const", "read_field"], 0.76),
    "SIG_SETOSST_HEAD": ("store", ["const", "write_field", "store"], 0.64),
    "SIG_GETPLAYERID_HEAD": ("read_field", ["const", "read_field"], 0.48),
    "SIG_PAD17": ("data", ["data"], 1.00),
    "SIG_PAD11_BR": ("branch", ["cmp", "branch"], 0.82),
    "SIG_PADRUN_BR": ("branch", ["cmp", "branch"], 0.86),
    "SIG_PADRUN_OPREF": ("load", ["data", "load"], 0.72),
    "SIG_USEOFF_CONST_CHAIN": ("const", ["const"], 0.72),
    "SIG_GETMODIFIERS_PADTAIL": ("load", ["load"], 0.68),
    "SIG_GETCASTLENUM_HEAD": ("load", ["load"], 0.68),
    "SIG_CONST_0100": ("const", ["const"], 1.00),
    "SIG_CONST_U32_REC62": ("make_record", ["const", "make_record"], 0.90),
    "SIG_CONST_U32_PFX_3D_REF": ("load", ["const", "load"], 0.90),
    "SIG_CONST_U32_IMM16": ("const", ["const"], 0.96),
    "SIG_CONST_U32_REF": ("load", ["const", "load"], 0.94),
    "SIG_TELEP_CREATEINFOPICTURE_TAIL": ("syscall", ["const", "make_record", "load", "syscall"], 0.76),
    "SIG_PLAYER_GETLEADER_TAIL": ("const", ["const", "const"], 0.56),
    "SIG_PLAYER_LOSTITEM2_TAIL": ("return", ["const", "load", "const", "syscall", "return"], 0.64),
    "SIG_MAIN_PARSECOMMAND_TAIL": ("branch", ["const", "const", "cmp", "branch"], 0.58),
    "SIG_CONST_U32_PFX_3D_30_REF": ("load", ["const", "load"], 0.88),
    "SIG_CONST_U32_PFX_5E_REF": ("load", ["const", "load"], 0.88),
    "SIG_CONST_U32_PFX_3D_PAIR72_23": ("return", ["const", "return"], 0.80),
    "SIG_CONST_U32_REF16": ("load", ["const", "load"], 0.93),
    "SIG_CONST_U32_IMM": ("const", ["const"], 0.96),
    "SIG_CONST_U32_CALL66": ("syscall", ["const", "syscall"], 0.98),
    "SIG_CONST_U32_CALL63A": ("call", ["const", "call"], 0.95),
    "SIG_CONST_U32_5E_IMM": ("const", ["const"], 0.88),
    "SIG_CONST_U32_26_CALL66": ("syscall", ["const", "syscall"], 0.92),
    "SIG_CONST_U32_26_REF": ("load", ["const", "load"], 0.88),
    "SIG_CONST_U32_REC41": ("make_record", ["const", "make_record"], 0.90),
    "SIG_U32_U8_CALL66_TAIL": ("syscall", ["const", "syscall"], 0.94),
}


@dataclass
class IRNode:
    index: int
    offset: int
    size: int
    raw_kind: str
    terminal_kind: str
    raw_pattern: str
    category: str
    opcode: str
    semantic_op: str
    semantic_components: list[str]
    lowering_rule: str
    confidence: float
    prefix_chain: list[str]
    operands: dict[str, Any]
    control: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IRFunction:
    name: str
    slice_mode: str
    span: dict[str, int]
    body_selection: dict[str, Any]
    nodes: list[IRNode]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportBodySelection:
    name: str
    slice_mode: str
    used_fallback: bool
    reason: str
    span: dict[str, int]
    public_span: dict[str, int] | None
    exact_span: dict[str, int] | None
    candidates: list[dict[str, Any]]
    entry: dict[str, Any] | None = None
    raw: bytes = field(repr=False, default=b"")
    tokens: list[Token] = field(repr=False, default_factory=list)
    nodes: list[IRNode] = field(repr=False, default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "slice_mode": self.slice_mode,
            "used_fallback": self.used_fallback,
            "reason": self.reason,
            "span": self.span,
            "public_span": self.public_span,
            "exact_span": self.exact_span,
            "candidates": self.candidates,
        }
        if self.entry is not None:
            payload["entry"] = self.entry
        return payload


def _hex_byte(value: int) -> str:
    return f"0x{value:02X}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _unique_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _choose_primary_semantic(components: list[str]) -> str:
    if not components:
        return "unknown"
    uniq = _unique_preserve(components)
    return sorted(uniq, key=lambda item: SEMANTIC_PRECEDENCE.get(item, 999))[0]


def _unwrap_prefixed_token(kind: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    terminal_kind = kind
    terminal_payload: dict[str, Any] = payload
    prefixes: list[str] = []

    while (
        isinstance(terminal_payload, dict)
        and "prefix_op" in terminal_payload
        and "nested_kind" in terminal_payload
        and "nested" in terminal_payload
    ):
        prefix_op = terminal_payload["prefix_op"]
        if isinstance(prefix_op, int):
            prefixes.append(_hex_byte(prefix_op))
        else:
            prefixes.append(str(prefix_op))
        terminal_kind = str(terminal_payload["nested_kind"])
        nested = terminal_payload["nested"]
        if not isinstance(nested, dict):
            terminal_payload = {"value": _json_safe(nested)}
            break
        terminal_payload = nested

    return terminal_kind, _json_safe(terminal_payload), prefixes


def _category_for_kind(raw_kind: str, terminal_kind: str) -> str:
    if raw_kind in AGGREGATE_KINDS or terminal_kind in AGGREGATE_KINDS:
        return "aggregate"
    if terminal_kind in CONST_KINDS or raw_kind.startswith("SIG_CONST_"):
        return "const"
    if terminal_kind in REF_KINDS:
        return "ref"
    if terminal_kind in RECORD_KINDS:
        return "record"
    if terminal_kind in CALL_KINDS:
        return "call"
    if terminal_kind in BRANCH_KINDS:
        return "branch"
    if terminal_kind in RETURN_KINDS:
        return "return"
    if terminal_kind in DATA_KINDS:
        return "data"
    if terminal_kind in RAW_OP_KINDS:
        return "op"
    if terminal_kind in UNKNOWN_KINDS:
        return "unknown"
    if raw_kind.startswith("SIG_") or terminal_kind.startswith("SIG_"):
        return "signature"
    return "op"


def _opcode_for_token(raw_kind: str, terminal_kind: str, category: str) -> str:
    if category == "aggregate":
        return "aggregate"
    if category == "const":
        if terminal_kind == "F32":
            return "const.float32"
        if terminal_kind == "IMM24S":
            return "const.s24"
        if terminal_kind == "IMM24U":
            return "const.u24"
        if terminal_kind == "IMM24Z":
            return "const.zero24"
        if terminal_kind == "IMM16":
            return "const.u16"
        if terminal_kind == "IMM32":
            return "const.u32"
        if terminal_kind == "OPU16":
            return "const.op_u16"
        if raw_kind.startswith("SIG_CONST_"):
            return "const.signature"
        return "const"
    if category == "ref":
        return "ref16" if terminal_kind == "REF16" else "ref"
    if category == "record":
        return f"record.{terminal_kind.lower()}"
    if category == "call":
        return {
            "CALL66": "call.native",
            "CALL63A": "call.rel_argv",
            "CALL63B": "call.rel",
        }.get(terminal_kind, "call")
    if category == "branch":
        return "branch"
    if category == "return":
        return "return"
    if category == "data":
        return {
            "PAD": "data.pad",
            "ASCII": "data.ascii",
            "DWBLOB": "data.blob",
        }.get(terminal_kind, "data")
    if category == "signature":
        family = raw_kind if raw_kind.startswith("SIG_") else terminal_kind
        return f"signature.{family[4:].lower()}"
    if category == "unknown":
        return "unknown"
    if category == "op":
        if terminal_kind == "OP":
            return "op.byte"
        return terminal_kind.lower()
    return category


def _normalize_operands(raw_kind: str, terminal_kind: str, payload: dict[str, Any], prefixes: list[str]) -> dict[str, Any]:
    operands: dict[str, Any] = {}

    if prefixes:
        operands["prefix_chain"] = prefixes

    if terminal_kind in {"IMM", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32"}:
        value = payload.get("value", payload.get("imm"))
        width = {
            "IMM": 8,
            "IMM16": 16,
            "IMM24S": 24,
            "IMM24U": 24,
            "IMM24Z": 24,
            "IMM32": 32,
        }[terminal_kind]
        operands.update(
            {
                "value": _json_safe(value),
                "width": width,
                "signed": terminal_kind == "IMM24S",
                "encoding": terminal_kind.lower(),
            }
        )
        if "op" in payload:
            operands["source_op"] = _hex_byte(int(payload["op"]))
    elif terminal_kind == "F32":
        operands.update(
            {
                "value": _json_safe(payload.get("value")),
                "bits": payload.get("bits"),
                "width": 32,
                "encoding": "float32",
            }
        )
    elif terminal_kind == "OPU16":
        operands.update(
            {
                "value": payload.get("value"),
                "width": 16,
                "source_op": _hex_byte(int(payload["op"])) if "op" in payload else None,
            }
        )
        operands = {k: v for k, v in operands.items() if v is not None}
    elif terminal_kind in {"REF", "REF16"}:
        ref_value = payload.get("ref")
        operands.update(
            {
                "ref": ref_value,
                "mode": _hex_byte(int(payload["mode"])) if "mode" in payload and isinstance(payload["mode"], int) else payload.get("mode"),
                "ref_width": 16 if terminal_kind == "REF16" else 32,
            }
        )
        if "op" in payload:
            operands["source_op"] = _hex_byte(int(payload["op"]))
    elif terminal_kind in RECORD_KINDS:
        operands.update(payload)
    elif terminal_kind in CALL_KINDS:
        call_kind = {
            "CALL66": "native",
            "CALL63A": "relative_with_arity",
            "CALL63B": "relative",
        }[terminal_kind]
        operands.update({"call_kind": call_kind, **payload})
    elif terminal_kind == "BR":
        operands.update(
            {
                "branch_op": _hex_byte(int(payload["op"])) if "op" in payload and isinstance(payload["op"], int) else payload.get("op"),
                "offset": payload.get("off"),
            }
        )
    elif raw_kind in AGGREGATE_KINDS or terminal_kind in AGGREGATE_KINDS:
        children = payload.get("children", [])
        operands.update(
            {
                "arity": payload.get("arity"),
                "children": children,
                "child_count": len(children),
                "aggregate_op": _hex_byte(int(payload["op"])) if "op" in payload and isinstance(payload["op"], int) else payload.get("op"),
            }
        )
    elif terminal_kind == "SIG_RETURN_TAIL":
        operands.update(
            {
                "imm": payload.get("imm"),
                "tail_form": payload.get("tail_form"),
                "has_f1_prefix": payload.get("has_f1_prefix", False),
            }
        )
    elif terminal_kind == "PAIR72_23":
        operands.update({"tail": payload.get("bytes", "72 23")})
    elif raw_kind.startswith("SIG_") or terminal_kind.startswith("SIG_"):
        family = raw_kind if raw_kind.startswith("SIG_") else terminal_kind
        operands.update({"family": family[4:], **payload})
    elif terminal_kind in DATA_KINDS:
        operands.update(payload)
    elif terminal_kind == "OP":
        if "op" in payload:
            operands["op"] = _hex_byte(int(payload["op"]))
        else:
            operands.update(payload)
    else:
        operands.update(payload)

    return _json_safe(operands)


def _lower_prefixed_basic(terminal_kind: str, prefix_chain: list[str]) -> tuple[str, list[str], float, str]:
    depth_penalty = 0.03 * len(prefix_chain)
    if terminal_kind in {"IMM", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32", "F32"}:
        return "const", ["const"], _clamp(1.0 - depth_penalty, 0.55, 1.0), "basic.const"
    if terminal_kind == "OPU16":
        return "const", ["const"], _clamp(0.90 - depth_penalty, 0.50, 0.90), "basic.opu16"
    if terminal_kind in REF_KINDS:
        return "load", ["load"], _clamp(0.94 - depth_penalty, 0.55, 0.94), "basic.ref"
    if terminal_kind in RECORD_KINDS:
        return "make_record", ["make_record"], _clamp(0.96 - depth_penalty, 0.60, 0.96), "basic.record"
    if terminal_kind == "CALL66":
        return "syscall", ["syscall"], _clamp(0.98 - depth_penalty, 0.60, 0.98), "basic.call66"
    if terminal_kind in {"CALL63A", "CALL63B"}:
        return "call", ["call"], _clamp(0.98 - depth_penalty, 0.60, 0.98), "basic.call63"
    if terminal_kind == "BR":
        comps = ["branch"] if not prefix_chain else ["cmp", "branch"]
        return "branch", comps, _clamp(0.96 - depth_penalty, 0.50, 0.96), "basic.branch"
    if terminal_kind in {"PAIR72_23", "SIG_RETURN_TAIL"}:
        return "return", ["return"], _clamp(0.98 - depth_penalty, 0.60, 0.98), "basic.return"
    if terminal_kind in AGGREGATE_KINDS:
        return "aggregate", ["aggregate"], _clamp(1.0 - depth_penalty, 0.60, 1.0), "basic.aggregate"
    if terminal_kind in DATA_KINDS:
        return "data", ["data"], 1.0, "basic.data"
    if terminal_kind == "OP":
        return "opaque_op", ["opaque_op"], _clamp(0.18 - depth_penalty, 0.05, 0.18), "basic.op"
    if terminal_kind == "UNK":
        return "unknown", ["unknown"], 0.0, "basic.unknown"
    return "opaque_op", ["opaque_op"], _clamp(0.20 - depth_penalty, 0.05, 0.20), "basic.fallback"


def _lower_signature_family(
    raw_kind: str,
    terminal_kind: str,
    payload: dict[str, Any],
    operands: dict[str, Any],
    prefix_chain: list[str],
) -> tuple[str, list[str], float, str]:
    exact_family = raw_kind if raw_kind in SIGNATURE_RULES else terminal_kind if terminal_kind in SIGNATURE_RULES else None
    if exact_family is not None:
        primary, components, confidence = SIGNATURE_RULES[exact_family]
        return primary, components, confidence, f"signature:{exact_family}"

    family = raw_kind if raw_kind.startswith("SIG_") else terminal_kind if terminal_kind.startswith("SIG_") else raw_kind
    nested_kind = payload.get("nested_kind")
    nested_payload = payload.get("nested")
    if isinstance(nested_kind, str) and isinstance(nested_payload, dict):
        nested_primary, nested_components, nested_conf, nested_rule = _lower_semantics(
            raw_kind=nested_kind,
            terminal_kind=nested_kind,
            payload=nested_payload,
            operands=_json_safe(nested_payload),
            prefix_chain=[],
        )
        components = ["const"] if "CONST" in family else []
        components.extend(nested_components)
        primary = _choose_primary_semantic(components)
        return primary, _unique_preserve(components), _clamp(nested_conf - 0.05, 0.25, 0.92), f"signature-nested:{family}->{nested_rule}"

    family_name = family[4:] if family.startswith("SIG_") else family
    components: list[str] = []
    confidence = 0.42

    if "CONST" in family_name or any(k in operands for k in ("value", "imm", "imm16", "bits")):
        components.append("const")
        confidence = max(confidence, 0.54)
    if "REC" in family_name:
        components.append("make_record")
        confidence = max(confidence, 0.62)
    if "CALL66" in family_name:
        components.append("syscall")
        confidence = max(confidence, 0.72)
    elif "CALL63" in family_name:
        components.append("call")
        confidence = max(confidence, 0.70)
    if "BR" in family_name or "offset" in operands or "off" in operands:
        components.extend(["cmp", "branch"])
        confidence = max(confidence, 0.64)
    if any(k in operands for k in ("ref", "ref_a", "ref_b", "ref_c", "first_ref")):
        components.append("load")
        confidence = max(confidence, 0.58)
    if any(k in operands for k in ("child_ref", "children", "slot_mode")) or family_name.startswith("GET"):
        components.append("read_field")
        confidence = max(confidence, 0.52)
    if family_name.startswith("SET"):
        components.extend(["write_field", "store"])
        confidence = max(confidence, 0.52)
    if family_name.startswith("USE") or family_name.startswith("PLAYER") or family_name.startswith("MAIN") or family_name.startswith("TELEP"):
        components.append("opaque_call")
        confidence = max(confidence, 0.40)
    if "AGG" in family_name or "WRAPPER" in family_name:
        components.append("aggregate")
        confidence = max(confidence, 0.52)
    if family_name.startswith("PAD") and not components:
        components.append("data")
        confidence = max(confidence, 0.70)

    if not components:
        return "opaque_op", ["opaque_op"], 0.25, f"signature-fallback:{family}"

    primary = _choose_primary_semantic(components)
    if primary in {"const", "load", "call", "syscall", "branch", "cmp", "make_record", "aggregate", "read_field", "write_field", "store", "data", "return"}:
        return primary, _unique_preserve(components), _clamp(confidence, 0.30, 0.90), f"signature-heuristic:{family}"
    return primary, _unique_preserve(components), _clamp(confidence, 0.20, 0.75), f"signature-opaque:{family}"


def _lower_semantics(
    raw_kind: str,
    terminal_kind: str,
    payload: dict[str, Any],
    operands: dict[str, Any],
    prefix_chain: list[str],
) -> tuple[str, list[str], float, str]:
    if raw_kind in AGGREGATE_KINDS or terminal_kind in AGGREGATE_KINDS:
        conf = _clamp(1.0 - 0.03 * len(prefix_chain), 0.60, 1.0)
        return "aggregate", ["aggregate"], conf, "basic.aggregate"

    primary, components, confidence, rule = _lower_prefixed_basic(terminal_kind, prefix_chain)
    if rule != "basic.fallback" or terminal_kind in {"OP", "UNK"}:
        return primary, _unique_preserve(components), confidence, rule

    if raw_kind.startswith("SIG_") or terminal_kind.startswith("SIG_"):
        return _lower_signature_family(raw_kind, terminal_kind, payload, operands, prefix_chain)

    return "opaque_op", ["opaque_op"], 0.20, "fallback.opaque_op"




def _to_signed16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


BRANCH_SIGNED_PREFERENCES: dict[str, list[tuple[str, int]]] = {
    "BR": [("start", 0), ("start", 1), ("start", 2), ("end", 0), ("end", 5), ("start", 8), ("absolute", 0)],
    "PFX_3D_BR": [("start", 1), ("start", 0), ("start", 2), ("end", 1), ("absolute", 1)],
    "PFX_EF_BR": [("start", 1), ("start", 0), ("start", 2), ("end", 1)],
    "PFX_72_BR": [("start", 1), ("start", 0), ("start", 2), ("end", 1)],
    "PFX_F3_BR": [("start", 1), ("start", 0), ("start", 2), ("end", 1)],
    "PFX_F6_BR": [("end", 5), ("start", 1), ("start", 0), ("start", 2)],
    "PFX_F1_72_BR": [("start", 2), ("start", 1), ("start", 0), ("end", 2)],
    "PFX_F1_3D_BR": [("start", 2), ("start", 1), ("start", 0), ("end", 2)],
}

def _resolve_branch_target(
    displacement: int,
    node_offset: int,
    node_size: int,
    valid_offsets: set[int],
    raw_pattern: str = "",
) -> dict[str, Any]:
    candidates: list[tuple[str, int]] = []

    def _extend(mode: str, base: int, deltas: list[int], signed: bool = False) -> None:
        prefix = "signed_" if signed else ""
        for delta in deltas:
            candidates.append((f"{prefix}{mode}+{delta}", base + delta))

    _extend("absolute", displacement, list(range(0, 9)))
    _extend("start", node_offset + displacement, list(range(0, 9)))
    _extend("end", node_offset + node_size + displacement, list(range(0, 9)))

    if isinstance(displacement, int) and displacement > 0x7FFF:
        signed_disp = _to_signed16(displacement)
        prefs = BRANCH_SIGNED_PREFERENCES.get(raw_pattern, [("start", 0), ("start", 1), ("start", 2), ("end", 0)])
        for mode, delta in prefs:
            if mode == "absolute":
                base = signed_disp
            elif mode == "end":
                base = node_offset + node_size + signed_disp
            else:
                base = node_offset + signed_disp
            candidates.append((f"signed_{mode}+{delta}", base + delta))

        _extend("start", node_offset + signed_disp, list(range(0, 9)), signed=True)
        _extend("end", node_offset + node_size + signed_disp, list(range(0, 9)), signed=True)
        _extend("absolute", signed_disp, list(range(0, 9)), signed=True)

    matched: list[tuple[str, int]] = []
    seen_targets: set[int] = set()
    for formula, target in candidates:
        if target in valid_offsets and target not in seen_targets:
            matched.append((formula, target))
            seen_targets.add(target)

    if not matched:
        return {
            "resolved": False,
            "target_offset": None,
            "formula": None,
            "confidence": 0.0,
            "candidate_count": 0,
        }

    best_formula, best_target = matched[0]
    immediate_fallthrough = node_offset + node_size
    degenerate_targets = {immediate_fallthrough, node_offset}
    if best_target in degenerate_targets and len(matched) > 1:
        relative_preference = [
            (formula, target)
            for formula, target in matched
            if target not in degenerate_targets and (formula.startswith("start+") or formula.startswith("end+") or formula.startswith("signed_start+") or formula.startswith("signed_end+"))
        ]
        if relative_preference:
            best_formula, best_target = relative_preference[0]
        else:
            for formula, target in matched:
                if target not in degenerate_targets:
                    best_formula, best_target = formula, target
                    break
    if len(matched) == 1:
        confidence = 0.94 if best_formula in {"start+0", "end+0", "absolute+0", "signed_start+0", "signed_start+1", "signed_start+2"} else 0.74
    else:
        confidence = 0.60 if best_formula.startswith("signed_") else 0.52

    return {
        "resolved": True,
        "target_offset": best_target,
        "formula": best_formula,
        "confidence": confidence,
        "candidate_count": len(matched),
    }


def _build_control(node: IRNode, valid_offsets: set[int]) -> dict[str, Any]:
    control = {
        "kind": None,
        "is_terminator": False,
        "fallthrough": True,
    }

    if node.semantic_op == "return":
        control.update({"kind": "return", "is_terminator": True, "fallthrough": False})
        return control

    branch_displacement = None
    if node.semantic_op == "branch":
        branch_displacement = node.operands.get("offset")
        if branch_displacement is None:
            branch_displacement = node.operands.get("off")
        if branch_displacement is None and isinstance(node.operands.get("nested"), dict):
            branch_displacement = node.operands["nested"].get("off")

    if branch_displacement is not None:
        branch_info = _resolve_branch_target(int(branch_displacement), node.offset, node.size, valid_offsets, raw_pattern=node.raw_pattern)
        control.update(
            {
                "kind": "branch",
                "is_terminator": True,
                "fallthrough": True,
                "branch_condition": "prefixed" if node.prefix_chain else "direct",
                "branch_displacement": int(branch_displacement),
                **branch_info,
            }
        )
        return control

    if node.semantic_op in {"call", "syscall"} and "rel" in node.operands:
        call_info = _resolve_branch_target(int(node.operands["rel"]), node.offset, node.size, valid_offsets, raw_pattern=node.raw_pattern)
        control.update(
            {
                "kind": "call",
                "is_terminator": False,
                "fallthrough": True,
                "resolved_target_offset": call_info.get("target_offset"),
                "resolved_target_formula": call_info.get("formula"),
                "resolved_target_confidence": call_info.get("confidence"),
            }
        )
        return control

    return control


def normalize_token(token: Token, index: int) -> IRNode:
    terminal_kind, terminal_payload, prefix_chain = _unwrap_prefixed_token(token.kind, token.payload)
    category = _category_for_kind(token.kind, terminal_kind)
    opcode = _opcode_for_token(token.kind, terminal_kind, category)
    operands = _normalize_operands(token.kind, terminal_kind, terminal_payload, prefix_chain)
    semantic_op, semantic_components, confidence, lowering_rule = _lower_semantics(
        raw_kind=token.kind,
        terminal_kind=terminal_kind,
        payload=token.payload,
        operands=operands,
        prefix_chain=prefix_chain,
    )
    return IRNode(
        index=index,
        offset=token.offset,
        size=token.size,
        raw_kind=token.kind,
        terminal_kind=terminal_kind,
        raw_pattern=token.kind,
        category=category,
        opcode=opcode,
        semantic_op=semantic_op,
        semantic_components=semantic_components,
        lowering_rule=lowering_rule,
        confidence=round(confidence, 4),
        prefix_chain=prefix_chain,
        operands=operands,
    )


def _summarize_body_candidate(mode: str, span: tuple[int, int], raw: bytes) -> tuple[dict[str, Any], list[Token], list[IRNode]]:
    tokens = tokenize_stream(raw)
    nodes = [normalize_token(token, index=idx) for idx, token in enumerate(tokens)]

    node_count = len(nodes)
    unknown_count = sum(1 for node in nodes if node.semantic_op == "unknown")
    data_count = sum(1 for node in nodes if node.semantic_op in {"data", "opaque_data"})
    opaque_count = sum(1 for node in nodes if node.semantic_op.startswith("opaque_"))
    return_count = sum(1 for node in nodes if node.semantic_op == "return")
    branch_count = sum(1 for node in nodes if node.semantic_op in {"branch", "opaque_branch"})
    call_count = sum(1 for node in nodes if node.semantic_op in {"call", "syscall", "opaque_call"})
    mean_confidence = statistics.mean([node.confidence for node in nodes]) if nodes else 0.0
    unknown_ratio = (unknown_count / node_count) if node_count else 1.0
    data_ratio = (data_count / node_count) if node_count else 1.0
    opaque_ratio = (opaque_count / node_count) if node_count else 1.0

    quality_score = (
        (mean_confidence * 100.0)
        - (unknown_ratio * 80.0)
        - (opaque_ratio * 24.0)
        - (data_ratio * 18.0)
        + (min(return_count, 1) * 6.0)
        + (min(call_count, 4) * 2.0)
        + (min(branch_count, 4) * 1.0)
        + min(math.log2(node_count + 1) * 6.0, 18.0)
        + (2.5 if mode == "definition_exact" else 0.0)
        - (10.0 if node_count <= 1 else 0.0)
        - (4.0 if raw and len(raw) < 12 else 0.0)
    )

    stats = {
        "slice_mode": mode,
        "span": {"start": span[0], "end": span[1]},
        "byte_size": len(raw),
        "token_count": len(tokens),
        "node_count": node_count,
        "mean_confidence": mean_confidence,
        "unknown_ratio": unknown_ratio,
        "data_ratio": data_ratio,
        "opaque_ratio": opaque_ratio,
        "return_count": return_count,
        "branch_count": branch_count,
        "call_count": call_count,
        "quality_score": quality_score,
    }
    return stats, tokens, nodes


def _span_dict(span: tuple[int, int] | None) -> dict[str, int] | None:
    if span is None:
        return None
    return {"start": span[0], "end": span[1]}


def _exact_body_is_stable(stats: dict[str, Any] | None) -> bool:
    if stats is None:
        return False
    return (
        stats["node_count"] >= 2
        and stats["mean_confidence"] >= 0.85
        and stats["unknown_ratio"] <= 0.10
        and stats["opaque_ratio"] <= 0.20
    )


def _entry_payload(entry: FunctionEntry | None) -> dict[str, Any] | None:
    return entry.to_dict() if entry is not None else None


def select_definition_body(mod: MBCModule, entry: FunctionEntry) -> ExportBodySelection:
    exact_span, exact_reason = mod.get_function_exact_code_span_with_reason(entry)
    exact_raw = mod._slice_code_span(*exact_span) if exact_span is not None else b""
    exact_stats: dict[str, Any] | None = None
    exact_tokens: list[Token] = []
    exact_nodes: list[IRNode] = []
    if exact_span is not None and exact_raw:
        exact_stats, exact_tokens, exact_nodes = _summarize_body_candidate("definition_exact", exact_span, exact_raw)

    if exact_stats is None:
        return ExportBodySelection(
            name=entry.name,
            slice_mode="definition_exact",
            used_fallback=False,
            reason=exact_reason or "empty_definition_slice",
            span=_span_dict(exact_span) or {"start": 0, "end": 0},
            public_span=_span_dict(exact_span),
            exact_span=_span_dict(exact_span),
            candidates=[],
            entry=_entry_payload(entry),
            raw=b"",
            tokens=[],
            nodes=[],
        )

    return ExportBodySelection(
        name=entry.name,
        slice_mode="definition_exact",
        used_fallback=False,
        reason="definition_exact",
        span=exact_stats["span"],
        public_span=exact_stats["span"],
        exact_span=_span_dict(exact_span),
        candidates=[exact_stats],
        entry=_entry_payload(entry),
        raw=exact_raw,
        tokens=exact_tokens,
        nodes=exact_nodes,
    )


def select_export_body(mod: MBCModule, export_name: str, entry: FunctionEntry | None = None) -> ExportBodySelection:
    public_span = mod.get_export_public_code_span(export_name)

    # Fast path: exact definitions are usually the best slice and avoid the old
    # public+exact double tokenization for every export.  Public slicing is now
    # only evaluated when exact slicing is missing or suspicious.
    exact_span, exact_reason = mod.get_export_exact_code_span_with_reason(export_name)
    exact_raw = mod._slice_code_span(*exact_span) if exact_span is not None else b""
    exact_stats: dict[str, Any] | None = None
    exact_tokens: list[Token] = []
    exact_nodes: list[IRNode] = []
    if exact_span is not None and exact_raw:
        exact_stats, exact_tokens, exact_nodes = _summarize_body_candidate("definition_exact", exact_span, exact_raw)

    if _exact_body_is_stable(exact_stats):
        return ExportBodySelection(
            name=entry.name if entry is not None else export_name,
            slice_mode="definition_exact",
            used_fallback=False,
            reason="definition_exact_fast_path",
            span=exact_stats["span"],
            public_span=_span_dict(public_span),
            exact_span=_span_dict(exact_span),
            candidates=[exact_stats],
            entry=_entry_payload(entry),
            raw=exact_raw,
            tokens=exact_tokens,
            nodes=exact_nodes,
        )

    public_raw = mod._slice_code_span(*public_span)
    public_stats: dict[str, Any] | None = None
    public_tokens: list[Token] = []
    public_nodes: list[IRNode] = []
    if public_raw:
        public_stats, public_tokens, public_nodes = _summarize_body_candidate("export_public", public_span, public_raw)

    candidate_stats = [candidate for candidate in (exact_stats, public_stats) if candidate is not None]

    if exact_stats is None and public_stats is None:
        empty_mode = "export_public"
        empty_span = public_span
        if exact_span is not None:
            empty_mode = "definition_exact"
            empty_span = exact_span
        return ExportBodySelection(
            name=entry.name if entry is not None else export_name,
            slice_mode=empty_mode,
            used_fallback=(empty_mode == "export_public"),
            reason=exact_reason or "empty_slice",
            span=_span_dict(empty_span) or {"start": 0, "end": 0},
            public_span=_span_dict(public_span),
            exact_span=_span_dict(exact_span),
            candidates=candidate_stats,
            entry=_entry_payload(entry),
            raw=b"",
            tokens=[],
            nodes=[],
        )

    if exact_stats is None:
        return ExportBodySelection(
            name=entry.name if entry is not None else export_name,
            slice_mode="export_public",
            used_fallback=True,
            reason=exact_reason or "definition_unavailable",
            span=public_stats["span"],
            public_span=_span_dict(public_span),
            exact_span=_span_dict(exact_span),
            candidates=candidate_stats,
            entry=_entry_payload(entry),
            raw=public_raw,
            tokens=public_tokens,
            nodes=public_nodes,
        )

    if public_stats is None or exact_raw == public_raw:
        return ExportBodySelection(
            name=entry.name if entry is not None else export_name,
            slice_mode="definition_exact",
            used_fallback=False,
            reason="definition_exact",
            span=exact_stats["span"],
            public_span=_span_dict(public_span),
            exact_span=_span_dict(exact_span),
            candidates=candidate_stats,
            entry=_entry_payload(entry),
            raw=exact_raw,
            tokens=exact_tokens,
            nodes=exact_nodes,
        )

    exact_delta = public_stats["quality_score"] - exact_stats["quality_score"]
    exact_tiny = exact_stats["node_count"] <= 1
    public_substantially_better = (
        exact_delta >= 12.0
        and public_stats["node_count"] >= max(exact_stats["node_count"] + 3, 4)
    )

    if public_substantially_better and (exact_tiny or not _exact_body_is_stable(exact_stats)):
        return ExportBodySelection(
            name=entry.name if entry is not None else export_name,
            slice_mode="export_public",
            used_fallback=True,
            reason="fallback_public_better_tokenization",
            span=public_stats["span"],
            public_span=_span_dict(public_span),
            exact_span=_span_dict(exact_span),
            candidates=candidate_stats,
            entry=_entry_payload(entry),
            raw=public_raw,
            tokens=public_tokens,
            nodes=public_nodes,
        )

    return ExportBodySelection(
        name=entry.name if entry is not None else export_name,
        slice_mode="definition_exact",
        used_fallback=False,
        reason="definition_exact_preferred",
        span=exact_stats["span"],
        public_span=_span_dict(public_span),
        exact_span=_span_dict(exact_span),
        candidates=candidate_stats,
        entry=_entry_payload(entry),
        raw=exact_raw,
        tokens=exact_tokens,
        nodes=exact_nodes,
    )


def select_function_body(mod: MBCModule, entry_or_name: FunctionEntry | str) -> ExportBodySelection:
    if isinstance(entry_or_name, FunctionEntry):
        entry = entry_or_name
    else:
        entry = mod.get_function_entry(entry_or_name)

    if entry.source_kind == "definition":
        return select_definition_body(mod, entry)
    return select_export_body(mod, entry.symbol, entry=entry)


def build_function_ir(mod: MBCModule, entry_or_name: FunctionEntry | str) -> IRFunction:
    try:
        selection = select_function_body(mod, entry_or_name)
    except KeyError:
        # Compatibility with the old API, where only export names existed.
        if not isinstance(entry_or_name, str):
            raise
        selection = select_export_body(mod, entry_or_name)

    nodes = list(selection.nodes)
    valid_offsets = {node.offset for node in nodes}
    for node in nodes:
        node.control = _build_control(node, valid_offsets)

    return IRFunction(
        name=selection.name,
        slice_mode=selection.slice_mode,
        span=selection.span,
        body_selection=selection.to_dict(),
        nodes=nodes,
    )
