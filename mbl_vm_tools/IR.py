from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from mbl_vm_tools.parser import MBCModule, TableRecord
from mbl_vm_tools.tokenizer import Token, tokenize_stream


CATEGORY_ORDER = (
    "aggregate",
    "const",
    "ref",
    "record",
    "call",
    "branch",
    "return",
    "signature",
    "data",
    "op",
    "unknown",
)

SEMANTIC_VOCABULARY = (
    "const",
    "load",
    "store",
    "call",
    "branch",
    "cmp",
    "return",
    "make_record",
    "read_field",
    "write_field",
    "aggregate",
    "syscall",
    "data",
    "opaque_const",
    "opaque_load",
    "opaque_store",
    "opaque_call",
    "opaque_branch",
    "opaque_cmp",
    "opaque_record",
    "opaque_aggregate",
    "opaque_data",
    "opaque_op",
    "unknown",
)

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
LOW_CONFIDENCE_THRESHOLD = 0.75

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
class IRFunctionSummary:
    name: str
    slice_mode: str
    span: dict[str, int]
    public_span: dict[str, int]
    byte_size: int
    token_count: int
    node_count: int
    category_histogram: dict[str, int]
    opcode_histogram: dict[str, int]
    semantic_histogram: dict[str, int]
    normalized_signature: str
    opcode_signature: str
    raw_signature: str
    unknown_count: int
    unknown_ratio: float
    data_ratio: float
    call_count: int
    branch_count: int
    return_count: int
    opaque_count: int
    opaque_ratio: float
    low_confidence_count: int
    mean_confidence: float
    basic_block_count: int
    cfg_edge_count: int
    cfg_anomaly_count: int
    sig_unlowered_count: int
    unresolved_branch_count: int
    unreachable_block_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IRFunction:
    name: str
    slice_mode: str
    span: dict[str, int]
    public_span: dict[str, int]
    byte_size: int
    raw_token_count: int
    normalized_signature: str
    opcode_signature: str
    raw_signature: str
    summary: IRFunctionSummary
    validation: dict[str, Any]
    cfg: dict[str, Any]
    basic_blocks: list[dict[str, Any]]
    nodes: list[IRNode]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_to_dict(record: TableRecord) -> dict[str, Any]:
    return {
        "offset": record.offset,
        "name": record.name,
        "a": record.a,
        "b": record.b,
        "c": record.c,
    }


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


def _compute_basic_blocks(nodes: list[IRNode], byte_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not nodes:
        cfg = {
            "entry_block": None,
            "blocks": [],
            "edges": [],
            "anomalies": [],
            "stats": {"block_count": 0, "edge_count": 0, "unreachable_blocks": 0, "unresolved_targets": 0},
        }
        return [], cfg

    valid_offsets = {node.offset for node in nodes}
    offset_to_index = {node.offset: idx for idx, node in enumerate(nodes)}

    for node in nodes:
        node.control = _build_control(node, valid_offsets)

    leaders: set[int] = {0}
    for idx, node in enumerate(nodes):
        ctrl = node.control
        if ctrl.get("kind") == "branch":
            target_offset = ctrl.get("target_offset")
            if target_offset in offset_to_index:
                leaders.add(offset_to_index[target_offset])
            if idx + 1 < len(nodes):
                leaders.add(idx + 1)
        elif ctrl.get("kind") == "return":
            if idx + 1 < len(nodes):
                leaders.add(idx + 1)

    sorted_leaders = sorted(leaders)
    blocks: list[dict[str, Any]] = []
    node_to_block: dict[int, str] = {}

    for block_idx, start_idx in enumerate(sorted_leaders):
        end_idx = (sorted_leaders[block_idx + 1] - 1) if block_idx + 1 < len(sorted_leaders) else len(nodes) - 1
        block_nodes = nodes[start_idx:end_idx + 1]
        block_id = f"bb{block_idx}"
        for node in block_nodes:
            node_to_block[node.index] = block_id
        blocks.append(
            {
                "id": block_id,
                "start_offset": block_nodes[0].offset,
                "end_offset": block_nodes[-1].offset + block_nodes[-1].size,
                "node_indices": [node.index for node in block_nodes],
                "node_offsets": [node.offset for node in block_nodes],
                "terminator_index": block_nodes[-1].index,
                "terminator_op": block_nodes[-1].semantic_op,
                "successors": [],
                "predecessors": [],
                "flags": [],
            }
        )

    block_index = {block["id"]: idx for idx, block in enumerate(blocks)}
    edges: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    def _add_edge(src: str, dst: Optional[str], edge_kind: str, confidence: float, target_offset: Optional[int] = None) -> None:
        if dst is None:
            edges.append({"src": src, "dst": None, "kind": edge_kind, "confidence": round(confidence, 4), "target_offset": target_offset})
            return
        edge = {"src": src, "dst": dst, "kind": edge_kind, "confidence": round(confidence, 4)}
        if target_offset is not None:
            edge["target_offset"] = target_offset
        edges.append(edge)
        block = blocks[block_index[src]]
        if dst not in block["successors"]:
            block["successors"].append(dst)
        pred_block = blocks[block_index[dst]]
        if src not in pred_block["predecessors"]:
            pred_block["predecessors"].append(src)

    for idx, block in enumerate(blocks):
        terminator = nodes[block["terminator_index"]]
        next_block_id = blocks[idx + 1]["id"] if idx + 1 < len(blocks) else None
        ctrl = terminator.control

        if ctrl.get("kind") == "return":
            block["flags"].append("returns")
            continue

        if ctrl.get("kind") == "branch":
            target_offset = ctrl.get("target_offset")
            target_block = node_to_block.get(offset_to_index[target_offset], None) if target_offset in offset_to_index else None
            if target_block is not None:
                _add_edge(block["id"], target_block, "branch", ctrl.get("confidence", 0.0), target_offset)
            else:
                anomalies.append(
                    {
                        "kind": "unresolved_branch_target",
                        "block_id": block["id"],
                        "node_index": terminator.index,
                        "node_offset": terminator.offset,
                        "displacement": ctrl.get("branch_displacement"),
                    }
                )
                block["flags"].append("unresolved_branch_target")
                _add_edge(block["id"], None, "branch_unresolved", ctrl.get("confidence", 0.0), target_offset)

            if next_block_id is not None:
                _add_edge(block["id"], next_block_id, "fallthrough", 1.0)
            else:
                block["flags"].append("open_exit")
            continue

        if next_block_id is not None:
            _add_edge(block["id"], next_block_id, "fallthrough", 1.0)
        else:
            block["flags"].append("open_exit")

    reachable: set[str] = set()
    if blocks:
        stack = [blocks[0]["id"]]
        while stack:
            block_id = stack.pop()
            if block_id in reachable:
                continue
            reachable.add(block_id)
            block = blocks[block_index[block_id]]
            for succ in block["successors"]:
                if succ not in reachable:
                    stack.append(succ)

    for block in blocks:
        if block["id"] not in reachable:
            block["flags"].append("unreachable")
            anomalies.append({"kind": "unreachable_block", "block_id": block["id"]})

    cfg = {
        "entry_block": blocks[0]["id"],
        "blocks": [{k: v for k, v in block.items() if k != "node_indices" and k != "node_offsets"} for block in blocks],
        "edges": edges,
        "anomalies": anomalies,
        "stats": {
            "block_count": len(blocks),
            "edge_count": len(edges),
            "unreachable_blocks": sum(1 for block in blocks if "unreachable" in block["flags"]),
            "unresolved_targets": sum(1 for item in anomalies if item["kind"] == "unresolved_branch_target"),
        },
    }
    return blocks, cfg


def _build_validation(nodes: list[IRNode], cfg: dict[str, Any]) -> dict[str, Any]:
    raw_to_semantic: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        raw_to_semantic[node.raw_pattern].add(node.semantic_op)

    semantic_to_raw: dict[str, set[str]] = defaultdict(set)
    unresolved_branch_patterns = Counter()
    for node in nodes:
        semantic_to_raw[node.semantic_op].add(node.raw_pattern)

    opaque_count = sum(1 for node in nodes if node.semantic_op.startswith("opaque_"))
    sig_raw_count = sum(1 for node in nodes if node.raw_kind.startswith("SIG_") or node.terminal_kind.startswith("SIG_"))
    sig_unlowered_count = sum(
        1
        for node in nodes
        if (node.raw_kind.startswith("SIG_") or node.terminal_kind.startswith("SIG_"))
        and node.semantic_op in {"opaque_op", "unknown"}
    )

    collisions = [
        {
            "semantic_op": semantic_op,
            "raw_pattern_count": len(raw_patterns),
            "raw_patterns": sorted(raw_patterns)[:24],
        }
        for semantic_op, raw_patterns in semantic_to_raw.items()
        if len(raw_patterns) > 1
    ]
    collisions.sort(key=lambda item: (-item["raw_pattern_count"], item["semantic_op"]))

    unresolved_branch_patterns = Counter(
        node.raw_pattern
        for node in nodes
        if node.control.get("kind") == "branch" and not node.control.get("resolved")
    )

    return {
        "semantic_vocabulary": list(SEMANTIC_VOCABULARY),
        "sig_raw_count": sig_raw_count,
        "sig_unlowered_count": sig_unlowered_count,
        "opaque_count": opaque_count,
        "opaque_ratio": round((opaque_count / len(nodes)) if nodes else 0.0, 4),
        "low_confidence_count": sum(1 for node in nodes if node.confidence < LOW_CONFIDENCE_THRESHOLD),
        "raw_patterns_per_semantic": collisions[:24],
        "unresolved_branch_patterns": [
            {"raw_pattern": raw_pattern, "count": count}
            for raw_pattern, count in unresolved_branch_patterns.most_common(16)
        ],
        "cfg_anomaly_count": len(cfg.get("anomalies", [])),
    }


def build_function_ir(mod: MBCModule, export_name: str) -> IRFunction:
    exact_span = mod.get_export_exact_code_span(export_name)
    public_span = mod.get_export_public_code_span(export_name)
    using_exact = exact_span is not None
    start, end = exact_span if using_exact else public_span

    raw = mod.get_export_body(export_name, exact=True)
    if not raw:
        raw = mod.get_export_body(export_name, exact=False)

    tokens = tokenize_stream(raw)
    nodes = [normalize_token(token, index=idx) for idx, token in enumerate(tokens)]
    basic_blocks, cfg = _compute_basic_blocks(nodes, len(raw))
    validation = _build_validation(nodes, cfg)

    category_hist = Counter(node.category for node in nodes)
    opcode_hist = Counter(node.opcode for node in nodes)
    semantic_hist = Counter(node.semantic_op for node in nodes)
    semantic_signature = " -> ".join(node.semantic_op for node in nodes)
    opcode_signature = " -> ".join(node.opcode for node in nodes)
    raw_signature = " -> ".join(token.kind for token in tokens)
    node_count = len(nodes)
    unknown_count = semantic_hist.get("unknown", 0)
    data_count = semantic_hist.get("data", 0) + semantic_hist.get("opaque_data", 0)
    opaque_count = validation["opaque_count"]

    summary = IRFunctionSummary(
        name=export_name,
        slice_mode="definition_exact" if using_exact else "export_public",
        span={"start": start, "end": end},
        public_span={"start": public_span[0], "end": public_span[1]},
        byte_size=len(raw),
        token_count=len(tokens),
        node_count=node_count,
        category_histogram={key: category_hist.get(key, 0) for key in CATEGORY_ORDER if category_hist.get(key, 0)},
        opcode_histogram=dict(sorted(opcode_hist.items())),
        semantic_histogram=dict(sorted(semantic_hist.items())),
        normalized_signature=semantic_signature,
        opcode_signature=opcode_signature,
        raw_signature=raw_signature,
        unknown_count=unknown_count,
        unknown_ratio=(unknown_count / node_count) if node_count else 0.0,
        data_ratio=(data_count / node_count) if node_count else 0.0,
        call_count=semantic_hist.get("call", 0) + semantic_hist.get("syscall", 0) + semantic_hist.get("opaque_call", 0),
        branch_count=semantic_hist.get("branch", 0) + semantic_hist.get("opaque_branch", 0),
        return_count=semantic_hist.get("return", 0),
        opaque_count=opaque_count,
        opaque_ratio=(opaque_count / node_count) if node_count else 0.0,
        low_confidence_count=sum(1 for node in nodes if node.confidence < LOW_CONFIDENCE_THRESHOLD),
        mean_confidence=(statistics.mean([node.confidence for node in nodes]) if nodes else 0.0),
        basic_block_count=cfg["stats"]["block_count"],
        cfg_edge_count=cfg["stats"]["edge_count"],
        cfg_anomaly_count=len(cfg["anomalies"]),
        sig_unlowered_count=validation["sig_unlowered_count"],
        unresolved_branch_count=cfg["stats"]["unresolved_targets"],
        unreachable_block_count=cfg["stats"]["unreachable_blocks"],
    )

    return IRFunction(
        name=export_name,
        slice_mode=summary.slice_mode,
        span=summary.span,
        public_span=summary.public_span,
        byte_size=len(raw),
        raw_token_count=len(tokens),
        normalized_signature=semantic_signature,
        opcode_signature=opcode_signature,
        raw_signature=raw_signature,
        summary=summary,
        validation=validation,
        cfg=cfg,
        basic_blocks=basic_blocks,
        nodes=nodes,
    )


def _module_summary(functions: list[IRFunction]) -> dict[str, Any]:
    export_count = len(functions)
    total_nodes = sum(fn.summary.node_count for fn in functions)
    total_bytes = sum(fn.summary.byte_size for fn in functions)
    category_hist = Counter()
    opcode_hist = Counter()
    semantic_hist = Counter()
    total_opaque = 0
    total_sig_unlowered = 0
    total_cfg_anomalies = 0
    total_unresolved_branches = 0

    for fn in functions:
        category_hist.update(fn.summary.category_histogram)
        opcode_hist.update(fn.summary.opcode_histogram)
        semantic_hist.update(fn.summary.semantic_histogram)
        total_opaque += fn.summary.opaque_count
        total_sig_unlowered += fn.summary.sig_unlowered_count
        total_cfg_anomalies += fn.summary.cfg_anomaly_count
        total_unresolved_branches += fn.summary.unresolved_branch_count

    return {
        "export_count": export_count,
        "total_nodes": total_nodes,
        "total_bytes": total_bytes,
        "avg_nodes_per_export": (total_nodes / export_count) if export_count else 0.0,
        "avg_bytes_per_export": (total_bytes / export_count) if export_count else 0.0,
        "category_histogram": {key: category_hist.get(key, 0) for key in CATEGORY_ORDER if category_hist.get(key, 0)},
        "top_opcodes": [{"opcode": opcode, "count": count} for opcode, count in opcode_hist.most_common(24)],
        "top_semantic_ops": [{"semantic_op": opcode, "count": count} for opcode, count in semantic_hist.most_common(24)],
        "opaque_count": total_opaque,
        "sig_unlowered_count": total_sig_unlowered,
        "cfg_anomaly_count": total_cfg_anomalies,
        "unresolved_branch_count": total_unresolved_branches,
    }


def build_module_ir(path: str | Path, include_nodes: bool = True) -> dict[str, Any]:
    path = Path(path)
    mod = MBCModule(path, collect_auxiliary=False)
    functions = [build_function_ir(mod, name) for name in mod.export_names()]

    exports_payload = [fn.to_dict() if include_nodes else {**fn.summary.to_dict(), "validation": fn.validation, "cfg": fn.cfg.get("stats", {})} for fn in functions]
    return {
        "path": str(path),
        "script_name": path.name,
        "has_magic_header": mod.has_magic_header,
        "code_base": mod.code_base,
        "code_size": mod.code_size,
        "data_blob_size": mod.data_blob_size,
        "definition_count": len(mod.definitions),
        "globals_count": len(mod.globals),
        "exports_count": len(mod.exports),
        "semantic_vocabulary": list(SEMANTIC_VOCABULARY),
        "definitions": [_record_to_dict(record) for record in mod.definitions],
        "globals": [_record_to_dict(record) for record in mod.globals],
        "summary": _module_summary(functions),
        "exports": exports_payload,
    }


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _z_score(value: float, mean: float, stdev: float) -> float:
    if stdev <= 1e-9:
        return 0.0
    return (value - mean) / stdev


def _init_corpus_state() -> dict[str, Any]:
    return {
        "function_rows": [],
        "module_rows": [],
        "global_categories": Counter(),
        "global_opcodes": Counter(),
        "global_semantics": Counter(),
        "signature_counts": Counter(),
        "semantic_to_raw": defaultdict(set),
        "unresolved_branch_patterns": Counter(),
    }


def _add_module_to_corpus_state(state: dict[str, Any], module: dict[str, Any]) -> None:
    function_rows: list[dict[str, Any]] = state["function_rows"]
    module_rows: list[dict[str, Any]] = state["module_rows"]
    global_categories: Counter = state["global_categories"]
    global_opcodes: Counter = state["global_opcodes"]
    global_semantics: Counter = state["global_semantics"]
    signature_counts: Counter = state["signature_counts"]
    semantic_to_raw: dict[str, set[str]] = state["semantic_to_raw"]
    unresolved_branch_patterns: Counter = state["unresolved_branch_patterns"]

    module_summary = module.get("summary", {})
    module_rows.append(
        {
            "script_name": module.get("script_name"),
            "path": module.get("path"),
            "export_count": int(module_summary.get("export_count", 0)),
            "total_nodes": int(module_summary.get("total_nodes", 0)),
            "avg_nodes_per_export": float(module_summary.get("avg_nodes_per_export", 0.0)),
            "cfg_anomaly_count": int(module_summary.get("cfg_anomaly_count", 0)),
        }
    )
    global_categories.update(module_summary.get("category_histogram", {}))
    for item in module_summary.get("top_opcodes", []):
        global_opcodes[item["opcode"]] += item["count"]
    for item in module_summary.get("top_semantic_ops", []):
        global_semantics[item["semantic_op"]] += item["count"]

    for fn in module.get("exports", []):
        validation = fn.get("validation", {})
        row = {
            "script_name": module.get("script_name"),
            "name": fn.get("name"),
            "node_count": int(fn.get("node_count", fn.get("summary", {}).get("node_count", 0))),
            "unknown_ratio": float(fn.get("unknown_ratio", fn.get("summary", {}).get("unknown_ratio", 0.0))),
            "data_ratio": float(fn.get("data_ratio", fn.get("summary", {}).get("data_ratio", 0.0))),
            "opaque_ratio": float(fn.get("opaque_ratio", fn.get("summary", {}).get("opaque_ratio", 0.0))),
            "mean_confidence": float(fn.get("mean_confidence", fn.get("summary", {}).get("mean_confidence", 0.0))),
            "return_count": int(fn.get("return_count", fn.get("summary", {}).get("return_count", 0))),
            "cfg_anomaly_count": int(fn.get("cfg_anomaly_count", fn.get("summary", {}).get("cfg_anomaly_count", 0))),
            "unresolved_branch_count": int(fn.get("unresolved_branch_count", fn.get("summary", {}).get("unresolved_branch_count", 0))),
            "sig_unlowered_count": int(fn.get("sig_unlowered_count", fn.get("summary", {}).get("sig_unlowered_count", validation.get("sig_unlowered_count", 0)))),
            "normalized_signature": fn.get("normalized_signature", fn.get("summary", {}).get("normalized_signature", "")),
        }
        for entry in validation.get("raw_patterns_per_semantic", []):
            for raw_pattern in entry.get("raw_patterns", []):
                semantic_to_raw[entry.get("semantic_op", "unknown")].add(raw_pattern)
        for entry in validation.get("unresolved_branch_patterns", []):
            unresolved_branch_patterns[entry.get("raw_pattern", "<unknown>")] += int(entry.get("count", 0))
        signature_counts[row["normalized_signature"]] += 1
        function_rows.append(row)


def _finalize_corpus_state(state: dict[str, Any]) -> dict[str, Any]:
    function_rows: list[dict[str, Any]] = state["function_rows"]
    module_rows: list[dict[str, Any]] = state["module_rows"]
    global_categories: Counter = state["global_categories"]
    global_opcodes: Counter = state["global_opcodes"]
    global_semantics: Counter = state["global_semantics"]
    signature_counts: Counter = state["signature_counts"]
    semantic_to_raw: dict[str, set[str]] = state["semantic_to_raw"]
    unresolved_branch_patterns: Counter = state["unresolved_branch_patterns"]

    node_counts = [row["node_count"] for row in function_rows]
    unknown_ratios = [row["unknown_ratio"] for row in function_rows]
    data_ratios = [row["data_ratio"] for row in function_rows]
    opaque_ratios = [row["opaque_ratio"] for row in function_rows]
    cfg_counts = [row["cfg_anomaly_count"] for row in function_rows]
    confidence_values = [row["mean_confidence"] for row in function_rows]

    mean_nodes = _safe_mean(node_counts)
    stdev_nodes = _safe_stdev(node_counts)
    mean_unknown = _safe_mean(unknown_ratios)
    stdev_unknown = _safe_stdev(unknown_ratios)
    mean_data = _safe_mean(data_ratios)
    stdev_data = _safe_stdev(data_ratios)
    mean_opaque = _safe_mean(opaque_ratios)
    stdev_opaque = _safe_stdev(opaque_ratios)
    mean_cfg = _safe_mean(cfg_counts)
    stdev_cfg = _safe_stdev(cfg_counts)

    anomalies: list[dict[str, Any]] = []
    for row in function_rows:
        structure_count = signature_counts[row["normalized_signature"]]
        node_z = _z_score(row["node_count"], mean_nodes, stdev_nodes)
        unknown_z = _z_score(row["unknown_ratio"], mean_unknown, stdev_unknown)
        data_z = _z_score(row["data_ratio"], mean_data, stdev_data)
        opaque_z = _z_score(row["opaque_ratio"], mean_opaque, stdev_opaque)
        cfg_z = _z_score(row["cfg_anomaly_count"], mean_cfg, stdev_cfg)

        score = 0.0
        reasons: list[str] = []
        if abs(node_z) >= 2.5:
            score += abs(node_z)
            reasons.append(f"node_count_z={node_z:.2f}")
        if unknown_z >= 2.0:
            score += unknown_z * 1.25
            reasons.append(f"unknown_ratio_z={unknown_z:.2f}")
        if data_z >= 2.5:
            score += data_z
            reasons.append(f"data_ratio_z={data_z:.2f}")
        if opaque_z >= 2.0:
            score += opaque_z * 1.5
            reasons.append(f"opaque_ratio_z={opaque_z:.2f}")
        if cfg_z >= 2.0 or row["cfg_anomaly_count"] >= 2:
            score += max(cfg_z, 1.0)
            reasons.append(f"cfg_anomalies={row['cfg_anomaly_count']}")
        if row["unresolved_branch_count"] > 0:
            score += row["unresolved_branch_count"] * 0.75
            reasons.append(f"unresolved_branches={row['unresolved_branch_count']}")
        if row["sig_unlowered_count"] > 0:
            score += row["sig_unlowered_count"] * 1.5
            reasons.append(f"sig_unlowered={row['sig_unlowered_count']}")
        if structure_count == 1 and row["node_count"] >= max(8, int(mean_nodes)):
            score += 1.0
            reasons.append("rare_semantic_signature")
        if row["return_count"] == 0 and row["node_count"] >= 8:
            score += 0.5
            reasons.append("no_explicit_return")

        if score > 0:
            anomalies.append(
                {
                    "script_name": row["script_name"],
                    "export_name": row["name"],
                    "score": round(score, 3),
                    "structure_frequency": structure_count,
                    "node_count": row["node_count"],
                    "opaque_ratio": round(row["opaque_ratio"], 4),
                    "mean_confidence": round(row["mean_confidence"], 4),
                    "cfg_anomaly_count": row["cfg_anomaly_count"],
                    "sig_unlowered_count": row["sig_unlowered_count"],
                    "normalized_signature": row["normalized_signature"],
                    "reasons": reasons,
                }
            )

    anomalies.sort(key=lambda item: (-item["score"], item["script_name"], item["export_name"]))

    module_node_values = [row["total_nodes"] for row in module_rows]
    module_mean = _safe_mean(module_node_values)
    module_stdev = _safe_stdev(module_node_values)
    module_anomalies: list[dict[str, Any]] = []
    for row in module_rows:
        z = _z_score(row["total_nodes"], module_mean, module_stdev)
        if abs(z) >= 2.5 or row["cfg_anomaly_count"] >= 4:
            reasons = []
            if abs(z) >= 2.5:
                reasons.append(f"module_total_nodes_z={z:.2f}")
            if row["cfg_anomaly_count"] >= 4:
                reasons.append(f"cfg_anomalies={row['cfg_anomaly_count']}")
            module_anomalies.append(
                {
                    "script_name": row["script_name"],
                    "total_nodes": row["total_nodes"],
                    "export_count": row["export_count"],
                    "score": round(max(abs(z), row["cfg_anomaly_count"] / 2), 3),
                    "reasons": reasons,
                }
            )

    summary = {
        "module_count": len(module_rows),
        "export_count": len(function_rows),
        "total_ir_nodes": sum(node_counts),
        "avg_nodes_per_export": _safe_mean(node_counts),
        "median_nodes_per_export": statistics.median(node_counts) if node_counts else 0.0,
        "avg_unknown_ratio": _safe_mean(unknown_ratios),
        "avg_data_ratio": _safe_mean(data_ratios),
        "avg_opaque_ratio": _safe_mean(opaque_ratios),
        "avg_mean_confidence": _safe_mean(confidence_values),
        "total_cfg_anomalies": sum(cfg_counts),
        "total_sig_unlowered": sum(row["sig_unlowered_count"] for row in function_rows),
        "top_categories": [{"category": category, "count": count} for category, count in global_categories.most_common()],
        "top_opcodes": [{"opcode": opcode, "count": count} for opcode, count in global_opcodes.most_common(24)],
        "top_semantic_ops": [{"semantic_op": opcode, "count": count} for opcode, count in global_semantics.most_common(24)],
        "top_normalized_structures": [
            {"normalized_signature": signature, "count": count}
            for signature, count in signature_counts.most_common(16)
        ],
        "raw_patterns_per_semantic": [
            {
                "semantic_op": semantic_op,
                "raw_pattern_count": len(raw_patterns),
                "raw_patterns": sorted(raw_patterns)[:24],
            }
            for semantic_op, raw_patterns in sorted(
                semantic_to_raw.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )[:24]
        ],
        "top_unresolved_branch_patterns": [
            {"raw_pattern": raw_pattern, "count": count}
            for raw_pattern, count in unresolved_branch_patterns.most_common(24)
        ],
    }

    return {
        "summary": summary,
        "anomalous_exports": anomalies[:32],
        "anomalous_modules": module_anomalies[:16],
    }


def summarize_corpus(module_irs: list[dict[str, Any]]) -> dict[str, Any]:
    state = _init_corpus_state()
    for module in module_irs:
        _add_module_to_corpus_state(state, module)
    return _finalize_corpus_state(state)


def write_json(payload: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
