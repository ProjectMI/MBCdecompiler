from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
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


@dataclass
class IRNode:
    index: int
    offset: int
    size: int
    raw_kind: str
    terminal_kind: str
    category: str
    opcode: str
    prefix_chain: list[str]
    operands: dict[str, Any]

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
    normalized_signature: str
    raw_signature: str
    unknown_count: int
    unknown_ratio: float
    data_ratio: float
    call_count: int
    branch_count: int
    return_count: int

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
    raw_signature: str
    summary: IRFunctionSummary
    nodes: list[IRNode]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


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


def normalize_token(token: Token, index: int) -> IRNode:
    terminal_kind, terminal_payload, prefix_chain = _unwrap_prefixed_token(token.kind, token.payload)
    category = _category_for_kind(token.kind, terminal_kind)
    opcode = _opcode_for_token(token.kind, terminal_kind, category)
    operands = _normalize_operands(token.kind, terminal_kind, terminal_payload, prefix_chain)
    return IRNode(
        index=index,
        offset=token.offset,
        size=token.size,
        raw_kind=token.kind,
        terminal_kind=terminal_kind,
        category=category,
        opcode=opcode,
        prefix_chain=prefix_chain,
        operands=operands,
    )


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

    category_hist = Counter(node.category for node in nodes)
    opcode_hist = Counter(node.opcode for node in nodes)
    normalized_signature = " -> ".join(node.opcode for node in nodes)
    raw_signature = " -> ".join(token.kind for token in tokens)
    node_count = len(nodes)
    unknown_count = category_hist.get("unknown", 0)
    data_count = category_hist.get("data", 0)

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
        normalized_signature=normalized_signature,
        raw_signature=raw_signature,
        unknown_count=unknown_count,
        unknown_ratio=(unknown_count / node_count) if node_count else 0.0,
        data_ratio=(data_count / node_count) if node_count else 0.0,
        call_count=category_hist.get("call", 0),
        branch_count=category_hist.get("branch", 0),
        return_count=category_hist.get("return", 0),
    )

    return IRFunction(
        name=export_name,
        slice_mode=summary.slice_mode,
        span=summary.span,
        public_span=summary.public_span,
        byte_size=len(raw),
        raw_token_count=len(tokens),
        normalized_signature=normalized_signature,
        raw_signature=raw_signature,
        summary=summary,
        nodes=nodes,
    )


def _module_summary(functions: list[IRFunction]) -> dict[str, Any]:
    export_count = len(functions)
    total_nodes = sum(fn.summary.node_count for fn in functions)
    total_bytes = sum(fn.summary.byte_size for fn in functions)
    category_hist = Counter()
    opcode_hist = Counter()
    for fn in functions:
        category_hist.update(fn.summary.category_histogram)
        opcode_hist.update(fn.summary.opcode_histogram)

    return {
        "export_count": export_count,
        "total_nodes": total_nodes,
        "total_bytes": total_bytes,
        "avg_nodes_per_export": (total_nodes / export_count) if export_count else 0.0,
        "avg_bytes_per_export": (total_bytes / export_count) if export_count else 0.0,
        "category_histogram": {key: category_hist.get(key, 0) for key in CATEGORY_ORDER if category_hist.get(key, 0)},
        "top_opcodes": [{"opcode": opcode, "count": count} for opcode, count in opcode_hist.most_common(24)],
    }


def build_module_ir(path: str | Path, include_nodes: bool = True) -> dict[str, Any]:
    path = Path(path)
    mod = MBCModule(path, collect_auxiliary=False)
    functions = [build_function_ir(mod, name) for name in mod.export_names()]

    exports_payload = [fn.to_dict() if include_nodes else {**fn.summary.to_dict()} for fn in functions]
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


def summarize_corpus(module_irs: list[dict[str, Any]]) -> dict[str, Any]:
    function_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    global_categories = Counter()
    global_opcodes = Counter()
    signature_counts = Counter()

    for module in module_irs:
        module_summary = module.get("summary", {})
        module_rows.append(
            {
                "script_name": module.get("script_name"),
                "path": module.get("path"),
                "export_count": int(module_summary.get("export_count", 0)),
                "total_nodes": int(module_summary.get("total_nodes", 0)),
                "avg_nodes_per_export": float(module_summary.get("avg_nodes_per_export", 0.0)),
            }
        )
        global_categories.update(module_summary.get("category_histogram", {}))
        for item in module_summary.get("top_opcodes", []):
            global_opcodes[item["opcode"]] += item["count"]

        for fn in module.get("exports", []):
            row = {
                "script_name": module.get("script_name"),
                "path": module.get("path"),
                "name": fn.get("name"),
                "node_count": int(fn.get("node_count", fn.get("summary", {}).get("node_count", 0))),
                "byte_size": int(fn.get("byte_size", fn.get("summary", {}).get("byte_size", 0))),
                "unknown_ratio": float(fn.get("unknown_ratio", fn.get("summary", {}).get("unknown_ratio", 0.0))),
                "data_ratio": float(fn.get("data_ratio", fn.get("summary", {}).get("data_ratio", 0.0))),
                "call_count": int(fn.get("call_count", fn.get("summary", {}).get("call_count", 0))),
                "branch_count": int(fn.get("branch_count", fn.get("summary", {}).get("branch_count", 0))),
                "return_count": int(fn.get("return_count", fn.get("summary", {}).get("return_count", 0))),
                "normalized_signature": fn.get("normalized_signature", fn.get("summary", {}).get("normalized_signature", "")),
                "raw_signature": fn.get("raw_signature", fn.get("summary", {}).get("raw_signature", "")),
            }
            signature_counts[row["normalized_signature"]] += 1
            function_rows.append(row)

    node_counts = [row["node_count"] for row in function_rows]
    unknown_ratios = [row["unknown_ratio"] for row in function_rows]
    data_ratios = [row["data_ratio"] for row in function_rows]

    mean_nodes = _safe_mean(node_counts)
    stdev_nodes = _safe_stdev(node_counts)
    mean_unknown = _safe_mean(unknown_ratios)
    stdev_unknown = _safe_stdev(unknown_ratios)
    mean_data = _safe_mean(data_ratios)
    stdev_data = _safe_stdev(data_ratios)

    anomalies: list[dict[str, Any]] = []
    for row in function_rows:
        structure_count = signature_counts[row["normalized_signature"]]
        node_z = _z_score(row["node_count"], mean_nodes, stdev_nodes)
        unknown_z = _z_score(row["unknown_ratio"], mean_unknown, stdev_unknown)
        data_z = _z_score(row["data_ratio"], mean_data, stdev_data)

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
        if structure_count == 1 and row["node_count"] >= max(8, int(mean_nodes)):
            score += 1.5
            reasons.append("rare_normalized_signature")
        if row["return_count"] == 0 and row["node_count"] >= 8:
            score += 0.75
            reasons.append("no_explicit_return")
        if row["branch_count"] >= max(4, int(mean_nodes * 0.2)):
            score += 0.5
            reasons.append("branch_dense")

        if score > 0:
            anomalies.append(
                {
                    "script_name": row["script_name"],
                    "export_name": row["name"],
                    "score": round(score, 3),
                    "structure_frequency": structure_count,
                    "node_count": row["node_count"],
                    "unknown_ratio": round(row["unknown_ratio"], 4),
                    "data_ratio": round(row["data_ratio"], 4),
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
        if abs(z) >= 2.5:
            module_anomalies.append(
                {
                    "script_name": row["script_name"],
                    "total_nodes": row["total_nodes"],
                    "export_count": row["export_count"],
                    "score": round(abs(z), 3),
                    "reason": f"module_total_nodes_z={z:.2f}",
                }
            )

    summary = {
        "module_count": len(module_irs),
        "export_count": len(function_rows),
        "total_ir_nodes": sum(node_counts),
        "avg_nodes_per_export": _safe_mean(node_counts),
        "median_nodes_per_export": statistics.median(node_counts) if node_counts else 0.0,
        "avg_unknown_ratio": _safe_mean(unknown_ratios),
        "avg_data_ratio": _safe_mean(data_ratios),
        "top_categories": [{"category": category, "count": count} for category, count in global_categories.most_common()],
        "top_opcodes": [{"opcode": opcode, "count": count} for opcode, count in global_opcodes.most_common(24)],
        "top_normalized_structures": [
            {"normalized_signature": signature, "count": count}
            for signature, count in signature_counts.most_common(16)
        ],
    }

    return {
        "summary": summary,
        "anomalous_exports": anomalies[:32],
        "anomalous_modules": module_anomalies[:16],
    }


def write_json(payload: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
