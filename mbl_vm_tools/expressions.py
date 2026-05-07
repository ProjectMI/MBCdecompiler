from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .ir import VMFunctionIR, VMModuleIR, build_callable_index, build_function_ir
from .parser import MBCModule


EXPRESSIONS_CONTRACT_VERSION = "vm-expressions-v4"
EXPRESSIONS_POLICY = (
    "Expression lifting consumes VM IR, CALL facts, and stack/dataflow lower "
    "operand-frame bindings. It reconstructs high-level consumers of local "
    "operand frames: script/native calls, branch predicates, and terminal return "
    "frames. It does not decode bytes, rebuild CFG, mutate dataflow, infer native "
    "syscall names, infer predicate polarity, or promote demand-bound call results "
    "to persistent SSA values. Residual deficits and unresolved targets remain "
    "diagnostics instead of accepted expression facts."
)

CALL_TERMINALS = {"CALL_NATIVE", "CALL_SCRIPT"}
RETURN_TERMINALS = {"RETURN_PAIR", "END"}
CONDITIONAL_BRANCH_OPS = {0x4B, 0x4C, 0x4D}
MAX_RENDERED_OPERANDS = 16


@dataclass(frozen=True)
class VMExpression:
    id: str
    kind: str
    text: str
    role: str
    block: Optional[str] = None
    word_index: Optional[int] = None
    offset: Optional[int] = None
    terminal_kind: Optional[str] = None
    operands: list[str] = field(default_factory=list)
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMExpressionConsumer:
    id: str
    kind: str
    block: Optional[str]
    word_index: Optional[int]
    offset: int
    terminal_kind: str
    expression: str
    argv: list[dict[str, Any]] = field(default_factory=list)
    non_argv: list[dict[str, Any]] = field(default_factory=list)
    auxiliary: list[dict[str, Any]] = field(default_factory=list)
    target: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMExpressionIssue:
    id: str
    kind: str
    block: Optional[str] = None
    word_index: Optional[int] = None
    offset: Optional[int] = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMFunctionExpressionsReport:
    contract: str
    function: str
    symbol: str
    span: dict[str, int]
    summary: dict[str, Any]
    expressions: list[VMExpression]
    consumers: list[VMExpressionConsumer]
    diagnostics: list[VMExpressionIssue] = field(default_factory=list)
    anomalies: list[VMExpressionIssue] = field(default_factory=list)

    def to_dict(self, *, include_expressions: bool = True, include_consumers: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract": self.contract,
            "function": self.function,
            "symbol": self.symbol,
            "span": dict(self.span),
            "summary": dict(self.summary),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "anomalies": [a.to_dict() for a in self.anomalies],
        }
        if include_expressions:
            payload["expressions"] = [e.to_dict() for e in self.expressions]
        if include_consumers:
            payload["consumers"] = [c.to_dict() for c in self.consumers]
        return payload


@dataclass(frozen=True)
class VMModuleExpressionsReport:
    contract: str
    module: str
    summary: dict[str, Any]
    functions: list[VMFunctionExpressionsReport]

    def to_dict(self, *, include_expressions: bool = True, include_consumers: bool = True) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "module": self.module,
            "summary": dict(self.summary),
            "functions": [
                f.to_dict(include_expressions=include_expressions, include_consumers=include_consumers)
                for f in self.functions
            ],
        }


# ---------------------------------------------------------------------------
# Stable helpers


def _safe_id_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)



def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def _hex_byte(value: Any) -> str:
    try:
        return f"0x{int(value) & 0xFF:02X}"
    except Exception:
        return str(value)



def _compact_text(items: Iterable[str], *, cap: int = MAX_RENDERED_OPERANDS) -> str:
    vals = [str(v) for v in items]
    if len(vals) <= cap:
        return ", ".join(vals)
    shown = ", ".join(vals[:cap])
    return f"{shown}, …(+{len(vals) - cap})"


def _word_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except Exception:
        return None
    return ivalue if ivalue >= 0 else None


def _word_for_value(value: dict[str, Any], words_by_index: dict[int, dict[str, Any]], words_by_offset: dict[int, dict[str, Any]]) -> dict[str, Any]:
    idx = _word_index(value.get("word_index"))
    if idx is not None and idx in words_by_index:
        return words_by_index[idx]
    offset = value.get("offset")
    try:
        off = int(offset)
    except Exception:
        return {}
    return words_by_offset.get(off, {})


def _word_for_operation(op: dict[str, Any], words_by_index: dict[int, dict[str, Any]], words_by_offset: dict[int, dict[str, Any]]) -> dict[str, Any]:
    idx = _word_index(op.get("word_index"))
    if idx is not None and idx in words_by_index:
        return words_by_index[idx]
    try:
        off = int(op.get("offset"))
    except Exception:
        return {}
    return words_by_offset.get(off, {})


def _merged_word_evidence(value: dict[str, Any], word: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(value.get("evidence") or {})
    operands = dict(word.get("operands") or {})
    if operands:
        evidence.setdefault("operands", operands)
        for key, val in operands.items():
            evidence.setdefault(key, val)
    if word.get("kind") is not None:
        evidence.setdefault("word_kind", word.get("kind"))
    if word.get("terminal_kind") is not None:
        evidence.setdefault("terminal_kind", word.get("terminal_kind"))
    if word.get("decoder_rule") is not None:
        evidence.setdefault("decoder_rule", word.get("decoder_rule"))
    if word.get("prefixes_hex"):
        evidence.setdefault("prefixes_hex", list(word.get("prefixes_hex") or []))
    elif word.get("prefixes"):
        evidence.setdefault("prefixes_hex", [_hex_byte(p) for p in word.get("prefixes") or []])
    return evidence


def _literal_value(evidence: dict[str, Any]) -> Any:
    operands = evidence.get("operands") if isinstance(evidence.get("operands"), dict) else {}
    if "value" in evidence:
        return evidence.get("value")
    return operands.get("value")


def _expression_kind_and_text(value: dict[str, Any], word: dict[str, Any], operand_texts: list[str]) -> tuple[str, str, dict[str, Any]]:
    evidence = _merged_word_evidence(value, word)
    terminal = str(evidence.get("terminal_kind") or value.get("kind") or "")
    operands = evidence.get("operands") if isinstance(evidence.get("operands"), dict) else {}
    prefix_text = ""
    if evidence.get("prefixes_hex"):
        prefix_text = f" pfx=[{','.join(str(p) for p in evidence.get('prefixes_hex') or [])}]"

    if terminal in {"IMM8", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32", "U16"}:
        kind = "literal_int"
        text = repr(_literal_value(evidence))
    elif terminal == "F32":
        kind = "literal_float"
        text = repr(_literal_value(evidence))
    elif terminal == "BARE_U32":
        kind = "auxiliary_literal"
        text = f"aux_u32({repr(_literal_value(evidence))})"
    elif terminal in {"REF", "REF16"}:
        mode = evidence.get("mode", operands.get("mode"))
        ref = evidence.get("ref", operands.get("ref"))
        if terminal == "REF16" or value.get("role") == "operand_frame_ref16_offset":
            kind = "reference_offset"
            text = f"ref16(mode={_hex_byte(mode)}, offset={ref})"
        else:
            kind = "reference"
            text = f"ref(mode={_hex_byte(mode)}, addr={ref})"
    elif terminal in {"REC41", "REC61", "REC62"}:
        kind = "record_reference"
        if terminal == "REC41":
            text = f"rec41(ref={evidence.get('ref', operands.get('ref'))}, imm={evidence.get('imm', operands.get('imm'))})"
        elif terminal == "REC61":
            text = (
                "rec61("
                f"mode={_hex_byte(evidence.get('mode', operands.get('mode')))}, "
                f"u16={evidence.get('u16', operands.get('u16'))}, "
                f"a={evidence.get('a', operands.get('a'))}, "
                f"b={evidence.get('b', operands.get('b'))}, "
                f"c={evidence.get('c', operands.get('c'))})"
            )
        else:
            text = (
                "rec62("
                f"mode={_hex_byte(evidence.get('mode', operands.get('mode')))}, "
                f"u16={evidence.get('u16', operands.get('u16'))}, "
                f"c={evidence.get('c', operands.get('c'))})"
            )
    elif terminal == "CODE_REF":
        kind = "code_reference"
        text = f"code_ref(rel={evidence.get('rel', operands.get('rel'))})"
    elif terminal in {"AGG", "AGG0"}:
        kind = "abi_prologue"
        text = f"abi_prologue(arity={evidence.get('arity', operands.get('arity'))})"
    elif terminal == "UNKNOWN":
        kind = "unknown_atom"
        text = f"unknown(byte={evidence.get('byte', operands.get('byte'))})"
    else:
        kind = str(value.get("kind") or "operand_atom")
        text = terminal.lower() if terminal else str(value.get("id"))

    if operand_texts:
        text = f"{text}{{aux=[{_compact_text(operand_texts)}]}}"
    if prefix_text:
        evidence.setdefault("prefix_note", prefix_text.strip())
    return kind, text, evidence


def _expr_ref(expr: VMExpression, *, role: Optional[str] = None, index: Optional[int] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": expr.id,
        "kind": expr.kind,
        "text": expr.text,
        "confidence": expr.confidence,
    }
    if role is not None:
        payload["role"] = role
    if index is not None:
        payload["index"] = index
    if expr.block is not None:
        payload["block"] = expr.block
    if expr.word_index is not None:
        payload["word_index"] = expr.word_index
    if expr.offset is not None:
        payload["offset"] = expr.offset
    if expr.terminal_kind is not None:
        payload["terminal_kind"] = expr.terminal_kind
    return payload


def _placeholder_expr(
    *,
    expression_id: str,
    kind: str,
    text: str,
    role: str,
    op: dict[str, Any],
    confidence: float,
    evidence: Optional[dict[str, Any]] = None,
) -> VMExpression:
    return VMExpression(
        id=expression_id,
        kind=kind,
        text=text,
        role=role,
        block=str(op.get("block")) if op.get("block") is not None else None,
        word_index=_word_index(op.get("word_index")),
        offset=int(op.get("offset", 0) or 0),
        terminal_kind=str(op.get("terminal_kind")),
        confidence=confidence,
        evidence=dict(evidence or {}),
    )


def _split_call_inputs(op: dict[str, Any]) -> tuple[list[str], list[str], list[str], int, int]:
    """Return auxiliary, non-argv, raw argv, pending-result argv count, unresolved argv count.

    The split mirrors dataflow.py's public operation contract: CALL inputs are
    pending auxiliary payloads, then non-argv frame prefix, then the encoded-argc
    suffix that is available as concrete lower atoms.  Demand-bound call-result
    candidates and real deficits are represented separately and never materialized
    as persistent SSA values.
    """

    contract = dict(op.get("contract") or {})
    inputs = [str(v) for v in op.get("inputs") or []]
    raw_argc_count = max(0, int(contract.get("frame_atom_pop", 0) or 0))
    non_arg_count = max(0, int(contract.get("frame_non_arg_pop", 0) or 0))
    pending_count = max(0, int(contract.get("frame_pending_call_result_argc_pop", 0) or 0))
    unresolved_count = max(0, int(contract.get("frame_argc_deficit", 0) or 0))

    # Clamp to the actual operation payload.  If upstream ever changes the
    # contract shape, expressions must report the inconsistency instead of
    # repairing the model by inventing new inputs.
    raw_argc_count = min(raw_argc_count, len(inputs))
    non_arg_count = min(non_arg_count, max(0, len(inputs) - raw_argc_count))
    aux_count = max(0, len(inputs) - raw_argc_count - non_arg_count)
    auxiliary = inputs[:aux_count]
    non_argv = inputs[aux_count:aux_count + non_arg_count]
    argv = inputs[aux_count + non_arg_count:]
    return auxiliary, non_argv, argv, pending_count, unresolved_count


def _consumer_input_refs(ids: Iterable[str], expr_by_id: dict[str, VMExpression], *, role: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, value_id in enumerate(ids):
        expr = expr_by_id.get(str(value_id))
        if expr is None:
            out.append({"id": str(value_id), "kind": "missing_expression", "text": str(value_id), "role": role, "index": idx, "confidence": 0.0})
            continue
        out.append(_expr_ref(expr, role=role, index=idx))
    return out


def _call_key(call: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    wi = _word_index(call.get("word_index"))
    offset: Optional[int]
    try:
        offset = int(call.get("offset"))
    except Exception:
        offset = None
    return wi, offset


def _call_indexes(calls: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_word: dict[int, dict[str, Any]] = {}
    by_offset: dict[int, dict[str, Any]] = {}
    for call in calls:
        wi, offset = _call_key(call)
        if wi is not None:
            by_word.setdefault(wi, call)
        if offset is not None:
            by_offset.setdefault(offset, call)
    return by_word, by_offset


def _find_call(op: dict[str, Any], by_word: dict[int, dict[str, Any]], by_offset: dict[int, dict[str, Any]]) -> dict[str, Any]:
    wi = _word_index(op.get("word_index"))
    if wi is not None and wi in by_word:
        return by_word[wi]
    try:
        offset = int(op.get("offset"))
    except Exception:
        return {}
    return by_offset.get(offset, {})


def _cfg_terminators(cfg: dict[str, Any]) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    by_block_word: dict[tuple[str, int], dict[str, Any]] = {}
    by_block_offset: dict[tuple[str, int], dict[str, Any]] = {}
    for block in cfg.get("blocks") or []:
        block_id = str(block.get("id"))
        term = block.get("terminator") or {}
        if not isinstance(term, dict) or not term:
            continue
        resolution = term.get("resolution") if isinstance(term.get("resolution"), dict) else {}
        wi = _word_index(resolution.get("word_index"))
        if wi is not None:
            by_block_word[(block_id, wi)] = term
        try:
            off = int(resolution.get("offset"))
            by_block_offset[(block_id, off)] = term
        except Exception:
            pass
    return by_block_word, by_block_offset


def _find_terminator(op: dict[str, Any], by_block_word: dict[tuple[str, int], dict[str, Any]], by_block_offset: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    block = str(op.get("block"))
    wi = _word_index(op.get("word_index"))
    if wi is not None and (block, wi) in by_block_word:
        return by_block_word[(block, wi)]
    try:
        offset = int(op.get("offset"))
    except Exception:
        return {}
    return by_block_offset.get((block, offset), {})


def _target_from_call(call: dict[str, Any], op: dict[str, Any]) -> dict[str, Any]:
    if call:
        if call.get("kind") == "native":
            native = call.get("native") if isinstance(call.get("native"), dict) else {}
            target = call.get("target") if isinstance(call.get("target"), dict) else {}
            return {
                "kind": "native",
                "name": None,
                "opid": call.get("opid"),
                "category": call.get("category") or native.get("category"),
                "category_name": call.get("category_name") or native.get("category_name"),
                "resolved": False,
                "source": "ir.native_call_fact",
                "raw": target,
            }
        target = call.get("target") if isinstance(call.get("target"), dict) else {}
        return {
            "kind": target.get("target_kind") or "script",
            "name": target.get("target_name"),
            "absolute_target": target.get("absolute_target"),
            "resolved": bool(target.get("resolved")),
            "confidence": target.get("confidence"),
            "source": "ir.script_call_target",
            "raw": target,
        }

    contract = dict(op.get("contract") or {})
    if op.get("terminal_kind") == "CALL_NATIVE":
        return {
            "kind": "native",
            "name": None,
            "opid": contract.get("opid"),
            "resolved": False,
            "source": "dataflow.operation_contract",
        }
    return {
        "kind": "script",
        "name": None,
        "resolved": False,
        "source": "dataflow.operation_contract",
    }


def _call_expression_text(target: dict[str, Any], argv: list[dict[str, Any]], non_argv: list[dict[str, Any]], auxiliary: list[dict[str, Any]]) -> str:
    arg_text = _compact_text(item.get("text", item.get("id")) for item in argv)
    ctx_text = _compact_text(item.get("text", item.get("id")) for item in non_argv)
    aux_text = _compact_text(item.get("text", item.get("id")) for item in auxiliary)
    if target.get("kind") == "native":
        name = f"native[opid={target.get('opid')}]"
        if target.get("category"):
            name += f"<{target.get('category')}>"
    else:
        name = str(target.get("name") or "script_call")
        if not target.get("resolved") and target.get("absolute_target") is not None:
            name += f"@{target.get('absolute_target')}"
    ctx_parts: list[str] = []
    if auxiliary:
        ctx_parts.append(f"aux=[{aux_text}]")
    if non_argv:
        ctx_parts.append(f"ctx=[{ctx_text}]")
    ctx = "{" + "; ".join(ctx_parts) + "}" if ctx_parts else ""
    return f"{name}{ctx}({arg_text})"


def _make_pending_argv_expressions(op: dict[str, Any], count: int) -> list[VMExpression]:
    op_id = str(op.get("id") or f"op_{op.get('offset')}")
    out: list[VMExpression] = []
    for idx in range(max(0, count)):
        out.append(_placeholder_expr(
            expression_id=f"x_{_safe_id_part(op_id)}_pending_call_result_argv_{idx}",
            kind="demand_bound_call_result_argv",
            text=f"pending_call_result_argv[{idx}]",
            role="argv_candidate",
            op=op,
            confidence=0.55,
            evidence={
                "demand_bound_result": True,
                "source": "dataflow.frame_pending_call_result_argc_pop",
                "rule": "not_persistent_ssa_value",
            },
        ))
    return out


def _make_unresolved_argv_expressions(op: dict[str, Any], count: int) -> list[VMExpression]:
    op_id = str(op.get("id") or f"op_{op.get('offset')}")
    out: list[VMExpression] = []
    for idx in range(max(0, count)):
        out.append(_placeholder_expr(
            expression_id=f"x_{_safe_id_part(op_id)}_unresolved_argv_{idx}",
            kind="unresolved_argv_slot",
            text=f"unresolved_argv[{idx}]",
            role="argv_deficit",
            op=op,
            confidence=0.0,
            evidence={
                "source": "dataflow.frame_argc_deficit",
                "diagnostic_only": True,
            },
        ))
    return out


def _call_consumer(
    op: dict[str, Any],
    *,
    expr_by_id: dict[str, VMExpression],
    expressions: list[VMExpression],
    call: dict[str, Any],
) -> tuple[VMExpressionConsumer, list[VMExpressionIssue]]:
    contract = dict(op.get("contract") or {})
    auxiliary_ids, non_argv_ids, argv_ids, pending_count, unresolved_count = _split_call_inputs(op)

    new_exprs = [*_make_pending_argv_expressions(op, pending_count), *_make_unresolved_argv_expressions(op, unresolved_count)]
    for expr in new_exprs:
        if expr.id not in expr_by_id:
            expr_by_id[expr.id] = expr
            expressions.append(expr)

    auxiliary = _consumer_input_refs(auxiliary_ids, expr_by_id, role="auxiliary")
    non_argv = _consumer_input_refs(non_argv_ids, expr_by_id, role="non_argv")
    argv = _consumer_input_refs(argv_ids, expr_by_id, role="argv")
    start_index = len(argv)
    argv.extend(_expr_ref(expr, role="argv", index=start_index + idx) for idx, expr in enumerate(new_exprs))

    target = _target_from_call(call, op)
    text = _call_expression_text(target, argv, non_argv, auxiliary)
    encoded_argc = int(contract.get("encoded_argc", call.get("encoded_argc", 0) or 0) or 0)
    diagnostics: dict[str, Any] = {}
    anomalies: list[VMExpressionIssue] = []
    subentry_deferred = max(0, int(contract.get("frame_subentry_argc_deficit_deferred", 0) or 0))
    argv_delta = encoded_argc - len(argv)
    if argv_delta != 0:
        if argv_delta > 0 and subentry_deferred >= argv_delta:
            # A prefixed/sub-entry CALL may expose an encoded argc owned by an
            # overlapping entry context.  This is a lower-level coordinate fact,
            # not an expression diagnostic; keep it as evidence and let
            # validation accept it explicitly.
            pass
        else:
            diagnostics["argv_count_mismatch"] = {"encoded_argc": encoded_argc, "expression_argv_count": len(argv)}
            anomalies.append(VMExpressionIssue(
                id=f"expression_argv_count_mismatch:{op.get('id')}",
                kind="expression_argv_count_mismatch",
                block=str(op.get("block")) if op.get("block") is not None else None,
                word_index=_word_index(op.get("word_index")),
                offset=int(op.get("offset", 0) or 0),
                detail=diagnostics["argv_count_mismatch"],
            ))
    if unresolved_count:
        diagnostics["unresolved_argv_slot_count"] = unresolved_count
        anomalies.append(VMExpressionIssue(
            id=f"unresolved_argv_slots:{op.get('id')}",
            kind="unresolved_argv_slots",
            block=str(op.get("block")) if op.get("block") is not None else None,
            word_index=_word_index(op.get("word_index")),
            offset=int(op.get("offset", 0) or 0),
            detail={"count": unresolved_count, "reason": "dataflow reported a real frame_argc_deficit"},
        ))
    if not target.get("resolved") and target.get("kind") != "native":
        diagnostics["unresolved_script_target"] = True
        anomalies.append(VMExpressionIssue(
            id=f"unresolved_script_target:{op.get('id')}",
            kind="unresolved_script_target",
            block=str(op.get("block")) if op.get("block") is not None else None,
            word_index=_word_index(op.get("word_index")),
            offset=int(op.get("offset", 0) or 0),
            detail={"target": target},
        ))

    consumer_kind = "native_call" if str(op.get("terminal_kind")) == "CALL_NATIVE" else "script_call"
    evidence = {
        "encoded_argc": encoded_argc,
        "frame_atom_pop": contract.get("frame_atom_pop"),
        "frame_non_arg_pop": contract.get("frame_non_arg_pop"),
        "frame_pending_call_result_argc_pop": pending_count,
        "frame_argc_deficit": unresolved_count,
        "frame_subentry_argc_deficit_deferred": contract.get("frame_subentry_argc_deficit_deferred"),
        "subentry_argc_deferred_count": subentry_deferred if subentry_deferred else None,
        "subentry_argc_deferred_rule": "prefixed/sub-entry CALL argv belong to an overlapping VM entry context below expression lifting" if subentry_deferred else None,
        "call_argument_binding_rule": contract.get("call_argument_binding_rule"),
        "call_non_arg_binding_rule": contract.get("call_non_arg_binding_rule"),
        "prefixes_hex": contract.get("prefixes_hex") or call.get("prefixes_hex") or [],
    }
    if target.get("kind") == "native":
        evidence["native_opid_is_module_local_evidence"] = True
    return VMExpressionConsumer(
        id=f"consumer_{_safe_id_part(op.get('id') or op.get('offset'))}",
        kind=consumer_kind,
        block=str(op.get("block")) if op.get("block") is not None else None,
        word_index=_word_index(op.get("word_index")),
        offset=int(op.get("offset", 0) or 0),
        terminal_kind=str(op.get("terminal_kind")),
        expression=text,
        argv=argv,
        non_argv=non_argv,
        auxiliary=auxiliary,
        target=target,
        result=contract.get("result"),
        confidence=1.0 if not unresolved_count else 0.75,
        evidence={k: v for k, v in evidence.items() if v is not None},
        diagnostics=diagnostics,
    ), anomalies




def _signed_u16_local(value: Any) -> int:
    try:
        ivalue = int(value) & 0xFFFF
    except Exception:
        return 0
    return ivalue - 0x10000 if ivalue & 0x8000 else ivalue


def _word_prefix_count(word: dict[str, Any]) -> int:
    prefixes = word.get("prefixes")
    if isinstance(prefixes, list):
        return len(prefixes)
    prefixes_hex = word.get("prefixes_hex")
    if isinstance(prefixes_hex, list):
        return len(prefixes_hex)
    return 0


def _branch_target_from_word(word: dict[str, Any]) -> Optional[int]:
    operands = word.get("operands") if isinstance(word.get("operands"), dict) else {}
    if word.get("terminal_kind") != "BR" or "off" not in operands:
        return None
    try:
        offset = int(word.get("offset"))
    except Exception:
        return None
    terminal_atom = offset + _word_prefix_count(word)
    return terminal_atom + 1 + _signed_u16_local(operands.get("off"))


def _branch_fallthrough_from_word(word: dict[str, Any]) -> Optional[int]:
    try:
        return int(word.get("offset")) + int(word.get("size"))
    except Exception:
        return None


def _is_predicate_no_transfer_word(word: dict[str, Any], op_int: int) -> bool:
    if op_int not in CONDITIONAL_BRANCH_OPS:
        return False
    target = _branch_target_from_word(word)
    fallthrough = _branch_fallthrough_from_word(word)
    return target is not None and fallthrough is not None and target == fallthrough

def _branch_consumer(
    op: dict[str, Any],
    *,
    expr_by_id: dict[str, VMExpression],
    terminator: dict[str, Any],
    word: dict[str, Any],
) -> VMExpressionConsumer:
    contract = dict(op.get("contract") or {})
    word_operands = dict(word.get("operands") or {})
    resolution = terminator.get("resolution") if isinstance(terminator.get("resolution"), dict) else {}
    semantic = terminator.get("semantic") if isinstance(terminator.get("semantic"), dict) else {}
    if not semantic and isinstance(resolution, dict):
        semantic = resolution.get("semantic") if isinstance(resolution.get("semantic"), dict) else {}
    op_code = semantic.get("op", word_operands.get("op"))
    try:
        op_int = int(op_code) & 0xFF
    except Exception:
        op_int = -1
    inputs = _consumer_input_refs([str(v) for v in op.get("inputs") or []], expr_by_id, role="predicate_input")
    op_role = str(op.get("role") or contract.get("role") or "")
    no_transfer = not terminator and (op_role == "predicate_no_transfer" or _is_predicate_no_transfer_word(word, op_int))
    is_conditional = op_int in CONDITIONAL_BRANCH_OPS or semantic.get("branch_kind") == "conditional_branch" or no_transfer

    if no_transfer:
        target_offset = _branch_target_from_word(word)
        fallthrough_offset = _branch_fallthrough_from_word(word)
        text = f"predicate_no_transfer[op={_hex_byte(op_int)}]({_compact_text(item.get('text', item.get('id')) for item in inputs)})"
        kind = "predicate_no_transfer"
        target = {
            "kind": "predicate_no_transfer",
            "target_offset": target_offset,
            "fallthrough_offset": fallthrough_offset,
            "resolved": True,
            "reason": "conditional BR target equals fallthrough; no CFG split is expected",
        }
        confidence = 1.0
        diagnostics: dict[str, Any] = {}
    elif is_conditional:
        text = f"branch[op={_hex_byte(op_int)}, polarity=unresolved]({_compact_text(item.get('text', item.get('id')) for item in inputs)})"
        kind = "branch_predicate"
        target = {
            "kind": semantic.get("branch_kind") or "conditional_branch",
            "target_offset": semantic.get("target_offset", resolution.get("selected_local_target")),
            "fallthrough_offset": semantic.get("fallthrough_offset"),
            "taken_edge_kind": semantic.get("taken_edge_kind"),
            "fallthrough_edge_kind": semantic.get("fallthrough_edge_kind"),
            "predicate_polarity": semantic.get("predicate_polarity"),
            "resolved": bool(resolution.get("selected_local_target") is not None or semantic.get("target_offset") is not None),
        }
        confidence = float(semantic.get("confidence", 0.95) or 0.95)
        diagnostics = {} if terminator else {"missing_cfg_terminator": True}
    else:
        text = f"jump(target={semantic.get('target_offset', resolution.get('selected_local_target'))})"
        kind = "jump"
        target = {
            "kind": semantic.get("branch_kind") or "unconditional_jump",
            "target_offset": semantic.get("target_offset", resolution.get("selected_local_target")),
            "fallthrough_offset": semantic.get("fallthrough_offset"),
            "taken_edge_kind": semantic.get("taken_edge_kind"),
            "fallthrough_edge_kind": semantic.get("fallthrough_edge_kind"),
            "predicate_polarity": semantic.get("predicate_polarity"),
            "resolved": bool(resolution.get("selected_local_target") is not None or semantic.get("target_offset") is not None),
        }
        confidence = float(semantic.get("confidence", 1.0) or 1.0)
        diagnostics = {} if terminator else {"missing_cfg_terminator": True}

    evidence = {
        "op": op_int if op_int >= 0 else op_code,
        "predicate_frame_input_count": contract.get("predicate_frame_input_count"),
        "predicate_frame_clause_count": contract.get("predicate_frame_clause_count"),
        "control_frame_input_count": contract.get("control_frame_input_count"),
        "branch_semantic_contract": semantic.get("contract"),
        "policy": semantic.get("policy"),
        "prefixes_hex": contract.get("prefixes_hex") or semantic.get("prefixes_hex") or word.get("prefixes_hex") or [],
        "predicate_no_transfer": no_transfer or None,
    }
    return VMExpressionConsumer(
        id=f"consumer_{_safe_id_part(op.get('id') or op.get('offset'))}",
        kind=kind,
        block=str(op.get("block")) if op.get("block") is not None else None,
        word_index=_word_index(op.get("word_index")),
        offset=int(op.get("offset", 0) or 0),
        terminal_kind=str(op.get("terminal_kind")),
        expression=text,
        argv=inputs if is_conditional else [],
        target={k: v for k, v in target.items() if v is not None},
        confidence=confidence,
        evidence={k: v for k, v in evidence.items() if v is not None},
        diagnostics=diagnostics,
    )

def _return_consumer(op: dict[str, Any], *, expr_by_id: dict[str, VMExpression]) -> VMExpressionConsumer:
    contract = dict(op.get("contract") or {})
    values = _consumer_input_refs([str(v) for v in op.get("inputs") or []], expr_by_id, role="return_input")
    text = f"return_frame({_compact_text(item.get('text', item.get('id')) for item in values)})"
    evidence = {
        "return_frame_input_count": contract.get("return_frame_input_count"),
        "return_frame_clause_count": contract.get("return_frame_clause_count"),
        "frame_atom_pop": contract.get("frame_atom_pop"),
        "stack_effect_rule": contract.get("stack_effect_rule"),
    }
    return VMExpressionConsumer(
        id=f"consumer_{_safe_id_part(op.get('id') or op.get('offset'))}",
        kind="return_frame",
        block=str(op.get("block")) if op.get("block") is not None else None,
        word_index=_word_index(op.get("word_index")),
        offset=int(op.get("offset", 0) or 0),
        terminal_kind=str(op.get("terminal_kind")),
        expression=text,
        argv=values,
        target={"kind": "function_exit", "resolved": True},
        confidence=1.0,
        evidence={k: v for k, v in evidence.items() if v is not None},
    )


def _event_from_dataflow(item: dict[str, Any], *, kind_prefix: str = "dataflow") -> VMExpressionIssue:
    return VMExpressionIssue(
        id=f"{kind_prefix}:{item.get('id')}",
        kind=f"{kind_prefix}_{item.get('kind')}",
        block=str(item.get("block")) if item.get("block") is not None else None,
        word_index=_word_index(item.get("word_index")),
        offset=int(item.get("offset")) if item.get("offset") is not None else None,
        detail={"source": "dataflow", **dict(item.get("detail") or {})},
    )


def _classify_dataflow_events(dataflow: dict[str, Any]) -> tuple[list[VMExpressionIssue], list[VMExpressionIssue], dict[str, Any]]:
    """Classify lower dataflow observations for expression reporting.

    Dataflow still stores many VM-transfer observations under its historical
    ``anomalies`` field.  Expressions must not blindly surface those as report
    events: call-result bindings, predicate-frame bindings and return-frame
    bindings are already represented by consumers.  Only unresolved lower-frame
    residue is surfaced as diagnostics/anomalies here.
    """

    diagnostics: list[VMExpressionIssue] = []
    anomalies: list[VMExpressionIssue] = []
    source_hist: Counter[str] = Counter()
    suppressed_hist: Counter[str] = Counter()
    pending_result_discarded = 0
    for item in dataflow.get("anomalies") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind"))
        source_hist[kind] += 1
        detail = dict(item.get("detail") or {})

        if kind in {"call_result_binding", "predicate_frame_bound", "terminal_return_frame_bound"}:
            # These are normal lower-transfer observations.  The corresponding
            # consumers already carry the bound expression inputs; an event per
            # consumer only inflates the report without adding model facts.
            suppressed_hist[kind] += 1
            continue

        if kind == "operand_frame_open_at_block_exit":
            frame_atoms = int(detail.get("frame_atom_count", 0) or 0)
            frame_clauses = int(detail.get("frame_clause_count", 0) or 0)
            auxiliary = int(detail.get("auxiliary_value_count", 0) or 0)
            pending = int(detail.get("pending_call_result_count", 0) or 0)
            if frame_atoms == 0 and frame_clauses == 0 and auxiliary == 0 and pending > 0:
                # This is not an open operand frame. It is an unconsumed possible
                # CALL result at a block boundary. CALL results are demand-bound,
                # not persistent stack values, so the candidate simply expires.
                pending_result_discarded += pending
                suppressed_hist["pending_call_result_open_at_block_exit"] += 1
                continue
            diagnostics.append(_event_from_dataflow(item))
            continue

        if kind == "auxiliary_literal_unbound_by_clause_boundary":
            diagnostics.append(_event_from_dataflow(item))
            continue

        if kind in {"operand_frame_arity_delta", "unknown_vm_word", "unresolved_block", "value_operand_cap", "operation_underflow", "unknown_stack_transfer", "dataflow_traversal_limit"}:
            anomalies.append(_event_from_dataflow(item))
            continue

        # Unknown future event kinds are kept as diagnostics, not silently hidden.
        diagnostics.append(_event_from_dataflow(item))

    summary = {
        "dataflow_event_kind_histogram": _counter_dict(source_hist),
        "suppressed_dataflow_event_kind_histogram": _counter_dict(suppressed_hist),
        "discarded_pending_call_result_count": pending_result_discarded,
    }
    return diagnostics, anomalies, summary


# ---------------------------------------------------------------------------
# Public analysis entry points


def analyze_function_expressions(fn: VMFunctionIR | dict[str, Any]) -> VMFunctionExpressionsReport:
    """Lift expression atoms and high-level operand-frame consumers for one VM IR function.

    This pass is intentionally a consumer of existing contracts.  It trusts the
    decoded word stream, CFG terminators, resolved call facts, and dataflow
    operation/value bindings that are already present in VM IR.  It does not run
    a byte decoder or a second stack simulator.
    """

    payload = _as_dict(fn)
    words = [_as_dict(w) for w in payload.get("words") or []]
    dataflow = _as_dict(payload.get("dataflow") or {})
    values = [_as_dict(v) for v in dataflow.get("values") or []]
    operations = [_as_dict(op) for op in dataflow.get("operations") or []]
    calls = [_as_dict(call) for call in payload.get("calls") or []]
    cfg = _as_dict(payload.get("cfg") or {})

    words_by_index: dict[int, dict[str, Any]] = {}
    words_by_offset: dict[int, dict[str, Any]] = {}
    for word in words:
        idx = _word_index(word.get("index"))
        if idx is not None:
            words_by_index[idx] = word
        try:
            words_by_offset[int(word.get("offset"))] = word
        except Exception:
            pass

    operations_by_id = {str(op.get("id")): op for op in operations if op.get("id") is not None}
    expressions: list[VMExpression] = []
    expr_by_id: dict[str, VMExpression] = {}

    for value in values:
        value_id = str(value.get("id"))
        word = _word_for_value(value, words_by_index, words_by_offset)
        producer = operations_by_id.get(str(value.get("producer")))
        operand_ids = [str(v) for v in (producer.get("inputs") if producer else []) or []]
        operand_texts = [expr_by_id.get(v).text if v in expr_by_id else v for v in operand_ids]
        kind, text, evidence = _expression_kind_and_text(value, word, operand_texts)
        terminal = str(evidence.get("terminal_kind") or value.get("terminal_kind") or "") or None
        expr = VMExpression(
            id=value_id,
            kind=kind,
            text=text,
            role=str(value.get("role") or value.get("kind") or "value"),
            block=str(value.get("block")) if value.get("block") is not None else None,
            word_index=_word_index(value.get("word_index")),
            offset=int(value.get("offset")) if value.get("offset") is not None else None,
            terminal_kind=terminal,
            operands=operand_ids,
            confidence=1.0 if kind != "unknown_atom" else 0.0,
            evidence=evidence,
        )
        expressions.append(expr)
        expr_by_id[value_id] = expr

    call_by_word, call_by_offset = _call_indexes(calls)
    term_by_block_word, term_by_block_offset = _cfg_terminators(cfg)

    consumers: list[VMExpressionConsumer] = []
    report_diagnostics, anomalies, dataflow_event_summary = _classify_dataflow_events(dataflow)

    for op in operations:
        terminal = str(op.get("terminal_kind"))
        if terminal in CALL_TERMINALS:
            consumer, new_anomalies = _call_consumer(
                op,
                expr_by_id=expr_by_id,
                expressions=expressions,
                call=_find_call(op, call_by_word, call_by_offset),
            )
            consumers.append(consumer)
            anomalies.extend(new_anomalies)
        elif terminal == "BR":
            word = _word_for_operation(op, words_by_index, words_by_offset)
            consumers.append(_branch_consumer(
                op,
                expr_by_id=expr_by_id,
                terminator=_find_terminator(op, term_by_block_word, term_by_block_offset),
                word=word,
            ))
        elif terminal in RETURN_TERMINALS:
            consumers.append(_return_consumer(op, expr_by_id=expr_by_id))

    expr_kind_hist = Counter(expr.kind for expr in expressions)
    consumer_kind_hist = Counter(consumer.kind for consumer in consumers)
    target_kind_hist = Counter(str(consumer.target.get("kind")) for consumer in consumers if consumer.target)
    diagnostics_hist = Counter()
    pending_arg_count = 0
    unresolved_arg_count = 0
    unresolved_script_target_count = 0
    missing_expression_ref_count = 0
    for consumer in consumers:
        if consumer.diagnostics:
            diagnostics_hist.update(consumer.diagnostics.keys())
        for arg in consumer.argv:
            if arg.get("kind") == "demand_bound_call_result_argv":
                pending_arg_count += 1
            if arg.get("kind") == "unresolved_argv_slot":
                unresolved_arg_count += 1
            if arg.get("kind") == "missing_expression":
                missing_expression_ref_count += 1
        if consumer.diagnostics.get("unresolved_script_target"):
            unresolved_script_target_count += 1

    report_diagnostics_hist = Counter(d.kind for d in report_diagnostics)
    anomaly_hist = Counter(a.kind for a in anomalies)
    diagnostics_hist.update(report_diagnostics_hist)
    summary = {
        "contract": EXPRESSIONS_CONTRACT_VERSION,
        "expression_count": len(expressions),
        "consumer_count": len(consumers),
        "call_consumer_count": consumer_kind_hist.get("native_call", 0) + consumer_kind_hist.get("script_call", 0),
        "native_call_consumer_count": consumer_kind_hist.get("native_call", 0),
        "script_call_consumer_count": consumer_kind_hist.get("script_call", 0),
        "branch_predicate_consumer_count": consumer_kind_hist.get("branch_predicate", 0),
        "predicate_no_transfer_consumer_count": consumer_kind_hist.get("predicate_no_transfer", 0),
        "jump_consumer_count": consumer_kind_hist.get("jump", 0),
        "return_frame_consumer_count": consumer_kind_hist.get("return_frame", 0),
        "pending_call_result_argv_count": pending_arg_count,
        "unresolved_argv_slot_count": unresolved_arg_count,
        "unresolved_script_target_count": unresolved_script_target_count,
        "missing_expression_ref_count": missing_expression_ref_count,
        "expression_kind_histogram": _counter_dict(expr_kind_hist),
        "consumer_kind_histogram": _counter_dict(consumer_kind_hist),
        "target_kind_histogram": _counter_dict(target_kind_hist),
        "diagnostic_count": len(report_diagnostics) + sum(1 for c in consumers if c.diagnostics),
        "report_diagnostic_count": len(report_diagnostics),
        "consumer_diagnostic_count": sum(1 for c in consumers if c.diagnostics),
        "diagnostic_kind_histogram": _counter_dict(diagnostics_hist),
        "report_diagnostic_kind_histogram": _counter_dict(report_diagnostics_hist),
        "anomaly_count": len(anomalies),
        "anomaly_kind_histogram": _counter_dict(anomaly_hist),
        **dataflow_event_summary,
        "policy": EXPRESSIONS_POLICY,
    }

    return VMFunctionExpressionsReport(
        contract=EXPRESSIONS_CONTRACT_VERSION,
        function=str(payload.get("name") or payload.get("function") or ""),
        symbol=str(payload.get("symbol") or payload.get("name") or ""),
        span=dict(payload.get("span") or {}),
        summary=summary,
        expressions=expressions,
        consumers=consumers,
        diagnostics=sorted(report_diagnostics, key=lambda a: (a.kind, a.block or "", -1 if a.offset is None else int(a.offset), a.id)),
        anomalies=sorted(anomalies, key=lambda a: (a.kind, a.block or "", -1 if a.offset is None else int(a.offset), a.id)),
    )


def analyze_module_expressions(
    mod_or_ir: MBCModule | VMModuleIR | dict[str, Any] | str | Path,
    *,
    function: Optional[str] = None,
    include_exports: bool = True,
    include_definitions: bool = True,
    limit_functions: Optional[int] = None,
) -> VMModuleExpressionsReport:
    """Build expression reports for a module or an already-built module IR."""

    if isinstance(mod_or_ir, VMModuleIR):
        module_payload = mod_or_ir.to_dict()
        module_name = str(module_payload.get("module_path") or "")
        function_payloads = [_as_dict(fn) for fn in module_payload.get("functions") or []]
        if function:
            function_payloads = [fn for fn in function_payloads if fn.get("name") == function or fn.get("symbol") == function]
        if limit_functions is not None:
            function_payloads = function_payloads[: max(0, int(limit_functions))]
        functions = [analyze_function_expressions(fn) for fn in function_payloads]
    elif isinstance(mod_or_ir, dict) and "functions" in mod_or_ir:
        module_name = str(mod_or_ir.get("module_path") or mod_or_ir.get("module") or "")
        function_payloads = [_as_dict(fn) for fn in mod_or_ir.get("functions") or []]
        if function:
            function_payloads = [fn for fn in function_payloads if fn.get("name") == function or fn.get("symbol") == function]
        if limit_functions is not None:
            function_payloads = function_payloads[: max(0, int(limit_functions))]
        functions = [analyze_function_expressions(fn) for fn in function_payloads]
    else:
        mod = mod_or_ir if isinstance(mod_or_ir, MBCModule) else MBCModule(mod_or_ir)
        module_name = str(mod.path)
        callable_index = build_callable_index(mod)
        if function:
            entries = [mod.get_function_entry(function)]
        else:
            entries = mod.function_entries(include_definitions=include_definitions, include_exports=include_exports, dedupe=True)
        if limit_functions is not None:
            entries = entries[: max(0, int(limit_functions))]
        functions = [analyze_function_expressions(build_function_ir(mod, entry, callable_index=callable_index)) for entry in entries]

    aggregate = Counter()
    expr_hist: Counter[str] = Counter()
    consumer_hist: Counter[str] = Counter()
    diagnostic_hist: Counter[str] = Counter()
    report_diagnostic_hist: Counter[str] = Counter()
    anomaly_hist: Counter[str] = Counter()
    dataflow_event_hist: Counter[str] = Counter()
    suppressed_dataflow_event_hist: Counter[str] = Counter()
    for report in functions:
        summary = report.summary
        for key in [
            "expression_count", "consumer_count", "call_consumer_count", "native_call_consumer_count",
            "script_call_consumer_count", "branch_predicate_consumer_count", "predicate_no_transfer_consumer_count", "jump_consumer_count",
            "return_frame_consumer_count", "pending_call_result_argv_count", "unresolved_argv_slot_count",
            "unresolved_script_target_count", "missing_expression_ref_count", "diagnostic_count",
            "report_diagnostic_count", "consumer_diagnostic_count", "discarded_pending_call_result_count", "anomaly_count",
        ]:
            aggregate[key] += int(summary.get(key, 0) or 0)
        expr_hist.update(summary.get("expression_kind_histogram", {}))
        consumer_hist.update(summary.get("consumer_kind_histogram", {}))
        diagnostic_hist.update(summary.get("diagnostic_kind_histogram", {}))
        report_diagnostic_hist.update(summary.get("report_diagnostic_kind_histogram", {}))
        anomaly_hist.update(summary.get("anomaly_kind_histogram", {}))
        dataflow_event_hist.update(summary.get("dataflow_event_kind_histogram", {}))
        suppressed_dataflow_event_hist.update(summary.get("suppressed_dataflow_event_kind_histogram", {}))

    summary = dict(sorted((str(k), int(v)) for k, v in aggregate.items()))
    summary.update({
        "contract": EXPRESSIONS_CONTRACT_VERSION,
        "function_count": len(functions),
        "expression_kind_histogram": _counter_dict(expr_hist),
        "consumer_kind_histogram": _counter_dict(consumer_hist),
        "diagnostic_kind_histogram": _counter_dict(diagnostic_hist),
        "report_diagnostic_kind_histogram": _counter_dict(report_diagnostic_hist),
        "anomaly_kind_histogram": _counter_dict(anomaly_hist),
        "dataflow_event_kind_histogram": _counter_dict(dataflow_event_hist),
        "suppressed_dataflow_event_kind_histogram": _counter_dict(suppressed_dataflow_event_hist),
        "policy": EXPRESSIONS_POLICY,
    })
    return VMModuleExpressionsReport(
        contract=EXPRESSIONS_CONTRACT_VERSION,
        module=module_name,
        summary=summary,
        functions=functions,
    )


def validate_expression_report(report: VMModuleExpressionsReport | VMFunctionExpressionsReport | dict[str, Any]) -> list[dict[str, Any]]:
    """Validate expression report invariants without repairing them."""

    payload = report.to_dict() if hasattr(report, "to_dict") else _as_dict(report)
    functions = payload.get("functions") if "functions" in payload else [payload]
    errors: list[dict[str, Any]] = []
    for fn in functions or []:
        fn_name = str(fn.get("function") or fn.get("name") or "")
        expression_ids = {str(expr.get("id")) for expr in fn.get("expressions") or []}
        for consumer in fn.get("consumers") or []:
            for role in ("argv", "non_argv", "auxiliary"):
                for item in consumer.get(role) or []:
                    expr_id = str(item.get("id"))
                    if item.get("kind") == "missing_expression" or expr_id not in expression_ids:
                        errors.append({
                            "kind": "missing_expression_reference",
                            "function": fn_name,
                            "consumer": consumer.get("id"),
                            "role": role,
                            "expression": expr_id,
                        })
            if consumer.get("kind") in {"native_call", "script_call"}:
                encoded = consumer.get("evidence", {}).get("encoded_argc")
                try:
                    encoded_i = int(encoded)
                except Exception:
                    encoded_i = None
                if encoded_i is not None and len(consumer.get("argv") or []) != encoded_i:
                    deferred = consumer.get("evidence", {}).get("frame_subentry_argc_deficit_deferred")
                    try:
                        deferred_i = int(deferred or 0)
                    except Exception:
                        deferred_i = 0
                    missing = encoded_i - len(consumer.get("argv") or [])
                    if not (missing > 0 and deferred_i >= missing):
                        errors.append({
                            "kind": "argv_count_mismatch",
                            "function": fn_name,
                            "consumer": consumer.get("id"),
                            "encoded_argc": encoded_i,
                            "argv_count": len(consumer.get("argv") or []),
                        })
    return errors


# ---------------------------------------------------------------------------
# Rendering / CLI


def render_function_expressions(report: VMFunctionExpressionsReport | dict[str, Any], *, max_consumers: Optional[int] = None) -> str:
    payload = report.to_dict() if hasattr(report, "to_dict") else _as_dict(report)
    summary = payload.get("summary") or {}
    lines = [
        f"function {payload.get('function')} expressions",
        f"  expressions={summary.get('expression_count', 0)} consumers={summary.get('consumer_count', 0)} diagnostics={summary.get('diagnostic_count', 0)} anomalies={summary.get('anomaly_count', 0)}",
    ]
    consumers = payload.get("consumers") or []
    if max_consumers is not None:
        consumers = consumers[: max(0, int(max_consumers))]
    for consumer in consumers:
        offset = consumer.get("offset")
        kind = consumer.get("kind")
        lines.append(f"  @{int(offset):04x} {kind}: {consumer.get('expression')}")
    return "\n".join(lines)


def render_module_expressions(report: VMModuleExpressionsReport | dict[str, Any], *, max_functions: Optional[int] = None, max_consumers_per_function: Optional[int] = 40) -> str:
    payload = report.to_dict() if hasattr(report, "to_dict") else _as_dict(report)
    summary = payload.get("summary") or {}
    lines = [
        f"module {payload.get('module')} expressions",
        f"  functions={summary.get('function_count', 0)} expressions={summary.get('expression_count', 0)} consumers={summary.get('consumer_count', 0)} anomalies={summary.get('anomaly_count', 0)}",
    ]
    functions = payload.get("functions") or []
    if max_functions is not None:
        functions = functions[: max(0, int(max_functions))]
    for fn in functions:
        lines.append("")
        lines.append(render_function_expressions(fn, max_consumers=max_consumers_per_function))
    return "\n".join(lines)


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Lift high-level expression consumers from VM IR/dataflow facts")
    parser.add_argument("module", type=Path, help=".mbc module")
    parser.add_argument("--json", type=Path, default=None, help="write expression report JSON")
    parser.add_argument("--text", type=Path, default=None, help="write readable expression consumer text")
    parser.add_argument("--function", default=None, help="only analyze one function")
    parser.add_argument("--limit-functions", type=int, default=None)
    parser.add_argument("--no-expressions", action="store_true", help="omit expression atoms from JSON")
    parser.add_argument("--no-consumers", action="store_true", help="omit expression consumers from JSON")
    parser.add_argument("--strict", action="store_true", help="return non-zero if expression report invariants fail")
    args = parser.parse_args(argv)

    report = analyze_module_expressions(
        args.module,
        function=args.function,
        limit_functions=args.limit_functions,
    )
    validation_errors = validate_expression_report(report)
    payload = report.to_dict(include_expressions=not args.no_expressions, include_consumers=not args.no_consumers)
    payload["validation"] = {"error_count": len(validation_errors), "errors": validation_errors}

    if args.json:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.text:
        args.text.write_text(render_module_expressions(report), encoding="utf-8")
    return 2 if args.strict and validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(_main())
