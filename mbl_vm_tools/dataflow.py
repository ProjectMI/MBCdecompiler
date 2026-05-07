from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .vm_spec import VMWord, decode_word_at, decode_words, is_lower_operand_atom, stack_contract, word_role
from .control import VMControlGraph, build_control_graph


DATAFLOW_CONTRACT_VERSION = "vm-stack-dataflow-ssa-v17"
MAX_SSA_VALUE_OPERANDS = 64

DATAFLOW_POLICY = (
    "Stack/dataflow SSA v17 consumes the lower operand/effect facts exposed by vm_spec instead of re-promoting decoded value shapes to persistent stack. "
    "AGG/AGG0 are ABI prologues and are no longer operand-frame atoms. "
    "CALL_* owns the complete local lower operand frame: encoded argc selects the argv suffix, while any older frame prefix is call-side non-argv effect/operator input. "
    "Pending call-result candidates are demand-bound lower values: one candidate may satisfy all missing argv slots required by the next CALL consumer without being promoted to persistent SSA. "
    "Residual uncertainty is reported as lower opcode/effect provenance, not post-join repair. "
    "Lower operand frames propagate across transparent CFG boundaries until an explicit lower-frame consumer or function boundary consumes/expires them. Conditional taken edges into prefixed/sub-entry CALL_* blocks transfer the branch predicate frame suffix required by the overlapping call entry; this is an edge-level VM rule, not an accepted argc deficit."
)


@dataclass
class VMSSAValue:
    id: str
    kind: str
    block: Optional[str] = None
    word_index: Optional[int] = None
    offset: Optional[int] = None
    role: Optional[str] = None
    producer: Optional[str] = None
    operands: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "block": self.block,
            "word_index": self.word_index,
            "offset": self.offset,
            "role": self.role,
            "producer": self.producer,
            "operands": list(self.operands),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class VMSSAOperation:
    id: str
    block: str
    word_index: Optional[int]
    offset: int
    terminal_kind: str
    role: str
    pop: Optional[int]
    push: int
    inputs: list[str]
    outputs: list[str]
    depth_before: Optional[int]
    depth_after: Optional[int]
    unknown_transfer: bool = False
    underflow: bool = False
    contract: dict[str, Any] = field(default_factory=dict)
    note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMSSABlock:
    id: str
    start_offset: int
    end_offset: int
    predecessors: list[str]
    successors: list[str]
    entry_stack: list[str]
    exit_stack: list[str]
    entry_depth: Optional[int]
    exit_depth: Optional[int]
    operation_ids: list[str]
    flags: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMDataflowAnomaly:
    id: str
    kind: str
    block: Optional[str] = None
    offset: Optional[int] = None
    word_index: Optional[int] = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMStackDataflowReport:
    contract: str
    summary: dict[str, Any]
    blocks: list[VMSSABlock]
    values: list[VMSSAValue]
    operations: list[VMSSAOperation]
    anomalies: list[VMDataflowAnomaly] = field(default_factory=list)

    def to_dict(self, *, include_operations: bool = True, include_values: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract": self.contract,
            "summary": self.summary,
            "blocks": [b.to_dict() for b in self.blocks],
            "anomalies": [a.to_dict() for a in self.anomalies],
        }
        if include_values:
            payload["values"] = [v.to_dict() for v in self.values]
        if include_operations:
            payload["operations"] = [op.to_dict() for op in self.operations]
        return payload

# ---------------------------------------------------------------------------
# SSA dataflow internals


def _as_dicts(cfg: VMControlGraph | dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(cfg, VMControlGraph):
        return [b.to_dict() for b in cfg.blocks], [e.to_dict() for e in cfg.edges]
    return list(cfg.get("blocks") or []), list(cfg.get("edges") or [])


def _dataflow_stack_contract(word: VMWord) -> dict[str, Any]:
    """Return the stack/dataflow transfer contract for one decoded VM word.

    The lower contract now lives in vm_spec.  Dataflow may append provenance, but
    it does not reinterpret REF/IMM/REC/AGG as persistent stack producers.
    """

    contract = dict(stack_contract(word))
    contract["pop"] = 0
    contract["push"] = 0
    return contract


def _reconstruct_raw(words: list[VMWord]) -> bytes:
    if not words:
        return b""
    size = max(int(w.offset) + int(w.size) for w in words)
    data = bytearray(size)
    for word in words:
        start = int(word.offset)
        end = start + int(word.size)
        data[start:end] = bytes(word.raw)[: max(0, end - start)]
    return bytes(data)


def _safe_id_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _limited_operands(values: Iterable[str], cap: int = MAX_SSA_VALUE_OPERANDS) -> tuple[list[str], int]:
    out: list[str] = []
    seen: set[str] = set()
    dropped = 0
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        if len(out) < cap:
            out.append(text)
        else:
            dropped += 1
    return out, dropped


def _word_index_or_none(word: VMWord) -> Optional[int]:
    idx = int(word.index)
    return idx if idx >= 0 else None


def _value_kind(word: VMWord, role: str, output_index: int) -> str:
    if role == "auxiliary_literal_payload":
        return "auxiliary_literal_payload"
    if role in {"operand_frame_value", "lower_operand_atom"}:
        return "operand_frame_value"
    if role == "operand_frame_ref16_offset":
        return "operand_frame_ref16_offset"
    if role == "native_call_frame_result":
        return "native_call_frame_result"
    if role == "script_call_frame_result":
        return "script_call_frame_result"
    if word.terminal_kind == "CALL_NATIVE":
        return "native_call_result"
    if word.terminal_kind == "CALL_SCRIPT":
        return "script_call_result"
    if role == "literal_ref16_offset":
        return "literal_ref16_offset"
    if role == "value":
        return "vm_stack_value"
    if role == "abi_prologue":
        return "abi_value"
    return f"{_safe_id_part(role)}_result" if output_index else "vm_result"


class _SSAContext:
    def __init__(self) -> None:
        self.values: dict[str, VMSSAValue] = {}
        self.value_order: list[str] = []
        self.anomalies: dict[str, VMDataflowAnomaly] = {}
        self.operand_cap_hits = 0
        self.max_operand_frame_depth = 0
        self.max_operand_frame_clause_count = 0

    def value(
        self,
        value_id: str,
        *,
        kind: str,
        block: Optional[str] = None,
        word_index: Optional[int] = None,
        offset: Optional[int] = None,
        role: Optional[str] = None,
        producer: Optional[str] = None,
        operands: Optional[Iterable[str]] = None,
        evidence: Optional[dict[str, Any]] = None,
    ) -> str:
        operands_list, dropped_operands = _limited_operands(str(v) for v in (operands or []) if str(v))
        if value_id not in self.values:
            initial_evidence = dict(evidence or {})
            if dropped_operands:
                self.operand_cap_hits += 1
                initial_evidence["operand_overflow_count"] = dropped_operands
                self.anomaly(VMDataflowAnomaly(
                    id=f"value_operand_cap:{value_id}",
                    kind="value_operand_cap",
                    block=block,
                    offset=offset,
                    word_index=word_index,
                    detail={"value": value_id, "kept_operand_count": len(operands_list), "dropped_operand_count": dropped_operands},
                ))
            self.values[value_id] = VMSSAValue(
                id=value_id,
                kind=kind,
                block=block,
                word_index=word_index,
                offset=offset,
                role=role,
                producer=producer,
                operands=operands_list,
                evidence=initial_evidence,
            )
            self.value_order.append(value_id)
            return value_id

        existing = self.values[value_id]
        if operands_list:
            merged, dropped = _limited_operands([*existing.operands, *operands_list])
            if dropped:
                self.operand_cap_hits += 1
                merged_evidence = dict(existing.evidence)
                merged_evidence["operand_overflow_count"] = int(merged_evidence.get("operand_overflow_count", 0) or 0) + dropped
                existing.evidence = merged_evidence
                self.anomaly(VMDataflowAnomaly(
                    id=f"value_operand_cap:{value_id}",
                    kind="value_operand_cap",
                    block=block or existing.block,
                    offset=offset if offset is not None else existing.offset,
                    word_index=word_index if word_index is not None else existing.word_index,
                    detail={"value": value_id, "kept_operand_count": len(merged), "dropped_operand_count": merged_evidence["operand_overflow_count"]},
                ))
            existing.operands = merged
        if evidence:
            merged_evidence = dict(existing.evidence)
            merged_evidence.update(evidence)
            existing.evidence = merged_evidence
        return value_id

    def produced_value(self, block_id: str, word: VMWord, role: str, output_index: int, op_id: str) -> str:
        value_id = f"v_{_safe_id_part(block_id)}_{int(word.offset):04x}_{output_index}"
        evidence: dict[str, Any] = {
            "terminal_kind": word.terminal_kind,
            "decoder_rule": word.decoder_rule,
        }
        if word.prefixes:
            evidence["prefixes_hex"] = [f"0x{p:02X}" for p in word.prefixes]
        for key in ("argc", "opid", "ref", "mode", "value"):
            if key in word.operands:
                evidence[key] = word.operands.get(key)
        return self.value(
            value_id,
            kind=_value_kind(word, role, output_index),
            block=block_id,
            word_index=_word_index_or_none(word),
            offset=int(word.offset),
            role=role,
            producer=op_id,
            evidence=evidence,
        )

    def anomaly(self, anomaly: VMDataflowAnomaly) -> None:
        self.anomalies[anomaly.id] = anomaly


def _decode_block_words(block: dict[str, Any], words_by_offset: dict[int, VMWord], raw: bytes) -> list[VMWord]:
    out: list[VMWord] = []
    for offset in list(block.get("instruction_offsets") or []):
        off = int(offset)
        if off in words_by_offset:
            out.append(words_by_offset[off])
            continue
        try:
            out.append(decode_word_at(raw, off, index=-1))
        except Exception:
            continue
    return out


def _entry_blocks(blocks: list[dict[str, Any]], preds: dict[str, set[str]]) -> list[str]:
    entries = [str(b.get("id")) for b in blocks if not preds.get(str(b.get("id")))]
    return sorted(entries, key=lambda bid: next((int(b.get("start_offset", 0) or 0) for b in blocks if str(b.get("id")) == bid), 0))


def _is_operand_frame_atom(word: VMWord, role: str, contract: dict[str, Any]) -> bool:
    """Delegate lower operand-frame atom identity to vm_spec."""

    return bool(contract.get("operand_frame_atom") or is_lower_operand_atom(word))


def _append_prefix_evidence(contract: dict[str, Any], word: VMWord) -> None:
    prefixes = [int(p) for p in word.prefixes]
    if not prefixes:
        return
    contract.update({
        "prefix_count": len(prefixes),
        "prefixes_hex": [f"0x{p:02X}" for p in prefixes],
        "has_prefix_frame_operator": True,
        "starts_new_operand_frame": 0x30 in prefixes,
        "prefix_effect_rule": "prefix_chain_is_executable_operand_frame_evidence",
    })


def _flatten_frame_clauses(clauses: list[list[str]]) -> list[str]:
    return [value_id for clause in clauses for value_id in clause]


def _dedupe_ordered(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _call_result_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result.get("producer_op"),
        result.get("terminal_kind"),
        result.get("offset"),
        result.get("word_index"),
        result.get("opid"),
        result.get("encoded_argc"),
    )


def _empty_frame_state() -> dict[str, Any]:
    return {
        "clauses": [],
        "pending_aux": [],
        "pending_call_results": [],
        "variant_count": 1,
        "empty_variant": True,
    }


def _frame_state_from_parts(
    clauses: list[list[str]],
    pending_aux: list[str],
    pending_call_results: list[dict[str, Any]],
    *,
    variant_count: int = 1,
    empty_variant: Optional[bool] = None,
) -> dict[str, Any]:
    clean_clauses = [list(dict.fromkeys(str(v) for v in clause if str(v))) for clause in clauses if clause]
    clean_aux = _dedupe_ordered(pending_aux)
    clean_pending: list[dict[str, Any]] = []
    seen_pending: set[tuple[Any, ...]] = set()
    for result in pending_call_results:
        payload = dict(result)
        sig = _call_result_signature(payload)
        if sig in seen_pending:
            continue
        seen_pending.add(sig)
        clean_pending.append(payload)
    has_payload = bool(clean_clauses or clean_aux or clean_pending)
    return {
        "clauses": clean_clauses,
        "pending_aux": clean_aux,
        "pending_call_results": clean_pending,
        "variant_count": max(1, int(variant_count)),
        "empty_variant": (not has_payload) if empty_variant is None else bool(empty_variant),
    }


def _clone_frame_state(state: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not state:
        return _empty_frame_state()
    return _frame_state_from_parts(
        [list(clause) for clause in state.get("clauses") or []],
        [str(v) for v in state.get("pending_aux") or []],
        [dict(result) for result in state.get("pending_call_results") or []],
        variant_count=int(state.get("variant_count", 1) or 1),
        empty_variant=bool(state.get("empty_variant", False)),
    )


def _frame_state_has_payload(state: Optional[dict[str, Any]]) -> bool:
    if not state:
        return False
    return bool(state.get("clauses") or state.get("pending_aux") or state.get("pending_call_results"))


def _frame_state_signature(state: Optional[dict[str, Any]]) -> tuple[Any, ...]:
    if not state:
        state = _empty_frame_state()
    return (
        tuple(tuple(str(v) for v in clause) for clause in state.get("clauses") or []),
        tuple(str(v) for v in state.get("pending_aux") or []),
        tuple(_call_result_signature(dict(result)) for result in state.get("pending_call_results") or []),
        bool(state.get("empty_variant", False)),
    )


def _merge_frame_states(*states: Optional[dict[str, Any]]) -> dict[str, Any]:
    clauses: list[list[str]] = []
    pending_aux: list[str] = []
    pending_results: list[dict[str, Any]] = []
    variant_count = 0
    empty_variant = False
    seen_clause_values: set[tuple[str, ...]] = set()
    for state in states:
        if state is None:
            continue
        clone = _clone_frame_state(state)
        variant_count += int(clone.get("variant_count", 1) or 1)
        empty_variant = empty_variant or bool(clone.get("empty_variant", False))
        for clause in clone.get("clauses") or []:
            cleaned = tuple(_dedupe_ordered(clause))
            if not cleaned or cleaned in seen_clause_values:
                continue
            seen_clause_values.add(cleaned)
            clauses.append(list(cleaned))
        pending_aux.extend(str(v) for v in clone.get("pending_aux") or [])
        pending_results.extend(dict(result) for result in clone.get("pending_call_results") or [])
    if variant_count <= 0:
        return _empty_frame_state()
    return _frame_state_from_parts(
        clauses,
        pending_aux,
        pending_results,
        variant_count=variant_count,
        empty_variant=empty_variant,
    )


def _edge_transferred_entry_state(
    edge: dict[str, Any],
    *,
    normal_exit_state: dict[str, Any],
    source_ops: list[VMSSAOperation],
    target_words: list[VMWord],
) -> dict[str, Any]:
    """Return the lower-frame state that flows over one CFG edge.

    Branch predicates are real lower operand-frame consumers.  There is one
    byte-level exception that belongs here rather than in CALL arity repair:
    when a conditional taken edge lands in a prefixed/sub-entry CALL_* block,
    the overlapping CALL entry consumes the suffix of the branch predicate frame
    carried by that taken edge.  Other edges receive the ordinary block exit
    state.
    """

    if str(edge.get("kind")) != "conditional_taken" or not target_words:
        return _clone_frame_state(normal_exit_state)
    first = target_words[0]
    if first.terminal_kind not in {"CALL_NATIVE", "CALL_SCRIPT"}:
        return _clone_frame_state(normal_exit_state)
    if _word_index_or_none(first) is not None or not first.prefixes:
        return _clone_frame_state(normal_exit_state)
    try:
        argc = int(first.operands.get("argc", 0) or 0)
    except Exception:
        argc = 0
    if argc <= 0:
        return _clone_frame_state(normal_exit_state)
    instruction_offset = edge.get("instruction_offset")
    try:
        instruction_offset = int(instruction_offset)
    except Exception:
        instruction_offset = None
    branch_ops = [
        op for op in source_ops
        if op.terminal_kind == "BR" and (instruction_offset is None or int(op.offset) == instruction_offset)
    ]
    if not branch_ops:
        return _clone_frame_state(normal_exit_state)
    branch_inputs = list(branch_ops[-1].inputs or [])
    if len(branch_inputs) < argc:
        return _clone_frame_state(normal_exit_state)
    argv_suffix = branch_inputs[-argc:]
    return _frame_state_from_parts(
        [argv_suffix],
        [],
        [],
        empty_variant=False,
    )


def _frame_state_evidence(state: Optional[dict[str, Any]]) -> dict[str, Any]:
    state = _clone_frame_state(state)
    clauses = [list(c) for c in state.get("clauses") or []]
    return {
        "entry_frame_clause_count": _nonempty_clause_count(clauses),
        "entry_frame_atom_count": _frame_atom_count(clauses),
        "entry_auxiliary_value_count": len(state.get("pending_aux") or []),
        "entry_pending_call_result_count": len(state.get("pending_call_results") or []),
        "entry_frame_variant_count": int(state.get("variant_count", 1) or 1),
        "entry_frame_has_empty_variant": bool(state.get("empty_variant", False)),
        "entry_frame_transfer_rule": "lower_operand_frame_flows_across_transparent_cfg_edges_until_a_real_consumer",
    }


def _nonempty_clause_count(clauses: list[list[str]]) -> int:
    return sum(1 for clause in clauses if clause)


def _frame_atom_count(clauses: list[list[str]]) -> int:
    return sum(len(clause) for clause in clauses)


def _record_call_result_binding(
    ctx: _SSAContext,
    *,
    block_id: str,
    result: dict[str, Any],
    binding: str,
    consumer_word: Optional[VMWord] = None,
    consumer_kind: Optional[str] = None,
    detail_extra: Optional[dict[str, Any]] = None,
) -> None:
    source_kind = str(result.get("terminal_kind"))
    kind = "call_result_binding"
    source_offset = int(result.get("offset", -1) or -1)
    consumer_offset = int(consumer_word.offset) if consumer_word is not None else None
    unique_consumer = "exit" if consumer_offset is None else f"{consumer_offset:04x}"
    anomaly_id = f"{kind}:{source_kind}:{binding}:{block_id}:{source_offset:04x}:{unique_consumer}:{result.get('value_id', 'candidate')}"
    detail = {
        "value": result.get("value_id"),
        "source_terminal_kind": source_kind,
        "source_offset": source_offset,
        "source_word_index": result.get("word_index"),
        "opid": result.get("opid"),
        "encoded_argc": result.get("encoded_argc"),
        "binding": binding,
    }
    if consumer_word is not None:
        detail.update({
            "consumer_terminal_kind": consumer_word.terminal_kind,
            "consumer_offset": int(consumer_word.offset),
            "consumer_word_index": _word_index_or_none(consumer_word),
            "consumer_prefixes_hex": [f"0x{int(p):02X}" for p in consumer_word.prefixes],
        })
    if consumer_kind is not None:
        detail["consumer_kind"] = consumer_kind
    if detail_extra:
        detail.update(detail_extra)
    ctx.anomaly(VMDataflowAnomaly(
        id=anomaly_id,
        kind=kind,
        block=block_id,
        offset=source_offset if source_offset >= 0 else None,
        word_index=result.get("word_index"),
        detail=detail,
    ))


def _record_pending_call_results(
    ctx: _SSAContext,
    *,
    block_id: str,
    pending_call_results: list[dict[str, Any]],
    binding: str,
    consumer_word: Optional[VMWord] = None,
    consumer_kind: Optional[str] = None,
    detail_extra: Optional[dict[str, Any]] = None,
) -> None:
    for result in pending_call_results:
        _record_call_result_binding(
            ctx,
            block_id=block_id,
            result=result,
            binding=binding,
            consumer_word=consumer_word,
            consumer_kind=consumer_kind,
            detail_extra=detail_extra,
        )


def _record_operand_frame_balance(
    ctx: _SSAContext,
    *,
    block_id: str,
    word: VMWord,
    ordinal: int,
    consumer_kind: str,
    encoded_argc: Optional[int],
    frame_clause_count: int,
    frame_atom_count: int,
    auxiliary_input_count: int,
    pending_call_result_count: int = 0,
    bound_non_arg_frame_atom_count: int = 0,
    unexplained_surplus_frame_atom_count: int = 0,
    argc_deficit: int = 0,
) -> None:
    if encoded_argc is None:
        return
    if int(argc_deficit) <= 0 and int(unexplained_surplus_frame_atom_count) <= 0:
        return
    delta = int(unexplained_surplus_frame_atom_count) - int(argc_deficit)
    ctx.anomaly(VMDataflowAnomaly(
        id=f"operand_frame_arity_delta:{block_id}:{int(word.offset):04x}:{ordinal}",
        kind="operand_frame_arity_delta",
        block=block_id,
        offset=int(word.offset),
        word_index=_word_index_or_none(word),
        detail={
            "consumer_kind": consumer_kind,
            "encoded_argc": int(encoded_argc),
            "frame_clause_count": frame_clause_count,
            "frame_atom_count": frame_atom_count,
            "auxiliary_input_count": auxiliary_input_count,
            "pending_call_result_count": pending_call_result_count,
            "bound_non_arg_frame_atom_count": int(bound_non_arg_frame_atom_count),
            "unexplained_surplus_frame_atom_count": int(unexplained_surplus_frame_atom_count),
            "argc_deficit": int(argc_deficit),
            "delta": delta,
            "atom_argc_delta": frame_atom_count - int(encoded_argc),
            "unexplained_atom_argc_delta": delta,
            "clause_argc_delta": frame_clause_count - int(encoded_argc),
            "opid": word.operands.get("opid"),
            "prefixes_hex": [f"0x{int(p):02X}" for p in word.prefixes],
            "available_surplus_or_deficit": "deficit" if int(argc_deficit) > 0 else "surplus",
            "reason": "encoded argc selects the argv suffix; 0x30-marked lower clauses are bound as non-argv call/effect inputs, leaving only unexplained surplus or deficit as lower-model uncertainty",
        },
    ))


def _transfer_block(
    ctx: _SSAContext,
    *,
    block_id: str,
    block_words: list[VMWord],
    initial_frame_state: Optional[dict[str, Any]] = None,
    report_open_frame_at_exit: bool = True,
) -> tuple[list[VMSSAOperation], dict[str, Any]]:
    """Transfer one block using the v16 lower operand-frame propagation model.

    Raw value-like atoms enter local operand-frame clauses.  A prefix-chain containing
    byte 0x30 starts a lower call/effect clause; this is not an argc rule.  CALL_*
    owns the complete local lower operand frame. It consumes exactly the suffix
    selected by encoded argc as argv, binds the older frame prefix as non-argv
    call-side operator/effect inputs, and may use pending
    a pending call-result candidate to explain all true argv deficits demanded by the next CALL consumer.  This records origin
    without promoting call return arity to persistent SSA.
    """

    initial_state = _clone_frame_state(initial_frame_state)
    clauses: list[list[str]] = [list(clause) for clause in initial_state.get("clauses") or []]
    clause_has_30_marker: list[bool] = [False for _ in clauses]
    clause_first_shape: list[Optional[str]] = [None for _ in clauses]
    pending_aux: list[str] = [str(v) for v in initial_state.get("pending_aux") or []]
    pending_call_results: list[dict[str, Any]] = [dict(result) for result in initial_state.get("pending_call_results") or []]
    operations: list[VMSSAOperation] = []
    initial_state_has_payload = _frame_state_has_payload(initial_state)

    def update_frame_max() -> None:
        ctx.max_operand_frame_depth = max(
            ctx.max_operand_frame_depth,
            _frame_atom_count(clauses) + len(pending_aux) + len(pending_call_results),
        )
        ctx.max_operand_frame_clause_count = max(
            ctx.max_operand_frame_clause_count,
            _nonempty_clause_count(clauses) + (1 if pending_call_results else 0),
        )

    def begin_clause(*, has_30_marker: bool = False) -> list[str]:
        if not clauses or clauses[-1]:
            clauses.append([])
            clause_has_30_marker.append(bool(has_30_marker))
            clause_first_shape.append(None)
        else:
            clause_has_30_marker[-1] = bool(clause_has_30_marker[-1] or has_30_marker)
        return clauses[-1]

    def ensure_clause() -> list[str]:
        if not clauses:
            clauses.append([])
            clause_has_30_marker.append(False)
            clause_first_shape.append(None)
        return clauses[-1]

    def start_new_clause(word: VMWord, ordinal: int) -> None:
        _ = (word, ordinal)
        # A 0x30 prefix starts a lower operand/effect clause, but it is not a
        # boundary that invalidates a pending bare-u32 payload.  The payload still
        # belongs to the following operand atom or CALL consumer.
        begin_clause(has_30_marker=True)

    def materialize_pending_call_results(word: VMWord, binding: str, consumer_kind: str) -> None:
        nonlocal pending_call_results
        if not pending_call_results:
            return
        _record_pending_call_results(
            ctx,
            block_id=block_id,
            pending_call_results=pending_call_results,
            binding=binding,
            consumer_word=word,
            consumer_kind=consumer_kind,
            detail_extra={
                "reason": "pending call-result candidates reached a lower-frame consumer; ABI return arity remains provenance unless a later argv deficit binds the candidate",
            },
        )
        pending_call_results = []

    def consume_pending_call_results_for_deficit(word: VMWord, count: int, consumer_kind: str) -> int:
        nonlocal pending_call_results
        remaining = max(0, int(count))
        if remaining <= 0 or not pending_call_results:
            return 0

        # A pending CALL result is not a proven scalar SSA value.  Full-corpus
        # evidence shows it behaves as a demand-bound lower value: the following
        # CALL consumer may destructure/use it to fill every missing argv slot
        # after raw lower atoms have been consumed.  This is lower provenance,
        # not a persistent stack push; unused capacity is not materialized.
        result = pending_call_results.pop()
        take = remaining
        _record_call_result_binding(
            ctx,
            block_id=block_id,
            result=result,
            binding="argv_deficit_candidate",
            consumer_word=word,
            consumer_kind=consumer_kind,
            detail_extra={
                "argc_deficit_candidate_count": take,
                "demand_bound_result": True,
                "reason": "encoded argc required more argv atoms than the raw lower operand-frame suffix provided; the immediately pending call-result candidate is a demand-bound lower value that supplies the missing argv slots without becoming a persistent SSA fact",
            },
        )
        return take

    def clear_frame() -> None:
        nonlocal pending_aux, pending_call_results
        clauses.clear()
        clause_has_30_marker.clear()
        clause_first_shape.clear()
        pending_aux = []
        pending_call_results = []

    def consume_frame_suffix_atoms(count: int) -> list[str]:
        remaining = max(0, int(count))
        if remaining <= 0:
            return []
        chunks: list[list[str]] = []
        while remaining > 0 and clauses:
            if not clauses[-1]:
                clauses.pop()
                clause_has_30_marker.pop()
                clause_first_shape.pop()
                continue
            take = min(len(clauses[-1]), remaining)
            chunk = clauses[-1][-take:]
            del clauses[-1][-take:]
            chunks.append(list(chunk))
            remaining -= take
            if clauses and not clauses[-1]:
                clauses.pop()
                clause_has_30_marker.pop()
                clause_first_shape.pop()
        consumed: list[str] = []
        for chunk in reversed(chunks):
            consumed.extend(chunk)
        return consumed

    def split_non_arg_call_clauses(*, call_has_prefix: bool = False) -> tuple[list[str], list[str], int, int]:
        bound: list[str] = []
        bound_clause_count = 0
        prefixed_call_bound_atom_count = 0
        for clause in clauses:
            if not clause:
                continue
            # CALL is the lower-frame consumer.  Encoded argc selects only the
            # argv suffix; the prefix that remains after suffix consumption is
            # still owned by the call as non-argv operator/effect data.  This is
            # a transfer rule, not an anomaly erasure: the prefix is not compared
            # against argc anymore because argc never described it.
            bound.extend(clause)
            bound_clause_count += 1
            prefixed_call_bound_atom_count += len(clause) if call_has_prefix else 0
        return bound, [], bound_clause_count, prefixed_call_bound_atom_count

    update_frame_max()

    for ordinal, word in enumerate(block_words):
        base_contract = _dataflow_stack_contract(word)
        role = str(base_contract.get("role", word_role(word)))
        before_depth = 0
        inputs: list[str] = []
        outputs: list[str] = []
        op_id = f"op_{_safe_id_part(block_id)}_{int(word.offset):04x}_{ordinal}"
        contract = dict(base_contract)
        if ordinal == 0 and initial_state_has_payload:
            contract.update(_frame_state_evidence(initial_state))
        _append_prefix_evidence(contract, word)

        prefixes = [int(p) for p in word.prefixes]
        starts_new_clause = bool(0x30 in prefixes)
        if starts_new_clause:
            start_new_clause(word, ordinal)
            contract["starts_new_operand_clause"] = True
            contract["operand_clause_boundary_rule"] = "prefix_0x30_clause_marker_is_provenance_not_argument_arity"

        if word.terminal_kind == "BARE_U32":
            aux_id = ctx.produced_value(block_id, word, "auxiliary_literal_payload", len(pending_aux), op_id)
            pending_aux.append(aux_id)
            outputs = [aux_id]
            contract.update({
                "pop": 0,
                "push": 0,
                "frame_push": 0,
                "frame_atom_push": 0,
                "auxiliary_frame_push": 1,
                "frame_clause_count_after": _nonempty_clause_count(clauses),
                "frame_atom_count_after": _frame_atom_count(clauses),
                "stack_effect_rule": "bare_u32_binds_to_next_operand_frame_atom",
            })

        elif _is_operand_frame_atom(word, role, contract):
            atom_role = "operand_frame_ref16_offset" if role == "literal_ref16_offset" else "operand_frame_value"
            inputs = list(pending_aux)
            value_id = ctx.produced_value(block_id, word, atom_role, 0, op_id)
            outputs = [value_id]
            clause = ensure_clause()
            if 0x30 in [int(p) for p in word.prefixes]:
                clause_has_30_marker[-1] = True
            if clause_first_shape[-1] is None:
                clause_first_shape[-1] = str(contract.get("shape_signature"))
            clause.append(value_id)
            pending_aux = []
            contract.update({
                "pop": 0,
                "push": 0,
                "frame_push": 1,
                "frame_atom_push": 1,
                "frame_clause_count_after": _nonempty_clause_count(clauses),
                "frame_atom_count_after": _frame_atom_count(clauses),
                "current_clause_atom_count": len(clauses[-1]) if clauses else 0,
                "stack_effect_rule": "decoded_value_atom_enters_local_operand_clause_not_persistent_stack",
            })

        elif word.terminal_kind in {"CALL_NATIVE", "CALL_SCRIPT"}:
            encoded_argc = int(word.operands.get("argc", 0) or 0)
            frame_atom_count_before = _frame_atom_count(clauses)
            frame_clause_count_before = _nonempty_clause_count(clauses)
            consumed_frame_values = consume_frame_suffix_atoms(encoded_argc)
            consumed_atom_count = len(consumed_frame_values)
            raw_argc_deficit = max(0, encoded_argc - consumed_atom_count)
            pending_result_argc_pop = consume_pending_call_results_for_deficit(word, raw_argc_deficit, word.terminal_kind)
            argc_deficit = max(0, raw_argc_deficit - pending_result_argc_pop)
            subentry_argc_deficit = 0
            non_arg_frame_values, unexplained_frame_values, non_arg_clause_count, prefixed_call_non_arg_atom_count = split_non_arg_call_clauses(call_has_prefix=bool(word.prefixes))
            remaining_atom_count = _frame_atom_count(clauses)
            remaining_clause_count = _nonempty_clause_count(clauses)
            inputs = [*pending_aux, *non_arg_frame_values, *consumed_frame_values]
            pending_aux_count = len(pending_aux)
            if argc_deficit or unexplained_frame_values:
                _record_operand_frame_balance(
                    ctx,
                    block_id=block_id,
                    word=word,
                    ordinal=ordinal,
                    consumer_kind=word.terminal_kind,
                    encoded_argc=encoded_argc,
                    frame_clause_count=frame_clause_count_before,
                    frame_atom_count=frame_atom_count_before,
                    auxiliary_input_count=pending_aux_count,
                    pending_call_result_count=pending_result_argc_pop,
                    bound_non_arg_frame_atom_count=len(non_arg_frame_values),
                    unexplained_surplus_frame_atom_count=len(unexplained_frame_values),
                    argc_deficit=argc_deficit,
                )
            result_id = None
            outputs = []
            pending_result = {
                "value_id": result_id,
                "terminal_kind": word.terminal_kind,
                "offset": int(word.offset),
                "word_index": _word_index_or_none(word),
                "opid": word.operands.get("opid"),
                "encoded_argc": encoded_argc,
                "producer_op": op_id,
            }
            # A call is a lower operand-frame boundary.  It consumes the encoded
            # argc suffix; any older frame prefix is retained only as provenance
            # in the operation contract/anomaly and is not allowed to pollute the
            # next call/predicate binding.
            clauses.clear()
            clause_has_30_marker.clear()
            clause_first_shape.clear()
            pending_aux = []
            pending_call_results.append(pending_result)
            contract.update({
                "pop": 0,
                "push": 0,
                "encoded_argc": encoded_argc,
                "frame_pop": len(inputs),
                "frame_clause_pop": frame_clause_count_before - remaining_clause_count,
                "frame_atom_pop": consumed_atom_count,
                "frame_non_arg_pop": len(non_arg_frame_values),
                "frame_non_arg_clause_pop": non_arg_clause_count,
                "frame_unexplained_surplus_atom_count": len(unexplained_frame_values),
                "frame_atom_count_before_call": frame_atom_count_before,
                "frame_clause_count_before_call": frame_clause_count_before,
                "frame_atom_count_after_call": remaining_atom_count,
                "frame_clause_count_after_call": remaining_clause_count,
                "frame_argc_deficit": argc_deficit,
                "frame_subentry_argc_deficit_deferred": subentry_argc_deficit,
                "frame_raw_argc_deficit": raw_argc_deficit,
                "frame_pending_call_result_argc_pop": pending_result_argc_pop,
                "frame_surplus_prefix_atom_count": remaining_atom_count,
                "frame_bound_non_arg_atom_count": len(non_arg_frame_values),
                "frame_prefixed_call_bound_non_arg_atom_count": prefixed_call_non_arg_atom_count,
                "frame_push": 0,
                "call_result_candidate": True,
                "argc_atom_delta": consumed_atom_count - encoded_argc,
                "argc_available_atom_delta": frame_atom_count_before - encoded_argc,
                "argc_clause_delta": frame_clause_count_before - encoded_argc,
                "frame_result_arity": "demand_bound_deferred_result_candidate",
                "call_argument_binding_rule": "call_consumes_encoded_argc_suffix_of_lower_operand_frame",
                "call_non_arg_binding_rule": "call_owns_complete_local_lower_frame_prefix_as_non_argv_effect_or_operator_inputs",
                "call_boundary_rule": "unexplained_surplus_prefix_is_archived_as_lower_provenance_not_carried_to_next_consumer",
                "subentry_call_frame_rule": "conditional_taken_edge_transfers_overlapping_prefixed_call_argv_suffix" if _word_index_or_none(word) is None and word.prefixes and consumed_atom_count else None,
                "stack_effect_rule": "call_consumes_encoded_argc_suffix_binds_remaining_frame_prefix_and_uses_demand_bound_pending_call_result_for_argc_deficits",
            })
            if word.terminal_kind == "CALL_NATIVE":
                contract["result"] = "pending_native_frame_result"
            else:
                contract["result"] = "pending_script_frame_result"

        elif word.terminal_kind == "BR":
            materialize_pending_call_results(word, "next_consumer", "branch")
            op = int(word.operands.get("op", -1) or -1) & 0xFF
            frame_values = _flatten_frame_clauses(clauses)
            inputs = [*pending_aux, *frame_values]
            clause_count = _nonempty_clause_count(clauses)
            atom_count = _frame_atom_count(clauses)
            contract.update({
                "pop": 0,
                "push": 0,
                "frame_pop": len(inputs),
                "frame_clause_pop": clause_count,
                "frame_atom_pop": atom_count,
                "frame_push": 0,
                "predicate_frame_clause_count": clause_count if op in {0x4B, 0x4C, 0x4D} else 0,
                "predicate_frame_input_count": len(inputs) if op in {0x4B, 0x4C, 0x4D} else 0,
                "control_frame_input_count": len(inputs),
                "stack_effect_rule": "branch_consumes_local_operand_clauses_as_control_or_predicate_data",
            })
            if op in {0x4B, 0x4C, 0x4D}:
                ctx.anomaly(VMDataflowAnomaly(
                    id=f"predicate_frame_bound:{block_id}:{int(word.offset):04x}:{ordinal}",
                    kind="predicate_frame_bound",
                    block=block_id,
                    offset=int(word.offset),
                    word_index=_word_index_or_none(word),
                    detail={
                        "op": op,
                        "frame_clause_count": clause_count,
                        "frame_input_count": len(inputs),
                        "prefixes_hex": [f"0x{int(p):02X}" for p in word.prefixes],
                        "reason": "conditional branch predicate is bound to local operand-frame clauses instead of an unknown persistent stack transfer",
                    },
                ))
            clear_frame()

        elif word.terminal_kind in {"RETURN_PAIR", "END"}:
            materialize_pending_call_results(word, "next_consumer", "return")
            frame_values = _flatten_frame_clauses(clauses)
            inputs = [*pending_aux, *frame_values]
            clause_count = _nonempty_clause_count(clauses)
            atom_count = _frame_atom_count(clauses)
            contract.update({
                "pop": 0,
                "push": 0,
                "frame_pop": len(inputs),
                "frame_clause_pop": clause_count,
                "frame_atom_pop": atom_count,
                "frame_push": 0,
                "return_frame_clause_count": clause_count,
                "return_frame_input_count": len(inputs),
                "stack_effect_rule": "terminal_return_consumes_local_operand_clauses",
            })
            if word.operands.get("optional_value"):
                ctx.anomaly(VMDataflowAnomaly(
                    id=f"terminal_return_frame_bound:{block_id}:{int(word.offset):04x}:{ordinal}",
                    kind="terminal_return_frame_bound",
                    block=block_id,
                    offset=int(word.offset),
                    word_index=_word_index_or_none(word),
                    detail={
                        "frame_clause_count": clause_count,
                        "frame_input_count": len(inputs),
                        "reason": "return payload is bound to local operand-frame clauses rather than a persistent CFG stack slot",
                    },
                ))
            clear_frame()

        else:
            if word.terminal_kind not in {"NOP", "MARK"}:
                materialize_pending_call_results(word, "next_structural_or_unknown_word", "structural_or_unknown_word")
            contract.update({
                "pop": 0,
                "push": 0,
                "frame_clause_count_after": _nonempty_clause_count(clauses),
                "frame_atom_count_after": _frame_atom_count(clauses),
                "stack_effect_rule": "structural_or_unknown_word_has_no_persistent_stack_effect",
            })
            if word.terminal_kind == "UNKNOWN":
                ctx.anomaly(VMDataflowAnomaly(
                    id=f"unknown_vm_word:{block_id}:{int(word.offset):04x}:{ordinal}",
                    kind="unknown_vm_word",
                    block=block_id,
                    offset=int(word.offset),
                    word_index=_word_index_or_none(word),
                    detail={"byte": word.operands.get("byte"), "reason": "decoder produced UNKNOWN below dataflow"},
                ))

        update_frame_max()
        operations.append(
            VMSSAOperation(
                id=op_id,
                block=block_id,
                word_index=_word_index_or_none(word),
                offset=int(word.offset),
                terminal_kind=word.terminal_kind,
                role=str(contract.get("role", role)),
                pop=0,
                push=0,
                inputs=inputs,
                outputs=outputs,
                depth_before=before_depth,
                depth_after=0,
                unknown_transfer=False,
                underflow=False,
                contract=dict(contract),
                note=None,
            )
        )
    final_state = _frame_state_from_parts(clauses, pending_aux, pending_call_results)
    if report_open_frame_at_exit and _frame_state_has_payload(final_state):
        if pending_call_results:
            _record_pending_call_results(
                ctx,
                block_id=block_id,
                pending_call_results=pending_call_results,
                binding="discarded_at_function_exit",
                consumer_word=None,
                consumer_kind="function_exit",
                detail_extra={
                    "function_exit_boundary": True,
                    "discarded_lower_frame_candidate": True,
                    "reason": "function ended before the pending call result candidate was consumed; demand-bound candidate expires at function exit",
                },
            )
        # Residual lower-frame atoms at a structural function boundary are not
        # promoted into model facts and are not expression diagnostics.  They have
        # no consumer, so the frame expires at the VM function boundary.

    return operations, final_state


# ---------------------------------------------------------------------------
# Public analysis entry point


def analyze_stack_dataflow(
    words: list[VMWord],
    *,
    cfg: Optional[VMControlGraph | dict[str, Any]] = None,
    raw: Optional[bytes] = None,
    function_start: int = 0,
    max_iterations: Optional[int] = None,
    max_state_depth: int = 96,
) -> VMStackDataflowReport:
    _ = max_state_depth  # retained for API compatibility; persistent stack joins are no longer modeled.
    raw_bytes = bytes(raw) if raw is not None else _reconstruct_raw(words)
    if cfg is None:
        cfg = build_control_graph(words, function_start=function_start, raw=raw_bytes)
    blocks, edges = _as_dicts(cfg)
    projection_summary = {
        "raw_block_count": len(blocks),
        "raw_edge_count": len([edge for edge in edges if str(edge.get("status", "proven")) != "candidate"]),
    }
    if not words or not blocks:
        return VMStackDataflowReport(
            contract=DATAFLOW_CONTRACT_VERSION,
            summary={
                "block_count": 0,
                "edge_count": 0,
                "operation_count": 0,
                "value_count": 0,
                **projection_summary,
                "policy": DATAFLOW_POLICY,
            },
            blocks=[],
            values=[],
            operations=[],
            anomalies=[],
        )

    block_by_id = {str(b.get("id")): b for b in blocks}
    block_order = {str(b.get("id")): idx for idx, b in enumerate(blocks)}
    successors: dict[str, set[str]] = defaultdict(set)
    predecessors: dict[str, set[str]] = defaultdict(set)
    proven_edges_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_kind_hist: Counter[str] = Counter()
    for edge in edges:
        if str(edge.get("status", "proven")) == "candidate":
            continue
        src = str(edge.get("source"))
        dst = str(edge.get("target"))
        if src not in block_by_id or dst not in block_by_id:
            continue
        successors[src].add(dst)
        predecessors[dst].add(src)
        proven_edges_by_source[src].append(dict(edge))
        edge_kind_hist[str(edge.get("kind", "unknown"))] += 1

    entries = _entry_blocks(blocks, predecessors)
    if not entries and blocks:
        entries = [str(blocks[0].get("id"))]

    words_by_offset = {int(w.offset): w for w in words}
    decoded_by_block = {str(b.get("id")): _decode_block_words(b, words_by_offset, raw_bytes) for b in blocks}

    entry_states: dict[str, dict[str, Any]] = {bid: _empty_frame_state() for bid in entries}
    exit_states: dict[str, dict[str, Any]] = {}
    reached_blocks: set[str] = set()

    q: deque[str] = deque(entries)
    queued = set(entries)
    iteration = 0
    limit = max_iterations if max_iterations is not None else max(1, len(blocks) * 4)
    converged = True

    # First pass: compute lower operand-frame entry states independently from
    # persistent SSA.  This pass is intentionally used only for propagation; its
    # temporary observations are discarded before the final report is built.
    while q:
        block_id = q.popleft()
        queued.discard(block_id)
        if block_id not in block_by_id:
            continue
        iteration += 1
        if iteration > limit:
            converged = False
            break
        reached_blocks.add(block_id)
        dry_ctx = _SSAContext()
        _ops, exit_state = _transfer_block(
            dry_ctx,
            block_id=block_id,
            block_words=decoded_by_block.get(block_id, []),
            initial_frame_state=entry_states.get(block_id),
            report_open_frame_at_exit=False,
        )
        exit_states[block_id] = exit_state
        outgoing_edges = sorted(
            proven_edges_by_source.get(block_id, []),
            key=lambda edge: block_order.get(str(edge.get("target")), 10**9),
        )
        for edge in outgoing_edges:
            succ = str(edge.get("target"))
            transfer_state = _edge_transferred_entry_state(
                edge,
                normal_exit_state=exit_state,
                source_ops=_ops,
                target_words=decoded_by_block.get(succ, []),
            )
            old_state = entry_states.get(succ)
            new_state = _merge_frame_states(old_state, transfer_state) if old_state is not None else _clone_frame_state(transfer_state)
            if old_state is None or _frame_state_signature(new_state) != _frame_state_signature(old_state):
                entry_states[succ] = new_state
                if succ not in queued:
                    q.append(succ)
                    queued.add(succ)

    ctx = _SSAContext()
    if not converged:
        ctx.anomaly(VMDataflowAnomaly(
            id="dataflow_traversal_limit",
            kind="dataflow_traversal_limit",
            detail={"iteration_limit": limit, "queued_block_count": len(q)},
        ))

    operations_by_block: dict[str, list[VMSSAOperation]] = {}
    final_exit_states: dict[str, dict[str, Any]] = {}
    for block in blocks:
        block_id = str(block.get("id"))
        if block_id not in reached_blocks:
            continue
        ops, exit_state = _transfer_block(
            ctx,
            block_id=block_id,
            block_words=decoded_by_block.get(block_id, []),
            initial_frame_state=entry_states.get(block_id),
            report_open_frame_at_exit=not bool(successors.get(block_id)),
        )
        operations_by_block[block_id] = ops
        final_exit_states[block_id] = exit_state

    # Any disconnected block that was not discovered from an entry is reported.
    unresolved_blocks = [bid for bid in block_by_id if bid not in reached_blocks]
    for bid in unresolved_blocks:
        ctx.anomaly(VMDataflowAnomaly(
            id=f"unresolved_block:{bid}",
            kind="unresolved_block",
            block=bid,
            detail={"reason": "no reached dataflow entry state"},
        ))

    ssa_blocks: list[VMSSABlock] = []
    all_operations: list[VMSSAOperation] = []
    for block in blocks:
        bid = str(block.get("id"))
        ops = operations_by_block.get(bid, [])
        all_operations.extend(ops)
        reached = bid in reached_blocks
        diagnostics: dict[str, Any] = {}
        block_anomaly_kinds = Counter(a.kind for a in ctx.anomalies.values() if a.block == bid)
        if block_anomaly_kinds:
            diagnostics["anomaly_kind_histogram"] = dict(sorted(block_anomaly_kinds.items()))
        ssa_blocks.append(VMSSABlock(
            id=bid,
            start_offset=int(block.get("start_offset", 0) or 0),
            end_offset=int(block.get("end_offset", 0) or 0),
            predecessors=sorted(predecessors.get(bid, set()), key=lambda x: block_order.get(x, 10**9)),
            successors=sorted(successors.get(bid, set()), key=lambda x: block_order.get(x, 10**9)),
            entry_stack=[] if reached else [],
            exit_stack=[] if reached else [],
            entry_depth=0 if reached else None,
            exit_depth=0 if reached else None,
            operation_ids=[op.id for op in ops],
            flags=list(block.get("flags") or []),
            diagnostics=diagnostics,
        ))

    value_list = [ctx.values[vid] for vid in ctx.value_order]
    anomalies = sorted(ctx.anomalies.values(), key=lambda a: (a.kind, a.block or "", -1 if a.offset is None else int(a.offset), a.id))

    op_role_hist = Counter(op.role for op in all_operations)
    terminal_hist = Counter(op.terminal_kind for op in all_operations)
    value_kind_hist = Counter(v.kind for v in value_list)
    join_blocks = [b for b in ssa_blocks if len(b.predecessors) > 1]
    subentry_ops = sum(1 for op in all_operations if op.word_index is None)
    call_ops = [op for op in all_operations if op.terminal_kind in {"CALL_NATIVE", "CALL_SCRIPT"}]
    anomaly_hist = Counter(a.kind for a in anomalies)
    call_result_binding_hist: Counter[str] = Counter()
    native_call_result_bound_count = 0
    native_call_result_discarded_count = 0
    native_call_result_open_count = 0
    script_call_result_bound_count = 0
    script_call_result_discarded_count = 0
    script_call_result_open_count = 0
    for anomaly in anomalies:
        if anomaly.kind != "call_result_binding":
            continue
        detail = anomaly.detail or {}
        source_kind = str(detail.get("source_terminal_kind"))
        binding = str(detail.get("binding"))
        key = f"{source_kind}:{binding}"
        call_result_binding_hist[key] += 1
        is_open = binding in {"open_at_block_exit", "open_at_function_exit"}
        is_discarded = binding in {"discarded_at_function_exit"}
        is_observed = not is_open and not is_discarded
        if source_kind == "CALL_NATIVE":
            native_call_result_open_count += 1 if is_open else 0
            native_call_result_discarded_count += 1 if is_discarded else 0
            native_call_result_bound_count += 1 if is_observed else 0
        elif source_kind == "CALL_SCRIPT":
            script_call_result_open_count += 1 if is_open else 0
            script_call_result_discarded_count += 1 if is_discarded else 0
            script_call_result_bound_count += 1 if is_observed else 0
    native_call_result_candidate_count = sum(
        1 for op in all_operations
        if op.terminal_kind == "CALL_NATIVE" and bool(op.contract.get("call_result_candidate"))
    )
    script_call_result_candidate_count = sum(
        1 for op in all_operations
        if op.terminal_kind == "CALL_SCRIPT" and bool(op.contract.get("call_result_candidate"))
    )
    operand_frame_producer_count = sum(1 for op in all_operations if int(op.contract.get("frame_push", 0) or 0) > 0)
    operand_frame_consumer_count = sum(1 for op in all_operations if int(op.contract.get("frame_pop", 0) or 0) > 0)
    operand_frame_consumed_value_count = sum(int(op.contract.get("frame_pop", 0) or 0) for op in all_operations)
    auxiliary_literal_frame_bind_count = sum(int(op.contract.get("auxiliary_frame_push", 0) or 0) for op in all_operations)
    prefix_frame_operator_count = sum(int(op.contract.get("prefix_count", 0) or 0) for op in all_operations)
    pending_call_result_argv_candidate_count = sum(int(op.contract.get("frame_pending_call_result_argc_pop", 0) or 0) for op in all_operations)
    raw_argc_deficit_count = sum(int(op.contract.get("frame_raw_argc_deficit", 0) or 0) for op in all_operations)
    final_argc_deficit_count = sum(int(op.contract.get("frame_argc_deficit", 0) or 0) for op in all_operations)
    subentry_argc_deficit_deferred_count = sum(int(op.contract.get("frame_subentry_argc_deficit_deferred", 0) or 0) for op in all_operations)
    prefix_frame_operator_hist = Counter()
    call_frame_balance_hist = Counter()
    for op in all_operations:
        for pref in op.contract.get("prefixes_hex", []) or []:
            prefix_frame_operator_hist[str(pref)] += 1
        if op.terminal_kind in {"CALL_NATIVE", "CALL_SCRIPT"}:
            encoded = int(op.contract.get("encoded_argc", 0) or 0)
            frame_atoms = int(op.contract.get("frame_atom_pop", op.contract.get("frame_pop", 0)) or 0)
            frame_clauses = int(op.contract.get("frame_clause_pop", frame_atoms) or 0)
            pending_argc = int(op.contract.get("frame_pending_call_result_argc_pop", 0) or 0)
            final_deficit = int(op.contract.get("frame_argc_deficit", 0) or 0)
            opid = op.contract.get("opid")
            key = f"{op.terminal_kind}:opid={opid}:argc={encoded}:atoms={frame_atoms}:pending={pending_argc}:deficit={final_deficit}:clauses={frame_clauses}:atom_delta={frame_atoms + pending_argc - encoded}:clause_delta={frame_clauses - encoded}"
            call_frame_balance_hist[key] += 1
    summary = {
        "block_count": len(blocks),
        "edge_count": sum(edge_kind_hist.values()),
        "entry_block_count": len(entries),
        "entry_blocks": entries,
        "unresolved_block_count": len(unresolved_blocks),
        "operation_count": len(all_operations),
        "subentry_operation_count": subentry_ops,
        "value_count": len(value_list),
        "phi_count": 0,
        "phi_join_block_count": 0,
        "join_block_count": len(join_blocks),
        "join_depth_mismatch_count": anomaly_hist.get("join_depth_mismatch", 0),
        "operation_underflow_count": anomaly_hist.get("operation_underflow", 0),
        "unknown_transfer_count": anomaly_hist.get("unknown_stack_transfer", 0),
        "state_depth_cap_hit_count": 0,
        "value_operand_cap_hit_count": ctx.operand_cap_hits,
        "fixed_point_converged": converged,
        "fixed_point_iteration_count": iteration,
        "max_stack_depth": 0,
        "max_operand_frame_depth": ctx.max_operand_frame_depth,
        "max_operand_frame_clause_count": ctx.max_operand_frame_clause_count,
        "max_phi_operand_count": 0,
        "operand_frame_producer_count": operand_frame_producer_count,
        "operand_frame_consumer_count": operand_frame_consumer_count,
        "operand_frame_consumed_value_count": operand_frame_consumed_value_count,
        "auxiliary_literal_frame_bind_count": auxiliary_literal_frame_bind_count,
        "prefix_frame_operator_count": prefix_frame_operator_count,
        "pending_call_result_argv_candidate_count": pending_call_result_argv_candidate_count,
        "raw_argc_deficit_count": raw_argc_deficit_count,
        "final_argc_deficit_count": final_argc_deficit_count,
        "subentry_argc_deficit_deferred_count": subentry_argc_deficit_deferred_count,
        "prefix_frame_operator_histogram": dict(sorted(prefix_frame_operator_hist.items())),
        "call_operand_frame_balance_histogram_top": dict(call_frame_balance_hist.most_common(32)),
        "call_operation_count": len(call_ops),
        "native_call_operation_count": sum(1 for op in call_ops if op.terminal_kind == "CALL_NATIVE"),
        "script_call_operation_count": sum(1 for op in call_ops if op.terminal_kind == "CALL_SCRIPT"),
        "native_return_deferred_count": native_call_result_open_count,
        "native_call_result_candidate_count": native_call_result_candidate_count,
        "native_call_result_bound_count": native_call_result_bound_count,
        "native_call_result_discarded_count": native_call_result_discarded_count,
        "native_call_result_open_count": native_call_result_open_count,
        "script_call_result_candidate_count": script_call_result_candidate_count,
        "script_call_result_bound_count": script_call_result_bound_count,
        "script_call_result_discarded_count": script_call_result_discarded_count,
        "script_call_result_open_count": script_call_result_open_count,
        "call_result_binding_histogram": dict(sorted(call_result_binding_hist.items())),
        "unresolved_predicate_stack_effect_count": anomaly_hist.get("unresolved_predicate_stack_effect", 0),
        "terminal_return_payload_deferred_count": anomaly_hist.get("terminal_return_payload_deferred", 0),
        "edge_kind_histogram": dict(sorted(edge_kind_hist.items())),
        "operation_role_histogram": dict(sorted(op_role_hist.items())),
        "terminal_kind_histogram": dict(sorted(terminal_hist.items())),
        "value_kind_histogram": dict(sorted(value_kind_hist.items())),
        "unknown_transfer_role_histogram": {},
        "underflow_role_histogram": {},
        "anomaly_kind_histogram": dict(sorted(anomaly_hist.items())),
        "join_depth_delta_histogram": {},
        "join_depth_mismatch_observation_count": 0,
        **projection_summary,
        "policy": DATAFLOW_POLICY,
    }

    return VMStackDataflowReport(
        contract=DATAFLOW_CONTRACT_VERSION,
        summary=summary,
        blocks=ssa_blocks,
        values=value_list,
        operations=all_operations,
        anomalies=anomalies,
    )


# ---------------------------------------------------------------------------
# CLI


def _module_dataflow_payload(module_path: Path, *, function: Optional[str] = None, limit_functions: Optional[int] = None, include_operations: bool = True, include_values: bool = True) -> dict[str, Any]:
    from .parser import MBCModule
    from .ir import select_function_body_vmir

    mod = MBCModule(module_path)
    entries = [mod.get_function_entry(function)] if function else mod.function_entries(include_definitions=True, include_exports=True, dedupe=True)
    if limit_functions is not None:
        entries = entries[: max(0, int(limit_functions))]
    functions: list[dict[str, Any]] = []
    aggregate = Counter()
    for entry in entries:
        raw, selection = select_function_body_vmir(mod, entry)
        words = decode_words(raw)
        span = selection.get("span") or {"start": 0, "end": 0}
        cfg = build_control_graph(words, function_start=int(span.get("start", 0)), raw=raw)
        report = analyze_stack_dataflow(words, cfg=cfg, raw=raw, function_start=int(span.get("start", 0)))
        summary = report.summary
        for key in [
            "operation_count", "value_count", "phi_count", "join_depth_mismatch_count", "operation_underflow_count",
            "unknown_transfer_count", "call_operation_count", "native_call_operation_count", "script_call_operation_count",
            "subentry_operation_count", "subentry_argc_deficit_deferred_count", "unresolved_block_count", "native_return_deferred_count",
            "unresolved_predicate_stack_effect_count", "terminal_return_payload_deferred_count",
            "join_depth_mismatch_observation_count",
        ]:
            aggregate[key] += int(summary.get(key, 0) or 0)
        if not summary.get("fixed_point_converged", False):
            aggregate["nonconverged_function_count"] += 1
        functions.append({
            "name": entry.name,
            "symbol": entry.symbol,
            "span": span,
            "dataflow": report.to_dict(include_operations=include_operations, include_values=include_values),
        })
    return {
        "contract": DATAFLOW_CONTRACT_VERSION,
        "module": str(module_path),
        "function_count": len(functions),
        "summary": dict(sorted(aggregate.items())),
        "functions": functions,
    }


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build CFG-aware stack/dataflow SSA facts for an MBL .mbc module")
    parser.add_argument("module", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="write stack/dataflow SSA JSON")
    parser.add_argument("--function", default=None, help="only emit one function")
    parser.add_argument("--limit-functions", type=int, default=None)
    parser.add_argument("--no-operations", action="store_true", help="omit per-word SSA operations")
    parser.add_argument("--no-values", action="store_true", help="omit SSA values")
    args = parser.parse_args(argv)

    payload = _module_dataflow_payload(
        args.module,
        function=args.function,
        limit_functions=args.limit_functions,
        include_operations=not args.no_operations,
        include_values=not args.no_values,
    )
    if args.json:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
