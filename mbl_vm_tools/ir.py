from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .parser import FunctionEntry, MBCModule, TableRecord
from .vm_spec import VMWord, branch_target_offset, decode_words, word_role
from .native import native_call_fact_for_word
from .control import build_control_graph
from .dataflow import analyze_stack_dataflow
from .semantic import classify_branch


IR_CONTRACT_VERSION = "vm-ir-v21"
VMIR_CONTRACT_VERSION = IR_CONTRACT_VERSION
CALL_REL_BIAS = -4


@dataclass(frozen=True)
class VMCallTarget:
    encoded_rel: int
    encoded_argc: int
    absolute_target: int
    formula: str
    resolved: bool
    target_name: Optional[str] = None
    target_kind: Optional[str] = None
    target_record: Optional[dict[str, Any]] = None
    confidence: float = 0.0
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VMFunctionIR:
    name: str
    symbol: str
    source_kind: str
    is_exported: bool
    span: dict[str, int]
    body_selection: dict[str, Any]
    abi: dict[str, Any]
    words: list[VMWord]
    dataflow: dict[str, Any]
    calls: list[dict[str, Any]]
    cfg: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": VMIR_CONTRACT_VERSION,
            "name": self.name,
            "symbol": self.symbol,
            "source_kind": self.source_kind,
            "is_exported": self.is_exported,
            "span": self.span,
            "body_selection": self.body_selection,
            "abi": self.abi,
            "words": [w.to_dict() for w in self.words],
            "dataflow": self.dataflow,
            "calls": self.calls,
            "cfg": self.cfg,
            "diagnostics": self.diagnostics,
        }


@dataclass
class VMModuleIR:
    module_path: str
    summary: dict[str, Any]
    callable_index: dict[str, Any]
    functions: list[VMFunctionIR]

    def to_dict(self, *, include_words: bool = True, include_dataflow: bool = True) -> dict[str, Any]:
        functions: list[dict[str, Any]] = []
        for fn in self.functions:
            payload = fn.to_dict()
            if not include_words:
                payload.pop("words", None)
            if not include_dataflow:
                payload.pop("dataflow", None)
            functions.append(payload)
        return {
            "contract": VMIR_CONTRACT_VERSION,
            "module_path": self.module_path,
            "summary": self.summary,
            "callable_index": self.callable_index,
            "functions": functions,
        }


# ---------------------------------------------------------------------------
# Function body selection


def _span_dict(span: tuple[int, int] | None) -> dict[str, int] | None:
    if span is None:
        return None
    return {"start": int(span[0]), "end": int(span[1])}


def select_function_body_vmir(mod: MBCModule, entry_or_name: FunctionEntry | str) -> tuple[bytes, dict[str, Any]]:
    """Select raw function bytes without source-level quality fallbacks.

    Definitions are authoritative.  Export-only symbols use their public export
    span.  No token-confidence scoring is used here; that kind of heuristic was
    one source of old IR instability.
    """

    entry = mod.get_function_entry(entry_or_name) if isinstance(entry_or_name, str) else entry_or_name
    if entry.source_kind == "export" and entry.definition_record is None:
        span = mod.get_export_public_code_span(entry.symbol)
        return mod._slice_code_span(*span), {
            "mode": "export_public",
            "reason": "export_without_definition",
            "span": _span_dict(span),
            "entry": entry.to_dict(),
        }
    span, reason = mod.get_function_exact_code_span_with_reason(entry)
    if span is None:
        return b"", {
            "mode": "empty",
            "reason": reason or "missing_span",
            "span": None,
            "entry": entry.to_dict(),
        }
    return mod._slice_code_span(*span), {
        "mode": "definition_exact",
        "reason": "definition_table_span",
        "span": _span_dict(span),
        "entry": entry.to_dict(),
    }


# ---------------------------------------------------------------------------
# Callable index and target resolution


def _record_payload(record: TableRecord) -> dict[str, Any]:
    return record.to_dict()


def build_callable_index(mod: MBCModule) -> dict[int, list[dict[str, Any]]]:
    """Build the VM callable address index.

    CALL63A's relative value points into a module-wide callable space.  A local
    definition and an external/import symbol both live in that space.  The key
    invariant is that we index the actual table address (`a`) and do not apply
    a compensating +4 for function prologues.
    """

    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, rec in enumerate(mod.definitions):
        out[int(rec.a)].append({
            "kind": "definition",
            "name": rec.name,
            "symbol": rec.name,
            "table": "definitions",
            "index": idx,
            "record": _record_payload(rec),
        })
    for idx, rec in enumerate(mod.globals):
        out[int(rec.a)].append({
            "kind": "external",
            "name": rec.name,
            "symbol": rec.name,
            "table": "globals",
            "index": idx,
            "record": _record_payload(rec),
        })
    for idx, rec in enumerate(mod.embedded_import_like_exports):
        out[int(rec.a)].append({
            "kind": "external",
            "name": rec.name,
            "symbol": rec.name,
            "table": "embedded_import_like_exports",
            "index": idx,
            "record": _record_payload(rec),
        })
    for idx, rec in enumerate(mod.exports):
        # Exports participate in two ways in the corpus: their code address and
        # their public ordinal (`b`).  Keep both, but prefer definitions if both
        # exist at the code address.
        out[int(rec.a)].append({
            "kind": "export_code",
            "name": rec.name,
            "symbol": rec.name,
            "table": "exports.a",
            "index": idx,
            "record": _record_payload(rec),
        })
        if rec.b != 0xFFFFFFFF:
            out[int(rec.b)].append({
                "kind": "export_ordinal",
                "name": rec.name,
                "symbol": rec.name,
                "table": "exports.b",
                "index": idx,
                "record": _record_payload(rec),
            })
    return dict(out)


def _index_to_json(index: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for addr in sorted(index):
        compact[str(addr)] = index[addr]
    return compact


def _rank_target(candidate: dict[str, Any]) -> int:
    order = {
        "definition": 0,
        "external": 1,
        "export_code": 2,
        "export_ordinal": 3,
    }
    return order.get(str(candidate.get("kind")), 99)


def resolve_call_target(
    mod: MBCModule,
    callable_index: dict[int, list[dict[str, Any]]],
    *,
    function_start: int,
    word: VMWord,
) -> VMCallTarget:
    rel = int(word.operands.get("rel", 0) or 0)
    argc = int(word.operands.get("argc", 0) or 0)
    # Critical VM invariant.  The 32-bit relative payload is relative to the
    # address just after the 32-bit rel field, then biased by -4 to point at the
    # callable table address.  Equivalently: function_start + word_end + rel - 4.
    absolute_target = int(function_start) + int(word.offset) + int(word.size) + rel + CALL_REL_BIAS
    matches = list(callable_index.get(absolute_target, []))
    matches.sort(key=_rank_target)
    alternatives: list[dict[str, Any]] = []
    if not matches:
        for delta in range(1, 9):
            for candidate_addr in (absolute_target - delta, absolute_target + delta):
                for candidate in callable_index.get(candidate_addr, []):
                    alt = dict(candidate)
                    alt["address"] = candidate_addr
                    alt["delta"] = candidate_addr - absolute_target
                    alternatives.append(alt)
            if alternatives:
                break
    if matches:
        best = matches[0]
        return VMCallTarget(
            encoded_rel=rel,
            encoded_argc=argc,
            absolute_target=absolute_target,
            formula="function_start + word.offset + word.size + rel - 4",
            resolved=True,
            target_name=str(best.get("name")),
            target_kind=str(best.get("kind")),
            target_record=best.get("record"),
            confidence=1.0,
            alternatives=[],
        )
    return VMCallTarget(
        encoded_rel=rel,
        encoded_argc=argc,
        absolute_target=absolute_target,
        formula="function_start + word.offset + word.size + rel - 4",
        resolved=False,
        confidence=0.0,
        alternatives=alternatives[:8],
    )


# ---------------------------------------------------------------------------
# ABI / stack / CFG


def _slot_name(ref: Any, mode: Any) -> str:
    if ref is None:
        return "slot<?>"
    if isinstance(mode, int):
        return f"slot_{ref}@0x{mode:02X}"
    return f"slot_{ref}@{mode}"


def infer_abi(words: list[VMWord]) -> dict[str, Any]:
    if not words or words[0].terminal_kind not in {"AGG", "AGG0"}:
        return {
            "source": "none",
            "arity": None,
            "params": [],
            "prologue_word_index": None,
            "note": "no aggregate ABI prologue; do not infer source args from stack balancing",
        }
    pro = words[0]
    children = list(pro.operands.get("children") or [])
    params = []
    for idx, child in enumerate(children):
        params.append({
            "index": idx,
            "tag": child.get("tag"),
            "ref": child.get("ref"),
            "slot": _slot_name(child.get("ref"), child.get("tag")),
        })
    return {
        "source": pro.terminal_kind,
        "arity": int(pro.operands.get("arity", len(params)) or 0),
        "params": params,
        "prologue_word_index": pro.index,
        "raw_arity": pro.operands.get("raw_arity"),
    }


def build_cfg(words: list[VMWord], *, function_start: int, raw: bytes | None = None) -> dict[str, Any]:
    """Build byte/sub-entry control facts using the independent control layer."""

    return build_control_graph(words, function_start=function_start, raw=raw).to_dict()


# ---------------------------------------------------------------------------
# Builders / renderers


def build_function_ir(
    mod: MBCModule | str | Path,
    entry_or_name: FunctionEntry | str,
    *,
    callable_index: Optional[dict[int, list[dict[str, Any]]]] = None,
) -> VMFunctionIR:
    mod = mod if isinstance(mod, MBCModule) else MBCModule(mod)
    entry = mod.get_function_entry(entry_or_name) if isinstance(entry_or_name, str) else entry_or_name
    raw, selection = select_function_body_vmir(mod, entry)
    words = decode_words(raw)
    native_facts_by_word = native_call_fact_for_word(
        module_name=Path(mod.path).name,
        function_name=entry.name,
        words=words,
    )
    span = selection.get("span") or {"start": 0, "end": 0}
    function_start = int(span.get("start", 0))
    if callable_index is None:
        callable_index = build_callable_index(mod)
    abi = infer_abi(words)
    calls: list[dict[str, Any]] = []
    unresolved_calls = 0
    for word in words:
        if word.terminal_kind == "CALL_SCRIPT":
            target = resolve_call_target(mod, callable_index, function_start=function_start, word=word)
            if not target.resolved:
                unresolved_calls += 1
            calls.append({
                "word_index": word.index,
                "offset": word.offset,
                "kind": "script",
                "prefixes_hex": [f"0x{p:02X}" for p in word.prefixes],
                "encoded_argc": int(word.operands.get("argc", 0) or 0),
                "target": target.to_dict(),
            })
        elif word.terminal_kind == "CALL_NATIVE":
            native_fact = native_facts_by_word.get(int(word.index))
            native_payload = native_fact.to_dict() if native_fact is not None else None
            calls.append({
                "word_index": word.index,
                "offset": word.offset,
                "kind": "native",
                "prefixes_hex": [f"0x{p:02X}" for p in word.prefixes],
                "encoded_argc": int(word.operands.get("argc", 0) or 0),
                "opid": word.operands.get("opid"),
                "category": native_fact.category if native_fact is not None else "unresolved_frame",
                "category_name": native_fact.category_name if native_fact is not None else "unresolved operand-frame binding",
                "native": native_payload,
                "target": {
                    "target_name": None,
                    "target_kind": "native",
                    "resolved": False,
                    "opid_is_module_local_evidence": True,
                    "reason": "CALL_NATIVE opid is script/module-local and is not a stable semantic target name",
                },
            })
    cfg = build_cfg(words, function_start=function_start, raw=raw)
    dataflow_report = analyze_stack_dataflow(words, cfg=cfg, raw=raw, function_start=function_start)
    dataflow = dataflow_report.to_dict()
    kind_hist = Counter(w.terminal_kind for w in words)
    prefix_hist = Counter(" ".join(f"0x{p:02X}" for p in w.prefixes) for w in words if w.prefixes)
    diagnostics = {
        "word_count": len(words),
        "unknown_word_count": kind_hist.get("UNKNOWN", 0),
        "kind_histogram": dict(sorted(kind_hist.items())),
        "prefix_histogram": dict(sorted(prefix_hist.items())),
        "dataflow": dataflow.get("summary", {}),
        "unresolved_script_call_count": unresolved_calls,
        "policy": [
            "Stack/dataflow SSA is the only stack model exposed by IR; legacy linear stack simulator was removed.",
            "CALL63A arity is the encoded argc byte only.",
            "Externs are call targets, not synthetic local definitions.",
            "Native calls expose VM-level operand-frame categories; native opid remains module-local evidence and is not emitted as a resolved syscall name.",
            "Control graph uses terminal-op operand-base branch targets and byte/sub-entry blocks; op 0x4A is a jump edge, while op 0x4B/0x4C/0x4D keep taken/fallthrough conditional edges.",
            "Parser v2 treats every definition-table record as an authoritative table entry; no parser-level sidecar suppression is applied.",
        ],
    }
    return VMFunctionIR(
        name=entry.name,
        symbol=entry.symbol,
        source_kind=entry.source_kind,
        is_exported=entry.is_exported,
        span=span,
        body_selection=selection,
        abi=abi,
        words=words,
        dataflow=dataflow,
        calls=calls,
        cfg=cfg,
        diagnostics=diagnostics,
    )


def build_module_ir(
    mod_or_path: MBCModule | str | Path,
    *,
    include_exports: bool = True,
    include_definitions: bool = True,
    limit_functions: Optional[int] = None,
) -> VMModuleIR:
    mod = mod_or_path if isinstance(mod_or_path, MBCModule) else MBCModule(mod_or_path)
    callable_index = build_callable_index(mod)
    entries = mod.function_entries(include_definitions=include_definitions, include_exports=include_exports, dedupe=True)
    if limit_functions is not None:
        entries = entries[: max(0, int(limit_functions))]
    functions: list[VMFunctionIR] = []
    for entry in entries:
        functions.append(build_function_ir(mod, entry, callable_index=callable_index))
    summary = summarize_functions(functions)
    summary.update({
        "module": str(mod.path),
        "function_count": len(functions),
        "callable_address_count": len(callable_index),
        "parser_contract": getattr(mod, "parser_contract", None),
        "parser_layout_policy": getattr(mod, "layout_diagnostics", {}).get("policy"),
        "contract": VMIR_CONTRACT_VERSION,
    })
    return VMModuleIR(str(mod.path), summary, _index_to_json(callable_index), functions)


def summarize_functions(functions: Iterable[VMFunctionIR]) -> dict[str, Any]:
    """Return a compact module-level summary for VM IR output."""

    funcs = list(functions)
    word_kind_hist: Counter[str] = Counter()
    branch_status_hist: Counter[str] = Counter()
    branch_semantic_hist: Counter[str] = Counter()
    edge_kind_hist: Counter[str] = Counter()
    dataflow_anomaly_hist: Counter[str] = Counter()
    dataflow_value_kind_hist: Counter[str] = Counter()
    dataflow_operation_role_hist: Counter[str] = Counter()
    native_category_hist: Counter[str] = Counter()

    totals = Counter({
        "unknown_word_count": 0,
        "script_call_count": 0,
        "native_call_count": 0,
        "resolved_definition_script_call_count": 0,
        "resolved_external_script_call_count": 0,
        "unresolved_script_call_count": 0,
        "proven_control_edge_count": 0,
        "candidate_control_edge_count": 0,
        "dataflow_operation_count": 0,
        "dataflow_value_count": 0,
        "dataflow_phi_count": 0,
        "dataflow_unknown_transfer_count": 0,
        "dataflow_operation_underflow_count": 0,
        "dataflow_join_depth_mismatch_count": 0,
        "dataflow_unresolved_block_count": 0,
        "dataflow_nonconverged_function_count": 0,
    })
    max_function_word_count = 0
    max_stack_depth = 0
    max_operand_frame_depth = 0

    for fn in funcs:
        max_function_word_count = max(max_function_word_count, len(fn.words))
        totals["unknown_word_count"] += int(fn.diagnostics.get("unknown_word_count", 0) or 0)
        word_kind_hist.update(word.terminal_kind for word in fn.words)

        cfg_summary = fn.cfg.get("summary", {}) if isinstance(fn.cfg, dict) else {}
        branch_status_hist.update(cfg_summary.get("branch_status_histogram", {}))
        branch_semantic_hist.update(cfg_summary.get("branch_semantic_kind_histogram", {}))
        edge_kind_hist.update(cfg_summary.get("edge_kind_histogram", {}))
        totals["proven_control_edge_count"] += int(cfg_summary.get("proven_edge_count", 0) or 0)
        totals["candidate_control_edge_count"] += int(cfg_summary.get("candidate_edge_count", 0) or 0)

        df_summary = fn.dataflow.get("summary", {}) if isinstance(fn.dataflow, dict) else {}
        totals["dataflow_operation_count"] += int(df_summary.get("operation_count", 0) or 0)
        totals["dataflow_value_count"] += int(df_summary.get("value_count", 0) or 0)
        totals["dataflow_phi_count"] += int(df_summary.get("phi_count", 0) or 0)
        totals["dataflow_unknown_transfer_count"] += int(df_summary.get("unknown_transfer_count", 0) or 0)
        totals["dataflow_operation_underflow_count"] += int(df_summary.get("operation_underflow_count", 0) or 0)
        totals["dataflow_join_depth_mismatch_count"] += int(df_summary.get("join_depth_mismatch_count", 0) or 0)
        totals["dataflow_unresolved_block_count"] += int(df_summary.get("unresolved_block_count", 0) or 0)
        max_stack_depth = max(max_stack_depth, int(df_summary.get("max_stack_depth", 0) or 0))
        max_operand_frame_depth = max(max_operand_frame_depth, int(df_summary.get("max_operand_frame_depth", 0) or 0))
        if not df_summary.get("fixed_point_converged", True):
            totals["dataflow_nonconverged_function_count"] += 1
        dataflow_anomaly_hist.update(df_summary.get("anomaly_kind_histogram", {}))
        dataflow_value_kind_hist.update(df_summary.get("value_kind_histogram", {}))
        dataflow_operation_role_hist.update(df_summary.get("operation_role_histogram", {}))

        for call in fn.calls:
            if call.get("kind") == "native":
                totals["native_call_count"] += 1
                native_category = call.get("category")
                if native_category:
                    native_category_hist[str(native_category)] += 1
                continue
            if call.get("kind") != "script":
                continue
            totals["script_call_count"] += 1
            target = call.get("target") or {}
            if not target.get("resolved"):
                totals["unresolved_script_call_count"] += 1
            elif target.get("target_kind") == "external":
                totals["resolved_external_script_call_count"] += 1
            elif target.get("target_kind") == "definition":
                totals["resolved_definition_script_call_count"] += 1

    return {
        "total_word_count": sum(len(fn.words) for fn in funcs),
        "max_function_word_count": max_function_word_count,
        "unknown_word_count": totals["unknown_word_count"],
        "script_call_count": totals["script_call_count"],
        "native_call_count": totals["native_call_count"],
        "resolved_definition_script_call_count": totals["resolved_definition_script_call_count"],
        "resolved_external_script_call_count": totals["resolved_external_script_call_count"],
        "unresolved_script_call_count": totals["unresolved_script_call_count"],
        "control_branch_status_histogram": dict(sorted(branch_status_hist.items())),
        "control_branch_semantic_kind_histogram": dict(sorted(branch_semantic_hist.items())),
        "control_edge_kind_histogram": dict(sorted(edge_kind_hist.items())),
        "proven_control_edge_count": totals["proven_control_edge_count"],
        "candidate_control_edge_count": totals["candidate_control_edge_count"],
        "dataflow_operation_count": totals["dataflow_operation_count"],
        "dataflow_value_count": totals["dataflow_value_count"],
        "dataflow_phi_count": totals["dataflow_phi_count"],
        "dataflow_unknown_transfer_count": totals["dataflow_unknown_transfer_count"],
        "dataflow_operation_underflow_count": totals["dataflow_operation_underflow_count"],
        "dataflow_join_depth_mismatch_count": totals["dataflow_join_depth_mismatch_count"],
        "dataflow_unresolved_block_count": totals["dataflow_unresolved_block_count"],
        "dataflow_nonconverged_function_count": totals["dataflow_nonconverged_function_count"],
        "dataflow_max_stack_depth": max_stack_depth,
        "dataflow_max_operand_frame_depth": max_operand_frame_depth,
        "dataflow_anomaly_kind_histogram": dict(sorted(dataflow_anomaly_hist.items())),
        "dataflow_value_kind_histogram": dict(sorted(dataflow_value_kind_hist.items())),
        "dataflow_operation_role_histogram": dict(sorted(dataflow_operation_role_hist.items())),
        "native_call_category_histogram": dict(sorted(native_category_hist.items())),
        "word_kind_histogram": dict(sorted(word_kind_hist.items())),
    }


def render_function_text(fn: VMFunctionIR) -> str:
    lines: list[str] = []
    arity = fn.abi.get("arity")
    arity_text = "unknown" if arity is None else str(arity)
    exported = " exported" if fn.is_exported else ""
    lines.append(f"fn {fn.name} @{fn.span.get('start')}..{fn.span.get('end')} abi={arity_text}{exported}")
    for word in fn.words:
        prefix = "" if not word.prefixes else "[" + " ".join(f"0x{p:02X}" for p in word.prefixes) + "] "
        detail = ""
        if word.terminal_kind == "CALL_SCRIPT":
            call = next((c for c in fn.calls if c.get("word_index") == word.index), None)
            target = (call or {}).get("target", {})
            target_name = target.get("target_name") or f"unresolved@{target.get('absolute_target')}"
            detail = f" argc={word.operands.get('argc')} -> {target_name} ({target.get('target_kind')})"
        elif word.terminal_kind == "CALL_NATIVE":
            call = next((c for c in fn.calls if c.get("word_index") == word.index), None)
            category = (call or {}).get("category") or "unresolved_frame"
            category_name = (call or {}).get("category_name") or category
            detail = f" argc={word.operands.get('argc')} native={category} ({category_name}); opid={word.operands.get('opid')} evidence-only"
        elif word.terminal_kind == "BR":
            op = int(word.operands.get("op", -1) or -1)
            role = word_role(word)
            if role == "branch":
                sem = classify_branch(word)
                kind = sem.branch_kind
            else:
                kind = role
            detail = f" op=0x{op:02X} off={word.operands.get('off')} target={branch_target_offset(word)} {kind}"
        elif word.terminal_kind in {"AGG", "AGG0"}:
            detail = f" arity={word.operands.get('arity')}"
        elif "value" in word.operands:
            detail = f" value={word.operands.get('value')}"
        elif "ref" in word.operands:
            detail = f" ref={word.operands.get('ref')} mode={word.operands.get('mode')}"
        lines.append(f"  {word.offset:04x}: {prefix}{word.terminal_kind:<12} size={word.size:<2}{detail}  ; {word.raw.hex(' ')}")
    return "\n".join(lines)


def render_module_text(module_ir: VMModuleIR, *, max_functions: Optional[int] = None) -> str:
    lines = [f"// contract: {VMIR_CONTRACT_VERSION}", f"// module: {module_ir.module_path}", ""]
    functions = module_ir.functions if max_functions is None else module_ir.functions[:max_functions]
    for fn in functions:
        lines.append(render_function_text(fn))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI



def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build byte-accurate VM IR for an MBL .mbc module")
    parser.add_argument("module", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="write JSON VMIR")
    parser.add_argument("--text", type=Path, default=None, help="write readable VMIR text")
    parser.add_argument("--function", default=None, help="only emit one function")
    parser.add_argument("--no-words", action="store_true", help="omit word streams from JSON")
    parser.add_argument("--no-dataflow", action="store_true", help="omit CFG-aware stack/dataflow SSA from JSON")
    parser.add_argument("--limit-functions", type=int, default=None)
    args = parser.parse_args(argv)

    mod = MBCModule(args.module)
    if args.function:
        callable_index = build_callable_index(mod)
        fn = build_function_ir(mod, args.function, callable_index=callable_index)
        payload: dict[str, Any] = fn.to_dict()
        text = render_function_text(fn)
    else:
        module_ir = build_module_ir(mod, limit_functions=args.limit_functions)
        payload = module_ir.to_dict(include_words=not args.no_words, include_dataflow=not args.no_dataflow)
        text = render_module_text(module_ir)

    if args.json:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.text:
        args.text.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
