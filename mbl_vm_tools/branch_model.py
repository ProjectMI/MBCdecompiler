from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .parser import FunctionEntry, MBCModule
from .vm_spec import (
    VMWord,
    CONDITIONAL_CONTROL_BRANCH_OPS,
    RETURN_WORDS,
    branch_target_offset,
    decode_word_at,
    decode_words,
    is_control_branch_word,
    signed_u16,
    terminal_atom_offset,
    word_role,
)


@dataclass(frozen=True)
class CFGEdge:
    source: int
    target: Optional[int]
    kind: str
    valid: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CFGNode:
    offset: int
    size: int
    end: int
    word: VMWord

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "size": self.size,
            "end": self.end,
            "word": self.word.to_dict(),
        }


@dataclass
class FunctionCFG:
    function_name: str
    symbol: str
    source_kind: str
    span: Optional[tuple[int, int]]
    body_len: int
    nodes: dict[int, CFGNode] = field(default_factory=dict)
    edges: list[CFGEdge] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "symbol": self.symbol,
            "source_kind": self.source_kind,
            "span": self.span,
            "body_len": self.body_len,
            "nodes": [node.to_dict() for _, node in sorted(self.nodes.items())],
            "edges": [edge.to_dict() for edge in self.edges],
            "anomalies": list(self.anomalies),
        }


@dataclass(frozen=True)
class TargetLanding:
    target: int
    in_range: bool
    at_function_end: bool
    linear_boundary: bool
    inside_linear_word: bool
    inside_offset: Optional[int]
    inside_size: Optional[int]
    subentry_kind: str
    target_terminal_kind: Optional[str]
    target_decoder_rule: Optional[str]
    unknown: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BRANCH_OP_NAMES = {
    0x4A: "JMP_4A",
    0x4B: "BR_4B",
    0x4C: "BR_4C",
    0x4D: "BR_4D",
}


BASE_MODELS = ("word_start", "word_start_plus1", "terminal_start", "operand_base", "word_end")


def branch_op_name(op: int) -> str:
    return BRANCH_OP_NAMES.get(int(op) & 0xFF, f"BR_0x{int(op) & 0xFF:02X}")


def is_unconditional_branch(word: VMWord) -> bool:
    if word.terminal_kind != "BR" or not is_control_branch_word(word):
        return False
    return (int(word.operands.get("op", -1) or -1) & 0xFF) == 0x4A


def _entry_label(entry: FunctionEntry) -> tuple[str, str, str]:
    return entry.name, entry.symbol, entry.source_kind


def _add_successor(queue: list[int], target: int, body_len: int) -> bool:
    if 0 <= target < body_len:
        queue.append(target)
        return True
    return 0 <= target <= body_len


def build_cfg_for_body(
    body: bytes,
    *,
    function_name: str = "<body>",
    symbol: Optional[str] = None,
    source_kind: str = "body",
    span: Optional[tuple[int, int]] = None,
    entry_offset: int = 0,
    max_nodes: int = 200_000,
) -> FunctionCFG:
    """Build a byte-entry CFG using the current VM decoder.

    The important point is that nodes are decoded with ``decode_word_at`` from
    the exact byte entry. Branch targets are not snapped to the linear tokenizer
    boundary because this VM can branch to prefix bytes or to terminal atoms
    inside a top-level fused word.
    """

    body_len = len(body)
    cfg = FunctionCFG(
        function_name=function_name,
        symbol=symbol or function_name,
        source_kind=source_kind,
        span=span,
        body_len=body_len,
    )
    if body_len == 0:
        cfg.anomalies.append({"kind": "empty_body"})
        return cfg

    queue: list[int] = [entry_offset]
    queued: set[int] = {entry_offset}
    while queue:
        offset = queue.pop(0)
        if offset in cfg.nodes:
            continue
        if len(cfg.nodes) >= max_nodes:
            cfg.anomalies.append({"kind": "max_nodes_reached", "limit": max_nodes})
            break
        if not (0 <= offset < body_len):
            cfg.anomalies.append({"kind": "entry_out_of_range", "offset": offset, "body_len": body_len})
            continue

        try:
            word = decode_word_at(body, offset, limit=body_len, index=len(cfg.nodes))
        except Exception as exc:  # Defensive: decoder errors are evidence.
            cfg.anomalies.append({"kind": "decode_error", "offset": offset, "error": repr(exc)})
            continue

        end = int(offset) + int(word.size)
        cfg.nodes[offset] = CFGNode(offset=offset, size=int(word.size), end=end, word=word)

        if end > body_len:
            cfg.anomalies.append({
                "kind": "word_overruns_body",
                "offset": offset,
                "size": int(word.size),
                "end": end,
                "body_len": body_len,
            })
            continue

        if word.terminal_kind in RETURN_WORDS:
            continue

        if word.terminal_kind == "BR" and is_control_branch_word(word):
            target = branch_target_offset(word)
            op = int(word.operands.get("op", -1) or -1) & 0xFF
            branch_kind = "jump" if op == 0x4A else "branch"
            valid = 0 <= target <= body_len
            note = "" if valid else "target_out_of_range"
            cfg.edges.append(CFGEdge(source=offset, target=target, kind=branch_kind, valid=valid, note=note))
            if valid and target < body_len and target not in queued:
                queue.append(target)
                queued.add(target)
            elif not valid:
                cfg.anomalies.append({
                    "kind": "branch_target_out_of_range",
                    "source": offset,
                    "target": target,
                    "body_len": body_len,
                    "op": op,
                })

            if op in CONDITIONAL_CONTROL_BRANCH_OPS:
                if end < body_len:
                    cfg.edges.append(CFGEdge(source=offset, target=end, kind="fallthrough", valid=True))
                    if end not in queued:
                        queue.append(end)
                        queued.add(end)
                elif end == body_len:
                    cfg.edges.append(CFGEdge(source=offset, target=end, kind="fallthrough_exit", valid=True))
            continue

        if end < body_len:
            cfg.edges.append(CFGEdge(source=offset, target=end, kind="fallthrough", valid=True))
            if end not in queued:
                queue.append(end)
                queued.add(end)
        elif end == body_len:
            # A non-return word can end exactly at the definition boundary. Keep
            # this visible; it often means a span/model issue, but not always.
            cfg.edges.append(CFGEdge(source=offset, target=end, kind="implicit_exit", valid=True))

    return cfg


def build_cfg_for_function(module: MBCModule, entry: FunctionEntry | str) -> FunctionCFG:
    if isinstance(entry, str):
        entry = module.get_function_entry(entry)
    body = module.get_function_body(entry)
    span, reason = module.get_function_exact_code_span_with_reason(entry)
    if span is None and entry.source_kind == "export" and entry.definition_record is None:
        span = module.get_export_public_code_span(entry.symbol)
    name, symbol, source_kind = _entry_label(entry)
    cfg = build_cfg_for_body(body, function_name=name, symbol=symbol, source_kind=source_kind, span=span)
    if reason is not None and body:
        cfg.anomalies.append({"kind": "span_reason", "reason": reason})
    return cfg


def _linear_intervals(linear_words: Iterable[VMWord]) -> list[tuple[int, int, VMWord]]:
    return [(int(w.offset), int(w.offset) + int(w.size), w) for w in linear_words]


def find_linear_container(target: int, linear_words: list[VMWord]) -> Optional[VMWord]:
    for start, end, word in _linear_intervals(linear_words):
        if start <= target < end:
            return word
    return None


def classify_target_landing(body: bytes, target: int, linear_words: list[VMWord]) -> TargetLanding:
    body_len = len(body)
    in_range = 0 <= target <= body_len
    at_function_end = target == body_len
    linear_offsets = {int(w.offset) for w in linear_words}
    boundary = target in linear_offsets
    container = find_linear_container(target, linear_words) if 0 <= target < body_len else None
    inside = container is not None and not boundary

    subentry_kind = "out_of_range"
    target_terminal_kind: Optional[str] = None
    target_decoder_rule: Optional[str] = None
    unknown = False
    if at_function_end:
        subentry_kind = "function_end"
    elif boundary:
        subentry_kind = "linear_boundary"
    elif inside and container is not None:
        term = terminal_atom_offset(container)
        if target == term:
            subentry_kind = "terminal_atom"
        elif target < term:
            subentry_kind = "prefix_byte"
        else:
            subentry_kind = "operand_byte"
    elif in_range:
        subentry_kind = "intertoken_gap"

    if in_range and not at_function_end:
        try:
            decoded = decode_word_at(body, target, limit=body_len)
            target_terminal_kind = decoded.terminal_kind
            target_decoder_rule = decoded.decoder_rule
            unknown = decoded.terminal_kind == "UNKNOWN"
        except Exception as exc:
            target_terminal_kind = "<decode_error>"
            target_decoder_rule = repr(exc)
            unknown = True
    elif not in_range:
        unknown = True

    return TargetLanding(
        target=target,
        in_range=in_range,
        at_function_end=at_function_end,
        linear_boundary=boundary,
        inside_linear_word=inside,
        inside_offset=int(container.offset) if container else None,
        inside_size=int(container.size) if container else None,
        subentry_kind=subentry_kind,
        target_terminal_kind=target_terminal_kind,
        target_decoder_rule=target_decoder_rule,
        unknown=unknown,
    )


def byte_coverage_runs(body_len: int, nodes: Iterable[CFGNode]) -> list[tuple[int, int]]:
    if body_len <= 0:
        return []
    covered = bytearray(body_len)
    for node in nodes:
        start = max(0, min(body_len, int(node.offset)))
        end = max(start, min(body_len, int(node.end)))
        covered[start:end] = b"\x01" * (end - start)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < body_len:
        if covered[i]:
            i += 1
            continue
        start = i
        while i < body_len and not covered[i]:
            i += 1
        runs.append((start, i))
    return runs


def validate_cfg(body: bytes, cfg: FunctionCFG, *, linear_words: Optional[list[VMWord]] = None) -> dict[str, Any]:
    linear_words = decode_words(body) if linear_words is None else linear_words
    linear_offsets = {int(w.offset) for w in linear_words}
    cfg_offsets = set(cfg.nodes.keys())
    cfg_branch_nodes = [node for node in cfg.nodes.values() if node.word.terminal_kind == "BR" and is_control_branch_word(node.word)]
    cfg_predicate_no_transfer = [node for node in cfg.nodes.values() if node.word.terminal_kind == "BR" and not is_control_branch_word(node.word)]
    target_landings: list[TargetLanding] = []
    for node in cfg_branch_nodes:
        target_landings.append(classify_target_landing(body, branch_target_offset(node.word), linear_words))

    uncovered = byte_coverage_runs(len(body), cfg.nodes.values())
    linear_unreached = [int(w.offset) for w in linear_words if int(w.offset) not in cfg_offsets]
    linear_shadowed = []
    linear_dead = []
    for word in linear_words:
        if int(word.offset) in cfg_offsets:
            continue
        start, end = int(word.offset), int(word.offset) + int(word.size)
        any_covered = False
        for node in cfg.nodes.values():
            if max(start, node.offset) < min(end, node.end):
                any_covered = True
                break
        if any_covered:
            linear_shadowed.append(start)
        else:
            linear_dead.append(start)

    issue_flags = []
    if any(not landing.in_range for landing in target_landings):
        issue_flags.append("branch_target_out_of_range")
    if any(landing.unknown for landing in target_landings):
        issue_flags.append("branch_target_unknown")
    if any(node.word.terminal_kind == "UNKNOWN" for node in cfg.nodes.values()):
        issue_flags.append("reachable_unknown_word")
    if cfg.anomalies:
        issue_flags.append("cfg_anomalies")
    if uncovered:
        issue_flags.append("uncovered_bytes")

    return {
        "body_len": len(body),
        "linear_words": len(linear_words),
        "cfg_nodes": len(cfg.nodes),
        "cfg_edges": len(cfg.edges),
        "linear_unreached_entries": len(linear_unreached),
        "linear_shadowed_entries": len(linear_shadowed),
        "linear_dead_entries": len(linear_dead),
        "uncovered_byte_count": sum(end - start for start, end in uncovered),
        "uncovered_runs": uncovered[:32],
        "uncovered_runs_truncated": max(0, len(uncovered) - 32),
        "reachable_unknown_words": sum(1 for node in cfg.nodes.values() if node.word.terminal_kind == "UNKNOWN"),
        "cfg_control_branches": len(cfg_branch_nodes),
        "cfg_predicate_no_transfer": len(cfg_predicate_no_transfer),
        "branch_targets_out_of_range": sum(1 for landing in target_landings if not landing.in_range),
        "branch_targets_unknown": sum(1 for landing in target_landings if landing.unknown),
        "branch_targets_linear_boundary": sum(1 for landing in target_landings if landing.linear_boundary),
        "branch_targets_subentry": sum(1 for landing in target_landings if landing.inside_linear_word),
        "branch_targets_terminal_atom": sum(1 for landing in target_landings if landing.subentry_kind == "terminal_atom"),
        "branch_targets_prefix_byte": sum(1 for landing in target_landings if landing.subentry_kind == "prefix_byte"),
        "branch_targets_operand_byte": sum(1 for landing in target_landings if landing.subentry_kind == "operand_byte"),
        "branch_targets_function_end": sum(1 for landing in target_landings if landing.at_function_end),
        "issue_flags": issue_flags,
        "target_landings": [landing.to_dict() for landing in target_landings],
        "linear_unreached_offsets": linear_unreached[:64],
        "linear_unreached_offsets_truncated": max(0, len(linear_unreached) - 64),
        "linear_dead_offsets": linear_dead[:64],
        "linear_dead_offsets_truncated": max(0, len(linear_dead) - 64),
    }


def _base_model_target(word: VMWord, model: str) -> int:
    delta = signed_u16(int(word.operands.get("off", 0) or 0))
    term = terminal_atom_offset(word)
    if model == "word_start":
        return int(word.offset) + delta
    if model == "word_start_plus1":
        return int(word.offset) + 1 + delta
    if model == "terminal_start":
        return int(term) + delta
    if model == "operand_base":
        return int(term) + 1 + delta
    if model == "word_end":
        return int(word.offset) + int(word.size) + delta
    raise KeyError(model)


def branch_base_audit_for_module(module: MBCModule) -> dict[str, Any]:
    """Compare plausible BR displacement bases against reachable CFG branches.

    The current spec model is ``operand_base``: the signed u16 displacement is
    relative to the first operand byte after the 4A/4B/4C/4D terminal atom. A
    useful sanity check is whether alternative bases produce out-of-range or
    UNKNOWN targets. The audit uses CFG-reachable branch nodes, not just the
    top-level linear stream, because valid branch entries may start inside a
    fused linear VM word.
    """

    model_stats: dict[str, dict[str, int]] = {
        model: {
            "total": 0,
            "in_range": 0,
            "at_function_end": 0,
            "known_target": 0,
            "unknown_target": 0,
            "linear_boundary": 0,
            "subentry": 0,
            "prefix_byte": 0,
            "terminal_atom": 0,
            "operand_byte": 0,
            "out_of_range": 0,
        }
        for model in BASE_MODELS
    }
    by_op: dict[str, dict[str, dict[str, int]]] = {}
    samples: list[dict[str, Any]] = []

    for entry in module.function_entries():
        body = module.get_function_body(entry)
        if not body:
            continue
        linear_words = decode_words(body)
        cfg = build_cfg_for_function(module, entry)
        for node in cfg.nodes.values():
            word = node.word
            if word.terminal_kind != "BR" or not is_control_branch_word(word):
                continue
            op = int(word.operands.get("op", -1) or -1) & 0xFF
            op_key = f"0x{op:02X}"
            by_op.setdefault(op_key, {model: {key: 0 for key in model_stats[model].keys()} for model in BASE_MODELS})
            for model in BASE_MODELS:
                target = _base_model_target(word, model)
                landing = classify_target_landing(body, target, linear_words)
                stats = model_stats[model]
                op_stats = by_op[op_key][model]
                for bucket in (stats, op_stats):
                    bucket["total"] += 1
                    if landing.in_range:
                        bucket["in_range"] += 1
                    else:
                        bucket["out_of_range"] += 1
                    if landing.at_function_end:
                        bucket["at_function_end"] += 1
                    if landing.in_range and not landing.unknown:
                        bucket["known_target"] += 1
                    if landing.unknown:
                        bucket["unknown_target"] += 1
                    if landing.linear_boundary:
                        bucket["linear_boundary"] += 1
                    if landing.inside_linear_word:
                        bucket["subentry"] += 1
                    if landing.subentry_kind == "prefix_byte":
                        bucket["prefix_byte"] += 1
                    elif landing.subentry_kind == "terminal_atom":
                        bucket["terminal_atom"] += 1
                    elif landing.subentry_kind == "operand_byte":
                        bucket["operand_byte"] += 1
                if model != "operand_base" and (not landing.in_range or landing.unknown):
                    if len(samples) < 20:
                        samples.append({
                            "function": entry.name,
                            "source_offset": int(word.offset),
                            "op": op_key,
                            "model": model,
                            "target": target,
                            "landing": landing.to_dict(),
                        })

    return {
        "module": str(module.path),
        "branch_base_model": "operand_base",
        "scope": "cfg_reachable_branches",
        "models": model_stats,
        "by_op": by_op,
        "bad_alternative_samples": samples,
    }


def analyze_module(module_or_path: MBCModule | str | Path) -> dict[str, Any]:
    module = module_or_path if isinstance(module_or_path, MBCModule) else MBCModule(module_or_path)
    functions: list[dict[str, Any]] = []
    totals = {
        "functions": 0,
        "body_bytes": 0,
        "linear_words": 0,
        "cfg_nodes": 0,
        "cfg_edges": 0,
        "cfg_control_branches": 0,
        "cfg_predicate_no_transfer": 0,
        "branch_targets_out_of_range": 0,
        "branch_targets_unknown": 0,
        "branch_targets_linear_boundary": 0,
        "branch_targets_subentry": 0,
        "branch_targets_terminal_atom": 0,
        "branch_targets_prefix_byte": 0,
        "branch_targets_operand_byte": 0,
        "reachable_unknown_words": 0,
        "uncovered_byte_count": 0,
        "functions_with_issues": 0,
    }
    for entry in module.function_entries():
        body = module.get_function_body(entry)
        cfg = build_cfg_for_function(module, entry)
        validation = validate_cfg(body, cfg)
        totals["functions"] += 1
        totals["body_bytes"] += len(body)
        for key in list(totals.keys()):
            if key in validation and isinstance(validation[key], int):
                if key == "body_len":
                    continue
                totals[key] += validation[key]
        if validation["issue_flags"]:
            totals["functions_with_issues"] += 1
        functions.append({
            "name": entry.name,
            "symbol": entry.symbol,
            "source_kind": entry.source_kind,
            "span": cfg.span,
            "validation": validation,
            "anomalies": cfg.anomalies,
        })
    return {
        "module": str(module.path),
        "parser_contract": module.parser_contract,
        "layout_diagnostics": module.layout_diagnostics,
        "definition_count": len(module.definitions),
        "export_count": len(module.exports),
        "adb_info": module.adb_info.to_dict(),
        "totals": totals,
        "base_audit": branch_base_audit_for_module(module),
        "functions": functions,
    }
