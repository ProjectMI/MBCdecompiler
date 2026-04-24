from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mbl_vm_tools.hir import (
    HIRBlock,
    build_function_hir,
    build_module_hir,
    _block_maps,
    _coerce_hir_block,
    _match_assignment,
    _normalize_hir_blocks,
    _ordered_unique,
    _rename_merge_params,
)
from mbl_vm_tools.parser import MBCModule


AST_CONTRACT_VERSION = "ast-v1"


@dataclass
class ASTFunction:
    name: str
    span: dict[str, int]
    slice_mode: str
    summary: dict[str, Any]
    ast: dict[str, Any]
    ast_text: str
    report: dict[str, Any]
    body_selection: dict[str, Any]
    hir_summary: dict[str, Any]
    input_contract: str
    normalized_hir_blocks: list[HIRBlock]

    def to_dict(self, include_hir: bool = False, include_text: bool = True) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "span": self.span,
            "slice_mode": self.slice_mode,
            "summary": self.summary,
            "ast": self.ast,
            "report": self.report,
            "body_selection": self.body_selection,
            "hir_summary": self.hir_summary,
            "input_contract": self.input_contract,
        }
        if include_text:
            payload["ast_text"] = self.ast_text
        if include_hir:
            payload["normalized_hir"] = {
                "hir_blocks": [block.to_dict() for block in self.normalized_hir_blocks],
            }
        return payload


_EXPR_CALL_RE = re.compile(r"^(?P<target>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")
_MEMBER_RE = re.compile(r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*)\.(?P<field>[A-Za-z_][A-Za-z0-9_]*)$")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d*|\d*\.\d+)$")


def _split_call_args(payload: str) -> list[str]:
    if not payload.strip():
        return []
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in payload:
        if ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
            continue
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth = max(0, depth - 1)
        current.append(ch)
    tail = ''.join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _lift_expr(text: Optional[str]) -> Optional[dict[str, Any]]:
    if text is None:
        return None
    source = str(text).strip()
    if not source:
        return {"kind": "empty", "text": ""}
    if source in {"break", "continue"}:
        return {"kind": source, "text": source}
    if source.startswith("'") or source.startswith('"'):
        return {"kind": "literal", "literal_type": "string", "value": source, "text": source}
    if _INT_RE.match(source):
        return {"kind": "literal", "literal_type": "int", "value": int(source), "text": source}
    if _FLOAT_RE.match(source):
        return {"kind": "literal", "literal_type": "float", "value": float(source), "text": source}
    match = _MEMBER_RE.match(source)
    if match is not None:
        return {
            "kind": "member",
            "base": match.group("base"),
            "field": match.group("field"),
            "text": source,
        }
    match = _EXPR_CALL_RE.match(source)
    if match is not None:
        return {
            "kind": "call_expr",
            "callee": match.group("target"),
            "args": [_lift_expr(item) for item in _split_call_args(match.group("args"))],
            "text": source,
        }
    if source.startswith("cond[") and source.endswith(")"):
        return {"kind": "predicate", "text": source}
    return {"kind": "symbol", "name": source, "text": source}


def _lift_statement(stmt: str) -> dict[str, Any]:
    text = stmt.strip()
    if not text:
        return {"kind": "empty", "text": stmt}
    if text == "break":
        return {"kind": "break", "text": text}
    if text == "continue":
        return {"kind": "continue", "text": text}
    if text.startswith("return"):
        value = text[len("return"):].strip()
        return {"kind": "return", "value": _lift_expr(value) if value else None, "text": text}
    matched = _match_assignment(text)
    if matched is not None:
        lhs, rhs = matched
        target_expr = _lift_expr(lhs)
        value_expr = _lift_expr(rhs)
        kind = "assign"
        if '.' in lhs:
            kind = "field_store"
        elif value_expr and value_expr.get("kind") == "call_expr":
            kind = "assign_call"
        return {
            "kind": kind,
            "target": target_expr,
            "value": value_expr,
            "text": text,
        }
    expr = _lift_expr(text)
    if expr and expr.get("kind") == "call_expr":
        return {"kind": "call", "expr": expr, "text": text}
    return {"kind": "expr", "expr": expr, "text": text}


def _lift_terminator(terminator: dict[str, Any]) -> dict[str, Any]:
    kind = terminator.get("kind") or "unknown"
    return {
        "kind": kind,
        "text": terminator.get("text"),
        "condition": _lift_expr(terminator.get("condition")),
        "successors": list(terminator.get("successors") or []),
        "branch_target": terminator.get("branch_target"),
        "fallthrough_target": terminator.get("fallthrough_target"),
    }


def _lift_region(region: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if region is None:
        return None
    kind = region.get("kind")
    if kind == "sequence":
        return {"kind": "sequence", "regions": [_lift_region(item) for item in region.get("regions", [])]}
    if kind == "block":
        return {
            "kind": "block",
            "block_id": region.get("block_id"),
            "label": region.get("label"),
            "block_params": list(region.get("block_params") or []),
            "body": [_lift_statement(stmt) for stmt in region.get("statements", [])],
            "terminator": _lift_terminator(region.get("terminator") or {}),
        }
    if kind in {"if", "if_else"}:
        return {
            "kind": kind,
            "header_block": region.get("header_block"),
            "block_params": list(region.get("block_params") or []),
            "condition": _lift_expr(region.get("condition")),
            "negated": bool(region.get("negated")),
            "prologue": [_lift_statement(stmt) for stmt in region.get("prologue", [])],
            "then": _lift_region(region.get("then")),
            "else": _lift_region(region.get("else")),
            "resume_block": region.get("resume_block"),
        }
    if kind == "while":
        return {
            "kind": "while",
            "header_block": region.get("header_block"),
            "block_params": list(region.get("block_params") or []),
            "condition": _lift_expr(region.get("condition")),
            "prologue": [_lift_statement(stmt) for stmt in region.get("prologue", [])],
            "rendering": region.get("rendering"),
            "body": _lift_region(region.get("body")),
            "continue_target": region.get("continue_target"),
            "break_target": region.get("break_target"),
        }
    if kind == "switch_like":
        return {
            "kind": "switch_like",
            "join_block": region.get("join_block"),
            "cases": [
                {
                    "kind": "case",
                    "dispatch_block": case.get("dispatch_block"),
                    "condition": _lift_expr(case.get("condition")),
                    "target": case.get("target"),
                    "body": _lift_region(case.get("body")),
                }
                for case in region.get("cases", [])
            ],
            "default": region.get("default"),
        }
    if kind == "goto_region":
        return {
            "kind": "goto_region",
            "style": region.get("style"),
            "entry_block": region.get("entry_block"),
            "block_ids": list(region.get("block_ids") or []),
            "blocks": [_lift_region(item) for item in region.get("blocks", [])],
            "external_predecessors": list(region.get("external_predecessors") or []),
            "external_successors": list(region.get("external_successors") or []),
            "residual": dict(region.get("residual") or {}),
        }
    return {"kind": kind or "unknown", "raw": region}


@dataclass
class _StructuredContext:
    blocks: list[HIRBlock]
    by_id: dict[str, HIRBlock]
    index_by_id: dict[str, int]
    dom: dict[str, set[str]]
    postdom: dict[str, set[str]]
    ipdom: dict[str, Optional[str]]
    loops: dict[str, dict[str, Any]]
    switches: dict[int, dict[str, Any]]


def _sequence_region(regions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "sequence", "regions": regions}


def _indent(lines: list[str], level: int) -> list[str]:
    prefix = "    " * level
    return [prefix + line if line else "" for line in lines]


def _compute_dominators(hir_blocks: list[HIRBlock]) -> dict[str, set[str]]:
    if not hir_blocks:
        return {}
    ids = [block.id for block in hir_blocks]
    _, _, _, preds = _block_maps(hir_blocks)
    entry = hir_blocks[0].id
    dom: dict[str, set[str]] = {block.id: set(ids) for block in hir_blocks}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for block in hir_blocks[1:]:
            pred_sets = [dom[pred] for pred in preds[block.id] if pred in dom]
            new_dom = {block.id}
            if pred_sets:
                new_dom |= set.intersection(*pred_sets)
            if new_dom != dom[block.id]:
                dom[block.id] = new_dom
                changed = True
    return dom


def _compute_postdominators(hir_blocks: list[HIRBlock]) -> dict[str, set[str]]:
    if not hir_blocks:
        return {}
    ids = [block.id for block in hir_blocks]
    _, _, succs, _ = _block_maps(hir_blocks)
    exits = [block.id for block in hir_blocks if not succs[block.id]]
    postdom: dict[str, set[str]] = {block.id: set(ids) for block in hir_blocks}
    for exit_id in exits:
        postdom[exit_id] = {exit_id}
    changed = True
    while changed:
        changed = False
        for block in hir_blocks:
            if block.id in exits:
                continue
            succ_sets = [postdom[succ] for succ in succs[block.id] if succ in postdom]
            new_postdom = {block.id}
            if succ_sets:
                new_postdom |= set.intersection(*succ_sets)
            if new_postdom != postdom[block.id]:
                postdom[block.id] = new_postdom
                changed = True
    return postdom


def _immediate_postdom(block_id: str, postdom: dict[str, set[str]], index_by_id: dict[str, int]) -> Optional[str]:
    candidates = list(postdom.get(block_id, set()) - {block_id})
    if not candidates:
        return None
    candidates.sort(key=lambda item: index_by_id.get(item, 10 ** 6))
    for candidate in candidates:
        if all(candidate not in postdom.get(other, set()) for other in candidates if other != candidate):
            return candidate
    return candidates[0]


def _edge_targets(block: HIRBlock, index_by_id: dict[str, int]) -> tuple[Optional[str], Optional[str]]:
    if block.terminator.get("kind") != "branch":
        return None, block.fallthrough_target or (block.successors[0] if block.successors else None)
    branch_target = block.branch_target
    fallthrough_target = block.fallthrough_target
    if branch_target is None and block.successors:
        ordered = sorted(block.successors, key=lambda item: index_by_id.get(item, 10 ** 6))
        branch_target = ordered[0]
        fallthrough_target = ordered[1] if len(ordered) > 1 else None
    return branch_target, fallthrough_target


def _natural_loops(hir_blocks: list[HIRBlock], dom: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    _, index_by_id, succs, preds = _block_maps(hir_blocks)
    loops: dict[str, dict[str, Any]] = {}
    for block in hir_blocks:
        for succ in succs[block.id]:
            if succ in dom.get(block.id, set()) and index_by_id.get(succ, 10 ** 6) <= block.index:
                loop = loops.setdefault(succ, {"header": succ, "latches": set(), "nodes": {succ}})
                loop["latches"].add(block.id)
                work = [block.id]
                while work:
                    node = work.pop()
                    if node in loop["nodes"]:
                        continue
                    loop["nodes"].add(node)
                    for pred in preds.get(node, []):
                        if pred not in loop["nodes"]:
                            work.append(pred)
    for header_id, loop in loops.items():
        node_indices = sorted(index_by_id[node] for node in loop["nodes"])
        loop["index_range"] = (node_indices[0], node_indices[-1]) if node_indices else (index_by_id[header_id], index_by_id[header_id])
        body_succs = [succ for succ in succs[header_id] if succ in loop["nodes"]]
        exit_succs = [succ for succ in succs[header_id] if succ not in loop["nodes"]]
        loop["body_succ"] = body_succs[0] if len(body_succs) == 1 else None
        loop["exit_succ"] = exit_succs[0] if len(exit_succs) == 1 else None
        idx0, idx1 = loop["index_range"]
        contiguous_ids = {hir_blocks[idx].id for idx in range(idx0, idx1 + 1)}
        loop["contiguous"] = contiguous_ids == set(loop["nodes"])
        exit_idx = index_by_id.get(loop["exit_succ"], -1) if loop["exit_succ"] is not None else -1
        body_idx = index_by_id.get(loop["body_succ"], -1) if loop["body_succ"] is not None else -1
        loop["safe"] = (
            hir_blocks[index_by_id[header_id]].terminator.get("kind") == "branch"
            and loop["body_succ"] is not None
            and loop["exit_succ"] is not None
            and loop["contiguous"]
            and body_idx >= idx0
            and exit_idx > idx1
        )
    return loops


def _next_allowed_block_id(blocks: list[HIRBlock], start_idx: int, limit: int, allowed_ids: Optional[set[str]]) -> Optional[str]:
    for idx in range(start_idx + 1, min(limit, len(blocks))):
        candidate = blocks[idx]
        if allowed_ids is None or candidate.id in allowed_ids:
            return candidate.id
    return None


def _label_needed(block: HIRBlock, by_id: dict[str, HIRBlock], index_by_id: dict[str, int], allowed_ids: Optional[set[str]]) -> bool:
    if block.index == 0:
        return bool(block.predecessors)
    preds = list(block.predecessors)
    if not preds:
        return True
    if len(preds) != 1:
        return True
    pred_id = preds[0]
    if allowed_ids is not None and pred_id not in allowed_ids:
        allowed_indices = [index_by_id[item] for item in allowed_ids if item in index_by_id]
        if allowed_indices and block.index == min(allowed_indices):
            return False
        return True
    pred = by_id.get(pred_id)
    if pred is None:
        return True
    return index_by_id.get(pred_id, -1) != block.index - 1


def _format_block_label(block: HIRBlock) -> str:
    if block.block_params:
        return f"{block.id}({', '.join(block.block_params)})"
    return block.id


def _format_jump_target(source: Optional[HIRBlock], target: Optional[str], by_id: dict[str, HIRBlock], jump_aliases: Optional[dict[str, str]]) -> Optional[str]:
    if target is None:
        return None
    if jump_aliases and target in jump_aliases:
        return jump_aliases[target]
    target_block = by_id.get(target)
    if target_block is None:
        return target
    if not target_block.block_params:
        return target
    if source is None:
        return _format_block_label(target_block)
    args = target_block.incoming_args.get(source.id)
    if not args:
        return _format_block_label(target_block)
    return f"{target}({', '.join(args)})"


def _jump_stmt(source: Optional[HIRBlock], target: Optional[str], by_id: dict[str, HIRBlock], jump_aliases: Optional[dict[str, str]]) -> tuple[Optional[str], int]:
    formatted = _format_jump_target(source, target, by_id, jump_aliases)
    if formatted is None:
        return None, 0
    if jump_aliases and target in jump_aliases:
        return formatted, 0
    return f"goto {formatted}", 1


def _conditional_jump(cond: str, source: Optional[HIRBlock], target: Optional[str], *, negate: bool, by_id: dict[str, HIRBlock], jump_aliases: Optional[dict[str, str]]) -> tuple[list[str], int]:
    stmt, cost = _jump_stmt(source, target, by_id, jump_aliases)
    if stmt is None:
        return [], 0
    if negate:
        return [f"if (!({cond})) {stmt}"], cost
    return [f"if ({cond}) {stmt}"], cost


def _generic_terminator_lines(
    block: HIRBlock,
    next_id: Optional[str],
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
    jump_aliases: Optional[dict[str, str]],
) -> tuple[list[str], int]:
    kind = block.terminator.get("kind")
    if kind == "return":
        return [], 0
    if kind == "branch":
        branch_target, fallthrough_target = _edge_targets(block, index_by_id)
        cond = block.terminator.get("condition") or f"cond_{block.id}"
        if branch_target and fallthrough_target:
            if fallthrough_target == next_id and branch_target != next_id:
                return _conditional_jump(cond, block, branch_target, negate=False, by_id=by_id, jump_aliases=jump_aliases)
            if branch_target == next_id and fallthrough_target != next_id:
                return _conditional_jump(cond, block, fallthrough_target, negate=True, by_id=by_id, jump_aliases=jump_aliases)
            if branch_target == fallthrough_target:
                stmt, cost = _jump_stmt(block, branch_target, by_id, jump_aliases)
                return ([stmt] if stmt else []), cost
            first_lines, first_cost = _conditional_jump(cond, block, branch_target, negate=False, by_id=by_id, jump_aliases=jump_aliases)
            second_stmt, second_cost = _jump_stmt(block, fallthrough_target, by_id, jump_aliases)
            lines = list(first_lines)
            if second_stmt is not None:
                lines.append(second_stmt)
            return lines, first_cost + second_cost
        if branch_target:
            return _conditional_jump(cond, block, branch_target, negate=False, by_id=by_id, jump_aliases=jump_aliases)
        if fallthrough_target and fallthrough_target != next_id:
            return _conditional_jump(cond, block, fallthrough_target, negate=True, by_id=by_id, jump_aliases=jump_aliases)
        return [f"if ({cond}) {{ /* unresolved edge */ }}"], 0
    successor = block.successors[0] if block.successors else None
    if successor and successor != next_id:
        stmt, cost = _jump_stmt(block, successor, by_id, jump_aliases)
        return ([stmt] if stmt else []), cost
    return [], 0


def _generic_block_lines(
    block: HIRBlock,
    *,
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
    next_id: Optional[str],
    allowed_ids: Optional[set[str]],
    jump_aliases: Optional[dict[str, str]],
) -> tuple[list[str], int, int]:
    lines: list[str] = []
    label_count = 0
    goto_count = 0
    stmt_indent = ""
    if _label_needed(block, by_id, index_by_id, allowed_ids):
        lines.append(f"{_format_block_label(block)}:")
        stmt_indent = "    "
        label_count = 1
    for stmt in block.statements:
        lines.append(f"{stmt_indent}{stmt}")
    term_lines, term_gotos = _generic_terminator_lines(block, next_id, by_id, index_by_id, jump_aliases)
    goto_count += term_gotos
    for line in term_lines:
        lines.append(f"{stmt_indent}{line}")
    return lines, label_count, goto_count


def _collect_linear_chain(
    start_id: Optional[str],
    *,
    join_id: Optional[str],
    entry_id: str,
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
    limit: int,
    allowed_ids: Optional[set[str]],
    blocked_ids: set[str],
) -> Optional[dict[str, Any]]:
    if start_id is None:
        return None
    if join_id is not None and start_id == join_id:
        return {"ids": [], "end": "join"}

    chain: list[str] = []
    local_seen = set(blocked_ids)
    prev_id = entry_id
    current_id = start_id
    join_index = index_by_id.get(join_id, 10 ** 6) if join_id is not None else 10 ** 6

    while True:
        if current_id in local_seen:
            return None
        block = by_id.get(current_id)
        if block is None:
            return None
        if allowed_ids is not None and current_id not in allowed_ids:
            return None
        current_index = index_by_id.get(current_id, 10 ** 6)
        if current_index >= limit:
            return None
        if join_id is not None and current_index >= join_index:
            return None
        if chain and current_index <= index_by_id.get(chain[-1], -1):
            return None
        if set(block.predecessors) != {prev_id}:
            return None
        if block.terminator.get("kind") == "branch":
            return None

        chain.append(current_id)
        local_seen.add(current_id)

        kind = block.terminator.get("kind")
        if kind == "return":
            return {"ids": chain, "end": "return"}
        if not block.successors:
            return {"ids": chain, "end": "exit"}
        if len(block.successors) != 1:
            return None
        successor = block.successors[0]
        if join_id is not None and successor == join_id:
            return {"ids": chain, "end": "join"}
        prev_id = current_id
        current_id = successor


def _render_linear_chain(chain_info: dict[str, Any], by_id: dict[str, HIRBlock], indent: int) -> list[str]:
    lines: list[str] = []
    for block_id in chain_info.get("ids", []):
        block = by_id[block_id]
        stmt_indent = indent
        if block.block_params:
            lines.extend(_indent([f"{_format_block_label(block)}:"], indent))
            stmt_indent = indent + 1
        lines.extend(_indent(block.statements, stmt_indent))
        if block.terminator.get("kind") == "return":
            lines.extend(_indent([block.terminator.get("text") or "return"], stmt_indent))
    return lines


def _collect_switch_like(start_index: int, hir_blocks: list[HIRBlock], index_by_id: dict[str, int], ipdom: dict[str, Optional[str]]) -> Optional[dict[str, Any]]:
    chain: list[HIRBlock] = []
    join: Optional[str] = None
    i = start_index
    while i < len(hir_blocks):
        block = hir_blocks[i]
        if block.terminator.get("kind") != "branch" or block.statements:
            break
        block_join = ipdom.get(block.id)
        branch_target, fallthrough_target = _edge_targets(block, index_by_id)
        ordered = [target for target in [branch_target, fallthrough_target] if target is not None]
        if len(ordered) != 2 or fallthrough_target != (hir_blocks[i + 1].id if i + 1 < len(hir_blocks) else None):
            break
        if join is None:
            join = block_join
        elif join != block_join:
            break
        chain.append(block)
        i += 1
        if join is not None and ordered[0] == join:
            break
    if len(chain) < 3 or join is None:
        return None
    return {"blocks": chain, "join": join, "end_index": i}


def _contiguous_span(ids: list[str], index_by_id: dict[str, int]) -> Optional[tuple[int, int]]:
    if not ids:
        return None
    indices = sorted(index_by_id[block_id] for block_id in ids)
    if indices != list(range(indices[0], indices[-1] + 1)):
        return None
    return indices[0], indices[-1] + 1


def _dominance_arm(
    before_join: list[str],
    primary_succ: Optional[str],
    other_succ: Optional[str],
    dom: dict[str, set[str]],
    index_by_id: dict[str, int],
) -> Optional[dict[str, Any]]:
    if primary_succ is None:
        return None
    ids = [
        block_id
        for block_id in before_join
        if primary_succ in dom.get(block_id, set()) and not (other_succ and other_succ in dom.get(block_id, set()))
    ]
    span = _contiguous_span(ids, index_by_id)
    if span is None:
        return None
    start, stop = span
    return {"mode": "range", "ids": set(ids), "start": start, "stop": stop}


def _linear_arm(
    start_id: Optional[str],
    *,
    join_id: Optional[str],
    entry_id: str,
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
    limit: int,
    allowed_ids: Optional[set[str]],
    blocked_ids: set[str],
    allowed_endings: set[str],
) -> Optional[dict[str, Any]]:
    chain = _collect_linear_chain(
        start_id,
        join_id=join_id,
        entry_id=entry_id,
        by_id=by_id,
        index_by_id=index_by_id,
        limit=limit,
        allowed_ids=allowed_ids,
        blocked_ids=blocked_ids,
    )
    if chain is None or chain.get("end") not in allowed_endings:
        return None
    return {"mode": "linear", "ids": set(chain.get("ids", [])), "chain": chain, "end": chain.get("end")}


def _render_arm(
    arm: dict[str, Any],
    *,
    render_range: Any,
    by_id: dict[str, HIRBlock],
    indent: int,
    jump_aliases: Optional[dict[str, str]],
) -> list[str]:
    if arm.get("mode") == "linear":
        return _render_linear_chain(arm["chain"], by_id, indent)
    return render_range(arm["start"], arm["stop"], indent, set(arm["ids"]), jump_aliases)


def _match_branch_region(
    *,
    block: HIRBlock,
    blocks: list[HIRBlock],
    limit: int,
    allowed_ids: Optional[set[str]],
    visited: set[str],
    loops: dict[str, dict[str, Any]],
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
    dom: dict[str, set[str]],
    ipdom: dict[str, Optional[str]],
) -> Optional[dict[str, Any]]:
    if block.terminator.get("kind") != "branch" or block.id in loops:
        return None

    succ_a, succ_b = _edge_targets(block, index_by_id)
    join_id = ipdom.get(block.id)
    join_idx = index_by_id.get(join_id, limit) if join_id is not None else limit

    def _prefer_linear(primary: Optional[dict[str, Any]], linear: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if primary is None:
            return None
        if linear is not None and set(linear.get("ids", [])) == set(primary.get("ids", [])):
            return linear
        return primary

    if join_id is not None and join_id in index_by_id and join_idx < limit:
        before_join = [candidate.id for candidate in blocks[block.index + 1:join_idx] if allowed_ids is None or candidate.id in allowed_ids]
        dom_arm_a = _dominance_arm(before_join, succ_a, succ_b, dom, index_by_id)
        dom_arm_b = _dominance_arm(before_join, succ_b, succ_a, dom, index_by_id)
        linear_arm_a = _linear_arm(
            succ_a,
            join_id=join_id,
            entry_id=block.id,
            by_id=by_id,
            index_by_id=index_by_id,
            limit=limit,
            allowed_ids=allowed_ids,
            blocked_ids=visited | {block.id},
            allowed_endings={"join"},
        )
        linear_arm_b = _linear_arm(
            succ_b,
            join_id=join_id,
            entry_id=block.id,
            by_id=by_id,
            index_by_id=index_by_id,
            limit=limit,
            allowed_ids=allowed_ids,
            blocked_ids=visited | {block.id},
            allowed_endings={"join"},
        )

        if succ_b == join_id and dom_arm_a is not None:
            then_arm = _prefer_linear(dom_arm_a, linear_arm_a)
            return {"kind": "if", "negate": False, "then_arm": then_arm, "else_arm": None, "resume_idx": join_idx, "consumed": set(then_arm["ids"])}
        if succ_a == join_id and dom_arm_b is not None:
            then_arm = _prefer_linear(dom_arm_b, linear_arm_b)
            return {"kind": "if", "negate": True, "then_arm": then_arm, "else_arm": None, "resume_idx": join_idx, "consumed": set(then_arm["ids"])}
        if dom_arm_a is not None and dom_arm_b is not None and dom_arm_a["ids"].isdisjoint(dom_arm_b["ids"]):
            then_arm = _prefer_linear(dom_arm_a, linear_arm_a)
            else_arm = _prefer_linear(dom_arm_b, linear_arm_b)
            return {
                "kind": "if_else",
                "negate": False,
                "then_arm": then_arm,
                "else_arm": else_arm,
                "resume_idx": join_idx,
                "consumed": set(then_arm["ids"]) | set(else_arm["ids"]),
            }

        if linear_arm_a is not None and linear_arm_b is not None:
            ids_a = set(linear_arm_a["ids"])
            ids_b = set(linear_arm_b["ids"])
            if not ids_a.isdisjoint(ids_b):
                return None
            if ids_a and not ids_b:
                return {"kind": "if", "negate": False, "then_arm": linear_arm_a, "else_arm": None, "resume_idx": join_idx, "consumed": ids_a}
            if ids_b and not ids_a:
                return {"kind": "if", "negate": True, "then_arm": linear_arm_b, "else_arm": None, "resume_idx": join_idx, "consumed": ids_b}
            if ids_a and ids_b:
                return {
                    "kind": "if_else",
                    "negate": False,
                    "then_arm": linear_arm_a,
                    "else_arm": linear_arm_b,
                    "resume_idx": join_idx,
                    "consumed": ids_a | ids_b,
                }

    next_id = _next_allowed_block_id(blocks, block.index, limit, allowed_ids)
    next_idx = index_by_id.get(next_id, limit) if next_id is not None else limit
    return_arm_a = _linear_arm(
        succ_a,
        join_id=None,
        entry_id=block.id,
        by_id=by_id,
        index_by_id=index_by_id,
        limit=limit,
        allowed_ids=allowed_ids,
        blocked_ids=visited | {block.id},
        allowed_endings={"return", "exit"},
    )
    return_arm_b = _linear_arm(
        succ_b,
        join_id=None,
        entry_id=block.id,
        by_id=by_id,
        index_by_id=index_by_id,
        limit=limit,
        allowed_ids=allowed_ids,
        blocked_ids=visited | {block.id},
        allowed_endings={"return", "exit"},
    )
    if succ_b == next_id and return_arm_a is not None and return_arm_a["ids"]:
        return {"kind": "if", "negate": False, "then_arm": return_arm_a, "else_arm": None, "resume_idx": next_idx, "consumed": set(return_arm_a["ids"])}
    if succ_a == next_id and return_arm_b is not None and return_arm_b["ids"]:
        return {"kind": "if", "negate": True, "then_arm": return_arm_b, "else_arm": None, "resume_idx": next_idx, "consumed": set(return_arm_b["ids"])}
    return None


def _render_branch_region(
    block: HIRBlock,
    region: dict[str, Any],
    *,
    render_range: Any,
    by_id: dict[str, HIRBlock],
    constructs: Counter[str],
    visited: set[str],
    indent: int,
    jump_aliases: Optional[dict[str, str]],
) -> list[str]:
    constructs[region["kind"]] += 1
    visited.add(block.id)
    cond = block.terminator.get("condition") or f"cond_{block.id}"
    if region.get("negate"):
        cond = f"!({cond})"

    lines: list[str] = []
    stmt_indent = indent
    if block.block_params:
        lines.extend(_indent([f"{_format_block_label(block)}:"], indent))
        stmt_indent = indent + 1
    lines.extend(_indent(block.statements, stmt_indent))
    lines.extend(_indent([f"if ({cond}) {{"], stmt_indent))
    lines.extend(_render_arm(region["then_arm"], render_range=render_range, by_id=by_id, indent=stmt_indent + 1, jump_aliases=jump_aliases))
    if region["kind"] == "if_else" and region.get("else_arm") is not None:
        lines.extend(_indent(["} else {"], stmt_indent))
        lines.extend(_render_arm(region["else_arm"], render_range=render_range, by_id=by_id, indent=stmt_indent + 1, jump_aliases=jump_aliases))
    lines.extend(_indent(["}"], stmt_indent))
    visited.update(region["consumed"])
    return lines


def _build_structured_context(blocks: list[HIRBlock]) -> _StructuredContext:
    by_id, index_by_id, _, _ = _block_maps(blocks)
    dom = _compute_dominators(blocks)
    postdom = _compute_postdominators(blocks)
    ipdom = {block.id: _immediate_postdom(block.id, postdom, index_by_id) for block in blocks}
    loops = _natural_loops(blocks, dom)
    switches = {
        idx: info
        for idx in range(len(blocks))
        if (info := _collect_switch_like(idx, blocks, index_by_id, ipdom)) is not None
    }
    return _StructuredContext(
        blocks=blocks,
        by_id=by_id,
        index_by_id=index_by_id,
        dom=dom,
        postdom=postdom,
        ipdom=ipdom,
        loops=loops,
        switches=switches,
    )


def _generic_block_region(
    block: HIRBlock,
    *,
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
    next_id: Optional[str],
    allowed_ids: Optional[set[str]],
    jump_aliases: Optional[dict[str, str]],
) -> tuple[dict[str, Any], list[str], int, int]:
    block_lines, label_count, goto_count = _generic_block_lines(
        block,
        by_id=by_id,
        index_by_id=index_by_id,
        next_id=next_id,
        allowed_ids=allowed_ids,
        jump_aliases=jump_aliases,
    )
    label = _format_block_label(block) if _label_needed(block, by_id, index_by_id, allowed_ids) else None
    term_lines, _ = _generic_terminator_lines(block, next_id, by_id, index_by_id, jump_aliases)
    region = {
        "kind": "block",
        "block_id": block.id,
        "label": label,
        "block_params": list(block.block_params),
        "statements": list(block.statements),
        "terminator": {
            "kind": block.terminator.get("kind"),
            "text": block.terminator.get("text"),
            "condition": block.terminator.get("condition"),
            "rendered_lines": term_lines,
            "successors": list(block.successors),
            "branch_target": block.branch_target,
            "fallthrough_target": block.fallthrough_target,
        },
    }
    return region, block_lines, label_count, goto_count


def _generic_block_counts(
    block: HIRBlock,
    *,
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
    next_id: Optional[str],
    allowed_ids: Optional[set[str]],
    jump_aliases: Optional[dict[str, str]],
) -> tuple[int, int]:
    label_count = 1 if _label_needed(block, by_id, index_by_id, allowed_ids) else 0
    _, goto_count = _generic_terminator_lines(block, next_id, by_id, index_by_id, jump_aliases)
    return label_count, goto_count


def _linear_chain_region(chain_info: dict[str, Any], by_id: dict[str, HIRBlock]) -> dict[str, Any]:
    regions: list[dict[str, Any]] = []
    for block_id in chain_info.get("ids", []):
        block = by_id[block_id]
        rendered_lines = [block.terminator.get("text") or "return"] if block.terminator.get("kind") == "return" else []
        regions.append(
            {
                "kind": "block",
                "block_id": block.id,
                "label": _format_block_label(block) if block.block_params else None,
                "block_params": list(block.block_params),
                "statements": list(block.statements),
                "terminator": {
                    "kind": block.terminator.get("kind"),
                    "text": block.terminator.get("text"),
                    "condition": block.terminator.get("condition"),
                    "rendered_lines": rendered_lines,
                    "successors": list(block.successors),
                    "branch_target": block.branch_target,
                    "fallthrough_target": block.fallthrough_target,
                },
            }
        )
    return {"kind": "sequence", "regions": regions, "end": chain_info.get("end")}


def _requires_goto_region(block: HIRBlock, label_count: int, goto_count: int) -> bool:
    if label_count or goto_count:
        return True
    if block.block_params:
        return True
    if block.terminator.get("kind") == "branch":
        return True
    return len(block.successors) > 1


def _build_goto_region(
    block_regions: list[dict[str, Any]],
    *,
    block_ids: list[str],
    by_id: dict[str, HIRBlock],
    label_count: int,
    goto_count: int,
) -> dict[str, Any]:
    member_ids = set(block_ids)
    external_successors = _ordered_unique(
        succ
        for block_id in block_ids
        for succ in by_id[block_id].successors
        if succ not in member_ids
    )
    external_predecessors = _ordered_unique(
        pred
        for block_id in block_ids
        for pred in by_id[block_id].predecessors
        if pred not in member_ids
    )
    return {
        "kind": "goto_region",
        "style": "explicit_cfg",
        "entry_block": block_ids[0] if block_ids else None,
        "block_ids": list(block_ids),
        "blocks": list(block_regions),
        "external_predecessors": external_predecessors,
        "external_successors": external_successors,
        "residual": {
            "label_count": label_count,
            "goto_count": goto_count,
        },
    }


def _analyze_structured_surface(
    blocks: list[HIRBlock],
    *,
    emit_text: bool,
    emit_tree: bool,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if not blocks:
        tree = _sequence_region([]) if emit_tree else {}
        return tree, "", {"constructs": {}, "fallback_block_count": 0, "residual_label_count": 0, "residual_goto_count": 0, "loop_header_count": 0}

    ctx = _build_structured_context(blocks)
    constructs: Counter[str] = Counter()
    visited: set[str] = set()
    residual_labels = 0
    residual_gotos = 0
    fallback_regions_total = 0
    fallback_blocks_total = 0

    def build_arm(
        arm: dict[str, Any],
        *,
        indent: int,
        jump_aliases: Optional[dict[str, str]],
    ) -> tuple[Optional[dict[str, Any]], list[str]]:
        if arm.get("mode") == "linear":
            region = _linear_chain_region(arm["chain"], ctx.by_id) if emit_tree else None
            lines = _render_linear_chain(arm["chain"], ctx.by_id, indent) if emit_text else []
            return region, lines
        arm_regions, arm_lines = walk_range(arm["start"], arm["stop"], indent, set(arm["ids"]), jump_aliases)
        return (_sequence_region(arm_regions) if emit_tree else None), arm_lines

    def walk_range(
        start_idx: int,
        stop_idx: Optional[int],
        indent: int,
        allowed_ids: Optional[set[str]] = None,
        jump_aliases: Optional[dict[str, str]] = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        nonlocal residual_labels, residual_gotos, fallback_regions_total, fallback_blocks_total
        regions: list[dict[str, Any]] = []
        lines: list[str] = []
        limit = stop_idx if stop_idx is not None else len(blocks)
        i = start_idx

        pending_block_regions: list[dict[str, Any]] = []
        pending_block_lines: list[str] = []
        pending_block_ids: list[str] = []
        pending_label_count = 0
        pending_goto_count = 0

        def flush_pending_goto_region() -> None:
            nonlocal pending_label_count, pending_goto_count, fallback_regions_total, fallback_blocks_total
            if not pending_block_ids:
                return
            fallback_regions_total += 1
            fallback_blocks_total += len(pending_block_ids)
            if emit_tree:
                regions.append(
                    _build_goto_region(
                        pending_block_regions,
                        block_ids=pending_block_ids,
                        by_id=ctx.by_id,
                        label_count=pending_label_count,
                        goto_count=pending_goto_count,
                    )
                )
            if emit_text:
                lines.extend(_indent(pending_block_lines, indent))
            pending_block_regions.clear()
            pending_block_lines.clear()
            pending_block_ids.clear()
            pending_label_count = 0
            pending_goto_count = 0

        while i < limit and i < len(blocks):
            block = blocks[i]
            if allowed_ids is not None and block.id not in allowed_ids:
                flush_pending_goto_region()
                i += 1
                continue
            if block.id in visited:
                flush_pending_goto_region()
                i += 1
                continue

            loop = ctx.loops.get(block.id)
            if loop and loop.get("safe") and loop["index_range"][1] < limit and (allowed_ids is None or set(loop["nodes"]).issubset(allowed_ids | {block.id})):
                flush_pending_goto_region()
                constructs["while"] += 1
                visited.add(block.id)
                header_lines = list(block.statements)
                cond = block.terminator.get("condition") or f"cond_{block.id}"
                body_ids = set(loop["nodes"]) - {block.id}
                body_start = ctx.index_by_id.get(loop.get("body_succ"), i + 1)
                body_end = loop["index_range"][1] + 1
                loop_aliases = dict(jump_aliases or {})
                loop_aliases[loop["header"]] = "continue"
                loop_aliases[loop["exit_succ"]] = "break"
                body_regions, body_lines = walk_range(body_start, body_end, indent + 1, body_ids, loop_aliases)
                if emit_tree:
                    regions.append(
                        {
                            "kind": "while",
                            "header_block": block.id,
                            "block_params": list(block.block_params),
                            "condition": cond,
                            "prologue": header_lines,
                            "rendering": "guarded_loop" if header_lines else "direct_while",
                            "body": _sequence_region(body_regions),
                            "continue_target": loop["header"],
                            "break_target": loop["exit_succ"],
                        }
                    )
                if emit_text:
                    if header_lines:
                        lines.extend(_indent(["while (true) {"], indent))
                        lines.extend(_indent(header_lines, indent + 1))
                        lines.extend(_indent([f"if (!({cond})) break"], indent + 1))
                    else:
                        lines.extend(_indent([f"while ({cond}) {{"], indent))
                    lines.extend(body_lines)
                    lines.extend(_indent(["}"], indent))
                visited.update(body_ids)
                i = loop["index_range"][1] + 1
                continue

            switch_info = ctx.switches.get(i)
            if switch_info and switch_info["end_index"] <= limit:
                flush_pending_goto_region()
                constructs["switch_like"] += 1
                case_regions: list[dict[str, Any]] = []
                if emit_text:
                    lines.extend(_indent(["switch_like {"], indent))
                for case_block in switch_info["blocks"]:
                    visited.add(case_block.id)
                    case_target, _ = _edge_targets(case_block, ctx.index_by_id)
                    inline_regions: list[dict[str, Any]] = []
                    inline_lines: list[str] = []
                    if case_target in ctx.by_id and case_target not in visited:
                        target_block = ctx.by_id[case_target]
                        if target_block.terminator.get("kind") == "return":
                            inline_regions.append(
                                {
                                    "kind": "block",
                                    "block_id": target_block.id,
                                    "label": None,
                                    "block_params": list(target_block.block_params),
                                    "statements": list(target_block.statements),
                                    "terminator": {
                                        "kind": target_block.terminator.get("kind"),
                                        "text": target_block.terminator.get("text"),
                                        "condition": target_block.terminator.get("condition"),
                                        "rendered_lines": [target_block.terminator.get("text") or "return"],
                                        "successors": list(target_block.successors),
                                        "branch_target": target_block.branch_target,
                                        "fallthrough_target": target_block.fallthrough_target,
                                    },
                                }
                            )
                            inline_lines = list(target_block.statements) + [target_block.terminator.get("text") or "return"]
                            visited.add(case_target)
                    if emit_tree:
                        case_regions.append(
                            {
                                "kind": "case",
                                "dispatch_block": case_block.id,
                                "condition": case_block.terminator.get("condition"),
                                "target": case_target,
                                "body": _sequence_region(inline_regions),
                            }
                        )
                    if emit_text:
                        lines.extend(_indent([f"when ({case_block.terminator.get('condition')}) -> {case_target or '<unresolved>'} {{"], indent + 1))
                        lines.extend(_indent(inline_lines, indent + 2))
                        lines.extend(_indent(["}"], indent + 1))
                if emit_tree:
                    regions.append(
                        {
                            "kind": "switch_like",
                            "join_block": switch_info.get("join"),
                            "cases": case_regions,
                            "default": "dispatch_fallthrough",
                        }
                    )
                if emit_text:
                    lines.extend(_indent(["default: { /* dispatch fallthrough */ }", "}"], indent))
                i = switch_info["end_index"]
                continue

            next_id = _next_allowed_block_id(blocks, i, limit, allowed_ids)
            region = _match_branch_region(
                block=block,
                blocks=blocks,
                limit=limit,
                allowed_ids=allowed_ids,
                visited=visited,
                loops=ctx.loops,
                by_id=ctx.by_id,
                index_by_id=ctx.index_by_id,
                dom=ctx.dom,
                ipdom=ctx.ipdom,
            )
            if region is not None:
                flush_pending_goto_region()
                constructs[region["kind"]] += 1
                visited.add(block.id)
                cond = block.terminator.get("condition") or f"cond_{block.id}"
                then_region, then_lines = build_arm(region["then_arm"], indent=indent + 1, jump_aliases=jump_aliases)
                else_region: Optional[dict[str, Any]] = None
                else_lines: list[str] = []
                if region["kind"] == "if_else" and region.get("else_arm") is not None:
                    else_region, else_lines = build_arm(region["else_arm"], indent=indent + 1, jump_aliases=jump_aliases)
                if emit_tree:
                    regions.append(
                        {
                            "kind": region["kind"],
                            "header_block": block.id,
                            "block_params": list(block.block_params),
                            "condition": cond,
                            "negated": bool(region.get("negate")),
                            "prologue": list(block.statements),
                            "then": then_region or _sequence_region([]),
                            "else": else_region,
                            "resume_block": blocks[region["resume_idx"]].id if region.get("resume_idx", len(blocks)) < len(blocks) else None,
                        }
                    )
                if emit_text:
                    rendered_cond = f"!({cond})" if region.get("negate") else cond
                    stmt_indent = indent
                    if block.block_params:
                        lines.extend(_indent([f"{_format_block_label(block)}:"], indent))
                        stmt_indent = indent + 1
                    lines.extend(_indent(block.statements, stmt_indent))
                    lines.extend(_indent([f"if ({rendered_cond}) {{"], stmt_indent))
                    lines.extend(then_lines)
                    if region["kind"] == "if_else" and region.get("else_arm") is not None:
                        lines.extend(_indent(["} else {"], stmt_indent))
                        lines.extend(else_lines)
                    lines.extend(_indent(["}"], stmt_indent))
                visited.update(region["consumed"])
                i = region["resume_idx"]
                continue

            visited.add(block.id)
            block_region: Optional[dict[str, Any]] = None
            block_lines: list[str] = []
            if emit_tree or emit_text:
                block_region, block_lines, label_count, goto_count = _generic_block_region(
                    block,
                    by_id=ctx.by_id,
                    index_by_id=ctx.index_by_id,
                    next_id=next_id,
                    allowed_ids=allowed_ids,
                    jump_aliases=jump_aliases,
                )
            else:
                label_count, goto_count = _generic_block_counts(
                    block,
                    by_id=ctx.by_id,
                    index_by_id=ctx.index_by_id,
                    next_id=next_id,
                    allowed_ids=allowed_ids,
                    jump_aliases=jump_aliases,
                )
            residual_labels += label_count
            residual_gotos += goto_count

            if pending_block_ids or _requires_goto_region(block, label_count, goto_count):
                pending_block_ids.append(block.id)
                pending_label_count += label_count
                pending_goto_count += goto_count
                if emit_tree and block_region is not None:
                    pending_block_regions.append(block_region)
                if emit_text:
                    pending_block_lines.extend(block_lines)
            else:
                if emit_tree and block_region is not None:
                    regions.append(block_region)
                if emit_text:
                    lines.extend(_indent(block_lines, indent))
            i += 1

        flush_pending_goto_region()
        return regions, lines

    body_regions, body_lines = walk_range(0, None, 1)
    tree = _sequence_region(body_regions) if emit_tree else {}
    text = "\n".join(body_lines) if emit_text else ""
    meta = {
        "constructs": dict(constructs),
        "fallback_region_count": fallback_regions_total,
        "fallback_block_count": fallback_blocks_total,
        "residual_label_count": residual_labels,
        "residual_goto_count": residual_gotos,
        "loop_header_count": sum(1 for loop in ctx.loops.values() if loop.get("safe")),
    }
    return tree, text, meta


def _coerce_function_hir(function_payload: dict[str, Any]) -> tuple[list[HIRBlock], str, dict[str, Any], dict[str, Any], str, dict[str, Any], str]:
    normalized = function_payload.get("normalized_hir", {})
    raw_blocks = normalized.get("hir_blocks") or function_payload.get("core_hir", {}).get("hir_blocks") or []
    blocks = [_coerce_hir_block(raw) for raw in raw_blocks]
    blocks = _rename_merge_params(_normalize_hir_blocks(blocks)) if blocks else []
    name = str(function_payload.get("name") or "<function>")
    span = dict(function_payload.get("span") or {})
    body_selection = dict(function_payload.get("body_selection") or {})
    slice_mode = str(function_payload.get("slice_mode") or "")
    hir_summary = dict(function_payload.get("summary") or {})
    input_contract = str(function_payload.get("input_contract") or function_payload.get("contract_version") or function_payload.get("contract", {}).get("version") or "")
    return blocks, name, span, body_selection, slice_mode, hir_summary, input_contract


def build_function_ast_from_payload(function_payload: dict[str, Any], *, include_hir: bool = False, include_text: bool = True) -> ASTFunction:
    t0 = time.perf_counter()
    blocks, name, span, body_selection, slice_mode, hir_summary, input_contract = _coerce_function_hir(function_payload)
    region_tree, ast_text_body, structured_meta = _analyze_structured_surface(blocks, emit_text=include_text, emit_tree=True)
    lifted_tree = _lift_region(region_tree)
    entry_args = blocks[0].entry_stack if blocks else []
    header = f"function {name}({', '.join(entry_args)}) {{" if entry_args else f"function {name}() {{"
    ast_text = "\n".join([header, ast_text_body, "}"]).rstrip() if include_text else ""
    summary = {
        "normalized_basic_block_count": len(blocks),
        "region_kind_histogram": structured_meta.get("constructs", {}),
        "fallback_region_count": structured_meta.get("fallback_region_count", 0),
        "fallback_block_count": structured_meta.get("fallback_block_count", 0),
        "residual_label_count": structured_meta.get("residual_label_count", 0),
        "residual_goto_count": structured_meta.get("residual_goto_count", 0),
        "loop_header_count": structured_meta.get("loop_header_count", 0),
    }
    report = {
        "structuring": structured_meta,
        "timings_ms": {
            "total": round((time.perf_counter() - t0) * 1000.0, 3),
        },
    }
    return ASTFunction(
        name=name,
        span=span,
        slice_mode=slice_mode,
        summary=summary,
        ast=lifted_tree or {"kind": "sequence", "regions": []},
        ast_text=ast_text,
        report=report,
        body_selection=body_selection,
        hir_summary=hir_summary,
        input_contract=input_contract,
        normalized_hir_blocks=blocks,
    )


def build_function_ast(mod: MBCModule, export_name: str, *, include_canonical: bool = False, include_hir: bool = False, include_text: bool = True) -> ASTFunction:
    hir = build_function_hir(mod, export_name, include_canonical=include_canonical, include_text=False)
    return build_function_ast_from_payload(hir.to_dict(include_canonical=include_canonical, include_text=False), include_hir=include_hir, include_text=include_text)


def build_module_ast(path: str | Path, *, include_canonical: bool = False, include_hir: bool = False, include_text: bool = True) -> dict[str, Any]:
    hir_payload = build_module_hir(path, include_canonical=include_canonical, include_text=True)
    functions = [
        build_function_ast_from_payload(fn, include_hir=include_hir, include_text=include_text)
        for fn in hir_payload.get("functions", [])
    ]
    functions_payload = [fn.to_dict(include_hir=include_hir, include_text=include_text) for fn in functions]
    region_hist = Counter()
    for fn in functions:
        region_hist.update(fn.summary.get("region_kind_histogram", {}))
    return {
        "contract": {
            "version": AST_CONTRACT_VERSION,
            "input_hir_contract": hir_payload.get("contract", {}).get("version"),
            "layers": ["normalized_hir", "structured_regions", "lifted_ast", "report"],
            "notes": [
                "AST consumes normalized CFG-like HIR blocks and performs final structuring above HIR",
                "structuring preserves explicit goto_region fallbacks where unsafe to force a source-shaped tree",
                "semantic lifting is conservative and keeps expression text when exact VM semantics remain unknown",
            ],
        },
        "path": hir_payload.get("path"),
        "script_name": hir_payload.get("script_name"),
        "summary": {
            "export_count": len(functions_payload),
            "total_normalized_basic_blocks": sum(fn.summary.get("normalized_basic_block_count", 0) for fn in functions),
            "total_fallback_regions": sum(fn.summary.get("fallback_region_count", 0) for fn in functions),
            "total_fallback_blocks": sum(fn.summary.get("fallback_block_count", 0) for fn in functions),
            "total_residual_labels": sum(fn.summary.get("residual_label_count", 0) for fn in functions),
            "total_residual_gotos": sum(fn.summary.get("residual_goto_count", 0) for fn in functions),
            "region_kind_histogram": dict(region_hist),
        },
        "functions": functions_payload,
    }


def render_function_ast_text_from_payload(function_payload: dict[str, Any]) -> str:
    ast_text = function_payload.get("ast_text") or ""
    return str(ast_text).rstrip()


def render_module_ast_text_from_payload(module_payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"// {module_payload['script_name']}")
    lines.append(f"// contract: {module_payload['contract']['version']}")
    lines.append("")
    for fn in module_payload.get("functions", []):
        ast_text = render_function_ast_text_from_payload(fn)
        if ast_text:
            lines.append(ast_text)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
