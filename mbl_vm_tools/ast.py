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
from mbl_vm_tools.parser import FunctionEntry, MBCModule


AST_CONTRACT_VERSION = "ast-v2"


@dataclass
class ASTFunction:
    name: str
    span: dict[str, int]
    slice_mode: str
    summary: dict[str, Any]
    ast: dict[str, Any]
    ast_text: str
    diagnostics: dict[str, Any]
    body_selection: dict[str, Any]
    hir_summary: dict[str, Any]
    input_contract: str
    normalized_hir_blocks: list[HIRBlock]

    def to_dict(self, include_hir: bool = False, include_text: bool = True, include_ast: bool = True, include_diagnostics: bool = True) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "span": self.span,
            "slice_mode": self.slice_mode,
            "summary": self.summary,
            "body_selection": self.body_selection,
            "hir_summary": self.hir_summary,
            "input_contract": self.input_contract,
        }
        if include_diagnostics:
            payload["diagnostics"] = self.diagnostics
        if include_ast:
            payload["ast"] = self.ast
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


def _split_inline_if_statement(text: str) -> Optional[tuple[str, str]]:
    if not text.startswith("if ("):
        return None
    depth = 1
    current: list[str] = []
    idx = len("if (")
    while idx < len(text):
        ch = text[idx]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                rest = text[idx + 1:].strip()
                condition = ''.join(current).strip()
                if not rest or rest.startswith("{"):
                    return None
                return condition, rest
        current.append(ch)
        idx += 1
    return None


def _lift_statement(stmt: str) -> dict[str, Any]:
    text = stmt.strip()
    if not text:
        return {"kind": "empty", "text": stmt}
    inline_if = _split_inline_if_statement(text)
    if inline_if is not None:
        condition, action = inline_if
        return {
            "kind": "conditional_jump",
            "condition": _lift_expr(condition),
            "jump": _lift_statement(action),
            "text": text,
        }
    if text == "break":
        return {"kind": "break", "text": text}
    if text.startswith("break "):
        return {"kind": "break", "target": text[len("break "):].strip(), "text": text}
    if text == "continue":
        return {"kind": "continue", "text": text}
    if text.startswith("continue "):
        return {"kind": "continue", "target": text[len("continue "):].strip(), "text": text}
    if text.startswith("goto "):
        return {"kind": "goto", "target": text[len("goto "):].strip(), "text": text}
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
    rendered_lines = list(terminator.get("rendered_lines") or [])
    return {
        "kind": kind,
        "text": terminator.get("text"),
        "condition": _lift_expr(terminator.get("condition")),
        "rendered_lines": rendered_lines,
        "rendered": [_lift_statement(line) for line in rendered_lines],
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
            "preheader": [_lift_statement(stmt) for stmt in region.get("preheader", [])],
            "condition": _lift_expr(region.get("condition")),
            "prologue": [_lift_statement(stmt) for stmt in region.get("prologue", [])],
            "rendering": region.get("rendering"),
            "body": _lift_region(region.get("body")),
            "continue_target": region.get("continue_target"),
            "continue_targets": list(region.get("continue_targets") or []),
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


_AST_REGION_KINDS = {
    "sequence",
    "block",
    "if",
    "if_else",
    "while",
    "switch_like",
    "case",
    "goto_region",
}


def _collect_ast_shape_metrics(ast_root: dict[str, Any]) -> dict[str, Any]:
    region_hist: Counter[str] = Counter()
    statement_hist: Counter[str] = Counter()
    expression_hist: Counter[str] = Counter()
    max_depth = 0
    parameterized_region_count = 0
    region_param_count = 0
    parameterized_block_count = 0
    block_param_count = 0
    targeted_continue_count = 0
    targeted_break_count = 0
    rendered_terminator_statement_count = 0

    def visit_expr(expr: Any) -> None:
        if not isinstance(expr, dict):
            return
        kind = str(expr.get("kind") or "unknown")
        expression_hist[kind] += 1
        for key in ("target", "value", "expr", "condition"):
            visit_expr(expr.get(key))
        for arg in expr.get("args") or []:
            visit_expr(arg)

    def visit_statement(stmt: Any) -> None:
        nonlocal targeted_continue_count, targeted_break_count
        if not isinstance(stmt, dict):
            return
        kind = str(stmt.get("kind") or "unknown")
        statement_hist[kind] += 1
        if kind == "continue" and stmt.get("target"):
            targeted_continue_count += 1
        if kind == "break" and stmt.get("target"):
            targeted_break_count += 1
        for key in ("target", "value", "expr", "condition"):
            visit_expr(stmt.get(key))
        jump = stmt.get("jump")
        if isinstance(jump, dict):
            visit_statement(jump)

    def visit_region(region: Any, depth: int = 1) -> None:
        nonlocal max_depth, parameterized_region_count, region_param_count, parameterized_block_count, block_param_count, rendered_terminator_statement_count
        if not isinstance(region, dict):
            return
        kind = str(region.get("kind") or "unknown")
        region_hist[kind] += 1
        max_depth = max(max_depth, depth)
        params = list(region.get("block_params") or [])
        if params:
            parameterized_region_count += 1
            region_param_count += len(params)

        if kind == "block":
            if params:
                parameterized_block_count += 1
                block_param_count += len(params)
            for stmt in region.get("body") or []:
                visit_statement(stmt)
            terminator = region.get("terminator") or {}
            rendered = list(terminator.get("rendered") or [])
            rendered_terminator_statement_count += len(rendered)
            for stmt in rendered:
                visit_statement(stmt)
            visit_expr(terminator.get("condition"))
            return

        if kind in {"if", "if_else", "while", "case"}:
            visit_expr(region.get("condition"))
        for stmt in region.get("preheader") or []:
            visit_statement(stmt)
        for stmt in region.get("prologue") or []:
            visit_statement(stmt)

        if kind == "sequence":
            for child in region.get("regions") or []:
                visit_region(child, depth + 1)
            return
        if kind in {"if", "if_else"}:
            visit_region(region.get("then"), depth + 1)
            visit_region(region.get("else"), depth + 1)
            return
        if kind == "while":
            visit_region(region.get("body"), depth + 1)
            return
        if kind == "switch_like":
            for case in region.get("cases") or []:
                visit_region(case, depth + 1)
            return
        if kind == "case":
            visit_region(region.get("body"), depth + 1)
            return
        if kind == "goto_region":
            for child in region.get("blocks") or []:
                visit_region(child, depth + 1)
            return

    visit_region(ast_root)
    return {
        "ast_region_count": sum(region_hist.values()),
        "ast_statement_count": sum(statement_hist.values()),
        "ast_expression_count": sum(expression_hist.values()),
        "ast_max_depth": max_depth,
        "ast_region_kind_histogram": dict(region_hist),
        "ast_statement_kind_histogram": dict(statement_hist),
        "ast_expression_kind_histogram": dict(expression_hist),
        "explicit_cfg_region_count": region_hist.get("goto_region", 0),
        "ast_parameterized_region_count": parameterized_region_count,
        "ast_region_param_count": region_param_count,
        "ast_parameterized_block_count": parameterized_block_count,
        "ast_block_param_count": block_param_count,
        "ast_targeted_continue_count": targeted_continue_count,
        "ast_targeted_break_count": targeted_break_count,
        "ast_rendered_terminator_statement_count": rendered_terminator_statement_count,
    }


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


def _loop_external_successors(loop: dict[str, Any], by_id: dict[str, HIRBlock]) -> list[str]:
    loop_nodes = set(loop.get("nodes") or set())
    return _ordered_unique(
        succ
        for block_id in sorted(loop_nodes, key=lambda item: by_id[item].index if item in by_id else 10 ** 9)
        for succ in (by_id[block_id].successors if block_id in by_id else [])
        if succ not in loop_nodes
    )


def _loopback_edges(loop: dict[str, Any], by_id: dict[str, HIRBlock], index_by_id: dict[str, int]) -> dict[str, set[str]]:
    loop_nodes = set(loop.get("nodes") or set())
    edges: dict[str, set[str]] = {}
    for source_id in loop_nodes:
        source = by_id.get(source_id)
        if source is None:
            continue
        source_idx = index_by_id.get(source_id, 10 ** 9)
        for succ in source.successors:
            if succ in loop_nodes and index_by_id.get(succ, 10 ** 9) <= source_idx:
                edges.setdefault(succ, set()).add(source_id)
    return edges


def _loop_jump_aliases(loop: dict[str, Any], by_id: dict[str, HIRBlock], index_by_id: dict[str, int], exit_id: Optional[str]) -> tuple[dict[str, str], dict[str, set[str]]]:
    aliases: dict[str, str] = {}
    settled: dict[str, set[str]] = {}
    header = str(loop.get("header"))
    for target_id, source_ids in _loopback_edges(loop, by_id, index_by_id).items():
        aliases[target_id] = "continue" if target_id == header else f"continue {target_id}"
        settled.setdefault(target_id, set()).update(source_ids)
    if exit_id is not None:
        aliases[exit_id] = "break"
    return aliases, settled


def _match_unconditional_loop(loop: dict[str, Any], *, limit: int, allowed_ids: Optional[set[str]], by_id: dict[str, HIRBlock], index_by_id: dict[str, int]) -> Optional[dict[str, Any]]:
    if not loop.get("contiguous"):
        return None
    loop_nodes = set(loop.get("nodes") or set())
    if allowed_ids is not None and not loop_nodes.issubset(allowed_ids):
        return None
    idx0, idx1 = loop.get("index_range") or (None, None)
    if idx0 is None or idx1 is None or idx1 >= limit:
        return None
    external_successors = _loop_external_successors(loop, by_id)
    if len(external_successors) != 1:
        return None
    exit_id = external_successors[0]
    exit_idx = index_by_id.get(exit_id, 10 ** 9)
    if exit_idx <= idx1:
        return None
    aliases, settled = _loop_jump_aliases(loop, by_id, index_by_id, exit_id)
    if not aliases:
        return None
    return {
        "kind": "while",
        "condition": "true",
        "rendering": "unconditional_loop",
        "body_start": idx0,
        "body_stop": idx1 + 1,
        "body_ids": loop_nodes,
        "exit_id": exit_id,
        "aliases": aliases,
        "settled_predecessors": settled,
    }


def _next_allowed_block_id(blocks: list[HIRBlock], start_idx: int, limit: int, allowed_ids: Optional[set[str]]) -> Optional[str]:
    for idx in range(start_idx + 1, min(limit, len(blocks))):
        candidate = blocks[idx]
        if allowed_ids is None or candidate.id in allowed_ids:
            return candidate.id
    return None


def _incoming_args_equivalent(source_id: str, linear_pred_id: str, target_id: str, by_id: dict[str, HIRBlock]) -> bool:
    target = by_id.get(target_id)
    if target is None or not target.block_params:
        return True
    return list(target.incoming_args.get(source_id) or []) == list(target.incoming_args.get(linear_pred_id) or [])


def _empty_fallthrough_tail(start_id: Optional[str], target_id: Optional[str], by_id: dict[str, HIRBlock], index_by_id: dict[str, int]) -> Optional[str]:
    if start_id is None or target_id is None:
        return None
    current_id = start_id
    previous_id: Optional[str] = None
    seen: set[str] = set()
    while current_id not in seen:
        if current_id == target_id:
            return previous_id
        seen.add(current_id)
        block = by_id.get(current_id)
        if block is None or block.statements or block.block_params:
            return None
        if block.terminator.get("kind") != "fallthrough" or len(block.successors) != 1:
            return None
        successor = block.successors[0]
        if index_by_id.get(successor, -1) <= index_by_id.get(current_id, -1):
            return None
        previous_id = current_id
        current_id = successor
    return None


def _empty_fallthrough_reaches(
    start_id: Optional[str],
    target_id: Optional[str],
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
    *,
    source_id: Optional[str] = None,
) -> bool:
    tail_id = _empty_fallthrough_tail(start_id, target_id, by_id, index_by_id)
    if tail_id is None:
        return start_id == target_id
    if source_id is not None:
        return _incoming_args_equivalent(source_id, tail_id, str(target_id), by_id)
    return True


def _empty_forward_skip(pred_id: str, target_id: str, by_id: dict[str, HIRBlock], index_by_id: dict[str, int]) -> bool:
    pred_idx = index_by_id.get(pred_id, -1)
    target_idx = index_by_id.get(target_id, -1)
    if pred_idx < 0 or target_idx < 0 or pred_idx >= target_idx:
        return False
    if target_idx == pred_idx + 1:
        return True
    next_id = None
    for block in by_id.values():
        if block.index == pred_idx + 1:
            next_id = block.id
            break
    tail_id = _empty_fallthrough_tail(next_id, target_id, by_id, index_by_id)
    if tail_id is None:
        return False
    if _incoming_args_equivalent(pred_id, tail_id, target_id, by_id):
        return True
    pred = by_id.get(pred_id)
    merge = _branch_param_merge(pred, next_id, by_id, index_by_id) if pred is not None else None
    return bool(merge and merge.get("target_id") == target_id and merge.get("lines"))


def _label_needed(
    block: HIRBlock,
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
    allowed_ids: Optional[set[str]],
    settled_predecessors: Optional[set[str]] = None,
) -> bool:
    settled = settled_predecessors or set()
    preds = [pred for pred in block.predecessors if pred not in settled]
    preds = [pred for pred in preds if not _empty_forward_skip(pred, block.id, by_id, index_by_id)]
    if block.index == 0:
        return bool(preds)
    if not preds:
        return False
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
    return not _empty_forward_skip(pred_id, block.id, by_id, index_by_id)


def _format_block_label(block: HIRBlock) -> str:
    if block.block_params:
        return f"{block.id}({', '.join(block.block_params)})"
    return block.id



def _linear_predecessor(block: HIRBlock, by_id: dict[str, HIRBlock]) -> Optional[HIRBlock]:
    expected_index = block.index - 1
    for candidate in by_id.values():
        if candidate.index == expected_index and candidate.id in block.predecessors:
            return candidate
    return None


def _block_param_assignment_lines(block: HIRBlock, source_id: Optional[str]) -> list[str]:
    if source_id is None or not block.block_params:
        return []
    args = list(block.incoming_args.get(source_id) or [])
    if len(args) != len(block.block_params):
        return []
    return [f"{param} = {arg}" for param, arg in zip(block.block_params, args)]


def _branch_param_merge(
    block: HIRBlock,
    next_id: Optional[str],
    by_id: dict[str, HIRBlock],
    index_by_id: dict[str, int],
) -> Optional[dict[str, Any]]:
    if block.terminator.get("kind") != "branch" or next_id is None:
        return None
    branch_target, fallthrough_target = _edge_targets(block, index_by_id)
    if not branch_target or not fallthrough_target:
        return None
    if fallthrough_target != next_id or branch_target == next_id:
        return None
    target = by_id.get(branch_target)
    if target is None or not target.block_params:
        return None
    fallthrough_tail = _empty_fallthrough_tail(fallthrough_target, branch_target, by_id, index_by_id)
    if fallthrough_tail is None:
        return None
    default_args = list(target.incoming_args.get(fallthrough_tail) or [])
    branch_args = list(target.incoming_args.get(block.id) or [])
    if len(default_args) != len(target.block_params) or len(branch_args) != len(target.block_params):
        return None
    if default_args == branch_args:
        return {
            "target_id": branch_target,
            "fallthrough_tail": fallthrough_tail,
            "lines": [],
        }
    condition = block.terminator.get("condition") or f"cond_{block.id}"
    return {
        "target_id": branch_target,
        "fallthrough_tail": fallthrough_tail,
        "lines": [
            f"{param} = select({condition}, {branch_arg}, {default_arg})"
            for param, branch_arg, default_arg in zip(target.block_params, branch_args, default_args)
        ],
    }


def _format_jump_target(source: Optional[HIRBlock], target: Optional[str], by_id: dict[str, HIRBlock], jump_aliases: Optional[dict[str, str]]) -> Optional[str]:
    if target is None:
        return None
    target_block = by_id.get(target)
    args = target_block.incoming_args.get(source.id) if source is not None and target_block is not None else None
    if jump_aliases and target in jump_aliases:
        alias = jump_aliases[target]
        alias_kind = alias.split(" ", 1)[0]
        is_forward_continue = (
            alias_kind == "continue"
            and source is not None
            and target_block is not None
            and target_block.index > source.index
        )
        if not is_forward_continue:
            if args and alias_kind in {"continue", "break"}:
                return f"{alias} {target}({', '.join(args)})" if alias in {"continue", "break"} else f"{alias}({', '.join(args)})"
            return alias
    if target_block is None:
        return target
    if not target_block.block_params:
        return target
    if source is None or not args:
        return _format_block_label(target_block)
    return f"{target}({', '.join(args)})"


def _jump_stmt(source: Optional[HIRBlock], target: Optional[str], by_id: dict[str, HIRBlock], jump_aliases: Optional[dict[str, str]]) -> tuple[Optional[str], int]:
    formatted = _format_jump_target(source, target, by_id, jump_aliases)
    if formatted is None:
        return None, 0
    if jump_aliases and target in jump_aliases:
        alias = jump_aliases[target]
        if formatted == alias or formatted.startswith(f"{alias}(") or formatted.startswith(f"{alias} "):
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
                merge = _branch_param_merge(block, next_id, by_id, index_by_id)
                if merge is not None:
                    return list(merge.get("lines") or []), 0
                fallthrough_tail = _empty_fallthrough_tail(fallthrough_target, branch_target, by_id, index_by_id)
                if fallthrough_tail is not None and _incoming_args_equivalent(block.id, fallthrough_tail, branch_target, by_id):
                    return [], 0
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
    settled_predecessors: Optional[set[str]] = None,
    handled_param_predecessors: Optional[set[str]] = None,
) -> tuple[list[str], int, int]:
    lines: list[str] = []
    label_count = 0
    goto_count = 0
    stmt_indent = ""
    needs_label = _label_needed(block, by_id, index_by_id, allowed_ids, settled_predecessors)
    if needs_label:
        lines.append(f"{_format_block_label(block)}:")
        stmt_indent = "    "
        label_count = 1
    else:
        predecessor = _linear_predecessor(block, by_id)
        predecessor_id = predecessor.id if predecessor is not None else None
        if predecessor_id is not None and predecessor_id not in (handled_param_predecessors or set()):
            for assignment in _block_param_assignment_lines(block, predecessor_id):
                lines.append(assignment)
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
    settled_predecessors: Optional[set[str]] = None,
    handled_param_predecessors: Optional[set[str]] = None,
) -> tuple[dict[str, Any], list[str], int, int]:
    block_lines, label_count, goto_count = _generic_block_lines(
        block,
        by_id=by_id,
        index_by_id=index_by_id,
        next_id=next_id,
        allowed_ids=allowed_ids,
        jump_aliases=jump_aliases,
        settled_predecessors=settled_predecessors,
        handled_param_predecessors=handled_param_predecessors,
    )
    needs_label = _label_needed(block, by_id, index_by_id, allowed_ids, settled_predecessors)
    label = _format_block_label(block) if needs_label else None
    predecessor = _linear_predecessor(block, by_id) if not needs_label else None
    predecessor_id = predecessor.id if predecessor is not None else None
    param_assignments = []
    if predecessor_id is not None and predecessor_id not in (handled_param_predecessors or set()):
        param_assignments = _block_param_assignment_lines(block, predecessor_id)
    term_lines, _ = _generic_terminator_lines(block, next_id, by_id, index_by_id, jump_aliases)
    region = {
        "kind": "block",
        "block_id": block.id,
        "label": label,
        "block_params": list(block.block_params),
        "statements": list(param_assignments) + list(block.statements),
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
    settled_predecessors: Optional[set[str]] = None,
) -> tuple[int, int]:
    label_count = 1 if _label_needed(block, by_id, index_by_id, allowed_ids, settled_predecessors) else 0
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
    return bool(label_count or goto_count)


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
        return tree, "", {"constructs": {}, "fallback_block_count": 0, "residual_label_count": 0, "residual_goto_count": 0, "loop_header_count": 0, "unconditional_loop_count": 0, "structured_loopback_count": 0}

    ctx = _build_structured_context(blocks)
    constructs: Counter[str] = Counter()
    visited: set[str] = set()
    residual_labels = 0
    residual_gotos = 0
    fallback_regions_total = 0
    fallback_blocks_total = 0
    unconditional_loops_total = 0
    structured_loopbacks_total = 0

    def build_arm(
        arm: dict[str, Any],
        *,
        indent: int,
        jump_aliases: Optional[dict[str, str]],
        suppressed_loop_headers: Optional[set[str]] = None,
    ) -> tuple[Optional[dict[str, Any]], list[str]]:
        if arm.get("mode") == "linear":
            region = _linear_chain_region(arm["chain"], ctx.by_id) if emit_tree else None
            lines = _render_linear_chain(arm["chain"], ctx.by_id, indent) if emit_text else []
            return region, lines
        arm_regions, arm_lines = walk_range(
            arm["start"],
            arm["stop"],
            indent,
            set(arm["ids"]),
            jump_aliases,
            suppressed_loop_headers=suppressed_loop_headers,
        )
        return (_sequence_region(arm_regions) if emit_tree else None), arm_lines

    def walk_range(
        start_idx: int,
        stop_idx: Optional[int],
        indent: int,
        allowed_ids: Optional[set[str]] = None,
        jump_aliases: Optional[dict[str, str]] = None,
        *,
        suppressed_loop_headers: Optional[set[str]] = None,
        settled_predecessors_seed: Optional[dict[str, set[str]]] = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        nonlocal residual_labels, residual_gotos, fallback_regions_total, fallback_blocks_total, unconditional_loops_total, structured_loopbacks_total
        regions: list[dict[str, Any]] = []
        lines: list[str] = []
        limit = stop_idx if stop_idx is not None else len(blocks)
        i = start_idx

        pending_block_regions: list[dict[str, Any]] = []
        pending_block_lines: list[str] = []
        pending_block_ids: list[str] = []
        pending_label_count = 0
        pending_goto_count = 0
        current_suppressed_loop_headers = suppressed_loop_headers or set()
        settled_predecessors_by_block: dict[str, set[str]] = {
            block_id: set(preds)
            for block_id, preds in (settled_predecessors_seed or {}).items()
        }
        handled_param_predecessors_by_block: dict[str, set[str]] = {}

        def mark_handled_param_predecessor(target_id: Optional[str], predecessor_id: Optional[str]) -> None:
            if target_id is None or predecessor_id is None:
                return
            handled_param_predecessors_by_block.setdefault(target_id, set()).add(predecessor_id)

        def mark_structured_predecessors(target_id: Optional[str], source_ids: set[str]) -> None:
            if target_id is None:
                return
            settled = {
                source_id
                for source_id in source_ids
                if source_id in ctx.by_id and target_id in ctx.by_id[source_id].successors
            }
            if not settled:
                return
            settled_predecessors_by_block.setdefault(target_id, set()).update(settled)

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
            if loop and block.id not in current_suppressed_loop_headers and loop.get("safe") and loop["index_range"][1] < limit and (allowed_ids is None or set(loop["nodes"]).issubset(allowed_ids | {block.id})):
                flush_pending_goto_region()
                constructs["while"] += 1
                visited.add(block.id)
                header_lines = list(block.statements)
                cond = block.terminator.get("condition") or f"cond_{block.id}"
                body_ids = set(loop["nodes"]) - {block.id}
                predecessor = _linear_predecessor(block, ctx.by_id)
                loop_param_assignments = _block_param_assignment_lines(
                    block,
                    predecessor.id if predecessor is not None and predecessor.id not in body_ids else None,
                )
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
                            "preheader": list(loop_param_assignments),
                            "condition": cond,
                            "prologue": header_lines,
                            "rendering": "guarded_loop" if header_lines else "direct_while",
                            "body": _sequence_region(body_regions),
                            "continue_target": loop["header"],
                            "break_target": loop["exit_succ"],
                        }
                    )
                if emit_text:
                    if loop_param_assignments:
                        lines.extend(_indent(loop_param_assignments, indent))
                    if header_lines:
                        lines.extend(_indent(["while (true) {"], indent))
                        lines.extend(_indent(header_lines, indent + 1))
                        lines.extend(_indent([f"if (!({cond})) break"], indent + 1))
                    else:
                        lines.extend(_indent([f"while ({cond}) {{"], indent))
                    lines.extend(body_lines)
                    lines.extend(_indent(["}"], indent))
                loop_node_ids = set(loop.get("nodes") or [])
                mark_structured_predecessors(loop.get("exit_succ"), loop_node_ids)
                visited.update(body_ids)
                i = loop["index_range"][1] + 1
                continue

            if loop and block.id not in current_suppressed_loop_headers and not loop.get("safe"):
                loop_region = _match_unconditional_loop(
                    loop,
                    limit=limit,
                    allowed_ids=allowed_ids,
                    by_id=ctx.by_id,
                    index_by_id=ctx.index_by_id,
                )
                if loop_region is not None:
                    flush_pending_goto_region()
                    constructs["while"] += 1
                    unconditional_loops_total += 1
                    structured_loopbacks_total += sum(len(preds) for preds in loop_region.get("settled_predecessors", {}).values())
                    loop_aliases = dict(jump_aliases or {})
                    loop_aliases.update(loop_region["aliases"])
                    body_regions, body_lines = walk_range(
                        loop_region["body_start"],
                        loop_region["body_stop"],
                        indent + 1,
                        set(loop_region["body_ids"]),
                        loop_aliases,
                        suppressed_loop_headers=current_suppressed_loop_headers | {block.id},
                        settled_predecessors_seed=loop_region.get("settled_predecessors", {}),
                    )
                    if emit_tree:
                        regions.append(
                            {
                                "kind": "while",
                                "header_block": block.id,
                                "block_params": list(block.block_params),
                                "condition": loop_region.get("condition", "true"),
                                "prologue": [],
                                "rendering": loop_region.get("rendering", "unconditional_loop"),
                                "body": _sequence_region(body_regions),
                                "continue_target": block.id,
                                "continue_targets": sorted(
                                    target_id
                                    for target_id, alias in loop_region.get("aliases", {}).items()
                                    if str(alias).startswith("continue")
                                ),
                                "break_target": loop_region.get("exit_id"),
                            }
                        )
                    if emit_text:
                        lines.extend(_indent(["while (true) {"], indent))
                        lines.extend(body_lines)
                        lines.extend(_indent(["}"], indent))
                    mark_structured_predecessors(loop_region.get("exit_id"), set(loop_region.get("body_ids", set())))
                    visited.update(loop_region["body_ids"])
                    i = loop_region["body_stop"]
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
                mark_structured_predecessors(
                    switch_info.get("join"),
                    {case_block.id for case_block in switch_info.get("blocks", [])},
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
                then_region, then_lines = build_arm(
                    region["then_arm"],
                    indent=indent + 1,
                    jump_aliases=jump_aliases,
                    suppressed_loop_headers=current_suppressed_loop_headers,
                )
                else_region: Optional[dict[str, Any]] = None
                else_lines: list[str] = []
                if region["kind"] == "if_else" and region.get("else_arm") is not None:
                    else_region, else_lines = build_arm(
                        region["else_arm"],
                        indent=indent + 1,
                        jump_aliases=jump_aliases,
                        suppressed_loop_headers=current_suppressed_loop_headers,
                    )
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
                resume_idx = int(region.get("resume_idx", len(blocks)))
                resume_id = blocks[resume_idx].id if resume_idx < len(blocks) else None
                mark_structured_predecessors(resume_id, {block.id} | set(region["consumed"]))
                visited.update(region["consumed"])
                i = region["resume_idx"]
                continue

            visited.add(block.id)
            block_region: Optional[dict[str, Any]] = None
            block_lines: list[str] = []
            settled_predecessors = settled_predecessors_by_block.get(block.id, set())
            handled_param_predecessors = handled_param_predecessors_by_block.get(block.id, set())
            merge = _branch_param_merge(block, next_id, ctx.by_id, ctx.index_by_id)
            if merge is not None and merge.get("lines"):
                mark_handled_param_predecessor(merge.get("target_id"), merge.get("fallthrough_tail"))
            if emit_tree or emit_text:
                block_region, block_lines, label_count, goto_count = _generic_block_region(
                    block,
                    by_id=ctx.by_id,
                    index_by_id=ctx.index_by_id,
                    next_id=next_id,
                    allowed_ids=allowed_ids,
                    jump_aliases=jump_aliases,
                    settled_predecessors=settled_predecessors,
                    handled_param_predecessors=handled_param_predecessors,
                )
            else:
                label_count, goto_count = _generic_block_counts(
                    block,
                    by_id=ctx.by_id,
                    index_by_id=ctx.index_by_id,
                    next_id=next_id,
                    allowed_ids=allowed_ids,
                    jump_aliases=jump_aliases,
                    settled_predecessors=settled_predecessors,
                )
            residual_labels += label_count
            residual_gotos += goto_count

            if _requires_goto_region(block, label_count, goto_count):
                pending_block_ids.append(block.id)
                pending_label_count += label_count
                pending_goto_count += goto_count
                if emit_tree and block_region is not None:
                    pending_block_regions.append(block_region)
                if emit_text:
                    pending_block_lines.extend(block_lines)
            else:
                flush_pending_goto_region()
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
        "unconditional_loop_count": unconditional_loops_total,
        "structured_loopback_count": structured_loopbacks_total,
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
    shape_metrics = _collect_ast_shape_metrics(lifted_tree or {"kind": "sequence", "regions": []})
    block_count = len(blocks)
    fallback_block_count = int(structured_meta.get("fallback_block_count", 0))
    residual_goto_count = int(structured_meta.get("residual_goto_count", 0))
    residual_label_count = int(structured_meta.get("residual_label_count", 0))
    structured_block_count = max(0, block_count - fallback_block_count)
    fallback_block_ratio = (fallback_block_count / block_count) if block_count else 0.0
    summary = {
        "normalized_basic_block_count": block_count,
        "structured_block_count": structured_block_count,
        "structured_block_ratio": (structured_block_count / block_count) if block_count else 1.0,
        "fallback_block_ratio": fallback_block_ratio,
        "region_kind_histogram": structured_meta.get("constructs", {}),
        "fallback_region_count": structured_meta.get("fallback_region_count", 0),
        "fallback_block_count": fallback_block_count,
        "residual_label_count": residual_label_count,
        "residual_goto_count": residual_goto_count,
        "residual_goto_density": (residual_goto_count / block_count) if block_count else 0.0,
        "residual_label_density": (residual_label_count / block_count) if block_count else 0.0,
        "loop_header_count": structured_meta.get("loop_header_count", 0),
        "unconditional_loop_count": structured_meta.get("unconditional_loop_count", 0),
        "structured_loopback_count": structured_meta.get("structured_loopback_count", 0),
        "source_shape_score": max(
            0.0,
            1.0
            - fallback_block_ratio
            - (0.25 * ((residual_goto_count / block_count) if block_count else 0.0))
            - (0.10 * ((residual_label_count / block_count) if block_count else 0.0)),
        ),
        **shape_metrics,
    }
    diagnostics = {
        "structuring": structured_meta,
        "ast_shape": shape_metrics,
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
        diagnostics=diagnostics,
        body_selection=body_selection,
        hir_summary=hir_summary,
        input_contract=input_contract,
        normalized_hir_blocks=blocks,
    )


def build_function_ast(mod: MBCModule, entry_or_name: FunctionEntry | str, *, include_canonical: bool = False, include_hir: bool = False, include_text: bool = True, validate: bool = False) -> ASTFunction:
    hir = build_function_hir(mod, entry_or_name, include_canonical=include_canonical, include_text=False, validate=validate)
    return build_function_ast_from_payload(hir.to_dict(include_canonical=include_canonical, include_text=False), include_hir=include_hir, include_text=include_text)


def build_module_ast(
    path: str | Path,
    *,
    include_canonical: bool = False,
    include_hir: bool = False,
    include_text: bool = True,
    include_ast: bool = True,
    include_definitions: bool = True,
    include_exports: bool = True,
    include_diagnostics: bool = True,
    validate: bool = False,
) -> dict[str, Any]:
    hir_payload = build_module_hir(
        path,
        include_canonical=include_canonical,
        include_text=True,
        include_definitions=include_definitions,
        include_exports=include_exports,
        validate=validate,
    )
    functions = [
        build_function_ast_from_payload(fn, include_hir=include_hir, include_text=include_text)
        for fn in hir_payload.get("functions", [])
    ]
    functions_payload = [fn.to_dict(include_hir=include_hir, include_text=include_text, include_ast=include_ast, include_diagnostics=include_diagnostics) for fn in functions]
    region_hist = Counter()
    ast_region_hist = Counter()
    statement_hist = Counter()
    expression_hist = Counter()
    for fn in functions:
        region_hist.update(fn.summary.get("region_kind_histogram", {}))
        ast_region_hist.update(fn.summary.get("ast_region_kind_histogram", {}))
        statement_hist.update(fn.summary.get("ast_statement_kind_histogram", {}))
        expression_hist.update(fn.summary.get("ast_expression_kind_histogram", {}))
    return {
        "contract": {
            "version": AST_CONTRACT_VERSION,
            "input_hir_contract": hir_payload.get("contract", {}).get("version"),
            "layers": ["normalized_hir", "structured_regions", "lifted_ast", "ast_summary"],
            "notes": [
                "AST consumes normalized CFG-like HIR blocks and performs final structuring above HIR",
                "structuring preserves explicit goto_region fallbacks where unsafe to force a source-shaped tree",
                "semantic lifting is conservative and keeps expression text when exact VM semantics remain unknown",
            ],
        },
        "path": hir_payload.get("path"),
        "script_name": hir_payload.get("script_name"),
        "summary": {
            "function_count": len(functions_payload),
            "total_normalized_basic_blocks": sum(fn.summary.get("normalized_basic_block_count", 0) for fn in functions),
            "total_structured_blocks": sum(fn.summary.get("structured_block_count", 0) for fn in functions),
            "total_fallback_regions": sum(fn.summary.get("fallback_region_count", 0) for fn in functions),
            "total_fallback_blocks": sum(fn.summary.get("fallback_block_count", 0) for fn in functions),
            "total_residual_labels": sum(fn.summary.get("residual_label_count", 0) for fn in functions),
            "total_residual_gotos": sum(fn.summary.get("residual_goto_count", 0) for fn in functions),
            "total_loop_headers": sum(fn.summary.get("loop_header_count", 0) for fn in functions),
            "total_unconditional_loops": sum(fn.summary.get("unconditional_loop_count", 0) for fn in functions),
            "total_structured_loopbacks": sum(fn.summary.get("structured_loopback_count", 0) for fn in functions),
            "avg_source_shape_score": (sum(fn.summary.get("source_shape_score", 1.0) for fn in functions) / len(functions)) if functions else 1.0,
            "max_ast_depth": max((fn.summary.get("ast_max_depth", 0) for fn in functions), default=0),
            "total_ast_regions": sum(fn.summary.get("ast_region_count", 0) for fn in functions),
            "total_ast_statements": sum(fn.summary.get("ast_statement_count", 0) for fn in functions),
            "total_ast_expressions": sum(fn.summary.get("ast_expression_count", 0) for fn in functions),
            "total_explicit_cfg_regions": sum(fn.summary.get("explicit_cfg_region_count", 0) for fn in functions),
            "total_ast_parameterized_regions": sum(fn.summary.get("ast_parameterized_region_count", 0) for fn in functions),
            "total_ast_region_params": sum(fn.summary.get("ast_region_param_count", 0) for fn in functions),
            "total_ast_parameterized_blocks": sum(fn.summary.get("ast_parameterized_block_count", 0) for fn in functions),
            "total_ast_block_params": sum(fn.summary.get("ast_block_param_count", 0) for fn in functions),
            "total_ast_targeted_continues": sum(fn.summary.get("ast_targeted_continue_count", 0) for fn in functions),
            "total_ast_targeted_breaks": sum(fn.summary.get("ast_targeted_break_count", 0) for fn in functions),
            "total_ast_rendered_terminator_statements": sum(fn.summary.get("ast_rendered_terminator_statement_count", 0) for fn in functions),
            "region_kind_histogram": dict(region_hist),
            "ast_region_kind_histogram": dict(ast_region_hist),
            "ast_statement_kind_histogram": dict(statement_hist),
            "ast_expression_kind_histogram": dict(expression_hist),
        },
        "functions": functions_payload,
    }


def _merge_counter(counter: Counter[str], payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    for key, value in payload.items():
        try:
            counter[str(key)] += int(value)
        except Exception:
            continue


def _entry_ref(module: dict[str, Any], function_payload: dict[str, Any]) -> dict[str, Any]:
    entry = (function_payload.get("body_selection", {}) or {}).get("entry", {}) or {}
    return {
        "script_name": module.get("script_name"),
        "function": function_payload.get("name"),
        "symbol": entry.get("symbol"),
        "source_kind": entry.get("source_kind"),
        "is_exported": bool(entry.get("is_exported")),
    }


def summarize_ast_corpus(module_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    module_count = len(module_payloads)
    function_count = 0
    total_blocks = 0
    total_structured_blocks = 0
    total_fallback_regions = 0
    total_fallback_blocks = 0
    total_residual_labels = 0
    total_residual_gotos = 0
    total_loop_headers = 0
    total_unconditional_loops = 0
    total_structured_loopbacks = 0
    total_ast_regions = 0
    total_ast_statements = 0
    total_ast_expressions = 0
    total_explicit_cfg_regions = 0
    total_ast_parameterized_regions = 0
    total_ast_region_params = 0
    total_ast_parameterized_blocks = 0
    total_ast_block_params = 0
    total_ast_targeted_continues = 0
    total_ast_targeted_breaks = 0
    total_ast_rendered_terminator_statements = 0
    max_ast_depth = 0
    score_sum = 0.0
    fully_structured_function_count = 0
    fallback_function_count = 0

    region_hist: Counter[str] = Counter()
    ast_region_hist: Counter[str] = Counter()
    statement_hist: Counter[str] = Counter()
    expression_hist: Counter[str] = Counter()

    worst_fallback_functions: list[dict[str, Any]] = []
    most_residual_goto_functions: list[dict[str, Any]] = []
    most_structured_loopback_functions: list[dict[str, Any]] = []
    deepest_ast_functions: list[dict[str, Any]] = []
    largest_ast_functions: list[dict[str, Any]] = []
    module_watchlist: list[dict[str, Any]] = []
    module_summaries: list[dict[str, Any]] = []

    for module in module_payloads:
        summary = module.get("summary", {}) or {}
        module_functions = list(module.get("functions", []) or [])
        module_function_count = int(summary.get("function_count", len(module_functions)))
        module_blocks = int(summary.get("total_normalized_basic_blocks", 0))
        module_fallback_blocks = int(summary.get("total_fallback_blocks", 0))
        module_residual_gotos = int(summary.get("total_residual_gotos", 0))
        module_score_sum = 0.0

        function_count += module_function_count
        total_blocks += module_blocks
        total_structured_blocks += int(summary.get("total_structured_blocks", max(0, module_blocks - module_fallback_blocks)))
        total_fallback_regions += int(summary.get("total_fallback_regions", 0))
        total_fallback_blocks += module_fallback_blocks
        total_residual_labels += int(summary.get("total_residual_labels", 0))
        total_residual_gotos += module_residual_gotos
        total_loop_headers += int(summary.get("total_loop_headers", 0))
        total_unconditional_loops += int(summary.get("total_unconditional_loops", 0))
        total_structured_loopbacks += int(summary.get("total_structured_loopbacks", 0))
        total_ast_regions += int(summary.get("total_ast_regions", 0))
        total_ast_statements += int(summary.get("total_ast_statements", 0))
        total_ast_expressions += int(summary.get("total_ast_expressions", 0))
        total_explicit_cfg_regions += int(summary.get("total_explicit_cfg_regions", 0))
        total_ast_parameterized_regions += int(summary.get("total_ast_parameterized_regions", 0))
        total_ast_region_params += int(summary.get("total_ast_region_params", 0))
        total_ast_parameterized_blocks += int(summary.get("total_ast_parameterized_blocks", 0))
        total_ast_block_params += int(summary.get("total_ast_block_params", 0))
        total_ast_targeted_continues += int(summary.get("total_ast_targeted_continues", 0))
        total_ast_targeted_breaks += int(summary.get("total_ast_targeted_breaks", 0))
        total_ast_rendered_terminator_statements += int(summary.get("total_ast_rendered_terminator_statements", 0))
        max_ast_depth = max(max_ast_depth, int(summary.get("max_ast_depth", 0)))
        _merge_counter(region_hist, summary.get("region_kind_histogram"))
        _merge_counter(ast_region_hist, summary.get("ast_region_kind_histogram"))
        _merge_counter(statement_hist, summary.get("ast_statement_kind_histogram"))
        _merge_counter(expression_hist, summary.get("ast_expression_kind_histogram"))

        for fn in module_functions:
            fn_summary = fn.get("summary", {}) or {}
            block_count = int(fn_summary.get("normalized_basic_block_count", 0))
            fallback_blocks = int(fn_summary.get("fallback_block_count", 0))
            residual_gotos = int(fn_summary.get("residual_goto_count", 0))
            score = float(fn_summary.get("source_shape_score", 1.0 if not fallback_blocks else 0.0))
            module_score_sum += score
            score_sum += score
            if fallback_blocks == 0 and residual_gotos == 0:
                fully_structured_function_count += 1
            if fallback_blocks or residual_gotos:
                fallback_function_count += 1
            ref = _entry_ref(module, fn)
            common = {
                **ref,
                "normalized_basic_block_count": block_count,
                "fallback_block_count": fallback_blocks,
                "fallback_block_ratio": (fallback_blocks / block_count) if block_count else 0.0,
                "fallback_region_count": int(fn_summary.get("fallback_region_count", 0)),
                "residual_goto_count": residual_gotos,
                "residual_label_count": int(fn_summary.get("residual_label_count", 0)),
                "unconditional_loop_count": int(fn_summary.get("unconditional_loop_count", 0)),
                "structured_loopback_count": int(fn_summary.get("structured_loopback_count", 0)),
                "ast_parameterized_region_count": int(fn_summary.get("ast_parameterized_region_count", 0)),
                "ast_region_param_count": int(fn_summary.get("ast_region_param_count", 0)),
                "ast_parameterized_block_count": int(fn_summary.get("ast_parameterized_block_count", 0)),
                "ast_block_param_count": int(fn_summary.get("ast_block_param_count", 0)),
                "ast_rendered_terminator_statement_count": int(fn_summary.get("ast_rendered_terminator_statement_count", 0)),
                "source_shape_score": score,
                "region_kind_histogram": fn_summary.get("region_kind_histogram", {}),
            }
            if fallback_blocks or residual_gotos:
                worst_fallback_functions.append(common)
            if residual_gotos:
                most_residual_goto_functions.append(common)
            if int(fn_summary.get("structured_loopback_count", 0)):
                most_structured_loopback_functions.append(common)
            deepest_ast_functions.append({
                **ref,
                "ast_max_depth": int(fn_summary.get("ast_max_depth", 0)),
                "ast_region_count": int(fn_summary.get("ast_region_count", 0)),
                "normalized_basic_block_count": block_count,
                "source_shape_score": score,
            })
            largest_ast_functions.append({
                **ref,
                "ast_region_count": int(fn_summary.get("ast_region_count", 0)),
                "ast_statement_count": int(fn_summary.get("ast_statement_count", 0)),
                "ast_expression_count": int(fn_summary.get("ast_expression_count", 0)),
                "normalized_basic_block_count": block_count,
                "source_shape_score": score,
            })

        module_score = (module_score_sum / module_function_count) if module_function_count else 1.0
        module_structured_blocks = int(summary.get("total_structured_blocks", max(0, module_blocks - module_fallback_blocks)))
        module_ref = {
            "script_name": module.get("script_name"),
            "function_count": module_function_count,
            "normalized_basic_block_count": module_blocks,
            "structured_block_count": module_structured_blocks,
            "structured_block_ratio": (module_structured_blocks / module_blocks) if module_blocks else 1.0,
            "fallback_region_count": int(summary.get("total_fallback_regions", 0)),
            "fallback_block_count": module_fallback_blocks,
            "fallback_block_ratio": (module_fallback_blocks / module_blocks) if module_blocks else 0.0,
            "residual_label_count": int(summary.get("total_residual_labels", 0)),
            "residual_goto_count": module_residual_gotos,
            "unconditional_loop_count": int(summary.get("total_unconditional_loops", 0)),
            "structured_loopback_count": int(summary.get("total_structured_loopbacks", 0)),
            "ast_region_count": int(summary.get("total_ast_regions", 0)),
            "ast_statement_count": int(summary.get("total_ast_statements", 0)),
            "ast_expression_count": int(summary.get("total_ast_expressions", 0)),
            "ast_max_depth": int(summary.get("max_ast_depth", 0)),
            "explicit_cfg_region_count": int(summary.get("total_explicit_cfg_regions", 0)),
            "ast_parameterized_region_count": int(summary.get("total_ast_parameterized_regions", 0)),
            "ast_region_param_count": int(summary.get("total_ast_region_params", 0)),
            "ast_parameterized_block_count": int(summary.get("total_ast_parameterized_blocks", 0)),
            "ast_block_param_count": int(summary.get("total_ast_block_params", 0)),
            "ast_targeted_continue_count": int(summary.get("total_ast_targeted_continues", 0)),
            "ast_targeted_break_count": int(summary.get("total_ast_targeted_breaks", 0)),
            "source_shape_score": module_score,
            "region_kind_histogram": summary.get("region_kind_histogram", {}),
        }
        module_summaries.append(module_ref)
        if module_fallback_blocks or module_residual_gotos:
            module_watchlist.append(module_ref)

    worst_fallback_functions.sort(key=lambda item: (-item["fallback_block_count"], -item["residual_goto_count"], item["source_shape_score"], str(item["script_name"]), str(item["function"])))
    most_residual_goto_functions.sort(key=lambda item: (-item["residual_goto_count"], -item["fallback_block_count"], str(item["script_name"]), str(item["function"])))
    most_structured_loopback_functions.sort(key=lambda item: (-item["structured_loopback_count"], -item["normalized_basic_block_count"], str(item["script_name"]), str(item["function"])))
    deepest_ast_functions.sort(key=lambda item: (-item["ast_max_depth"], -item["ast_region_count"], str(item["script_name"]), str(item["function"])))
    largest_ast_functions.sort(key=lambda item: (-item["ast_region_count"], -item["ast_statement_count"], str(item["script_name"]), str(item["function"])))
    module_watchlist.sort(key=lambda item: (-item["fallback_block_count"], -item["residual_goto_count"], item["source_shape_score"], str(item["script_name"])))
    module_summaries.sort(key=lambda item: str(item["script_name"]))

    structured_block_ratio = (total_structured_blocks / total_blocks) if total_blocks else 1.0
    fallback_block_ratio = (total_fallback_blocks / total_blocks) if total_blocks else 0.0
    return {
        "contract": {
            "version": "ast-report-v1",
            "ast_contract": AST_CONTRACT_VERSION,
            "layers": ["normalized_hir", "structured_regions", "lifted_ast", "ast_corpus_metrics"],
            "notes": [
                "Corpus report is intentionally based on AST summaries rather than pre-AST HIR coverage reports.",
                "Fallback metrics measure residual explicit-CFG surface that still needs rule-based structuring.",
                "Source-shape score is a corpus triage metric; lower values should be investigated first.",
            ],
        },
        "summary": {
            "module_count": module_count,
            "function_count": function_count,
            "fully_structured_function_count": fully_structured_function_count,
            "fallback_function_count": fallback_function_count,
            "total_normalized_basic_blocks": total_blocks,
            "total_structured_blocks": total_structured_blocks,
            "structured_block_ratio": structured_block_ratio,
            "total_fallback_regions": total_fallback_regions,
            "total_fallback_blocks": total_fallback_blocks,
            "fallback_block_ratio": fallback_block_ratio,
            "total_residual_labels": total_residual_labels,
            "total_residual_gotos": total_residual_gotos,
            "residual_goto_density_per_1k_blocks": (1000.0 * total_residual_gotos / total_blocks) if total_blocks else 0.0,
            "total_loop_headers": total_loop_headers,
            "total_unconditional_loops": total_unconditional_loops,
            "total_structured_loopbacks": total_structured_loopbacks,
            "avg_source_shape_score": (score_sum / function_count) if function_count else 1.0,
            "max_ast_depth": max_ast_depth,
            "total_ast_regions": total_ast_regions,
            "total_ast_statements": total_ast_statements,
            "total_ast_expressions": total_ast_expressions,
            "total_explicit_cfg_regions": total_explicit_cfg_regions,
            "total_ast_parameterized_regions": total_ast_parameterized_regions,
            "total_ast_region_params": total_ast_region_params,
            "total_ast_parameterized_blocks": total_ast_parameterized_blocks,
            "total_ast_block_params": total_ast_block_params,
            "total_ast_targeted_continues": total_ast_targeted_continues,
            "total_ast_targeted_breaks": total_ast_targeted_breaks,
            "total_ast_rendered_terminator_statements": total_ast_rendered_terminator_statements,
        },
        "ast_metrics": {
            "region_kind_histogram": dict(region_hist.most_common()),
            "ast_region_kind_histogram": dict(ast_region_hist.most_common()),
            "ast_statement_kind_histogram": dict(statement_hist.most_common()),
            "ast_expression_kind_histogram": dict(expression_hist.most_common()),
            "control_flow_counters": {
                "loop_headers": total_loop_headers,
                "unconditional_loops": total_unconditional_loops,
                "structured_loopbacks": total_structured_loopbacks,
                "explicit_cfg_regions": total_explicit_cfg_regions,
                "parameterized_regions": total_ast_parameterized_regions,
                "region_params": total_ast_region_params,
                "parameterized_blocks": total_ast_parameterized_blocks,
                "block_params": total_ast_block_params,
                "targeted_continues": total_ast_targeted_continues,
                "targeted_breaks": total_ast_targeted_breaks,
                "rendered_terminator_statements": total_ast_rendered_terminator_statements,
            },
        },
        "rankings": {
            "worst_fallback_functions": worst_fallback_functions[:64],
            "most_residual_goto_functions": most_residual_goto_functions[:64],
            "most_structured_loopback_functions": most_structured_loopback_functions[:64],
            "deepest_ast_functions": deepest_ast_functions[:32],
            "largest_ast_functions": largest_ast_functions[:32],
            "module_structuring_watchlist": module_watchlist[:64],
        },
        "modules": module_summaries,
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
