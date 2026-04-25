from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mbl_vm_tools.hir import (
    HIRBlock,
    HIR_CONTRACT_VERSION,
    build_function_hir,
    _block_maps,
    _coerce_hir_block,
    _match_assignment,
    _normalize_hir_blocks,
    _ordered_unique,
    _rename_merge_params,
)
from mbl_vm_tools.parser import FunctionEntry, MBCModule


AST_CONTRACT_VERSION = "ast-v7"


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

    def to_dict(
        self,
        include_hir: bool = False,
        include_text: bool = True,
        include_ast: bool = True,
        include_diagnostics: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
            payload["normalized_hir"] = {"hir_blocks": [block.to_dict() for block in self.normalized_hir_blocks]}
        return payload


_EXPR_CALL_RE = re.compile(r"^(?P<target>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")
_MEMBER_RE = re.compile(r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*)\.(?P<field>[A-Za-z_][A-Za-z0-9_]*)$")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d*|\d*\.\d+)$")


def _split_top_level(payload: str, separator: str = ",") -> list[str]:
    if not payload.strip():
        return []
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: Optional[str] = None
    escape = False
    for ch in payload:
        if quote is not None:
            current.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            current.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == separator and depth == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _expr_node(text: Optional[str]) -> Optional[dict[str, Any]]:
    if text is None:
        return None
    source = str(text).strip()
    if not source:
        return {"kind": "empty"}
    if source.startswith("'") or source.startswith('"'):
        return {"kind": "literal", "literal_type": "string", "value": source}
    if _INT_RE.match(source):
        return {"kind": "literal", "literal_type": "int", "value": int(source)}
    if _FLOAT_RE.match(source):
        return {"kind": "literal", "literal_type": "float", "value": float(source)}
    member = _MEMBER_RE.match(source)
    if member is not None:
        return {"kind": "member", "base": {"kind": "symbol", "name": member.group("base")}, "field": member.group("field")}
    call = _EXPR_CALL_RE.match(source)
    if call is not None:
        return {
            "kind": "call_expr",
            "callee": call.group("target"),
            "args": [_expr_node(item) for item in _split_top_level(call.group("args"))],
        }
    if source.startswith("cond[") and source.endswith(")"):
        return {"kind": "predicate", "source": source}
    return {"kind": "symbol", "name": source}


def _statement_node(stmt: Any) -> dict[str, Any]:
    if isinstance(stmt, dict):
        return stmt
    text = str(stmt).strip()
    if not text:
        return {"kind": "empty"}
    if text == "break" or text.startswith("break "):
        return {"kind": "break", "target": text[6:].strip() or None}
    if text == "continue" or text.startswith("continue "):
        return {"kind": "continue", "target": text[9:].strip() or None}
    if text.startswith("goto "):
        return {"kind": "goto", "target": text[5:].strip()}
    if text.startswith("return"):
        value = text[6:].strip()
        return {"kind": "return", "value": _expr_node(value) if value else None}
    matched = _match_assignment(text)
    if matched is not None:
        lhs, rhs = matched
        value = _expr_node(rhs)
        kind = "field_store" if "." in lhs else "assign_call" if value and value.get("kind") == "call_expr" else "assign"
        return {"kind": kind, "target": _expr_node(lhs), "value": value}
    expr = _expr_node(text)
    if expr and expr.get("kind") == "call_expr":
        return {"kind": "call", "expr": expr}
    return {"kind": "expr", "expr": expr}


def _assignment_node(lhs: str, rhs: str) -> dict[str, Any]:
    value = _expr_node(rhs)
    kind = "field_store" if "." in lhs else "assign_call" if value and value.get("kind") == "call_expr" else "assign"
    return {"kind": kind, "target": _expr_node(lhs), "value": value}


def _conditional_node(condition: Any, body: list[dict[str, Any]], *, negate: bool = False) -> Optional[dict[str, Any]]:
    if not body:
        return None
    cond_expr = condition if isinstance(condition, dict) else _expr_node(str(condition))
    if negate:
        cond_expr = {"kind": "not", "expr": cond_expr}
    if len(body) == 1:
        return {"kind": "conditional_jump", "condition": cond_expr, "jump": body[0]}
    return {"kind": "conditional_block", "condition": cond_expr, "body": body}


def _render_expr(expr: Any) -> str:
    if expr is None:
        return ""
    if not isinstance(expr, dict):
        return str(expr)
    kind = expr.get("kind")
    if kind == "empty":
        return ""
    if kind == "literal":
        return str(expr.get("value"))
    if kind == "symbol":
        return str(expr.get("name") or "")
    if kind == "member":
        base = _render_expr(expr.get("base"))
        return f"{base}.{expr.get('field')}" if base else str(expr.get("field") or "")
    if kind == "call_expr":
        return f"{expr.get('callee')}({', '.join(_render_expr(arg) for arg in expr.get('args') or [])})"
    if kind == "predicate":
        return str(expr.get("source") or "")
    if kind == "not":
        return f"!({_render_expr(expr.get('expr'))})"
    return str(expr.get("source") or expr.get("name") or kind or "")


def _render_statement(stmt: Any) -> str:
    if not isinstance(stmt, dict):
        return str(stmt).strip()
    kind = stmt.get("kind")
    if kind == "empty":
        return ""
    if kind == "call":
        return _render_expr(stmt.get("expr"))
    if kind == "expr":
        return _render_expr(stmt.get("expr"))
    if kind in {"assign", "assign_call", "field_store"}:
        return f"{_render_expr(stmt.get('target'))} = {_render_expr(stmt.get('value'))}"
    if kind == "return":
        value = _render_expr(stmt.get("value"))
        return f"return {value}" if value else "return"
    if kind == "goto":
        return f"goto {stmt.get('target')}"
    if kind == "break":
        target = stmt.get("target")
        return f"break {target}" if target else "break"
    if kind == "continue":
        target = stmt.get("target")
        return f"continue {target}" if target else "continue"
    if kind in {"conditional_jump", "conditional_block"}:
        return f"if ({_render_expr(stmt.get('condition'))}) {{ ... }}"
    return str(stmt.get("source") or kind or "")


def _render_statement_lines(stmt: Any) -> list[str]:
    if not isinstance(stmt, dict):
        text = str(stmt).strip()
        return [text] if text else []
    kind = stmt.get("kind")
    if kind == "conditional_jump":
        body = _render_statement_lines(stmt.get("jump"))
        return [f"if ({_render_expr(stmt.get('condition'))}) {{", *(f"    {line}" for line in body), "}"]
    if kind == "conditional_block":
        lines: list[str] = [f"if ({_render_expr(stmt.get('condition'))}) {{"]
        for child in stmt.get("body") or []:
            lines.extend(f"    {line}" for line in _render_statement_lines(child))
        lines.append("}")
        return lines
    rendered = _render_statement(stmt)
    return [rendered] if rendered else []


def _render_statement_list(statements: list[dict[str, Any]], indent: int) -> list[str]:
    prefix = "    " * indent
    lines: list[str] = []
    for stmt in statements:
        lines.extend(prefix + line for line in _render_statement_lines(stmt))
    return lines

def _base_label(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    value = value.split("(", 1)[0].strip()
    return value or None


def _statement_target(stmt: dict[str, Any]) -> Optional[str]:
    target = stmt.get("target")
    if isinstance(target, dict):
        target = _render_expr(target)
    if not target:
        return None
    return _base_label(str(target))


def _collect_ast_shape_metrics(ast_root: dict[str, Any]) -> dict[str, Any]:
    region_hist: Counter[str] = Counter()
    statement_hist: Counter[str] = Counter()
    expression_hist: Counter[str] = Counter()
    rendered_labels: set[str] = set()
    goto_targets: set[str] = set()
    jump_targets: set[str] = set()
    max_depth = 0
    parameterized_region_count = region_param_count = 0
    parameterized_block_count = block_param_count = 0
    targeted_continue_count = targeted_break_count = 0
    rendered_terminator_statement_count = 0

    def visit_expr(expr: Any) -> None:
        if not isinstance(expr, dict):
            return
        expression_hist[str(expr.get("kind") or "unknown")] += 1
        for arg in expr.get("args") or []:
            visit_expr(arg)
        for key in ("condition", "value", "target", "expr", "base"):
            visit_expr(expr.get(key))

    def visit_stmt(stmt: Any) -> None:
        nonlocal targeted_continue_count, targeted_break_count
        if not isinstance(stmt, dict):
            return
        kind = str(stmt.get("kind") or "unknown")
        statement_hist[kind] += 1
        if kind == "goto":
            target = _statement_target(stmt)
            if target:
                goto_targets.add(target)
        elif kind in {"break", "continue"}:
            target = _statement_target(stmt)
            if target:
                jump_targets.add(target)
                if kind == "break":
                    targeted_break_count += 1
                else:
                    targeted_continue_count += 1
        for key in ("expr", "condition", "value", "target"):
            visit_expr(stmt.get(key))
        visit_stmt(stmt.get("jump"))
        for child in stmt.get("body") or []:
            visit_stmt(child)

    def visit_region(region: Any, depth: int = 1) -> None:
        nonlocal max_depth, parameterized_region_count, region_param_count
        nonlocal parameterized_block_count, block_param_count, rendered_terminator_statement_count
        if not isinstance(region, dict):
            return
        kind = str(region.get("kind") or "unknown")
        region_hist[kind] += 1
        max_depth = max(max_depth, depth)
        label = _base_label(region.get("label"))
        if label:
            rendered_labels.add(label)
        params = list(region.get("block_params") or [])
        if kind == "block":
            if params:
                parameterized_block_count += 1
                block_param_count += len(params)
            for stmt in region.get("prelabel") or []:
                visit_stmt(stmt)
            for stmt in region.get("body") or []:
                visit_stmt(stmt)
            term = region.get("terminator") or {}
            for stmt in term.get("rendered") or []:
                rendered_terminator_statement_count += 1
                visit_stmt(stmt)
            return
        if kind in {"if", "if_else", "while", "do_while"}:
            if params:
                parameterized_region_count += 1
                region_param_count += len(params)
            visit_expr(region.get("condition"))
            for name in ("prologue", "preheader", "guard_exit", "resume_param_merge"):
                for stmt in region.get(name) or []:
                    visit_stmt(stmt)
        if kind == "sequence":
            for child in region.get("regions") or []:
                visit_region(child, depth + 1)
        elif kind in {"if", "if_else"}:
            visit_region(region.get("then"), depth + 1)
            visit_region(region.get("else"), depth + 1)
        elif kind == "while":
            visit_region(region.get("body"), depth + 1)
        elif kind == "do_while":
            for stmt in region.get("body") or []:
                visit_stmt(stmt)
            for stmt in region.get("loopback") or []:
                visit_stmt(stmt)
        elif kind == "threaded_decision_tree":
            for stmt in region.get("entry_param_merge") or []:
                visit_stmt(stmt)
            for node in region.get("nodes") or []:
                for stmt in node.get("prelabel") or []:
                    visit_stmt(stmt)
                for stmt in node.get("body") or []:
                    visit_stmt(stmt)
                visit_expr(node.get("condition"))
                for edge_name in ("true_edge", "false_edge"):
                    edge = node.get(edge_name) or {}
                    for stmt in edge.get("statements") or []:
                        visit_stmt(stmt)
            for stmt in region.get("exit_param_merge") or []:
                visit_stmt(stmt)
        elif kind == "switch_like":
            for case in region.get("cases") or []:
                visit_region(case, depth + 1)
        elif kind == "case":
            visit_region(region.get("body"), depth + 1)
        elif kind == "goto_region":
            for child in region.get("blocks") or []:
                visit_region(child, depth + 1)

    visit_region(ast_root)
    dangling_gotos = sorted(target for target in goto_targets if target not in rendered_labels)
    dangling_jumps = sorted(target for target in jump_targets if target not in rendered_labels)
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
        "ast_rendered_label_count": len(rendered_labels),
        "ast_goto_target_count": len(goto_targets),
        "ast_dangling_goto_count": len(dangling_gotos),
        "ast_jump_target_count": len(jump_targets),
        "ast_dangling_jump_count": len(dangling_jumps),
        "ast_dangling_jump_targets": dangling_gotos + dangling_jumps,
    }


def _statement_text(stmt: dict[str, Any]) -> str:
    return _render_statement(stmt).strip()


def _is_data_nop_statement(stmt: dict[str, Any]) -> bool:
    expr = stmt.get("expr") if isinstance(stmt, dict) and stmt.get("kind") in {"call", "expr"} else None
    if not isinstance(expr, dict) or expr.get("kind") != "call_expr" or expr.get("callee") != "data":
        return bool(isinstance(stmt, dict) and stmt.get("kind") == "empty")
    payload = " ".join(_render_expr(arg) for arg in expr.get("args") or [])
    return '"role": "nop"' in payload or "'role': 'nop'" in payload


def _collect_simple_body_statements(region: Any) -> Optional[list[dict[str, Any]]]:
    if not isinstance(region, dict):
        return None
    if region.get("kind") == "sequence":
        out: list[dict[str, Any]] = []
        for child in region.get("regions") or []:
            items = _collect_simple_body_statements(child)
            if items is None:
                return None
            out.extend(items)
        return out
    if region.get("kind") != "block":
        return None
    if region.get("label") or region.get("block_params") or region.get("prelabel"):
        return None
    term = region.get("terminator") or {}
    if term.get("rendered"):
        return None
    if term.get("kind") not in {None, "return", "fallthrough"}:
        return None
    return list(region.get("body") or [])


def _classify_trivial_function_body(ast_root: dict[str, Any], entry_args: list[str]) -> dict[str, Any]:
    statements = _collect_simple_body_statements(ast_root)
    if statements is None:
        return {"kind": "nontrivial", "is_trivial": False, "reason": "structured_or_explicit_control", "semantic_statement_count": None, "ignored_noop_statement_count": 0}
    ignored = sum(1 for stmt in statements if _is_data_nop_statement(stmt))
    significant = [stmt for stmt in statements if not _is_data_nop_statement(stmt)]
    base = {"is_trivial": False, "semantic_statement_count": len(significant), "ignored_noop_statement_count": ignored, "raw_statement_count": len(statements)}
    if not significant:
        return {**base, "kind": "empty", "is_trivial": True, "reason": "no_semantic_statements"}
    if len(significant) == 1 and significant[0].get("kind") == "return":
        value = significant[0].get("value")
        value_text = _render_expr(value) if value is not None else ""
        if isinstance(value, dict) and value.get("kind") == "literal" and value.get("literal_type") == "int" and value.get("value") == 0:
            return {**base, "kind": "return_zero", "is_trivial": True, "reason": "single_zero_return_after_noops", "return_value_text": value_text}
        if isinstance(value, dict) and value.get("kind") == "symbol" and value.get("name") in entry_args:
            arg = str(value.get("name"))
            return {**base, "kind": "return_argument", "is_trivial": True, "reason": "single_entry_argument_return_after_noops", "return_value_text": value_text, "return_argument": arg, "return_argument_index": entry_args.index(arg)}
        return {**base, "kind": "return_only_other", "reason": "single_non_requested_return", "return_value_text": value_text}
    return {**base, "kind": "nontrivial", "reason": "semantic_statements_before_or_after_return", "sample_statements": [_statement_text(stmt) for stmt in significant[:4]]}


def _validate_ast_semantics(ast_root: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    targets = list(metrics.get("ast_dangling_jump_targets") or [])
    if targets:
        errors.append({"kind": "dangling_ast_jump", "message": "AST has visible jumps to labels not rendered in the same tree.", "targets": targets[:32]})
    return {"ok": not errors, "error_count": len(errors), "kind_histogram": dict(Counter(error["kind"] for error in errors)), "errors": errors[:32]}


@dataclass
class _Cfg:
    blocks: list[HIRBlock]
    by_id: dict[str, HIRBlock]
    index_by_id: dict[str, int]
    succs: dict[str, list[str]]
    preds: dict[str, list[str]]
    dom: dict[str, set[str]]
    postdom: dict[str, set[str]]
    ipdom: dict[str, Optional[str]]
    loops: dict[str, dict[str, Any]]
    switches: dict[int, dict[str, Any]]


def _compute_dominators(blocks: list[HIRBlock]) -> dict[str, set[str]]:
    if not blocks:
        return {}
    ids = [block.id for block in blocks]
    _, _, _, preds = _block_maps(blocks)
    dom = {block.id: set(ids) for block in blocks}
    dom[blocks[0].id] = {blocks[0].id}
    for _ in range(max(1, len(blocks) * 3)):
        changed = False
        for block in blocks[1:]:
            pred_sets = [dom[pred] for pred in preds[block.id] if pred in dom]
            new = {block.id} | (set.intersection(*pred_sets) if pred_sets else set())
            if new != dom[block.id]:
                dom[block.id] = new
                changed = True
        if not changed:
            break
    return dom


def _compute_postdominators(blocks: list[HIRBlock]) -> dict[str, set[str]]:
    if not blocks:
        return {}
    ids = [block.id for block in blocks]
    _, _, succs, _ = _block_maps(blocks)
    exits = [block.id for block in blocks if not succs[block.id]]
    if not exits:
        return {block.id: {block.id} for block in blocks}
    post = {block.id: set(ids) for block in blocks}
    for exit_id in exits:
        post[exit_id] = {exit_id}
    for _ in range(max(1, len(blocks) * 4)):
        changed = False
        for block in reversed(blocks):
            if block.id in exits:
                continue
            block_succs = [succ for succ in succs[block.id] if succ in post]
            if block_succs:
                acc = set(post[block_succs[0]])
                for succ in block_succs[1:]:
                    acc &= post[succ]
                new = {block.id} | acc
            else:
                new = {block.id}
            if new != post[block.id]:
                post[block.id] = new
                changed = True
        if not changed:
            break
    return post


def _immediate_postdom(block_id: str, postdom: dict[str, set[str]], index_by_id: dict[str, int]) -> Optional[str]:
    candidates = list(postdom.get(block_id, set()) - {block_id})
    if not candidates:
        return None
    candidates.sort(key=lambda item: index_by_id.get(item, 10**9))
    for candidate in candidates:
        if all(candidate not in postdom.get(other, set()) for other in candidates if other != candidate):
            return candidate
    return candidates[0]


def _edge_targets(block: HIRBlock, index_by_id: dict[str, int]) -> tuple[Optional[str], Optional[str]]:
    if block.terminator.get("kind") != "branch":
        return None, block.fallthrough_target or (block.successors[0] if block.successors else None)
    branch = block.branch_target
    fallthrough = block.fallthrough_target
    if branch is None and block.successors:
        ordered = sorted(block.successors, key=lambda item: index_by_id.get(item, 10**9))
        branch = ordered[0]
        fallthrough = ordered[1] if len(ordered) > 1 else None
    return branch, fallthrough


def _natural_loops(blocks: list[HIRBlock], dom: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    _, index_by_id, succs, preds = _block_maps(blocks)
    loops: dict[str, dict[str, Any]] = {}
    for block in blocks:
        for succ in succs[block.id]:
            if succ in dom.get(block.id, set()) and index_by_id.get(succ, 10**9) <= block.index:
                loop = loops.setdefault(succ, {"header": succ, "latches": set(), "nodes": {succ}})
                loop["latches"].add(block.id)
                work = [block.id]
                while work:
                    node = work.pop()
                    if node in loop["nodes"]:
                        continue
                    loop["nodes"].add(node)
                    work.extend(pred for pred in preds.get(node, []) if pred not in loop["nodes"])
    for header_id, loop in loops.items():
        indices = sorted(index_by_id[node] for node in loop["nodes"] if node in index_by_id)
        idx0, idx1 = (indices[0], indices[-1]) if indices else (index_by_id[header_id], index_by_id[header_id])
        loop["index_range"] = (idx0, idx1)
        contiguous_ids = {blocks[idx].id for idx in range(idx0, idx1 + 1)}
        nodes = set(loop["nodes"])
        contiguous = contiguous_ids == nodes
        header_succs = list(succs.get(header_id, []))
        body_succs = [succ for succ in header_succs if succ in nodes]
        exit_succs = [succ for succ in header_succs if succ not in nodes]
        loop["body_succ"] = body_succs[0] if len(body_succs) == 1 else None
        loop["exit_succ"] = exit_succs[0] if len(exit_succs) == 1 else None
        external_entries = [
            (pred, node)
            for node in nodes - {header_id}
            for pred in preds.get(node, [])
            if pred not in nodes
        ]
        loop["contiguous"] = contiguous
        loop["external_entries"] = external_entries
        exit_idx = index_by_id.get(loop["exit_succ"], -1) if loop.get("exit_succ") else -1
        body_idx = index_by_id.get(loop.get("body_succ"), -1) if loop.get("body_succ") else -1
        loop["safe"] = bool(
            blocks[index_by_id[header_id]].terminator.get("kind") == "branch"
            and loop.get("body_succ")
            and loop.get("exit_succ")
            and contiguous
            and body_idx >= idx0
            and exit_idx > idx1
        )
    return loops



def _collect_switch_like(
    start_index: int,
    blocks: list[HIRBlock],
    index_by_id: dict[str, int],
    ipdom: dict[str, Optional[str]],
    preds: dict[str, list[str]],
) -> Optional[dict[str, Any]]:
    chain: list[HIRBlock] = []
    join: Optional[str] = None
    i = start_index
    while i < len(blocks):
        block = blocks[i]
        if block.terminator.get("kind") != "branch" or block.statements:
            break
        branch_target, fallthrough_target = _edge_targets(block, index_by_id)
        next_textual = blocks[i + 1].id if i + 1 < len(blocks) else None
        if branch_target is None or fallthrough_target != next_textual:
            break
        block_join = ipdom.get(block.id)
        if block_join is None:
            break
        if join is None:
            join = block_join
        elif join != block_join:
            break
        chain.append(block)
        i += 1
        if branch_target == join:
            break
    if len(chain) < 3 or join is None:
        return None
    chain_ids = {block.id for block in chain}
    for offset, chain_block in enumerate(chain):
        for pred_id in preds.get(chain_block.id, []):
            if pred_id in chain_ids:
                continue
            pred_idx = index_by_id.get(pred_id)
            allowed_textual_entry = offset == 0 and pred_idx is not None and pred_idx + 1 == chain_block.index
            if not allowed_textual_entry:
                return None
    return {"blocks": chain, "join": join, "end_index": i}

def _build_cfg(blocks: list[HIRBlock]) -> _Cfg:
    by_id, index_by_id, succs, preds = _block_maps(blocks)
    dom = _compute_dominators(blocks)
    postdom = _compute_postdominators(blocks)
    ipdom = {block.id: _immediate_postdom(block.id, postdom, index_by_id) for block in blocks}
    loops = _natural_loops(blocks, dom)
    switches = {
        idx: info
        for idx in range(len(blocks))
        if (info := _collect_switch_like(idx, blocks, index_by_id, ipdom, preds)) is not None
    }
    return _Cfg(blocks, by_id, index_by_id, succs, preds, dom, postdom, ipdom, loops, switches)


def _format_block_label(block: HIRBlock) -> str:
    return f"{block.id}({', '.join(block.block_params)})" if block.block_params else block.id


def _incoming_assignments(target: Optional[HIRBlock], source_id: Optional[str]) -> list[dict[str, Any]]:
    if target is None or source_id is None or not target.block_params:
        return []
    args = list(target.incoming_args.get(source_id) or [])
    if len(args) != len(target.block_params):
        return []
    return [_assignment_node(param, arg) for param, arg in zip(target.block_params, args) if param != arg]


def _target_call(target: Optional[HIRBlock], source: Optional[HIRBlock]) -> Optional[str]:
    if target is None:
        return None
    if source is None or not target.block_params:
        return target.id
    args = list(target.incoming_args.get(source.id) or [])
    if len(args) == len(target.block_params):
        return f"{target.id}({', '.join(args)})"
    return _format_block_label(target)


def _external_successors(loop: dict[str, Any], cfg: _Cfg) -> list[str]:
    nodes = set(loop.get("nodes") or set())
    return _ordered_unique([
        succ
        for block_id in sorted(nodes, key=lambda item: cfg.index_by_id.get(item, 10**9))
        for succ in cfg.succs.get(block_id, [])
        if succ not in nodes
    ])


def _loopback_source_count(loop: dict[str, Any], cfg: _Cfg) -> int:
    return sum(len(sources) for sources in _loopback_edges(loop, cfg).values())


def _loopback_edges(loop: dict[str, Any], cfg: _Cfg) -> dict[str, set[str]]:
    nodes = set(loop.get("nodes") or set())
    edges: dict[str, set[str]] = {}
    for source_id in nodes:
        source_idx = cfg.index_by_id.get(source_id, 10**9)
        for succ in cfg.succs.get(source_id, []):
            if succ in nodes and cfg.index_by_id.get(succ, 10**9) <= source_idx:
                edges.setdefault(succ, set()).add(source_id)
    return edges


class _SurfaceBuilder:
    def __init__(self, blocks: list[HIRBlock]):
        self.blocks = blocks
        self.cfg = _build_cfg(blocks)
        self.constructs: Counter[str] = Counter()
        self.fallback_regions = 0
        self.fallback_blocks = 0
        self.residual_labels = 0
        self.residual_gotos = 0
        self.loop_headers = 0
        self.unconditional_loops = 0
        self.structured_loopbacks = 0
        self.settled_preds_by_block: dict[str, set[str]] = defaultdict(set)
        self.handled_param_predecessors_by_block: dict[str, set[str]] = defaultdict(set)
        self.needed_label_ids: set[str] = set()
        self.explicit_label_ids = {
            succ
            for block in blocks
            for succ in block.successors
            if self.cfg.index_by_id.get(succ, -1) != block.index + 1
        }
        self.explicit_label_ids.update(block.id for block in blocks if block.block_params or len(block.predecessors) > 1)

    def build(self) -> tuple[dict[str, Any], dict[str, Any]]:
        tree = {"kind": "sequence", "regions": self._range(0, len(self.blocks), None, {}, set(), None, set())}
        meta = {
            "constructs": dict(self.constructs),
            "fallback_region_count": self.fallback_regions,
            "fallback_block_count": self.fallback_blocks,
            "residual_label_count": self.residual_labels,
            "residual_goto_count": self.residual_gotos,
            "loop_header_count": self.loop_headers,
            "unconditional_loop_count": self.unconditional_loops,
            "structured_loopback_count": self.structured_loopbacks,
        }
        return tree, meta

    def _next_allowed_id(self, idx: int, stop: int, allowed: Optional[set[str]]) -> Optional[str]:
        for pos in range(idx + 1, min(stop, len(self.blocks))):
            block = self.blocks[pos]
            if allowed is None or block.id in allowed:
                return block.id
        return None

    def _label_needed(self, block: HIRBlock, hidden_labels: set[str], previous_source_id: Optional[str] = None) -> bool:
        if block.id in hidden_labels and block.id not in self.needed_label_ids:
            return False
        if block.id in self.needed_label_ids:
            return True
        if any(self.cfg.index_by_id.get(pred, -1) >= block.index for pred in block.predecessors):
            return True
        settled = set(self.settled_preds_by_block.get(block.id, set()))
        if previous_source_id is not None:
            settled.add(previous_source_id)
        preds = [
            pred
            for pred in block.predecessors
            if pred not in settled and not self._empty_forward_skip(pred, block.id)
        ]
        if not preds:
            return False
        nonlinear = [pred for pred in preds if self.cfg.index_by_id.get(pred, -10) + 1 != block.index]
        if nonlinear:
            return True
        return len(preds) > 1

    def _incoming_args_equivalent(self, source_id: str, linear_pred_id: str, target_id: str) -> bool:
        target = self.cfg.by_id.get(target_id)
        if target is None or not target.block_params:
            return True
        return list(target.incoming_args.get(source_id) or []) == list(target.incoming_args.get(linear_pred_id) or [])

    def _empty_fallthrough_tail(self, start_id: Optional[str], target_id: Optional[str]) -> Optional[str]:
        if start_id is None or target_id is None:
            return None
        current_id = start_id
        previous_id: Optional[str] = None
        seen: set[str] = set()
        while current_id not in seen:
            if current_id == target_id:
                return previous_id
            seen.add(current_id)
            block = self.cfg.by_id.get(current_id)
            if block is None or block.statements or block.block_params:
                return None
            if block.terminator.get("kind") != "fallthrough" or len(block.successors) != 1:
                return None
            successor = block.successors[0]
            if self.cfg.index_by_id.get(successor, -1) <= self.cfg.index_by_id.get(current_id, -1):
                return None
            previous_id = current_id
            current_id = successor
        return None

    def _empty_fallthrough_reaches(self, start_id: Optional[str], target_id: Optional[str], *, source_id: Optional[str] = None) -> bool:
        tail_id = self._empty_fallthrough_tail(start_id, target_id)
        if tail_id is None:
            return start_id == target_id
        if source_id is not None and target_id is not None:
            return self._incoming_args_equivalent(source_id, tail_id, target_id)
        return True

    def _empty_forward_skip(self, pred_id: str, target_id: str) -> bool:
        pred_idx = self.cfg.index_by_id.get(pred_id, -1)
        target_idx = self.cfg.index_by_id.get(target_id, -1)
        if pred_idx < 0 or target_idx < 0 or pred_idx >= target_idx:
            return False
        if target_idx == pred_idx + 1:
            return True
        next_id = self.blocks[pred_idx + 1].id if pred_idx + 1 < len(self.blocks) else None
        tail_id = self._empty_fallthrough_tail(next_id, target_id)
        if tail_id is None:
            return False
        if self._incoming_args_equivalent(pred_id, tail_id, target_id):
            return True
        merge = self._branch_param_merge(self.cfg.by_id.get(pred_id), next_id)
        return bool(merge and merge.get("target_id") == target_id)

    def _branch_empty_convergence(self, block: HIRBlock) -> Optional[str]:
        if block.terminator.get("kind") != "branch":
            return None
        branch_target, fallthrough_target = _edge_targets(block, self.cfg.index_by_id)
        if branch_target is None or fallthrough_target is None:
            return None
        if self._empty_fallthrough_reaches(fallthrough_target, branch_target, source_id=block.id):
            return branch_target
        if self._empty_fallthrough_reaches(branch_target, fallthrough_target, source_id=block.id):
            return fallthrough_target
        return None

    def _branch_param_merge(self, block: Optional[HIRBlock], next_id: Optional[str]) -> Optional[dict[str, Any]]:
        if block is None or block.terminator.get("kind") != "branch" or next_id is None:
            return None
        branch_target, fallthrough_target = _edge_targets(block, self.cfg.index_by_id)
        if not branch_target or not fallthrough_target or fallthrough_target != next_id or branch_target == next_id:
            return None
        target = self.cfg.by_id.get(branch_target)
        if target is None:
            return None
        fallthrough_tail = self._empty_fallthrough_tail(fallthrough_target, branch_target)
        if fallthrough_tail is None:
            return None
        if not target.block_params:
            return {"target_id": branch_target, "fallthrough_tail": fallthrough_tail, "nodes": [], "sources": {block.id, fallthrough_tail}}
        default_args = list(target.incoming_args.get(fallthrough_tail) or [])
        branch_args = list(target.incoming_args.get(block.id) or [])
        if len(default_args) != len(target.block_params) or len(branch_args) != len(target.block_params):
            return None
        if default_args == branch_args:
            nodes: list[dict[str, Any]] = []
        else:
            condition = block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}"
            nodes = [
                _assignment_node(param, f"select({condition}, {branch_arg}, {default_arg})")
                for param, branch_arg, default_arg in zip(target.block_params, branch_args, default_args)
                if param != branch_arg or param != default_arg
            ]
        return {
            "target_id": branch_target,
            "fallthrough_tail": fallthrough_tail,
            "nodes": nodes,
            "sources": {block.id, fallthrough_tail},
        }

    def _mark_param_merge(self, merge: Optional[dict[str, Any]]) -> None:
        if not merge:
            return
        target_id = merge.get("target_id")
        if not target_id:
            return
        sources = set(merge.get("sources") or set())
        if sources:
            self.settled_preds_by_block[str(target_id)].update(str(source) for source in sources)
        tail = merge.get("fallthrough_tail")
        if tail:
            self.handled_param_predecessors_by_block[str(target_id)].add(str(tail))

    def _transparent_fallthrough_target(self, target_id: Optional[str]) -> Optional[str]:
        seen: set[str] = set()
        current = target_id
        while current is not None and current not in seen:
            seen.add(current)
            block = self.cfg.by_id.get(current)
            if block is None:
                return current
            if block.statements or block.block_params or block.terminator.get("kind") != "fallthrough":
                return current
            nxt = block.fallthrough_target or (block.successors[0] if block.successors else None)
            if nxt is None:
                return current
            current = nxt
        return current

    def _alias_for_source(self, target_id: Optional[str], source: Optional[HIRBlock], aliases: dict[str, Any]) -> Any:
        if target_id is None:
            return None
        alias = aliases.get(target_id)
        if not isinstance(alias, dict):
            return alias
        by_source = alias.get("prelude_by_source") or {}
        if by_source:
            source_key = source.id if source is not None else None
            if source_key not in by_source:
                return None
        return alias

    def _jump_nodes(
        self,
        source: Optional[HIRBlock],
        target_id: Optional[str],
        aliases: dict[str, Any],
        *,
        source_specific_alias: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        if target_id is None:
            return [], 0
        target = self.cfg.by_id.get(target_id)
        alias = self._alias_for_source(target_id, source, aliases) if source_specific_alias else aliases.get(target_id)
        if alias is None:
            if target_id in self.cfg.by_id:
                self.needed_label_ids.add(target_id)
            return [{"kind": "goto", "target": _target_call(target, source) or target_id}], 1
        if isinstance(alias, dict):
            source_key = source.id if source is not None else None
            by_source = alias.get("prelude_by_source") or {}
            prelude = list(by_source.get(source_key) or [])
            jump = alias.get("jump")
            return prelude + ([jump] if isinstance(jump, dict) else []), int(isinstance(jump, dict) and jump.get("kind") == "goto")
        node = _statement_node(alias)
        return ([node] if node.get("kind") != "empty" else []), int(node.get("kind") == "goto")

    def _terminator_nodes(
        self,
        block: HIRBlock,
        next_id: Optional[str],
        aliases: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        kind = block.terminator.get("kind")
        if kind == "return":
            return ([_statement_node(block.terminator.get("text") or "return")], 0)
        if kind == "stop":
            return ([{"kind": "stop", "source": str(block.terminator.get("text") or "stop").strip()}], 0)
        if kind == "fallthrough":
            target_id = block.fallthrough_target or (block.successors[0] if block.successors else None)
            if target_id is None:
                return [], 0
            if self._alias_for_source(target_id, block, aliases) is None:
                if target_id == next_id:
                    return [], 0
                if next_id is not None and self._transparent_fallthrough_target(next_id) == target_id:
                    return [], 0
            return self._jump_nodes(block, target_id, aliases, source_specific_alias=True)
        if kind == "branch":
            cond = _expr_node(block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}")
            branch, fallthrough = _edge_targets(block, self.cfg.index_by_id)
            if branch == fallthrough:
                if branch is None or branch == next_id:
                    return [], 0
                return self._jump_nodes(block, branch, aliases)
            branch_alias = self._alias_for_source(branch, block, aliases)
            fallthrough_alias = self._alias_for_source(fallthrough, block, aliases)
            if branch_alias is None and fallthrough_alias is None and self._branch_empty_convergence(block) is not None:
                return [], 0
            if fallthrough == next_id and branch_alias is None and fallthrough_alias is None:
                merge = self._branch_param_merge(block, next_id)
                if merge is not None:
                    self._mark_param_merge(merge)
                    return list(merge.get("nodes") or []), 0
            if branch_alias is None and fallthrough_alias is None:
                branch_final = self._transparent_fallthrough_target(branch)
                fallthrough_final = self._transparent_fallthrough_target(fallthrough)
                next_final = self._transparent_fallthrough_target(next_id)
                if branch_final is not None and branch_final == fallthrough_final and next_final == branch_final:
                    return [], 0
            nodes: list[dict[str, Any]] = []
            goto_count = 0
            if branch == next_id and fallthrough is not None:
                edge_nodes, edge_gotos = self._jump_nodes(block, fallthrough, aliases)
                conditional = _conditional_node(cond, edge_nodes, negate=True)
                if conditional is not None:
                    nodes.append(conditional)
                return nodes, edge_gotos
            if fallthrough == next_id and branch is not None:
                edge_nodes, edge_gotos = self._jump_nodes(block, branch, aliases)
                conditional = _conditional_node(cond, edge_nodes)
                if conditional is not None:
                    nodes.append(conditional)
                return nodes, edge_gotos
            if branch is not None:
                edge_nodes, edge_gotos = self._jump_nodes(block, branch, aliases)
                conditional = _conditional_node(cond, edge_nodes)
                if conditional is not None:
                    nodes.append(conditional)
                goto_count += edge_gotos
            if fallthrough is not None:
                edge_nodes, edge_gotos = self._jump_nodes(block, fallthrough, aliases)
                nodes.extend(edge_nodes)
                goto_count += edge_gotos
            return nodes, goto_count
        if not block.successors:
            return [], 0
        return self._jump_nodes(block, block.successors[0], aliases)

    def _block_region(
        self,
        block: HIRBlock,
        next_id: Optional[str],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
    ) -> tuple[dict[str, Any], Optional[str]]:
        label = _format_block_label(block) if self._label_needed(block, hidden_labels, previous_source_id) else None
        handled_param_pred = previous_source_id in self.handled_param_predecessors_by_block.get(block.id, set())
        prelabel = _incoming_assignments(block, previous_source_id) if previous_source_id in block.predecessors and block.id not in aliases and not handled_param_pred else []
        statements = [_statement_node(stmt) for stmt in block.statements]
        term_nodes, goto_count = self._terminator_nodes(block, next_id, aliases)
        if len(term_nodes) == 1 and statements and _render_statement(statements[-1]) == _render_statement(term_nodes[0]):
            if term_nodes[0].get("kind") == "goto" and goto_count:
                goto_count = 0
            term_nodes = []
        label_count = 1 if label and block.id not in self.needed_label_ids else 0
        self.residual_labels += label_count
        self.residual_gotos += goto_count
        region = {
            "kind": "block",
            "block_id": block.id,
            "label": label,
            "block_params": list(block.block_params),
            "prelabel": prelabel,
            "body": statements,
            "terminator": {
                "kind": block.terminator.get("kind"),
                "condition": _expr_node(block.terminator.get("condition")),
                "rendered": term_nodes,
                "successors": list(block.successors),
                "branch_target": block.branch_target,
                "fallthrough_target": block.fallthrough_target,
            },
        }
        if label_count or goto_count:
            self.fallback_regions += 1
            self.fallback_blocks += 1
            region = {"kind": "goto_region", "label_count": label_count, "goto_count": goto_count, "blocks": [region]}
        return region, block.id

    def _edge_alias_to_join(self, join_id: str, source_ids: set[str]) -> dict[str, Any]:
        join_block = self.cfg.by_id.get(join_id)
        return {
            "jump": None,
            "prelude_by_source": {
                source_id: _incoming_assignments(join_block, source_id)
                for source_id in source_ids
            },
        }

    def _arm_ids(self, start_id: Optional[str], join_id: str, stop: int, allowed: Optional[set[str]]) -> Optional[set[str]]:
        if start_id is None:
            return None
        if start_id == join_id:
            return set()
        if start_id not in self.cfg.index_by_id or join_id not in self.cfg.index_by_id:
            return None
        start_idx = self.cfg.index_by_id[start_id]
        join_idx = self.cfg.index_by_id[join_id]
        if start_idx <= 0 or start_idx >= join_idx or join_idx > stop:
            return None
        ids: set[str] = set()
        stack = [start_id]
        while stack:
            node = stack.pop()
            if node == join_id:
                continue
            node_idx = self.cfg.index_by_id.get(node)
            if node_idx is None or node_idx <= start_idx - 1 or node_idx >= join_idx:
                return None
            if allowed is not None and node not in allowed:
                return None
            if node in ids:
                continue
            ids.add(node)
            stack.extend(succ for succ in self.cfg.succs.get(node, []) if succ != join_id)
        return ids

    def _has_external_entry(self, ids: set[str], allowed_preds: set[str]) -> bool:
        for block_id in ids:
            for pred in self.cfg.preds.get(block_id, []):
                if pred not in ids and pred not in allowed_preds:
                    return True
        return False

    def _arm_exit_sources(self, ids: set[str], direct_source: str, join_id: str) -> set[str]:
        if not ids:
            return {direct_source}
        return {block_id for block_id in ids if join_id in self.cfg.succs.get(block_id, [])}

    def _edge_statement_block(self, block_id: str, statements: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not statements:
            return None
        return {
            "kind": "block",
            "block_id": block_id,
            "label": None,
            "block_params": [],
            "prelabel": [],
            "body": statements,
            "terminator": {"kind": "fallthrough", "condition": None, "rendered": [], "successors": [], "branch_target": None, "fallthrough_target": None},
        }

    def _decision_edge_node(
        self,
        source: HIRBlock,
        target_id: Optional[str],
        tree_ids: set[str],
        aliases: dict[str, Any],
        next_after_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if target_id is None:
            return {"kind": "exit", "target": None, "statements": []}
        if target_id in tree_ids:
            return {"kind": "internal", "target": target_id, "statements": []}
        if target_id == next_after_id:
            return {"kind": "fallthrough_exit", "target": target_id, "statements": []}
        if self._alias_for_source(target_id, source, aliases) is not None:
            statements, _ = self._jump_nodes(source, target_id, aliases)
            return {"kind": "structured_jump", "target": target_id, "statements": statements}
        return None

    def _same_incoming_args_for_sources(self, target_id: str, sources: set[str]) -> Optional[list[str]]:
        target = self.cfg.by_id.get(target_id)
        if target is None or not target.block_params:
            return []
        selected: Optional[list[str]] = None
        for source_id in sources:
            args = list(target.incoming_args.get(source_id) or [])
            if len(args) != len(target.block_params):
                return None
            if selected is None:
                selected = args
            elif selected != args:
                return None
        return selected or []

    def _try_threaded_decision_tree(
        self,
        idx: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
    ) -> Optional[tuple[dict[str, Any], int]]:
        if idx in self.cfg.switches:
            return None
        run: list[HIRBlock] = []
        i = idx
        while i < min(stop, len(self.blocks)):
            block = self.blocks[i]
            if allowed is not None and block.id not in allowed:
                break
            if block.terminator.get("kind") != "branch":
                break
            run.append(block)
            i += 1
        if len(run) < 4:
            return None
        tree_ids = {block.id for block in run}
        next_after_id = self.blocks[i].id if i < min(stop, len(self.blocks)) else None

        complex_internal_edge = False
        backward_internal_edge = False
        for block in run:
            for succ in self.cfg.succs.get(block.id, []):
                if succ not in tree_ids:
                    continue
                succ_idx = self.cfg.index_by_id.get(succ, 10**9)
                if succ_idx != block.index + 1:
                    complex_internal_edge = True
                if succ_idx <= block.index:
                    backward_internal_edge = True
        if not complex_internal_edge:
            return None
        if not backward_internal_edge and len(run) < 8:
            return None

        for offset, block in enumerate(run):
            for pred_id in self.cfg.preds.get(block.id, []):
                if pred_id in tree_ids:
                    continue
                pred_idx = self.cfg.index_by_id.get(pred_id)
                allowed_entry = offset == 0 and (pred_id == previous_source_id or (pred_idx is not None and pred_idx + 1 == block.index))
                if not allowed_entry:
                    return None

        exit_sources_by_target: dict[str, set[str]] = defaultdict(set)
        nodes: list[dict[str, Any]] = []
        for block in run:
            branch, fallthrough = _edge_targets(block, self.cfg.index_by_id)
            true_edge = self._decision_edge_node(block, branch, tree_ids, aliases, next_after_id)
            false_edge = self._decision_edge_node(block, fallthrough, tree_ids, aliases, next_after_id)
            if true_edge is None or false_edge is None:
                return None
            for target in (branch, fallthrough):
                if target is not None and target not in tree_ids and target not in aliases:
                    exit_sources_by_target[target].add(block.id)
            nodes.append(
                {
                    "kind": "decision_node",
                    "block_id": block.id,
                    "block_params": list(block.block_params),
                    "prelabel": _incoming_assignments(block, previous_source_id) if block.id == run[0].id and previous_source_id in block.predecessors else [],
                    "body": [_statement_node(stmt) for stmt in block.statements],
                    "condition": _expr_node(block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}"),
                    "true_edge": true_edge,
                    "false_edge": false_edge,
                }
            )

        exit_param_merge: list[dict[str, Any]] = []
        for target_id, sources in exit_sources_by_target.items():
            if target_id != next_after_id:
                return None
            args = self._same_incoming_args_for_sources(target_id, sources)
            if args is None:
                return None
            target = self.cfg.by_id.get(target_id)
            if target is not None and target.block_params:
                exit_param_merge.extend(
                    _assignment_node(param, arg)
                    for param, arg in zip(target.block_params, args)
                    if param != arg
                )
                self.handled_param_predecessors_by_block[target_id].update(sources)
            self.settled_preds_by_block[target_id].update(sources)

        label = _format_block_label(run[0]) if self._label_needed(run[0], hidden_labels, previous_source_id) else None
        self.constructs["threaded_decision_tree"] += 1
        region = {
            "kind": "threaded_decision_tree",
            "label": label,
            "entry_block": run[0].id,
            "exit_block": next_after_id,
            "node_count": len(nodes),
            "nodes": nodes,
            "exit_param_merge": exit_param_merge,
        }
        return region, i

    def _try_switch_like(
        self,
        idx: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
    ) -> Optional[tuple[dict[str, Any], int]]:
        info = self.cfg.switches.get(idx)
        if info is None:
            return None
        end_index = int(info.get("end_index", idx))
        if end_index <= idx or end_index > stop:
            return None
        dispatch_blocks = list(info.get("blocks") or [])
        if not dispatch_blocks:
            return None
        dispatch_ids = {block.id for block in dispatch_blocks}
        if allowed is not None and not dispatch_ids.issubset(allowed):
            return None
        join_id = info.get("join")
        if join_id is not None:
            self.settled_preds_by_block[str(join_id)].update(dispatch_ids)
        label = _format_block_label(dispatch_blocks[0]) if self._label_needed(dispatch_blocks[0], hidden_labels, previous_source_id) else None
        cases: list[dict[str, Any]] = []
        for case_block in dispatch_blocks:
            case_target, _ = _edge_targets(case_block, self.cfg.index_by_id)
            cases.append(
                {
                    "kind": "case",
                    "dispatch_block": case_block.id,
                    "condition": _expr_node(case_block.terminator.get("condition")),
                    "target": case_target,
                    "body": {"kind": "sequence", "regions": []},
                }
            )
        self.constructs["switch_like"] += 1
        region = {
            "kind": "switch_like",
            "label": label,
            "join_block": join_id,
            "cases": cases,
            "default": "dispatch_fallthrough",
        }
        return region, end_index

    def _try_if(
        self,
        idx: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
    ) -> Optional[tuple[dict[str, Any], int]]:
        block = self.blocks[idx]
        if block.terminator.get("kind") != "branch":
            return None
        join_id = self.cfg.ipdom.get(block.id)
        if join_id is None or join_id not in self.cfg.index_by_id:
            return None
        join_idx = self.cfg.index_by_id[join_id]
        if join_idx <= idx or join_idx > stop:
            return None
        branch, fallthrough = _edge_targets(block, self.cfg.index_by_id)
        if branch is None or fallthrough is None:
            return None
        branch_ids = self._arm_ids(branch, join_id, stop, allowed)
        fall_ids = self._arm_ids(fallthrough, join_id, stop, allowed)
        if branch_ids is None or fall_ids is None or not branch_ids.isdisjoint(fall_ids):
            return None
        span_ids = {candidate.id for candidate in self.blocks[idx + 1 : join_idx] if allowed is None or candidate.id in allowed}
        if (branch_ids | fall_ids) != span_ids or not span_ids:
            return None
        if self._has_external_entry(branch_ids, {block.id}) or self._has_external_entry(fall_ids, {block.id}):
            return None

        join_sources = self._arm_exit_sources(branch_ids, block.id, join_id) | self._arm_exit_sources(fall_ids, block.id, join_id)
        join_alias = self._edge_alias_to_join(join_id, join_sources)
        self.settled_preds_by_block[join_id].update(join_sources)
        arm_aliases = dict(aliases)
        arm_aliases[join_id] = join_alias
        cond = _expr_node(block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}")
        label = _format_block_label(block) if self._label_needed(block, hidden_labels, previous_source_id) else None
        prologue = _incoming_assignments(block, previous_source_id) if previous_source_id in block.predecessors else []
        prologue.extend(_statement_node(stmt) for stmt in block.statements)

        def build_arm(ids: set[str], direct_edge: bool, suffix: str) -> list[dict[str, Any]]:
            if ids:
                start_index = min(self.cfg.index_by_id[item] for item in ids)
                return self._range(start_index, join_idx, ids, arm_aliases, set(), block.id, set())
            if direct_edge:
                assignments = _incoming_assignments(self.cfg.by_id.get(join_id), block.id)
                edge_block = self._edge_statement_block(f"edge_{block.id}_{join_id}_{suffix}", assignments)
                return [edge_block] if edge_block else []
            return []

        then_ids = branch_ids
        else_ids = fall_ids
        rendered_cond = cond
        branch_direct = branch == join_id
        fall_direct = fallthrough == join_id
        then_regions = build_arm(then_ids, branch_direct, "then")
        else_regions = build_arm(else_ids, fall_direct, "else")
        if not then_regions and not else_regions:
            if prologue:
                region = {
                    "kind": "block",
                    "block_id": block.id,
                    "label": label,
                    "block_params": list(block.block_params),
                    "prelabel": [],
                    "body": prologue,
                    "terminator": {"kind": "fallthrough", "condition": None, "rendered": [], "successors": [], "branch_target": None, "fallthrough_target": None},
                }
                return region, join_idx
            return {"kind": "sequence", "regions": []}, join_idx
        kind = "if_else" if else_regions else "if"
        self.constructs[kind] += 1
        region = {
            "kind": kind,
            "header_block": block.id,
            "label": label,
            "block_params": list(block.block_params),
            "condition": rendered_cond,
            "prologue": prologue,
            "then": {"kind": "sequence", "regions": then_regions},
            "else": {"kind": "sequence", "regions": else_regions} if else_regions else None,
            "resume_block": join_id,
            "resume_param_merge": [],
        }
        return region, join_idx

    def _loop_aliases(self, loop: dict[str, Any], exit_ids: Optional[list[str]] = None) -> dict[str, Any]:
        nodes = set(loop.get("nodes") or set())
        header_id = str(loop.get("header"))
        aliases: dict[str, Any] = {}
        for target_id, sources in _loopback_edges(loop, self.cfg).items():
            target = self.cfg.by_id.get(target_id)
            jump = {"kind": "continue", "target": None if target_id == header_id else target_id}
            if target_id != header_id:
                self.needed_label_ids.add(target_id)
            aliases[target_id] = {
                "jump": jump,
                "prelude_by_source": {source_id: _incoming_assignments(target, source_id) for source_id in sources},
            }
        exits = list(exit_ids) if exit_ids is not None else _external_successors(loop, self.cfg)
        idx0, idx1 = loop.get("index_range") or (None, None)
        next_after = self.blocks[idx1 + 1].id if isinstance(idx1, int) and idx1 + 1 < len(self.blocks) else None
        if next_after in exits:
            exits.remove(next_after)
            exits.insert(0, next_after)
        for pos, exit_id in enumerate(exits):
            exit_block = self.cfg.by_id.get(exit_id)
            sources = [source_id for source_id in nodes if exit_id in self.cfg.succs.get(source_id, [])]
            jump = {"kind": "break", "target": None if pos == 0 else exit_id}
            if pos != 0:
                self.needed_label_ids.add(exit_id)
            aliases[exit_id] = {
                "jump": jump,
                "prelude_by_source": {source_id: _incoming_assignments(exit_block, source_id) for source_id in sources},
            }
        return aliases

    def _settle_alias_predecessors(self, alias_map: dict[str, Any]) -> None:
        for target_id, alias in alias_map.items():
            if isinstance(alias, dict):
                self.settled_preds_by_block[target_id].update((alias.get("prelude_by_source") or {}).keys())

    def _try_self_loop(
        self,
        idx: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
    ) -> Optional[tuple[dict[str, Any], int]]:
        block = self.blocks[idx]
        loop = self.cfg.loops.get(block.id)
        if not loop or not loop.get("safe"):
            return None
        nodes = set(loop.get("nodes") or set())
        if nodes != {block.id}:
            return None
        idx0, idx1 = loop.get("index_range") or (idx, idx)
        if idx0 != idx or idx1 != idx or idx + 1 > stop or (allowed is not None and block.id not in allowed):
            return None
        if loop.get("body_succ") != block.id or loop.get("exit_succ") is None:
            return None
        exit_succ = str(loop.get("exit_succ"))
        # Keep this rule intentionally conservative. Parameterized self-loops need a
        # phi-update model; plain self-loops can be lifted to a real do/while node.
        if block.block_params:
            return None
        if _incoming_assignments(block, block.id):
            return None
        if _incoming_assignments(self.cfg.by_id.get(exit_succ), block.id):
            return None
        cond = _expr_node(block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}")
        label = _format_block_label(block) if self._label_needed(block, hidden_labels, previous_source_id) else None
        preheader = _incoming_assignments(block, previous_source_id) if previous_source_id in block.predecessors else []
        self.constructs["do_while"] += 1
        self.loop_headers += 1
        self.structured_loopbacks += _loopback_source_count(loop, self.cfg)
        region = {
            "kind": "do_while",
            "header_block": block.id,
            "label": label,
            "block_params": [],
            "preheader": preheader,
            "body": [_statement_node(stmt) for stmt in block.statements],
            "condition": cond,
            "loopback": [],
            "continue_target": block.id,
            "break_target": exit_succ,
        }
        return region, idx + 1

    def _try_loop(
        self,
        idx: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
    ) -> Optional[tuple[dict[str, Any], int]]:
        block = self.blocks[idx]
        loop = self.cfg.loops.get(block.id)
        if not loop or not loop.get("safe"):
            return None
        idx0, idx1 = loop.get("index_range") or (idx, idx)
        nodes = set(loop.get("nodes") or set())
        if idx0 != idx or idx1 >= stop or (allowed is not None and not nodes.issubset(allowed)):
            return None
        branch, fallthrough = _edge_targets(block, self.cfg.index_by_id)
        body_succ = loop.get("body_succ")
        exit_succ = loop.get("exit_succ")
        if body_succ not in self.cfg.index_by_id or exit_succ is None:
            return None
        cond = _expr_node(block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}")
        body_cond = cond if branch == body_succ else {"kind": "not", "expr": cond} if fallthrough == body_succ else cond
        loop_aliases = dict(aliases)
        local_aliases = self._loop_aliases(loop)
        self._settle_alias_predecessors(local_aliases)
        loop_aliases.update(local_aliases)
        body_ids = nodes - {block.id}
        if not body_ids:
            return None
        body_start = min(self.cfg.index_by_id[item] for item in body_ids)
        body = self._range(body_start, idx1 + 1, body_ids, loop_aliases, set(), block.id, set())
        preheader = _incoming_assignments(block, previous_source_id) if previous_source_id in block.predecessors else []
        header_exit_assignments = _incoming_assignments(self.cfg.by_id.get(exit_succ), block.id)
        guarded = bool(block.statements or header_exit_assignments)
        self.constructs["while"] += 1
        self.loop_headers += 1
        self.structured_loopbacks += _loopback_source_count(loop, self.cfg)
        label = _format_block_label(block) if self._label_needed(block, hidden_labels, previous_source_id) else None
        region = {
            "kind": "while",
            "header_block": block.id,
            "label": label,
            "block_params": list(block.block_params),
            "preheader": preheader,
            "condition": _expr_node("true") if guarded else body_cond,
            "prologue": [_statement_node(stmt) for stmt in block.statements],
            "guard_condition": body_cond,
            "guard_exit": header_exit_assignments,
            "rendering": "guarded_loop" if guarded else "direct_while",
            "body": {"kind": "sequence", "regions": body},
            "continue_target": block.id,
            "break_target": exit_succ,
            "break_targets": _external_successors(loop, self.cfg),
        }
        return region, idx1 + 1

    def _try_unconditional_loop(
        self,
        idx: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
        suppressed_loops: set[str],
    ) -> Optional[tuple[dict[str, Any], int]]:
        block = self.blocks[idx]
        loop = self.cfg.loops.get(block.id)
        if not loop or loop.get("safe") or block.id in suppressed_loops or not loop.get("contiguous"):
            return None
        idx0, idx1 = loop.get("index_range") or (idx, idx)
        nodes = set(loop.get("nodes") or set())
        if idx0 != idx or idx1 >= stop or (allowed is not None and not nodes.issubset(allowed)):
            return None
        exits = [succ for succ in _external_successors(loop, self.cfg) if self.cfg.index_by_id.get(succ, -1) > idx1]
        if not exits or not _loopback_edges(loop, self.cfg):
            return None
        loop_aliases = dict(aliases)
        local_aliases = self._loop_aliases(loop, exits)
        self._settle_alias_predecessors(local_aliases)
        loop_aliases.update(local_aliases)
        body = self._range(idx, idx1 + 1, nodes, loop_aliases, hidden_labels | {block.id}, None, suppressed_loops | {block.id})
        self.constructs["while"] += 1
        self.unconditional_loops += 1
        self.structured_loopbacks += _loopback_source_count(loop, self.cfg)
        label = _format_block_label(block) if self._label_needed(block, hidden_labels, previous_source_id) else None
        region = {
            "kind": "while",
            "header_block": block.id,
            "label": label,
            "block_params": list(block.block_params),
            "preheader": _incoming_assignments(block, previous_source_id) if previous_source_id in block.predecessors else [],
            "condition": _expr_node("true"),
            "prologue": [],
            "guard_exit": [],
            "rendering": "unconditional_loop",
            "body": {"kind": "sequence", "regions": body},
            "continue_target": block.id,
            "break_target": exits[0] if exits else None,
            "break_targets": exits,
        }
        return region, idx1 + 1

    def _range(
        self,
        start: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
        suppressed_loops: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        regions: list[dict[str, Any]] = []
        i = start
        prev = previous_source_id
        suppressed_loops = suppressed_loops or set()
        limit = max(1, (min(stop, len(self.blocks)) - start + 1) * 8)
        steps = 0
        while i < min(stop, len(self.blocks)):
            steps += 1
            if steps > limit:
                raise RuntimeError(f"AST range walk did not converge at block index {i}")
            block = self.blocks[i]
            if allowed is not None and block.id not in allowed:
                i += 1
                prev = None
                continue
            self_loop = None if block.id in suppressed_loops else self._try_self_loop(i, stop, allowed, aliases, hidden_labels, prev)
            if self_loop is not None:
                region, next_i = self_loop
                if next_i > i:
                    regions.append(region)
                    i = next_i
                    prev = None
                    continue
            loop = None if block.id in suppressed_loops else self._try_loop(i, stop, allowed, aliases, hidden_labels, prev)
            if loop is not None:
                region, next_i = loop
                if next_i > i:
                    regions.append(region)
                    i = next_i
                    prev = None
                    continue
            switch = self._try_switch_like(i, stop, allowed, aliases, hidden_labels, prev)
            if switch is not None:
                region, next_i = switch
                if next_i > i:
                    regions.append(region)
                    i = next_i
                    prev = None
                    continue
            threaded = self._try_threaded_decision_tree(i, stop, allowed, aliases, hidden_labels, prev)
            if threaded is not None:
                region, next_i = threaded
                if next_i > i:
                    regions.append(region)
                    i = next_i
                    prev = None
                    continue
            uncond = self._try_unconditional_loop(i, stop, allowed, aliases, hidden_labels, prev, suppressed_loops)
            if uncond is not None:
                region, next_i = uncond
                if next_i > i:
                    regions.append(region)
                    i = next_i
                    prev = None
                    continue
            branch = self._try_if(i, stop, allowed, aliases, hidden_labels, prev)
            if branch is not None:
                region, next_i = branch
                if next_i > i:
                    regions.append(region)
                    i = next_i
                    prev = None
                    continue
            next_id = self._next_allowed_id(i, stop, allowed)
            region, prev = self._block_region(block, next_id, aliases, hidden_labels, prev)
            regions.append(region)
            i += 1
        return regions


def _render_region(region: Optional[dict[str, Any]], indent: int = 0) -> list[str]:
    if not region:
        return []
    prefix = "    " * indent
    kind = region.get("kind")
    if kind == "sequence":
        lines: list[str] = []
        for child in region.get("regions") or []:
            lines.extend(_render_region(child, indent))
        return lines
    if kind == "goto_region":
        lines: list[str] = []
        for child in region.get("blocks") or []:
            lines.extend(_render_region(child, indent))
        return lines
    if kind == "block":
        lines = _render_statement_list(region.get("prelabel") or [], indent)
        label = region.get("label")
        stmt_indent = indent
        if label:
            lines.append(prefix + f"{label}:")
            stmt_indent += 1
        lines.extend(_render_statement_list(region.get("body") or [], stmt_indent))
        lines.extend(_render_statement_list((region.get("terminator") or {}).get("rendered") or [], stmt_indent))
        return lines
    if kind in {"if", "if_else"}:
        lines: list[str] = []
        label = region.get("label")
        stmt_indent = indent
        if label:
            lines.append(prefix + f"{label}:")
            stmt_indent += 1
        sp = "    " * stmt_indent
        lines.extend(_render_statement_list(region.get("prologue") or [], stmt_indent))
        lines.append(sp + f"if ({_render_expr(region.get('condition'))}) {{")
        lines.extend(_render_region(region.get("then"), stmt_indent + 1))
        if kind == "if_else" and region.get("else"):
            lines.append(sp + "} else {")
            lines.extend(_render_region(region.get("else"), stmt_indent + 1))
        lines.append(sp + "}")
        lines.extend(_render_statement_list(region.get("resume_param_merge") or [], stmt_indent))
        return lines
    if kind == "do_while":
        lines = _render_statement_list(region.get("preheader") or [], indent)
        label = region.get("label")
        loop_indent = indent
        if label:
            lines.append(prefix + f"{label}:")
            loop_indent += 1
        lp = "    " * loop_indent
        lines.append(lp + "do {")
        inner = loop_indent + 1
        lines.extend(_render_statement_list(region.get("body") or [], inner))
        lines.extend(_render_statement_list(region.get("loopback") or [], inner))
        lines.append(lp + f"}} while ({_render_expr(region.get('condition')) or 'true'})")
        return lines
    if kind == "while":
        lines = _render_statement_list(region.get("preheader") or [], indent)
        label = region.get("label")
        loop_indent = indent
        if label:
            lines.append(prefix + f"{label}:")
            loop_indent += 1
        lp = "    " * loop_indent
        lines.append(lp + f"while ({_render_expr(region.get('condition')) or 'true'}) {{")
        inner = loop_indent + 1
        ip = "    " * inner
        lines.extend(_render_statement_list(region.get("prologue") or [], inner))
        if region.get("rendering") == "guarded_loop":
            guard = _render_expr(region.get("guard_condition")) or "true"
            lines.append(ip + f"if (!({guard})) {{")
            lines.extend(_render_statement_list(region.get("guard_exit") or [], inner + 1))
            lines.append("    " * (inner + 1) + "break")
            lines.append(ip + "}")
        lines.extend(_render_region(region.get("body"), inner))
        lines.append(lp + "}")
        return lines
    if kind == "threaded_decision_tree":
        lines: list[str] = []
        label = region.get("label")
        tree_indent = indent
        if label:
            lines.append(prefix + f"{label}:")
            tree_indent += 1
        tp = "    " * tree_indent
        header = f"threaded_decision_tree {region.get('entry_block')}"
        if region.get("exit_block"):
            header += f" -> {region.get('exit_block')}"
        lines.append(tp + header + " {")

        def edge_text(edge: dict[str, Any]) -> str:
            target = edge.get("target")
            edge_kind = edge.get("kind")
            rendered: list[str] = []
            for stmt in edge.get("statements") or []:
                rendered.extend(_render_statement_lines(stmt))
            suffix = ""
            if rendered:
                suffix = " { " + "; ".join(rendered) + " }"
            if edge_kind == "internal":
                return f"node {target}"
            if edge_kind == "fallthrough_exit":
                return f"exit {target}" if target else "exit"
            if edge_kind == "structured_jump":
                return f"structured {target}" + suffix
            return str(target or "exit") + suffix

        for node in region.get("nodes") or []:
            np = "    " * (tree_indent + 1)
            lines.append(np + f"node {node.get('block_id')} {{")
            body_indent = tree_indent + 2
            for stmt in node.get("prelabel") or []:
                lines.extend("    " * body_indent + line for line in _render_statement_lines(stmt))
            for stmt in node.get("body") or []:
                lines.extend("    " * body_indent + line for line in _render_statement_lines(stmt))
            cond = _render_expr(node.get("condition")) or "<cond>"
            lines.append("    " * body_indent + f"if ({cond}) -> {edge_text(node.get('true_edge') or {})}")
            lines.append("    " * body_indent + f"else -> {edge_text(node.get('false_edge') or {})}")
            lines.append(np + "}")
        lines.append(tp + "}")
        lines.extend(_render_statement_list(region.get("exit_param_merge") or [], tree_indent))
        return lines
    if kind == "switch_like":
        lines: list[str] = []
        label = region.get("label")
        switch_indent = indent
        if label:
            lines.append(prefix + f"{label}:")
            switch_indent += 1
        sp = "    " * switch_indent
        lines.append(sp + "switch_like {")
        for case in region.get("cases") or []:
            cond = _render_expr(case.get("condition")) or "<cond>"
            target = case.get("target") or "<unresolved>"
            lines.append("    " * (switch_indent + 1) + f"when ({cond}) -> {target} {{")
            lines.extend(_render_region(case.get("body"), switch_indent + 2))
            lines.append("    " * (switch_indent + 1) + "}")
        lines.append("    " * (switch_indent + 1) + "default: { /* dispatch fallthrough */ }")
        lines.append(sp + "}")
        return lines
    return [prefix + f"/* unsupported AST region: {kind} */"]

def _render_function_text(name: str, entry_args: list[str], tree: dict[str, Any], include_text: bool) -> str:
    if not include_text:
        return ""
    header = f"function {name}({', '.join(entry_args)}) {{" if entry_args else f"function {name}() {{"
    body = _render_region(tree, 1)
    return "\n".join([header, *body, "}"]).rstrip()


def _coerce_function_hir(function_payload: dict[str, Any]) -> tuple[list[HIRBlock], str, dict[str, int], dict[str, Any], str, dict[str, Any], str]:
    normalized = function_payload.get("normalized_hir", {})
    raw_blocks = normalized.get("hir_blocks") or function_payload.get("core_hir", {}).get("hir_blocks") or []
    blocks = _rename_merge_params(_normalize_hir_blocks([_coerce_hir_block(raw) for raw in raw_blocks])) if raw_blocks else []
    name = str(function_payload.get("name") or "<function>")
    span = dict(function_payload.get("span") or {})
    body_selection = dict(function_payload.get("body_selection") or {})
    slice_mode = str(function_payload.get("slice_mode") or body_selection.get("slice_mode") or "")
    hir_summary = dict(function_payload.get("summary") or {})
    input_contract = str(function_payload.get("input_contract") or function_payload.get("contract_version") or function_payload.get("contract", {}).get("version") or "")
    return blocks, name, span, body_selection, slice_mode, hir_summary, input_contract


def build_function_ast_from_payload(function_payload: dict[str, Any], *, include_hir: bool = False, include_text: bool = True) -> ASTFunction:
    t0 = time.perf_counter()
    blocks, name, span, body_selection, slice_mode, hir_summary, input_contract = _coerce_function_hir(function_payload)
    entry_args = blocks[0].entry_stack if blocks else []
    builder = _SurfaceBuilder(blocks)
    region_tree, structured_meta = builder.build()
    ast_root = region_tree or {"kind": "sequence", "regions": []}
    shape_metrics = _collect_ast_shape_metrics(ast_root)
    semantic_validation = _validate_ast_semantics(ast_root, shape_metrics)
    ast_text = _render_function_text(name, list(entry_args), ast_root, include_text)
    trivial_body = _classify_trivial_function_body(ast_root, list(entry_args))
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
        "source_shape_score": max(0.0, 1.0 - fallback_block_ratio - (0.25 * ((residual_goto_count / block_count) if block_count else 0.0))),
        **shape_metrics,
        "semantic_body_kind": trivial_body.get("kind"),
        "semantic_trivial_body": bool(trivial_body.get("is_trivial")),
        "semantic_statement_count": trivial_body.get("semantic_statement_count"),
        "ignored_noop_statement_count": trivial_body.get("ignored_noop_statement_count", 0),
        "ast_semantic_error_count": int(semantic_validation.get("error_count", 0)),
        "ast_forced_label_count": 0,
    }
    diagnostics = {
        "structuring": {**structured_meta, "forced_label_count": 0, "forced_labels": []},
        "ast_shape": shape_metrics,
        "semantic_body": trivial_body,
        "semantic_validation": semantic_validation,
        "timings_ms": {"total": round((time.perf_counter() - t0) * 1000.0, 3)},
    }
    return ASTFunction(
        name=name,
        span=span,
        slice_mode=slice_mode,
        summary=summary,
        ast=ast_root,
        ast_text=ast_text,
        diagnostics=diagnostics,
        body_selection=body_selection,
        hir_summary=hir_summary,
        input_contract=input_contract,
        normalized_hir_blocks=blocks,
    )


def build_function_ast(
    mod: MBCModule,
    entry_or_name: FunctionEntry | str,
    *,
    include_canonical: bool = False,
    include_hir: bool = False,
    include_text: bool = True,
    validate: bool = False,
) -> ASTFunction:
    hir = build_function_hir(mod, entry_or_name, include_canonical=include_canonical, include_text=False, validate=validate, include_analysis_hints=False)
    return build_function_ast_from_payload(hir.to_dict(include_canonical=include_canonical, include_text=False), include_hir=include_hir, include_text=include_text)


def _merge_counter(counter: Counter[str], payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    for key, value in payload.items():
        try:
            counter[str(key)] += int(value)
        except Exception:
            continue


def _empty_module_summary(function_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    semantic_body_hist = Counter(str(summary.get("semantic_body_kind") or "unknown") for summary in function_summaries)
    region_hist: Counter[str] = Counter()
    ast_region_hist: Counter[str] = Counter()
    statement_hist: Counter[str] = Counter()
    expression_hist: Counter[str] = Counter()
    semantic_error_hist: Counter[str] = Counter()
    for summary in function_summaries:
        _merge_counter(region_hist, summary.get("region_kind_histogram", {}))
        _merge_counter(ast_region_hist, summary.get("ast_region_kind_histogram", {}))
        _merge_counter(statement_hist, summary.get("ast_statement_kind_histogram", {}))
        _merge_counter(expression_hist, summary.get("ast_expression_kind_histogram", {}))
    return {
        "function_count": len(function_summaries),
        "total_normalized_basic_blocks": sum(summary.get("normalized_basic_block_count", 0) for summary in function_summaries),
        "total_structured_blocks": sum(summary.get("structured_block_count", 0) for summary in function_summaries),
        "total_fallback_regions": sum(summary.get("fallback_region_count", 0) for summary in function_summaries),
        "total_fallback_blocks": sum(summary.get("fallback_block_count", 0) for summary in function_summaries),
        "total_residual_labels": sum(summary.get("residual_label_count", 0) for summary in function_summaries),
        "total_residual_gotos": sum(summary.get("residual_goto_count", 0) for summary in function_summaries),
        "total_loop_headers": sum(summary.get("loop_header_count", 0) for summary in function_summaries),
        "total_unconditional_loops": sum(summary.get("unconditional_loop_count", 0) for summary in function_summaries),
        "total_structured_loopbacks": sum(summary.get("structured_loopback_count", 0) for summary in function_summaries),
        "avg_source_shape_score": (sum(summary.get("source_shape_score", 1.0) for summary in function_summaries) / len(function_summaries)) if function_summaries else 1.0,
        "max_ast_depth": max((summary.get("ast_max_depth", 0) for summary in function_summaries), default=0),
        "total_ast_regions": sum(summary.get("ast_region_count", 0) for summary in function_summaries),
        "total_ast_statements": sum(summary.get("ast_statement_count", 0) for summary in function_summaries),
        "total_ast_expressions": sum(summary.get("ast_expression_count", 0) for summary in function_summaries),
        "total_explicit_cfg_regions": sum(summary.get("explicit_cfg_region_count", 0) for summary in function_summaries),
        "total_ast_parameterized_regions": sum(summary.get("ast_parameterized_region_count", 0) for summary in function_summaries),
        "total_ast_region_params": sum(summary.get("ast_region_param_count", 0) for summary in function_summaries),
        "total_ast_parameterized_blocks": sum(summary.get("ast_parameterized_block_count", 0) for summary in function_summaries),
        "total_ast_block_params": sum(summary.get("ast_block_param_count", 0) for summary in function_summaries),
        "total_ast_targeted_continues": sum(summary.get("ast_targeted_continue_count", 0) for summary in function_summaries),
        "total_ast_targeted_breaks": sum(summary.get("ast_targeted_break_count", 0) for summary in function_summaries),
        "total_ast_rendered_terminator_statements": sum(summary.get("ast_rendered_terminator_statement_count", 0) for summary in function_summaries),
        "total_ast_rendered_labels": sum(summary.get("ast_rendered_label_count", 0) for summary in function_summaries),
        "total_ast_goto_targets": sum(summary.get("ast_goto_target_count", 0) for summary in function_summaries),
        "total_ast_dangling_gotos": sum(summary.get("ast_dangling_goto_count", 0) for summary in function_summaries),
        "total_ast_jump_targets": sum(summary.get("ast_jump_target_count", 0) for summary in function_summaries),
        "total_ast_dangling_jumps": sum(summary.get("ast_dangling_jump_count", 0) for summary in function_summaries),
        "total_ast_semantic_errors": sum(summary.get("ast_semantic_error_count", 0) for summary in function_summaries),
        "semantic_body_kind_histogram": dict(semantic_body_hist),
        "ast_semantic_error_kind_histogram": dict(semantic_error_hist),
        "region_kind_histogram": dict(region_hist),
        "ast_region_kind_histogram": dict(ast_region_hist),
        "ast_statement_kind_histogram": dict(statement_hist),
        "ast_expression_kind_histogram": dict(expression_hist),
    }


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
    path = Path(path)
    mod = MBCModule(path, collect_auxiliary=False)
    entries = mod.function_entries(include_definitions=include_definitions, include_exports=include_exports, dedupe=True)
    functions_payload: list[dict[str, Any]] = []
    function_summaries: list[dict[str, Any]] = []
    for entry in entries:
        fn = build_function_ast(mod, entry, include_canonical=include_canonical, include_hir=include_hir, include_text=include_text, validate=validate)
        payload: dict[str, Any] = {
            "name": fn.name,
            "span": fn.span,
            "slice_mode": fn.slice_mode,
            "body_selection": fn.body_selection,
            "hir_summary": fn.hir_summary,
            "input_contract": fn.input_contract,
        }
        if include_diagnostics:
            payload["diagnostics"] = fn.diagnostics
        if include_ast:
            payload["ast"] = fn.ast
        if include_text:
            payload["ast_text"] = fn.ast_text
        if include_hir:
            payload["normalized_hir"] = {"hir_blocks": [block.to_dict() for block in fn.normalized_hir_blocks]}
        functions_payload.append(payload)
        function_summaries.append(dict(fn.summary))
        del fn
    for payload, summary in zip(functions_payload, function_summaries):
        payload["summary"] = summary
    return {
        "contract": {
            "version": AST_CONTRACT_VERSION,
            "input_hir_contract": HIR_CONTRACT_VERSION,
            "layers": ["normalized_hir", "node_ast_builder", "ast_renderer", "ast_summary"],
            "notes": [
                "AST v7 builds statement/expression nodes directly from normalized HIR CFG blocks and applies source-specific edge aliases and conservative self-loop lifting before rendering.",
                "Text is an output-only renderer concern; branch, loop, alias, and edge-parameter normalization operate on nodes.",
                "Unsafe shapes remain explicit goto_region instead of being repaired after the fact.",
            ],
        },
        "path": str(path),
        "script_name": path.name,
        "summary": _empty_module_summary(function_summaries),
        "functions": functions_payload,
    }


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
    function_payloads = [fn for module in module_payloads for fn in module.get("functions", [])]
    summaries = [fn.get("summary", {}) for fn in function_payloads]
    summary = _empty_module_summary(summaries)
    summary["module_count"] = len(module_payloads)
    summary["failed_module_count"] = 0
    summary["fully_structured_function_count"] = sum(1 for item in summaries if int(item.get("fallback_block_count", 0)) == 0 and int(item.get("residual_goto_count", 0)) == 0)
    summary["fallback_function_count"] = sum(1 for item in summaries if int(item.get("fallback_block_count", 0)) or int(item.get("residual_goto_count", 0)))
    total_blocks = int(summary.get("total_normalized_basic_blocks", 0))
    summary["structured_block_ratio"] = (summary.get("total_structured_blocks", 0) / total_blocks) if total_blocks else 1.0
    summary["fallback_block_ratio"] = (summary.get("total_fallback_blocks", 0) / total_blocks) if total_blocks else 0.0
    summary["residual_goto_density_per_1k_blocks"] = (1000.0 * summary.get("total_residual_gotos", 0) / total_blocks) if total_blocks else 0.0

    worst: list[dict[str, Any]] = []
    dangling: list[dict[str, Any]] = []
    for module in module_payloads:
        for fn in module.get("functions", []):
            ref = _entry_ref(module, fn)
            item = {**ref, **{k: fn.get("summary", {}).get(k) for k in (
                "normalized_basic_block_count", "fallback_block_count", "residual_goto_count", "ast_dangling_goto_count", "ast_dangling_jump_count", "ast_semantic_error_count", "source_shape_score"
            )}}
            if item.get("fallback_block_count") or item.get("residual_goto_count"):
                worst.append(item)
            if item.get("ast_dangling_goto_count") or item.get("ast_dangling_jump_count") or item.get("ast_semantic_error_count"):
                dangling.append(item)
    worst.sort(key=lambda item: (-int(item.get("fallback_block_count") or 0), -int(item.get("residual_goto_count") or 0), str(item.get("script_name")), str(item.get("function"))))
    dangling.sort(key=lambda item: (-int(item.get("ast_semantic_error_count") or 0), -int(item.get("ast_dangling_goto_count") or 0), str(item.get("script_name")), str(item.get("function"))))
    return {
        "contract": {
            "version": "ast-report-v2",
            "ast_contract": AST_CONTRACT_VERSION,
            "layers": ["normalized_hir", "node_ast_builder", "ast_renderer", "ast_corpus_metrics"],
            "notes": [
                "AST v6 keeps residual explicit CFG as a first-class result, not as an input to a repair loop.",
                "Dangling jump counters are semantic validation failures, and text rendering is output-only.",
            ],
        },
        "summary": summary,
        "ast_metrics": {
            "region_kind_histogram": summary.get("region_kind_histogram", {}),
            "ast_region_kind_histogram": summary.get("ast_region_kind_histogram", {}),
            "ast_statement_kind_histogram": summary.get("ast_statement_kind_histogram", {}),
            "ast_expression_kind_histogram": summary.get("ast_expression_kind_histogram", {}),
            "semantic_body_kind_histogram": summary.get("semantic_body_kind_histogram", {}),
            "ast_semantic_error_kind_histogram": summary.get("ast_semantic_error_kind_histogram", {}),
            "control_flow_counters": {
                "loop_headers": summary.get("total_loop_headers", 0),
                "structured_loopbacks": summary.get("total_structured_loopbacks", 0),
                "explicit_cfg_regions": summary.get("total_explicit_cfg_regions", 0),
                "rendered_labels": summary.get("total_ast_rendered_labels", 0),
                "goto_targets": summary.get("total_ast_goto_targets", 0),
                "dangling_gotos": summary.get("total_ast_dangling_gotos", 0),
                "jump_targets": summary.get("total_ast_jump_targets", 0),
                "dangling_jumps": summary.get("total_ast_dangling_jumps", 0),
                "semantic_errors": summary.get("total_ast_semantic_errors", 0),
            },
        },
        "rankings": {
            "worst_fallback_functions": worst[:64],
            "most_residual_goto_functions": sorted(worst, key=lambda item: -int(item.get("residual_goto_count") or 0))[:64],
            "most_dangling_goto_functions": dangling[:64],
            "most_dangling_jump_functions": dangling[:64],
            "most_semantic_error_functions": dangling[:64],
            "module_structuring_watchlist": [
                {
                    "script_name": module.get("script_name"),
                    "fallback_block_count": module.get("summary", {}).get("total_fallback_blocks", 0),
                    "residual_goto_count": module.get("summary", {}).get("total_residual_gotos", 0),
                    "ast_semantic_error_count": module.get("summary", {}).get("total_ast_semantic_errors", 0),
                }
                for module in sorted(module_payloads, key=lambda m: -int(m.get("summary", {}).get("total_fallback_blocks", 0)))[:64]
            ],
        },
        "modules": [
            {
                "script_name": module.get("script_name"),
                "function_count": module.get("summary", {}).get("function_count", 0),
                "normalized_basic_block_count": module.get("summary", {}).get("total_normalized_basic_blocks", 0),
                "fallback_block_count": module.get("summary", {}).get("total_fallback_blocks", 0),
                "residual_goto_count": module.get("summary", {}).get("total_residual_gotos", 0),
                "ast_semantic_error_count": module.get("summary", {}).get("total_ast_semantic_errors", 0),
                "region_kind_histogram": module.get("summary", {}).get("region_kind_histogram", {}),
            }
            for module in sorted(module_payloads, key=lambda item: str(item.get("script_name")))
        ],
    }


def render_function_ast_text_from_payload(function_payload: dict[str, Any]) -> str:
    return str(function_payload.get("ast_text") or "").rstrip()


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
