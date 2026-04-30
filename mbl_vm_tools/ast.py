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
    _coerce_hir_block,
    _match_assignment,
    _normalize_hir_blocks,
    _rename_merge_params,
)
from mbl_vm_tools.ast_cfg import (
    build_cfg,
    edge_targets,
    external_successors,
    fold_constant_branches,
    prune_unreachable_blocks,
)
from mbl_vm_tools.ast_flow import (
    alias_for_source as _flow_alias_for_source,
    classify_cyclic_component as _classify_cyclic_component,
    classify_linear_cyclic_span as _classify_linear_cyclic_span,
    classify_transfer as _classify_transfer,
    merge_scoped_aliases as _merge_scoped_aliases,
    plan_loop_aliases as _plan_loop_aliases,
)
from mbl_vm_tools.parser import FunctionEntry, MBCModule


AST_CONTRACT_VERSION = "ast-v5"


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


_EXPR_CALL_RE = re.compile(r"^(?P<target>call_rel_-?\d+|[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")
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


def _negated_expr(expr: Any) -> dict[str, Any]:
    cond_expr = expr if isinstance(expr, dict) else _expr_node(str(expr))
    if isinstance(cond_expr, dict) and cond_expr.get("kind") == "not":
        inner = cond_expr.get("expr")
        return inner if isinstance(inner, dict) else _expr_node(str(inner))
    return {"kind": "not", "expr": cond_expr}


def _conditional_node(condition: Any, body: list[dict[str, Any]], *, negate: bool = False) -> Optional[dict[str, Any]]:
    if not body:
        return None
    cond_expr = condition if isinstance(condition, dict) else _expr_node(str(condition))
    if negate:
        cond_expr = _negated_expr(cond_expr)
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


def _collect_ast_metrics(ast_root: dict[str, Any]) -> dict[str, Any]:
    rendered_labels: set[str] = set()
    jump_targets: set[str] = set()
    max_depth = 0
    parameterized_region_count = region_param_count = 0
    parameterized_block_count = block_param_count = 0
    unknown_branch_predicate_count = 0
    unresolved_call_rel_count = 0

    def visit_expr(expr: Any) -> None:
        nonlocal unknown_branch_predicate_count, unresolved_call_rel_count
        if not isinstance(expr, dict):
            return
        kind = str(expr.get("kind") or "unknown")
        if kind == "predicate":
            source = str(expr.get("source") or "")
            match = re.match(r"^cond\[(?P<op>[^\]]+)\]\(", source)
            if match is not None and match.group("op").lower() == "0x??":
                unknown_branch_predicate_count += 1
        elif kind == "call_expr":
            callee = str(expr.get("callee") or "")
            if callee.startswith("call_rel_"):
                unresolved_call_rel_count += 1
        for arg in expr.get("args") or []:
            visit_expr(arg)
        for key in ("condition", "value", "target", "expr", "base"):
            visit_expr(expr.get(key))

    def visit_stmt(stmt: Any) -> None:
        if not isinstance(stmt, dict):
            return
        kind = str(stmt.get("kind") or "unknown")
        if kind in {"goto", "break", "continue"}:
            target = _statement_target(stmt)
            if target:
                jump_targets.add(target)
        for key in ("expr", "condition", "value", "target"):
            visit_expr(stmt.get(key))
        visit_stmt(stmt.get("jump"))
        for child in stmt.get("body") or []:
            visit_stmt(child)

    def visit_region(region: Any, depth: int = 1) -> None:
        nonlocal max_depth, parameterized_region_count, region_param_count
        nonlocal parameterized_block_count, block_param_count
        if not isinstance(region, dict):
            return
        kind = str(region.get("kind") or "unknown")
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
                visit_stmt(stmt)
            return
        if kind in {"if", "if_else", "while", "do_while", "empty_loop"}:
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
        elif kind == "empty_loop":
            pass
        elif kind == "decision_loop":
            for node in region.get("nodes") or []:
                node_label = _base_label(node.get("block_id"))
                if node_label:
                    rendered_labels.add(node_label)
                for stmt in node.get("prelabel") or []:
                    visit_stmt(stmt)
                for stmt in node.get("body") or []:
                    visit_stmt(stmt)
                visit_expr(node.get("condition"))
                for stmt in node.get("terminal") or []:
                    visit_stmt(stmt)
                for edge_name in ("true_edge", "false_edge", "next_edge"):
                    edge = node.get(edge_name) or {}
                    for stmt in edge.get("statements") or []:
                        visit_stmt(stmt)
        elif kind == "goto_region":
            for child in region.get("blocks") or []:
                visit_region(child, depth + 1)

    visit_region(ast_root)
    dangling_jumps = sorted(target for target in jump_targets if target not in rendered_labels)
    return {
        "ast_max_depth": max_depth,
        "ast_parameterized_region_count": parameterized_region_count,
        "ast_region_param_count": region_param_count,
        "ast_parameterized_block_count": parameterized_block_count,
        "ast_block_param_count": block_param_count,
        "ast_dangling_jump_targets": dangling_jumps,
        "ast_unknown_branch_predicate_count": unknown_branch_predicate_count,
        "ast_unresolved_call_rel_count": unresolved_call_rel_count,
    }


def _is_nonsemantic_data_statement(stmt: dict[str, Any]) -> bool:
    if not isinstance(stmt, dict):
        return False
    if stmt.get("kind") == "empty":
        return True
    expr = stmt.get("expr") if stmt.get("kind") in {"call", "expr"} else None
    if not isinstance(expr, dict) or expr.get("kind") != "call_expr" or expr.get("callee") != "data":
        return False
    payload = " ".join(_render_expr(arg) for arg in expr.get("args") or [])
    normalized = payload.replace("'", '"')
    if '"role": "nop"' in normalized or '"role": "marker"' in normalized:
        return True
    if '"family": "PAD' in normalized:
        return True
    if '"byte":' in normalized and '"len":' in normalized and '"family":' not in normalized:
        return True
    return False


def _statement_nodes(statements: list[Any]) -> list[dict[str, Any]]:
    nodes = [_statement_node(stmt) for stmt in statements]
    return [node for node in nodes if not _is_nonsemantic_data_statement(node)]


def _region_has_visible_content(region: Any, keep_block_ids: Optional[set[str]] = None) -> bool:
    keep_block_ids = keep_block_ids or set()
    if not isinstance(region, dict):
        return bool(region)
    kind = region.get("kind")
    if kind == "sequence":
        return any(_region_has_visible_content(child, keep_block_ids) for child in region.get("regions") or [])
    if kind == "goto_region":
        return any(_region_has_visible_content(child, keep_block_ids) for child in region.get("blocks") or [])
    if kind == "block":
        term = region.get("terminator") or {}
        return bool(
            region.get("block_id") in keep_block_ids
            or region.get("label")
            or region.get("prelabel")
            or region.get("body")
            or term.get("rendered")
        )
    return True


def _drop_empty_regions(regions: list[dict[str, Any]], keep_block_ids: Optional[set[str]] = None) -> list[dict[str, Any]]:
    return [region for region in regions if _region_has_visible_content(region, keep_block_ids)]


def _validate_ast_semantics(ast_root: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    targets = list(metrics.get("ast_dangling_jump_targets") or [])
    if targets:
        errors.append({"kind": "dangling_ast_jump", "message": "AST has visible jumps to labels not rendered in the same tree.", "targets": targets[:32]})

    unknown_predicates = int(metrics.get("ast_unknown_branch_predicate_count", 0) or 0)
    if unknown_predicates:
        warnings.append(
            {
                "kind": "unknown_branch_predicate",
                "message": "AST contains branch predicates whose opcode was not decoded.",
                "count": unknown_predicates,
            }
        )

    unresolved_calls = int(metrics.get("ast_unresolved_call_rel_count", 0) or 0)
    if unresolved_calls:
        warnings.append(
            {
                "kind": "unresolved_call_target",
                "message": "AST contains relative call placeholders that were not symbolized.",
                "count": unresolved_calls,
            }
        )

    return {
        "ok": not errors,
        "error_count": len(errors),
        "kind_histogram": dict(Counter(error["kind"] for error in errors)),
        "errors": errors[:32],
        "warning_count": len(warnings),
        "warning_kind_histogram": dict(Counter(warning["kind"] for warning in warnings)),
        "warnings": warnings[:32],
    }


def _format_block_label(block: HIRBlock) -> str:
    return block.id


def _incoming_assignments(target: Optional[HIRBlock], source_id: Optional[str]) -> list[dict[str, Any]]:
    if target is None or source_id is None or not target.block_params:
        return []
    args = list(target.incoming_args.get(source_id) or [])
    if len(args) != len(target.block_params):
        return []
    return [_assignment_node(param, arg) for param, arg in zip(target.block_params, args) if param != arg]



class _SurfaceBuilder:
    def __init__(self, blocks: list[HIRBlock]):
        self.blocks = prune_unreachable_blocks(fold_constant_branches(blocks))
        self.cfg = build_cfg(self.blocks)
        self.constructs: Counter[str] = Counter()
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
        meta = {"constructs": dict(self.constructs)}
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

    def _substitute_path_args(self, args: list[str], env: dict[str, str]) -> Optional[list[str]]:
        resolved: list[str] = []
        for arg in args:
            if arg in env:
                resolved.append(env[arg])
                continue
            if any(param in arg for param in env):
                return None
            resolved.append(arg)
        return resolved

    def _fallthrough_path_tail_args(
        self,
        start_id: Optional[str],
        target_id: Optional[str],
        source_id: Optional[str],
    ) -> Optional[tuple[str, list[str]]]:
        if start_id is None or target_id is None or source_id is None:
            return None
        current_id = start_id
        previous_id = source_id
        env: dict[str, str] = {}
        seen: set[str] = set()
        while current_id not in seen:
            if current_id == target_id:
                target = self.cfg.by_id.get(target_id)
                if target is None:
                    return None
                args = list(target.incoming_args.get(previous_id) or [])
                resolved = self._substitute_path_args(args, env)
                if resolved is None:
                    return None
                return previous_id, resolved
            seen.add(current_id)
            block = self.cfg.by_id.get(current_id)
            if block is None or block.statements:
                return None
            if block.block_params:
                incoming = list(block.incoming_args.get(previous_id) or [])
                if len(incoming) != len(block.block_params):
                    return None
                resolved_incoming = self._substitute_path_args(incoming, env)
                if resolved_incoming is None:
                    return None
                env.update(dict(zip(block.block_params, resolved_incoming)))
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
        branch_target, fallthrough_target = edge_targets(block, self.cfg.index_by_id)
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
        branch_target, fallthrough_target = edge_targets(block, self.cfg.index_by_id)
        if not branch_target or not fallthrough_target or fallthrough_target != next_id or branch_target == next_id:
            return None
        target = self.cfg.by_id.get(branch_target)
        if target is None:
            return None
        fallthrough_path = self._fallthrough_path_tail_args(fallthrough_target, branch_target, block.id)
        if fallthrough_path is None:
            return None
        fallthrough_tail, default_args = fallthrough_path
        if not target.block_params:
            return {"target_id": branch_target, "fallthrough_tail": fallthrough_tail, "nodes": [], "sources": {block.id, fallthrough_tail}}
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
        source_id = source.id if source is not None else None
        return _flow_alias_for_source(target_id, source_id, aliases)

    def _jump_nodes(
        self,
        source: Optional[HIRBlock],
        target_id: Optional[str],
        aliases: dict[str, Any],
        *,
        source_specific_alias: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        source_id = source.id if source is not None else None
        action = _classify_transfer(
            source_id=source_id,
            source_index=source.index if source is not None else None,
            target_id=target_id,
            target_index=self.cfg.index_by_id.get(target_id) if target_id is not None else None,
            aliases=aliases,
            source_specific_alias=source_specific_alias,
        )
        if action.kind == "none" or target_id is None:
            return [], 0
        if action.kind == "alias":
            alias = action.alias
            if isinstance(alias, dict):
                by_source = alias.get("prelude_by_source") or {}
                prelude = list(by_source.get(source_id) or [])
                jump = alias.get("jump")
                return prelude + ([jump] if isinstance(jump, dict) else []), int(isinstance(jump, dict) and jump.get("kind") == "goto")
            node = _statement_node(alias)
            return ([node] if node.get("kind") != "empty" else []), int(node.get("kind") == "goto")

        target = self.cfg.by_id.get(target_id)
        prelude = _incoming_assignments(target, source_id)
        if action.needs_label and target_id in self.cfg.by_id:
            self.needed_label_ids.add(target_id)
        return [*prelude, {"kind": action.kind, "target": action.target_label or target_id}], int(action.counts_as_goto)

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
            branch, fallthrough = edge_targets(block, self.cfg.index_by_id)
            if branch == fallthrough:
                if branch is None or branch == next_id:
                    return [], 0
                return self._jump_nodes(block, branch, aliases)
            branch_alias = self._alias_for_source(branch, block, aliases)
            fallthrough_alias = self._alias_for_source(fallthrough, block, aliases)
            if branch_alias is None and fallthrough_alias is None and self._branch_empty_convergence(block) is not None:
                return [], 0
            merge_next_id = next_id
            if merge_next_id is None and fallthrough is not None and self.cfg.index_by_id.get(fallthrough) == block.index + 1:
                merge_next_id = fallthrough
            if fallthrough == merge_next_id and branch_alias is None and (fallthrough_alias is None or merge_next_id == fallthrough):
                merge = self._branch_param_merge(block, merge_next_id)
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
        prelabel = self._entry_assignments(block, previous_source_id, aliases)
        statements = _statement_nodes(list(block.statements))
        term_nodes, goto_count = self._terminator_nodes(block, next_id, aliases)
        if len(term_nodes) == 1 and statements and _render_statement(statements[-1]) == _render_statement(term_nodes[0]):
            if term_nodes[0].get("kind") == "goto" and goto_count:
                goto_count = 0
            term_nodes = []
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
        if goto_count:
            region = {"kind": "goto_region", "goto_count": goto_count, "blocks": [region]}
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

    def _arm_ids(
        self,
        start_id: Optional[str],
        join_id: str,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
    ) -> Optional[set[str]]:
        if start_id is None:
            return None
        if start_id == join_id or start_id in aliases:
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
            if node == join_id or node in aliases:
                continue
            node_idx = self.cfg.index_by_id.get(node)
            if node_idx is None or node_idx <= start_idx - 1 or node_idx >= join_idx:
                return None
            if allowed is not None and node not in allowed:
                return None
            if node in ids:
                continue
            ids.add(node)
            stack.extend(succ for succ in self.cfg.succs.get(node, []) if succ != join_id and succ not in aliases)
        return ids

    def _has_external_entry(self, ids: set[str], allowed_preds: set[str]) -> bool:
        for block_id in ids:
            settled = set(self.settled_preds_by_block.get(block_id, set()))
            for pred in self.cfg.preds.get(block_id, []):
                if pred in settled:
                    continue
                if pred not in ids and pred not in allowed_preds:
                    return True
        return False

    def _is_duplicable_shared_arm_path(
        self,
        start_id: str,
        join_id: str,
        ids: set[str],
        source_id: str,
        sibling_ids: set[str],
    ) -> bool:
        if start_id not in ids or start_id not in self.cfg.index_by_id or len(ids) > 4:
            return False
        seen: set[str] = set()
        current = start_id
        while current != join_id:
            if current in seen or current not in ids:
                return False
            block = self.cfg.by_id.get(current)
            if block is None or len(block.statements) > 4:
                return False
            seen.add(current)
            current_idx = self.cfg.index_by_id.get(current, 10**9)
            for pred in self.cfg.preds.get(current, []):
                if pred in ids or pred == source_id or pred in sibling_ids:
                    continue
                if pred in self.settled_preds_by_block.get(current, set()):
                    continue
                return False
            succs = list(self.cfg.succs.get(current, []))
            if len(succs) != 1:
                return False
            succ = succs[0]
            if succ != join_id and succ not in ids:
                return False
            if succ != join_id and self.cfg.index_by_id.get(succ, -1) <= current_idx:
                return False
            current = succ
        return seen == ids

    def _entry_assignments(
        self,
        block: HIRBlock,
        previous_source_id: Optional[str],
        aliases: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if previous_source_id not in block.predecessors:
            return []
        previous_source = self.cfg.by_id.get(previous_source_id) if previous_source_id is not None else None
        if previous_source_id in self.handled_param_predecessors_by_block.get(block.id, set()):
            return []
        if self._alias_for_source(block.id, previous_source, aliases) is not None:
            return []
        return _incoming_assignments(block, previous_source_id)

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

    def _decision_loop_edge_node(
        self,
        source: HIRBlock,
        target_id: Optional[str],
        loop_ids: set[str],
        aliases: dict[str, Any],
        next_after_id: Optional[str],
    ) -> dict[str, Any]:
        if target_id is None:
            return {"kind": "exit", "target": None, "statements": []}
        target = self.cfg.by_id.get(target_id)
        if target_id in loop_ids:
            return {
                "kind": "internal",
                "target": target_id,
                "statements": _incoming_assignments(target, source.id),
            }
        if self._alias_for_source(target_id, source, aliases) is not None:
            statements, _ = self._jump_nodes(source, target_id, aliases, source_specific_alias=True)
            return {"kind": "structured_jump", "target": target_id, "statements": statements}
        statements = _incoming_assignments(target, source.id)
        if target_id in self.cfg.by_id:
            self.settled_preds_by_block[target_id].add(source.id)
            if statements:
                self.handled_param_predecessors_by_block[target_id].add(source.id)
        return {
            "kind": "fallthrough_exit" if target_id == next_after_id else "external_exit",
            "target": target_id,
            "statements": statements,
        }

    def _decision_loop_region(
        self,
        *,
        label_block: HIRBlock,
        entry_id: str,
        ordered_ids: list[str],
        loop_ids: set[str],
        exit_ids: list[str],
        classification: str,
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
        next_after_id: Optional[str],
        backward_edge_count: int = 0,
    ) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        for block_id in ordered_ids:
            node_block = self.cfg.by_id.get(str(block_id))
            if node_block is None:
                continue
            node: dict[str, Any] = {
                "kind": "decision_loop_node",
                "block_id": node_block.id,
                "block_params": list(node_block.block_params),
                "prelabel": self._entry_assignments(node_block, previous_source_id, aliases) if node_block.id == entry_id else [],
                "body": _statement_nodes(list(node_block.statements)),
            }
            term_kind = node_block.terminator.get("kind")
            if term_kind == "branch":
                branch, fallthrough = edge_targets(node_block, self.cfg.index_by_id)
                node.update(
                    {
                        "terminator_kind": "branch",
                        "condition": _expr_node(node_block.terminator.get("condition") or node_block.terminator.get("text") or f"cond_{node_block.id}"),
                        "true_edge": self._decision_loop_edge_node(node_block, branch, loop_ids, aliases, next_after_id),
                        "false_edge": self._decision_loop_edge_node(node_block, fallthrough, loop_ids, aliases, next_after_id),
                    }
                )
            elif term_kind == "fallthrough":
                target_id = node_block.fallthrough_target or (node_block.successors[0] if node_block.successors else None)
                node.update(
                    {
                        "terminator_kind": "fallthrough",
                        "next_edge": self._decision_loop_edge_node(node_block, target_id, loop_ids, aliases, next_after_id),
                    }
                )
            elif term_kind == "return":
                node.update({"terminator_kind": "return", "terminal": [_statement_node(node_block.terminator.get("text") or "return")]})
            elif term_kind == "stop":
                node.update({"terminator_kind": "stop", "terminal": [{"kind": "stop", "source": str(node_block.terminator.get("text") or "stop").strip()}]})
            else:
                target_id = node_block.successors[0] if node_block.successors else None
                node.update(
                    {
                        "terminator_kind": term_kind or "none",
                        "next_edge": self._decision_loop_edge_node(node_block, target_id, loop_ids, aliases, next_after_id) if target_id else None,
                    }
                )
            nodes.append(node)

        label = _format_block_label(label_block) if self._label_needed(label_block, hidden_labels, previous_source_id) else None
        self.constructs["decision_loop"] += 1
        return {
            "kind": "decision_loop",
            "label": label,
            "entry_block": entry_id,
            "exit_blocks": list(exit_ids),
            "classification": classification,
            "node_count": len(nodes),
            "nodes": nodes,
        }

    def _try_decision_loop(
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
        if block.id in suppressed_loops:
            return None
        component = self.cfg.cyclic_components.get(block.id)
        if component is None:
            return None
        plan = _classify_cyclic_component(component, self.cfg)
        if plan is None:
            return None
        idx0, idx1 = component.get("index_range") or (idx, idx)
        loop_ids = set(plan.nodes)
        if idx0 != idx or idx1 >= stop or (allowed is not None and not loop_ids.issubset(allowed)):
            return None

        entry_id = plan.entry_id
        for pred_id, node_id in component.get("external_entries") or []:
            pred_id = str(pred_id)
            node_id = str(node_id)
            settled = set(self.settled_preds_by_block.get(node_id, set()))
            pred_idx = self.cfg.index_by_id.get(pred_id)
            allowed_entry = (
                node_id == entry_id
                and (pred_id == previous_source_id or pred_id in settled or (pred_idx is not None and pred_idx + 1 == idx))
            )
            if not allowed_entry:
                return None

        next_after_id = self.blocks[idx1 + 1].id if idx1 + 1 < min(stop, len(self.blocks)) else None
        region = self._decision_loop_region(
            label_block=block,
            entry_id=entry_id,
            ordered_ids=[str(item) for item in (component.get("ordered_nodes") or [])],
            loop_ids=loop_ids,
            exit_ids=list(plan.exit_ids),
            classification=plan.classification,
            aliases=aliases,
            hidden_labels=hidden_labels,
            previous_source_id=previous_source_id,
            next_after_id=next_after_id,
            backward_edge_count=len(component.get("backward_edges") or []),
        )
        return region, idx1 + 1


    def _forward_skip_arm_ids(
        self,
        start_id: Optional[str],
        join_id: str,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
    ) -> Optional[set[str]]:
        if start_id is None:
            return None
        if start_id == join_id:
            return set()
        start_idx = self.cfg.index_by_id.get(start_id)
        join_idx = self.cfg.index_by_id.get(join_id)
        if start_idx is None or join_idx is None or start_idx <= 0 or start_idx >= join_idx or join_idx > stop:
            return None
        ids: set[str] = set()
        stack = [start_id]
        while stack:
            node = stack.pop()
            if node == join_id:
                continue
            node_idx = self.cfg.index_by_id.get(node)
            if node_idx is None or node_idx < start_idx or node_idx >= join_idx:
                return None
            if allowed is not None and node not in allowed:
                return None
            if node in ids:
                continue
            ids.add(node)
            block = self.cfg.by_id.get(node)
            for succ in self.cfg.succs.get(node, []):
                if succ == join_id:
                    continue
                succ_idx = self.cfg.index_by_id.get(succ)
                if succ_idx is not None and start_idx <= succ_idx < join_idx:
                    stack.append(succ)
                    continue
                if succ_idx is not None and join_idx <= succ_idx:
                    continue
                if self._alias_for_source(succ, block, aliases) is not None:
                    continue
                return None
        return ids

    def _forward_skip_bypass_targets(
        self,
        ids: set[str],
        join_id: str,
        aliases: dict[str, Any],
        stop: int,
    ) -> Optional[set[str]]:
        join_idx = self.cfg.index_by_id.get(join_id, 10**9)
        bypass_targets: set[str] = set()
        for block_id in ids:
            block = self.cfg.by_id.get(block_id)
            for succ in self.cfg.succs.get(block_id, []):
                if succ in ids or succ == join_id:
                    continue
                if self._alias_for_source(succ, block, aliases) is not None:
                    continue
                succ_idx = self.cfg.index_by_id.get(succ)
                if succ_idx is not None and join_idx <= succ_idx:
                    bypass_targets.add(succ)
                    continue
                return None
        return bypass_targets

    def _try_forward_skip_if(
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
        branch, fallthrough = edge_targets(block, self.cfg.index_by_id)
        if branch is None or fallthrough is None or branch == fallthrough:
            return None
        cond = _expr_node(block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}")

        candidates = [
            (branch, fallthrough, True),
            (fallthrough, branch, False),
        ]
        for join_id, body_id, direct_is_true in candidates:
            join_idx = self.cfg.index_by_id.get(join_id)
            body_idx = self.cfg.index_by_id.get(body_id)
            if join_idx is None or body_idx is None:
                continue
            if join_idx <= idx or body_idx <= idx or body_idx >= join_idx or join_idx > stop:
                continue
            if allowed is not None and (join_id not in allowed or body_id not in allowed):
                continue
            body_ids = self._forward_skip_arm_ids(body_id, join_id, stop, allowed, aliases)
            if body_ids is None or not body_ids:
                continue
            span_ids = {
                candidate.id
                for candidate in self.blocks[idx + 1 : join_idx]
                if allowed is None or candidate.id in allowed
            }
            if body_ids != span_ids:
                continue
            if self._has_external_entry(body_ids, {block.id}):
                continue
            bypass_targets = self._forward_skip_bypass_targets(body_ids, join_id, aliases, stop)
            if bypass_targets is None or len(bypass_targets) > 1:
                continue

            body_exit_sources = self._arm_exit_sources(body_ids, block.id, join_id)
            join_sources = {block.id} | body_exit_sources
            join_alias = self._edge_alias_to_join(join_id, join_sources)
            self.settled_preds_by_block[str(join_id)].update(join_sources)

            bypass_aliases: dict[str, Any] = {}
            for bypass_target in bypass_targets:
                bypass_sources = {
                    source_id
                    for source_id in body_ids
                    if bypass_target in self.cfg.succs.get(source_id, [])
                }
                if not bypass_sources:
                    continue
                target_block = self.cfg.by_id.get(bypass_target)
                self.needed_label_ids.add(bypass_target)
                bypass_aliases[bypass_target] = {
                    "source_specific": True,
                    "jump": {"kind": "break", "target": bypass_target},
                    "prelude_by_source": {
                        source_id: _incoming_assignments(target_block, source_id)
                        for source_id in bypass_sources
                    },
                }
            self._settle_alias_predecessors(bypass_aliases)

            arm_aliases = dict(aliases)
            arm_aliases[str(join_id)] = join_alias
            arm_aliases.update(bypass_aliases)
            body_regions = self._range(
                body_idx,
                join_idx,
                body_ids,
                arm_aliases,
                hidden_labels,
                block.id,
                set(),
            )
            body_regions = _drop_empty_regions(body_regions, self.explicit_label_ids)
            if not body_regions:
                continue

            direct_assignments = _incoming_assignments(self.cfg.by_id.get(join_id), block.id)
            direct_block = self._edge_statement_block(f"edge_{block.id}_{join_id}_skip", direct_assignments)
            direct_regions = [direct_block] if direct_block else []

            rendered_cond = cond
            if direct_is_true:
                if direct_regions:
                    then_regions = direct_regions
                    else_regions = body_regions
                else:
                    rendered_cond = _negated_expr(rendered_cond)
                    then_regions = body_regions
                    else_regions = []
            else:
                then_regions = body_regions
                else_regions = direct_regions

            label = _format_block_label(block) if self._label_needed(block, hidden_labels, previous_source_id) else None
            prologue = self._entry_assignments(block, previous_source_id, aliases)
            prologue.extend(_statement_nodes(list(block.statements)))
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
                "rendering": "forward_skip_if",
            }
            return region, join_idx
        return None


    def _empty_decision_span_ids(
        self,
        start_id: str,
        join_id: str,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        previous_source_id: Optional[str],
    ) -> Optional[set[str]]:
        start_idx = self.cfg.index_by_id.get(start_id)
        join_idx = self.cfg.index_by_id.get(join_id)
        if start_idx is None or join_idx is None or join_idx <= start_idx or join_idx > stop:
            return None
        ids: set[str] = set()
        stack = [start_id]
        while stack:
            node = stack.pop()
            if node == join_id:
                continue
            node_idx = self.cfg.index_by_id.get(node)
            if node_idx is None or node_idx < start_idx or node_idx >= join_idx:
                return None
            if allowed is not None and node not in allowed:
                return None
            if node in aliases and node != start_id:
                return None
            if node in ids:
                continue
            block = self.cfg.by_id.get(node)
            if block is None or block.statements:
                return None
            if block.terminator.get("kind") not in {"branch", "fallthrough"}:
                return None
            succs = list(self.cfg.succs.get(node, []))
            if not succs:
                return None
            ids.add(node)
            for succ in succs:
                if succ == join_id:
                    continue
                if succ in aliases:
                    return None
                stack.append(succ)
        if not ids:
            return None
        for node in ids:
            settled = set(self.settled_preds_by_block.get(node, set()))
            for pred in self.cfg.preds.get(node, []):
                if pred in ids or pred in settled:
                    continue
                if node == start_id and pred == previous_source_id:
                    continue
                return None
        join_block = self.cfg.by_id.get(join_id)
        for source_id in ids:
            if join_id in self.cfg.succs.get(source_id, []) and _incoming_assignments(join_block, source_id):
                return None
        return ids

    def _try_empty_decision_span(
        self,
        idx: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
    ) -> Optional[tuple[dict[str, Any], int]]:
        block = self.blocks[idx]
        if block.statements or block.terminator.get("kind") not in {"branch", "fallthrough"}:
            return None
        if self._label_needed(block, hidden_labels, previous_source_id):
            return None
        join_id = self.cfg.ipdom.get(block.id)
        if join_id is None:
            return None
        ids = self._empty_decision_span_ids(block.id, join_id, stop, allowed, aliases, previous_source_id)
        if ids is None:
            return None
        join_sources = {source_id for source_id in ids if join_id in self.cfg.succs.get(source_id, [])}
        if join_sources:
            self.settled_preds_by_block[str(join_id)].update(join_sources)
        self.constructs["empty_decision_span"] += 1
        return {"kind": "sequence", "regions": []}, self.cfg.index_by_id[join_id]

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
        branch, fallthrough = edge_targets(block, self.cfg.index_by_id)
        if branch is None or fallthrough is None:
            return None
        branch_ids = self._arm_ids(branch, join_id, stop, allowed, aliases)
        fall_ids = self._arm_ids(fallthrough, join_id, stop, allowed, aliases)
        duplicate_branch_arm = False
        duplicate_fall_arm = False
        if branch_ids is None or fall_ids is None:
            return None
        shared_ids = branch_ids & fall_ids
        if shared_ids:
            if branch_ids == shared_ids and self._is_duplicable_shared_arm_path(
                branch,
                join_id,
                branch_ids,
                block.id,
                fall_ids - shared_ids,
            ):
                duplicate_branch_arm = True
            elif fall_ids == shared_ids and self._is_duplicable_shared_arm_path(
                fallthrough,
                join_id,
                fall_ids,
                block.id,
                branch_ids - shared_ids,
            ):
                duplicate_fall_arm = True
            else:
                return None
        span_ids = {candidate.id for candidate in self.blocks[idx + 1 : join_idx] if allowed is None or candidate.id in allowed}
        if (branch_ids | fall_ids) != span_ids or not span_ids:
            return None
        branch_allowed_preds = {block.id} | ((fall_ids - shared_ids) if duplicate_branch_arm else set())
        fall_allowed_preds = {block.id} | ((branch_ids - shared_ids) if duplicate_fall_arm else set())
        if self._has_external_entry(branch_ids, branch_allowed_preds) or self._has_external_entry(fall_ids, fall_allowed_preds):
            return None

        if duplicate_branch_arm:
            self.settled_preds_by_block[branch].add(block.id)
        if duplicate_fall_arm:
            self.settled_preds_by_block[fallthrough].add(block.id)

        join_sources = self._arm_exit_sources(branch_ids, block.id, join_id) | self._arm_exit_sources(fall_ids, block.id, join_id)
        join_alias = self._edge_alias_to_join(join_id, join_sources)
        self.settled_preds_by_block[join_id].update(join_sources)
        arm_aliases = dict(aliases)
        arm_aliases[join_id] = join_alias
        cond = _expr_node(block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}")
        label = _format_block_label(block) if self._label_needed(block, hidden_labels, previous_source_id) else None
        prologue = self._entry_assignments(block, previous_source_id, aliases)
        prologue.extend(_statement_nodes(list(block.statements)))

        def build_arm(ids: set[str], target_id: Optional[str], suffix: str, hidden_arm_labels: set[str]) -> list[dict[str, Any]]:
            if ids:
                start_index = min(self.cfg.index_by_id[item] for item in ids)
                return self._range(start_index, join_idx, ids, arm_aliases, hidden_arm_labels, block.id, set())
            if target_id == join_id:
                assignments = _incoming_assignments(self.cfg.by_id.get(join_id), block.id)
                edge_block = self._edge_statement_block(f"edge_{block.id}_{join_id}_{suffix}", assignments)
                return [edge_block] if edge_block else []
            if self._alias_for_source(target_id, block, arm_aliases) is not None:
                statements, _ = self._jump_nodes(block, target_id, arm_aliases, source_specific_alias=True)
                edge_block = self._edge_statement_block(f"edge_{block.id}_{target_id}_{suffix}", statements)
                return [edge_block] if edge_block else []
            return []

        then_ids = branch_ids
        else_ids = fall_ids
        rendered_cond = cond
        then_hidden = set(shared_ids) if duplicate_branch_arm else set()
        else_hidden = set(shared_ids) if duplicate_fall_arm else set()
        then_regions = _drop_empty_regions(build_arm(then_ids, branch, "then", then_hidden), self.explicit_label_ids)
        else_regions = _drop_empty_regions(build_arm(else_ids, fallthrough, "else", else_hidden), self.explicit_label_ids)
        if not then_regions and else_regions:
            rendered_cond = _negated_expr(rendered_cond)
            then_regions, else_regions = else_regions, []
        if not then_regions and not else_regions:
            if prologue or label:
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
        aliases: dict[str, Any] = {}
        for plan in _plan_loop_aliases(loop, self.cfg, exit_ids):
            target = self.cfg.by_id.get(plan.target_id)
            if plan.needs_label:
                self.needed_label_ids.add(plan.target_id)
            alias: dict[str, Any] = {
                "jump": {"kind": plan.jump_kind, "target": plan.jump_target},
                "prelude_by_source": {source_id: _incoming_assignments(target, source_id) for source_id in plan.sources},
            }
            if plan.source_specific:
                alias["source_specific"] = True
            aliases[plan.target_id] = alias
        return aliases

    def _settle_alias_predecessors(self, alias_map: dict[str, Any]) -> None:
        for target_id, alias in alias_map.items():
            if isinstance(alias, dict):
                self.settled_preds_by_block[target_id].update((alias.get("prelude_by_source") or {}).keys())


    def _try_linear_decision_loop(
        self,
        idx: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
    ) -> Optional[tuple[dict[str, Any], int]]:
        plan = _classify_linear_cyclic_span(
            self.cfg,
            idx=idx,
            stop=stop,
            allowed=allowed,
            previous_source_id=previous_source_id,
            settled_preds_by_block=self.settled_preds_by_block,
        )
        if plan is None:
            return None
        loop_ids = set(plan.nodes)
        end = int(plan.end_index)
        entry_block = self.blocks[idx]
        next_after_id = plan.exit_ids[0] if plan.exit_ids else None
        backward_edge_count = 0
        for source_id in loop_ids:
            source_idx = self.cfg.index_by_id.get(source_id, 10**9)
            for succ in self.cfg.succs.get(source_id, []):
                if succ in loop_ids and self.cfg.index_by_id.get(succ, 10**9) <= source_idx:
                    backward_edge_count += 1
        region = self._decision_loop_region(
            label_block=entry_block,
            entry_id=plan.entry_id,
            ordered_ids=[self.blocks[pos].id for pos in range(idx, end + 1) if self.blocks[pos].id in loop_ids],
            loop_ids=loop_ids,
            exit_ids=list(plan.exit_ids),
            classification=plan.classification,
            aliases=aliases,
            hidden_labels=hidden_labels,
            previous_source_id=previous_source_id,
            next_after_id=str(next_after_id) if next_after_id is not None else None,
            backward_edge_count=backward_edge_count,
        )
        return region, end + 1

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
        loopback_assignments = _incoming_assignments(block, block.id)
        outside_preds = [pred for pred in block.predecessors if pred != block.id]
        if block.block_params:
            settled = set(self.settled_preds_by_block.get(block.id, set()))
            allowed_entries = set(outside_preds)
            if previous_source_id is not None:
                allowed_entries.discard(previous_source_id)
            if allowed_entries and not allowed_entries.issubset(settled):
                return None
            if previous_source_id is None and not outside_preds and block.index != 0:
                return None
        exit_assignments = _incoming_assignments(self.cfg.by_id.get(exit_succ), block.id)
        cond = _expr_node(block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}")
        branch, fallthrough = edge_targets(block, self.cfg.index_by_id)
        if branch == block.id:
            continue_condition = cond
        elif fallthrough == block.id:
            continue_condition = _negated_expr(cond)
        else:
            return None
        label = _format_block_label(block) if self._label_needed(block, hidden_labels, previous_source_id) else None
        preheader = self._entry_assignments(block, previous_source_id, aliases)
        self.settled_preds_by_block[exit_succ].add(block.id)
        if exit_assignments:
            self.handled_param_predecessors_by_block[exit_succ].add(block.id)
        if loopback_assignments or exit_assignments:
            body_nodes = _statement_nodes(list(block.statements))
            body_nodes.append(_conditional_node(_negated_expr(continue_condition), [*exit_assignments, {"kind": "break", "target": None}]))
            body_nodes.extend(loopback_assignments)
            self.constructs["while"] += 1
            region = {
                "kind": "while",
                "header_block": block.id,
                "label": label,
                "block_params": list(block.block_params),
                "preheader": preheader,
                "condition": _expr_node("true"),
                "prologue": [],
                "guard_condition": continue_condition,
                "guard_exit": exit_assignments,
                "rendering": "parameterized_self_loop",
                "body": {
                    "kind": "sequence",
                    "regions": [
                        {
                            "kind": "block",
                            "block_id": block.id,
                            "label": None,
                            "block_params": [],
                            "prelabel": [],
                            "body": [node for node in body_nodes if node is not None],
                            "terminator": {
                                "kind": "fallthrough",
                                "condition": None,
                                "rendered": [],
                                "successors": [],
                                "branch_target": None,
                                "fallthrough_target": None,
                            },
                        }
                    ],
                },
                "continue_target": block.id,
                "break_target": exit_succ,
                "break_targets": [exit_succ],
            }
            return region, idx + 1
        if not block.statements:
            self.constructs["empty_loop"] += 1
            region = {
                "kind": "empty_loop",
                "header_block": block.id,
                "label": label,
                "block_params": [],
                "preheader": preheader,
                "condition": continue_condition,
                "continue_target": block.id,
                "break_target": exit_succ,
            }
            return region, idx + 1
        self.constructs["do_while"] += 1
        region = {
            "kind": "do_while",
            "header_block": block.id,
            "label": label,
            "block_params": [],
            "preheader": preheader,
            "body": _statement_nodes(list(block.statements)),
            "condition": continue_condition,
            "loopback": [],
            "continue_target": block.id,
            "break_target": exit_succ,
        }
        return region, idx + 1

    def _try_transparent_header_loop(
        self,
        idx: int,
        stop: int,
        allowed: Optional[set[str]],
        aliases: dict[str, Any],
        hidden_labels: set[str],
        previous_source_id: Optional[str],
    ) -> Optional[tuple[dict[str, Any], int]]:
        header = self.blocks[idx]
        if header.statements or header.block_params or header.terminator.get("kind") != "fallthrough":
            return None
        body_id = header.fallthrough_target or (header.successors[0] if header.successors else None)
        body_idx = self.cfg.index_by_id.get(body_id)
        if body_id is None or body_idx != idx + 1 or body_idx >= stop:
            return None
        if allowed is not None and (header.id not in allowed or body_id not in allowed):
            return None
        body = self.cfg.by_id.get(body_id)
        if body is None or body.block_params or body.terminator.get("kind") != "branch":
            return None
        if header.id in aliases or body_id in aliases:
            return None
        branch, fallthrough = edge_targets(body, self.cfg.index_by_id)
        cond = _expr_node(body.terminator.get("condition") or body.terminator.get("text") or f"cond_{body.id}")
        if branch == header.id and fallthrough is not None:
            loop_cond = cond
            exit_succ = fallthrough
        elif fallthrough == header.id and branch is not None:
            loop_cond = _negated_expr(cond)
            exit_succ = branch
        else:
            return None
        exit_idx = self.cfg.index_by_id.get(exit_succ)
        if exit_idx is None or exit_idx <= body_idx or exit_idx > stop:
            return None
        if _incoming_assignments(header, body.id):
            return None
        if _incoming_assignments(self.cfg.by_id.get(exit_succ), body.id):
            return None

        header_entry_preds = {pred for pred in header.predecessors if pred != body.id}
        body_entry_preds = {pred for pred in body.predecessors if pred != header.id}
        if previous_source_id is not None:
            header_entry_preds.discard(previous_source_id)
            body_entry_preds.discard(previous_source_id)
        settled_header = set(self.settled_preds_by_block.get(header.id, set()))
        if header_entry_preds - settled_header:
            return None
        if body_entry_preds:
            return None

        preheader = self._entry_assignments(header, previous_source_id, aliases)
        label = _format_block_label(header) if self._label_needed(header, hidden_labels, previous_source_id) else None
        if not body.statements:
            self.constructs["empty_loop"] += 1
            region = {
                "kind": "empty_loop",
                "header_block": header.id,
                "label": label,
                "block_params": [],
                "preheader": preheader,
                "condition": loop_cond,
                "continue_target": header.id,
                "break_target": exit_succ,
                "rendering": "transparent_header_loop",
            }
            return region, body_idx + 1
        self.constructs["do_while"] += 1
        region = {
            "kind": "do_while",
            "header_block": header.id,
            "label": label,
            "block_params": [],
            "preheader": preheader,
            "body": _statement_nodes(list(body.statements)),
            "condition": loop_cond,
            "loopback": [],
            "continue_target": header.id,
            "break_target": exit_succ,
            "rendering": "transparent_header_loop",
        }
        return region, body_idx + 1

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
        branch, fallthrough = edge_targets(block, self.cfg.index_by_id)
        body_succ = loop.get("body_succ")
        exit_succ = loop.get("exit_succ")
        if body_succ not in self.cfg.index_by_id or exit_succ is None:
            return None
        cond = _expr_node(block.terminator.get("condition") or block.terminator.get("text") or f"cond_{block.id}")
        body_cond = cond if branch == body_succ else _negated_expr(cond) if fallthrough == body_succ else cond
        local_aliases = self._loop_aliases(loop)
        self._settle_alias_predecessors(local_aliases)
        loop_aliases = _merge_scoped_aliases(aliases, local_aliases, nodes)
        body_ids = nodes - {block.id}
        if not body_ids:
            return None
        body_start = min(self.cfg.index_by_id[item] for item in body_ids)
        body = self._range(body_start, idx1 + 1, body_ids, loop_aliases, set(), block.id, set())
        preheader = self._entry_assignments(block, previous_source_id, aliases)
        header_exit_assignments = _incoming_assignments(self.cfg.by_id.get(exit_succ), block.id)
        guarded = bool(block.statements or header_exit_assignments)
        self.constructs["while"] += 1
        label = _format_block_label(block) if self._label_needed(block, hidden_labels, previous_source_id) else None
        region = {
            "kind": "while",
            "header_block": block.id,
            "label": label,
            "block_params": list(block.block_params),
            "preheader": preheader,
            "condition": _expr_node("true") if guarded else body_cond,
            "prologue": _statement_nodes(list(block.statements)),
            "guard_condition": body_cond,
            "guard_exit": header_exit_assignments,
            "rendering": "guarded_loop" if guarded else "direct_while",
            "body": {"kind": "sequence", "regions": body},
            "continue_target": block.id,
            "break_target": exit_succ,
            "break_targets": external_successors(loop, self.cfg),
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
            transparent_loop = None if block.id in suppressed_loops else self._try_transparent_header_loop(
                i, stop, allowed, aliases, hidden_labels, prev
            )
            if transparent_loop is not None:
                region, next_i = transparent_loop
                if next_i > i:
                    regions.append(region)
                    i = next_i
                    prev = None
                    continue
            decision_loop = self._try_decision_loop(i, stop, allowed, aliases, hidden_labels, prev, suppressed_loops)
            if decision_loop is not None:
                region, next_i = decision_loop
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
            empty_decision = self._try_empty_decision_span(i, stop, allowed, aliases, hidden_labels, prev)
            if empty_decision is not None:
                region, next_i = empty_decision
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
            forward_skip = self._try_forward_skip_if(i, stop, allowed, aliases, hidden_labels, prev)
            if forward_skip is not None:
                region, next_i = forward_skip
                if next_i > i:
                    regions.append(region)
                    i = next_i
                    prev = None
                    continue
            linear_decision_loop = self._try_linear_decision_loop(i, stop, allowed, aliases, hidden_labels, prev)
            if linear_decision_loop is not None:
                region, next_i = linear_decision_loop
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
    if kind == "empty_loop":
        lines = _render_statement_list(region.get("preheader") or [], indent)
        label = region.get("label")
        loop_indent = indent
        if label:
            lines.append(prefix + f"{label}:")
            loop_indent += 1
        lp = "    " * loop_indent
        cond = _render_expr(region.get("condition")) or "true"
        header = f"empty_loop {region.get('header_block')} while ({cond})"
        if region.get("break_target"):
            header += f" -> {region.get('break_target')}"
        lines.append(lp + header)
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
            guard = _render_expr(_negated_expr(region.get("guard_condition"))) or "true"
            lines.append(ip + f"if ({guard}) {{")
            lines.extend(_render_statement_list(region.get("guard_exit") or [], inner + 1))
            lines.append("    " * (inner + 1) + "break")
            lines.append(ip + "}")
        lines.extend(_render_region(region.get("body"), inner))
        lines.append(lp + "}")
        return lines
    if kind == "decision_loop":
        lines: list[str] = []
        label = region.get("label")
        loop_indent = indent
        if label:
            lines.append(prefix + f"{label}:")
            loop_indent += 1
        lp = "    " * loop_indent
        exits = [str(item) for item in region.get("exit_blocks") or []]
        header = f"decision_loop {region.get('entry_block')}"
        if region.get("classification"):
            header += f" [{region.get('classification')}]"
        if exits:
            header += " -> " + ", ".join(exits)
        lines.append(lp + header + " {")

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
                return f"node {target}" + suffix
            if edge_kind == "fallthrough_exit":
                return (f"exit {target}" if target else "exit") + suffix
            if edge_kind == "external_exit":
                return f"external {target}" + suffix
            if edge_kind == "structured_jump":
                return f"structured {target}" + suffix
            if edge_kind == "exit":
                return "exit" + suffix
            return str(target or "exit") + suffix

        for node in region.get("nodes") or []:
            np = "    " * (loop_indent + 1)
            lines.append(np + f"node {node.get('block_id')} {{")
            body_indent = loop_indent + 2
            for stmt in node.get("prelabel") or []:
                lines.extend("    " * body_indent + line for line in _render_statement_lines(stmt))
            for stmt in node.get("body") or []:
                lines.extend("    " * body_indent + line for line in _render_statement_lines(stmt))
            if node.get("terminator_kind") == "branch":
                cond = _render_expr(node.get("condition")) or "<cond>"
                lines.append("    " * body_indent + f"if ({cond}) -> {edge_text(node.get('true_edge') or {})}")
                lines.append("    " * body_indent + f"else -> {edge_text(node.get('false_edge') or {})}")
            elif node.get("next_edge"):
                lines.append("    " * body_indent + f"next -> {edge_text(node.get('next_edge') or {})}")
            for stmt in node.get("terminal") or []:
                lines.extend("    " * body_indent + line for line in _render_statement_lines(stmt))
            lines.append(np + "}")
        lines.append(lp + "}")
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
    ast_metrics = _collect_ast_metrics(ast_root)
    semantic_validation = _validate_ast_semantics(ast_root, ast_metrics)
    ast_text = _render_function_text(name, list(entry_args), ast_root, include_text)
    summary = {
        "normalized_basic_block_count": len(blocks),
        "max_ast_depth": int(ast_metrics.get("ast_max_depth", 0)),
        "region_kind_histogram": structured_meta.get("constructs", {}),
        "ast_parameterized_region_count": int(ast_metrics.get("ast_parameterized_region_count", 0)),
        "ast_region_param_count": int(ast_metrics.get("ast_region_param_count", 0)),
        "ast_parameterized_block_count": int(ast_metrics.get("ast_parameterized_block_count", 0)),
        "ast_block_param_count": int(ast_metrics.get("ast_block_param_count", 0)),
        "ast_semantic_error_count": int(semantic_validation.get("error_count", 0)),
        "ast_semantic_error_kind_histogram": semantic_validation.get("kind_histogram", {}),
        "ast_semantic_warning_count": int(semantic_validation.get("warning_count", 0)),
        "ast_semantic_warning_kind_histogram": semantic_validation.get("warning_kind_histogram", {}),
        "ast_unresolved_call_rel_count": int(ast_metrics.get("ast_unresolved_call_rel_count", 0)),
    }
    diagnostics = {
        "structuring": structured_meta,
        "ast_metrics": ast_metrics,
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
    region_hist: Counter[str] = Counter()
    semantic_error_hist: Counter[str] = Counter()
    semantic_warning_hist: Counter[str] = Counter()
    for summary in function_summaries:
        _merge_counter(region_hist, summary.get("region_kind_histogram", {}))
        _merge_counter(semantic_error_hist, summary.get("ast_semantic_error_kind_histogram", {}))
        _merge_counter(semantic_warning_hist, summary.get("ast_semantic_warning_kind_histogram", {}))
    return {
        "function_count": len(function_summaries),
        "total_normalized_basic_blocks": sum(summary.get("normalized_basic_block_count", 0) for summary in function_summaries),
        "max_ast_depth": max((summary.get("max_ast_depth", 0) for summary in function_summaries), default=0),
        "total_ast_parameterized_regions": sum(summary.get("ast_parameterized_region_count", 0) for summary in function_summaries),
        "total_ast_region_params": sum(summary.get("ast_region_param_count", 0) for summary in function_summaries),
        "total_ast_parameterized_blocks": sum(summary.get("ast_parameterized_block_count", 0) for summary in function_summaries),
        "total_ast_block_params": sum(summary.get("ast_block_param_count", 0) for summary in function_summaries),
        "total_ast_semantic_errors": sum(summary.get("ast_semantic_error_count", 0) for summary in function_summaries),
        "total_ast_semantic_warnings": sum(summary.get("ast_semantic_warning_count", 0) for summary in function_summaries),
        "total_ast_unresolved_call_rels": sum(summary.get("ast_unresolved_call_rel_count", 0) for summary in function_summaries),
        "region_kind_histogram": dict(region_hist),
        "ast_semantic_error_kind_histogram": dict(semantic_error_hist),
        "ast_semantic_warning_kind_histogram": dict(semantic_warning_hist),
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
                "AST builds statement/expression nodes directly from normalized HIR CFG blocks and keeps source-specific edge aliases from hiding unrelated linear merge-param assignments.",
                "Text is an output-only renderer concern; branch, loop, alias, and edge-parameter normalization operate on nodes.",
                "Complex cyclic CFG regions are represented as decision_loop data instead of renderer-side targeted jumps.",
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

    semantic_errors: list[dict[str, Any]] = []
    unresolved_calls: list[dict[str, Any]] = []
    for module in module_payloads:
        for fn in module.get("functions", []):
            fn_summary = fn.get("summary", {})
            ref = _entry_ref(module, fn)
            item = {
                **ref,
                "normalized_basic_block_count": fn_summary.get("normalized_basic_block_count"),
                "ast_semantic_error_count": fn_summary.get("ast_semantic_error_count"),
                "ast_semantic_warning_count": fn_summary.get("ast_semantic_warning_count"),
                "ast_unresolved_call_rel_count": fn_summary.get("ast_unresolved_call_rel_count"),
            }
            if item.get("ast_semantic_error_count"):
                semantic_errors.append(item)
            if item.get("ast_unresolved_call_rel_count"):
                unresolved_calls.append(item)

    semantic_errors.sort(key=lambda item: (-int(item.get("ast_semantic_error_count") or 0), str(item.get("script_name")), str(item.get("function"))))
    unresolved_calls.sort(key=lambda item: (-int(item.get("ast_unresolved_call_rel_count") or 0), str(item.get("script_name")), str(item.get("function"))))
    return {
        "contract": {
            "version": "ast-report-v3",
            "ast_contract": AST_CONTRACT_VERSION,
            "layers": ["normalized_hir", "node_ast_builder", "ast_renderer", "ast_summary"],
            "notes": [
                "Report counters are limited to AST normalization, block-param materialization, and semantic validation signals.",
                "Renderer labels, fallback/residual-goto accounting, and raw statement/expression histograms are intentionally not report metrics.",
            ],
        },
        "summary": summary,
        "rankings": {
            "most_semantic_error_functions": semantic_errors[:64],
            "most_unresolved_call_rel_functions": unresolved_calls[:64],
        },
        "modules": [
            {
                "script_name": module.get("script_name"),
                "function_count": module.get("summary", {}).get("function_count", 0),
                "normalized_basic_block_count": module.get("summary", {}).get("total_normalized_basic_blocks", 0),
                "ast_semantic_error_count": module.get("summary", {}).get("total_ast_semantic_errors", 0),
                "ast_semantic_warning_count": module.get("summary", {}).get("total_ast_semantic_warnings", 0),
                "ast_unresolved_call_rel_count": module.get("summary", {}).get("total_ast_unresolved_call_rels", 0),
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
