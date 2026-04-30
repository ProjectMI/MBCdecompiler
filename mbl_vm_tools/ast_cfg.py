from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Optional

from mbl_vm_tools.hir import HIRBlock, _block_maps, _ordered_unique


@dataclass
class AstCFG:
    blocks: list[HIRBlock]
    by_id: dict[str, HIRBlock]
    index_by_id: dict[str, int]
    succs: dict[str, list[str]]
    preds: dict[str, list[str]]
    dom: dict[str, set[str]]
    postdom: dict[str, set[str]]
    ipdom: dict[str, Optional[str]]
    loops: dict[str, dict[str, Any]]
    cyclic_components: dict[str, dict[str, Any]]


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


def compute_dominators(blocks: list[HIRBlock]) -> dict[str, set[str]]:
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


def compute_postdominators(blocks: list[HIRBlock]) -> dict[str, set[str]]:
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


def immediate_postdom(block_id: str, postdom: dict[str, set[str]], index_by_id: dict[str, int]) -> Optional[str]:
    candidates = list(postdom.get(block_id, set()) - {block_id})
    if not candidates:
        return None
    candidates.sort(key=lambda item: index_by_id.get(item, 10**9))
    for candidate in candidates:
        if all(candidate not in postdom.get(other, set()) for other in candidates if other != candidate):
            return candidate
    return candidates[0]


def edge_targets(block: HIRBlock, index_by_id: dict[str, int]) -> tuple[Optional[str], Optional[str]]:
    if block.terminator.get("kind") != "branch":
        return None, block.fallthrough_target or (block.successors[0] if block.successors else None)
    branch = block.branch_target
    fallthrough = block.fallthrough_target
    if branch is None and block.successors:
        ordered = sorted(block.successors, key=lambda item: index_by_id.get(item, 10**9))
        branch = ordered[0]
        fallthrough = ordered[1] if len(ordered) > 1 else None
    return branch, fallthrough


_COND_CMP_RE = re.compile(r"^cond\[(?P<op>0x[0-9A-Fa-f]+)\]\((?P<payload>.*)\)$")


def constant_branch_value(block: HIRBlock) -> Optional[bool]:
    if block.terminator.get("kind") != "branch":
        return None
    condition = str(block.terminator.get("condition") or block.terminator.get("text") or "").strip()
    match = _COND_CMP_RE.match(condition)
    if match is None:
        return None
    payload = match.group("payload").strip()
    if not (payload.startswith("cmp(") and payload.endswith(")")):
        return None
    args = _split_top_level(payload[4:-1])
    if len(args) != 2 or args[0].strip() != args[1].strip():
        return None
    op = str(block.terminator.get("branch_op") or match.group("op")).upper()
    if op == "0X4A":
        return True
    return None


def fold_constant_branches(blocks: list[HIRBlock]) -> list[HIRBlock]:
    if not blocks:
        return blocks
    index_by_id = {block.id: block.index for block in blocks}
    folded: list[HIRBlock] = []
    changed = False
    for block in blocks:
        constant = constant_branch_value(block)
        if constant is None:
            folded.append(block)
            continue
        kept_target = block.branch_target if constant else block.fallthrough_target
        kept_index = index_by_id.get(kept_target, 10**9) if kept_target is not None else 10**9
        if kept_index <= block.index:
            folded.append(block)
            continue
        term = dict(block.terminator)
        term.update(
            {
                "kind": "fallthrough",
                "text": None,
                "condition": None,
                "folded_branch_condition": block.terminator.get("condition") or block.terminator.get("text"),
                "folded_branch_value": constant,
            }
        )
        successors = [kept_target] if kept_target else []
        folded.append(
            replace(
                block,
                terminator=term,
                branch_target=None,
                fallthrough_target=kept_target,
                successors=successors,
            )
        )
        changed = True
    if not changed:
        return blocks
    ids = {block.id for block in folded}
    preds: dict[str, list[str]] = {block.id: [] for block in folded}
    for block in folded:
        for succ in block.successors:
            if succ in ids:
                preds[succ].append(block.id)
    return [replace(block, predecessors=preds.get(block.id, [])) for block in folded]


def prune_unreachable_blocks(blocks: list[HIRBlock]) -> list[HIRBlock]:
    if not blocks:
        return blocks
    by_id = {block.id: block for block in blocks}
    reachable: set[str] = set()
    stack = [blocks[0].id]
    while stack:
        block_id = stack.pop()
        if block_id in reachable or block_id not in by_id:
            continue
        reachable.add(block_id)
        stack.extend(succ for succ in by_id[block_id].successors if succ in by_id and succ not in reachable)
    if len(reachable) == len(blocks):
        return blocks

    pruned: list[HIRBlock] = []
    for new_index, block in enumerate(block for block in blocks if block.id in reachable):
        successors = [succ for succ in block.successors if succ in reachable]
        incoming_args = {pred: list(args) for pred, args in block.incoming_args.items() if pred in reachable}
        branch_target = block.branch_target if block.branch_target in reachable else None
        fallthrough_target = block.fallthrough_target if block.fallthrough_target in reachable else None
        term = dict(block.terminator)
        if isinstance(term.get("successors"), list):
            term["successors"] = [succ for succ in term.get("successors", []) if succ in reachable]
        if term.get("branch_target") not in reachable:
            term["branch_target"] = None
        if term.get("fallthrough_target") not in reachable:
            term["fallthrough_target"] = None
        pruned.append(
            replace(
                block,
                index=new_index,
                incoming_args=incoming_args,
                branch_target=branch_target,
                fallthrough_target=fallthrough_target,
                successors=successors,
                terminator=term,
            )
        )

    ids = {block.id for block in pruned}
    preds: dict[str, list[str]] = {block.id: [] for block in pruned}
    for block in pruned:
        for succ in block.successors:
            if succ in ids:
                preds[succ].append(block.id)
    return [replace(block, predecessors=preds.get(block.id, [])) for block in pruned]


def natural_loops(blocks: list[HIRBlock], dom: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
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


def cyclic_components(blocks: list[HIRBlock]) -> dict[str, dict[str, Any]]:
    if not blocks:
        return {}
    by_id, index_by_id, succs, preds = _block_maps(blocks)
    preorder: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    raw_components: list[list[str]] = []

    def visit(block_id: str) -> None:
        preorder[block_id] = len(preorder)
        lowlink[block_id] = preorder[block_id]
        stack.append(block_id)
        on_stack.add(block_id)
        for succ in succs.get(block_id, []):
            if succ not in by_id:
                continue
            if succ not in preorder:
                visit(succ)
                lowlink[block_id] = min(lowlink[block_id], lowlink[succ])
            elif succ in on_stack:
                lowlink[block_id] = min(lowlink[block_id], preorder[succ])
        if lowlink[block_id] != preorder[block_id]:
            return
        component: list[str] = []
        while stack:
            item = stack.pop()
            on_stack.remove(item)
            component.append(item)
            if item == block_id:
                break
        raw_components.append(component)

    for block in blocks:
        if block.id not in preorder:
            visit(block.id)

    components: dict[str, dict[str, Any]] = {}
    for component in raw_components:
        nodes = set(component)
        if len(nodes) == 1:
            only = next(iter(nodes))
            if only not in succs.get(only, []):
                continue
        ordered_nodes = sorted(nodes, key=lambda item: index_by_id.get(item, 10**9))
        idx0 = index_by_id[ordered_nodes[0]]
        idx1 = index_by_id[ordered_nodes[-1]]
        contiguous_ids = {blocks[idx].id for idx in range(idx0, idx1 + 1)}
        internal_edges = [
            (source, succ)
            for source in ordered_nodes
            for succ in succs.get(source, [])
            if succ in nodes
        ]
        backward_edges = [
            (source, succ)
            for source, succ in internal_edges
            if index_by_id.get(succ, 10**9) <= index_by_id.get(source, -1)
        ]
        nonlinear_internal_edges = [
            (source, succ)
            for source, succ in internal_edges
            if index_by_id.get(succ, 10**9) != index_by_id.get(source, -1) + 1
        ]
        external_entries = [
            (pred, node)
            for node in ordered_nodes
            for pred in preds.get(node, [])
            if pred not in nodes
        ]
        exit_ids = _ordered_unique([
            succ
            for source in ordered_nodes
            for succ in succs.get(source, [])
            if succ not in nodes
        ])
        entry_id = ordered_nodes[0]
        components[entry_id] = {
            "entry": entry_id,
            "nodes": nodes,
            "ordered_nodes": ordered_nodes,
            "index_range": (idx0, idx1),
            "contiguous": contiguous_ids == nodes,
            "external_entries": external_entries,
            "exit_ids": exit_ids,
            "internal_edges": internal_edges,
            "backward_edges": backward_edges,
            "backward_targets": {target for _, target in backward_edges},
            "nonlinear_internal_edges": nonlinear_internal_edges,
            "branch_nodes": [block_id for block_id in ordered_nodes if by_id[block_id].terminator.get("kind") == "branch"],
        }
    return components

def build_cfg(blocks: list[HIRBlock]) -> AstCFG:
    by_id, index_by_id, succs, preds = _block_maps(blocks)
    dom = compute_dominators(blocks)
    postdom = compute_postdominators(blocks)
    ipdom = {block.id: immediate_postdom(block.id, postdom, index_by_id) for block in blocks}
    loops = natural_loops(blocks, dom)
    cyclic = cyclic_components(blocks)
    return AstCFG(blocks, by_id, index_by_id, succs, preds, dom, postdom, ipdom, loops, cyclic)


def external_successors(loop: dict[str, Any], cfg: AstCFG) -> list[str]:
    nodes = set(loop.get("nodes") or set())
    return _ordered_unique([
        succ
        for block_id in sorted(nodes, key=lambda item: cfg.index_by_id.get(item, 10**9))
        for succ in cfg.succs.get(block_id, [])
        if succ not in nodes
    ])


def loopback_edges(loop: dict[str, Any], cfg: AstCFG) -> dict[str, set[str]]:
    nodes = set(loop.get("nodes") or set())
    edges: dict[str, set[str]] = {}
    for source_id in nodes:
        source_idx = cfg.index_by_id.get(source_id, 10**9)
        for succ in cfg.succs.get(source_id, []):
            if succ in nodes and cfg.index_by_id.get(succ, 10**9) <= source_idx:
                edges.setdefault(succ, set()).add(source_id)
    return edges


