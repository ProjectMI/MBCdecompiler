from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .control import VMControlGraph, build_control_graph
from .parser import MBCModule
from .ir import select_function_body_vmir
from .vm_spec import decode_words
from .regions import (
    REGION_CONTRACT_VERSION,
    VIRTUAL_FUNCTION_EXIT,
    CYCLIC_MULTI_EXIT,
    REQUIRED_CONDITIONAL_EDGE_KINDS,
    _as_dicts,
    _build_scc_model,
    _conditional_edge_kind_set,
    _conditional_edge_targets,
    _conditional_out_edges,
    _canonical_conditional_contexts,
    _cyclic_branch_model,
    _nearest_common_postdominator_by_order,
    _stable_sorted,
)


BRANCH_CONTRACT_VERSION = "vm-branch-v13"
BRANCH_LIFT_POLICY = (
    "Branch lifting consumes hierarchical vm-region-v13 SCC facts. It lifts "
    "cross-SCC conditional regions only when canonical conditional successors "
    "enter distinct SCCs on the augmented SCC DAG. Same-SCC conditionals are "
    "lifted only inside their owning cyclic SCC using a local projection that "
    "distinguishes forward joins, feedback/boundary joins, and cyclic multi-exit "
    "frontiers. The pass preserves VM taken/fallthrough edge identity, branch-arm "
    "role, hierarchy parent, and unresolved predicate polarity. BR-shaped "
    "predicate_no_transfer words live below CFG and are not branch-lift inputs."
)


@dataclass(frozen=True)
class VMBranchArm:
    id: str
    edge_kind: str
    successor: str
    successor_scc: str
    sccs: list[str]
    nodes: list[str]
    entry_edges: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    role: str = "vm_edge"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMLiftedBranch:
    id: str
    kind: str
    header: str
    header_scc: str
    exit: str
    exit_kind: str
    join_scc: str
    arms: list[VMBranchArm]
    nodes: list[str]
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)
    region_kind: str = "conditional_region"
    hierarchy_parent: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["arms"] = [a.to_dict() for a in self.arms]
        return payload


@dataclass(frozen=True)
class VMBranchLiftReport:
    contract: str
    summary: dict[str, Any]
    branches: list[VMLiftedBranch]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "summary": self.summary,
            "branches": [b.to_dict() for b in self.branches],
        }


# ---------------------------------------------------------------------------


def _edge_sort_key(edge: dict[str, Any]) -> tuple[int, str, str, int]:
    return (
        0 if edge.get("kind") == "conditional_taken" else 1 if edge.get("kind") == "conditional_fallthrough" else 2,
        str(edge.get("source")),
        str(edge.get("target")),
        int(edge.get("instruction_offset", -1) or -1),
    )


def _edge_payload(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        key: edge.get(key)
        for key in ["source", "target", "kind", "status", "word_index", "instruction_offset", "local_target", "formula"]
        if edge.get(key) is not None
    }


def _arm_sort_key(item: tuple[str, list[dict[str, Any]], str]) -> tuple[int, str]:
    successor, edges, _scc = item
    kinds = {str(e.get("kind")) for e in edges}
    if "conditional_taken" in kinds:
        rank = 0
    elif "conditional_fallthrough" in kinds:
        rank = 1
    else:
        rank = 2
    return rank, successor


def _arm_role(edge_kind: str) -> str:
    kinds = set(str(edge_kind).split("|"))
    if "conditional_taken" in kinds:
        return "vm_taken"
    if "conditional_fallthrough" in kinds:
        return "vm_fallthrough"
    if "jump" in kinds:
        return "vm_jump"
    return "vm_edge"


def _reachable_sccs_until(
    *,
    start_scc: str,
    join_scc: str,
    scc_succs: dict[str, set[str]],
    scc_nodes: set[str],
) -> set[str]:
    if start_scc == join_scc or start_scc == VIRTUAL_FUNCTION_EXIT:
        return set()
    out: set[str] = set()
    q: deque[str] = deque([start_scc])
    while q:
        cur = q.popleft()
        if cur == join_scc or cur == VIRTUAL_FUNCTION_EXIT or cur in out or cur not in scc_nodes:
            continue
        out.add(cur)
        for nxt in sorted(scc_succs.get(cur, set())):
            if nxt != join_scc and nxt not in out:
                q.append(nxt)
    return out


def _build_arm(
    *,
    arm_id: str,
    successor: str,
    successor_scc: str,
    join_scc: str,
    entry_edges: list[dict[str, Any]],
    scc_succs: dict[str, set[str]],
    scc_nodes: set[str],
    blocks_by_scc: dict[str, list[str]],
) -> VMBranchArm:
    arm_sccs = _reachable_sccs_until(
        start_scc=successor_scc,
        join_scc=join_scc,
        scc_succs=scc_succs,
        scc_nodes=scc_nodes,
    )
    nodes: list[str] = []
    for scc in _stable_sorted(arm_sccs):
        nodes.extend(blocks_by_scc.get(scc, []))
    kinds = _stable_sorted(str(e.get("kind")) for e in entry_edges)
    edge_kind = "|".join(kinds) if kinds else "unknown"
    return VMBranchArm(
        id=arm_id,
        edge_kind=edge_kind,
        successor=successor,
        successor_scc=successor_scc,
        sccs=_stable_sorted(arm_sccs),
        nodes=_stable_sorted(nodes),
        entry_edges=[_edge_payload(e) for e in sorted(entry_edges, key=_edge_sort_key)],
        diagnostics={
            "direct_to_join": successor_scc == join_scc,
            "scc_count": len(arm_sccs),
            "node_count": len(nodes),
        },
        role=_arm_role(edge_kind),
    )


def _build_cyclic_arm(
    *,
    arm_id: str,
    successor: str,
    successor_scc: str,
    entry_edges: list[dict[str, Any]],
    cyclic_model: dict[str, Any],
) -> VMBranchArm:
    kinds = _stable_sorted(str(e.get("kind")) for e in entry_edges)
    edge_kind = "|".join(kinds) if kinds else "unknown"
    local_join = cyclic_model.get("local_join")
    feedback_successors = set(cyclic_model.get("feedback_successors") or [])
    arm_nodes_by_successor = cyclic_model.get("arm_nodes_by_successor") or {}
    nodes = list(arm_nodes_by_successor.get(successor) or [])
    return VMBranchArm(
        id=arm_id,
        edge_kind=edge_kind,
        successor=successor,
        successor_scc=successor_scc,
        sccs=[successor_scc],
        nodes=_stable_sorted(nodes),
        entry_edges=[_edge_payload(e) for e in sorted(entry_edges, key=_edge_sort_key)],
        diagnostics={
            "direct_to_join": local_join is not None and successor == local_join,
            "local_join": local_join,
            "local_join_kind": cyclic_model.get("local_join_kind"),
            "cyclic_feedback_arm": successor in feedback_successors,
            "scc_count": 1,
            "node_count": len(nodes),
        },
        role=_arm_role(edge_kind),
    )


# ---------------------------------------------------------------------------
# Branch lifting


def analyze_branch_lifts(cfg: VMControlGraph | dict[str, Any], *, include_branches: bool = True) -> VMBranchLiftReport:
    blocks, edges = _as_dicts(cfg)
    if not blocks:
        return VMBranchLiftReport(
            contract=BRANCH_CONTRACT_VERSION,
            summary={
                "block_count": 0,
                "proven_edge_count": 0,
                "conditional_branch_count": 0,
                "conditional_two_edge_count": 0,
                "lifted_branch_count": 0,
                "lifted_cross_scc_branch_count": 0,
                "lifted_cyclic_scc_branch_count": 0,
                "cyclic_local_join_branch_count": 0,
                "cyclic_feedback_join_branch_count": 0,
                "cyclic_boundary_join_branch_count": 0,
                "cyclic_multi_exit_branch_count": 0,
                "cyclic_feedback_arm_count": 0,
                "arm_role_histogram": {},
                "policy": BRANCH_LIFT_POLICY,
            },
            branches=[],
        )

    model = _build_scc_model(blocks, edges)
    proven_edges: list[dict[str, Any]] = model["proven_edges"]
    succs: dict[str, set[str]] = model["succs"]
    preds: dict[str, set[str]] = model["preds"]
    out_edges_by_source: dict[str, list[dict[str, Any]]] = model["out_edges_by_source"]
    reachable: set[str] = model["reachable"]
    scc_by_block: dict[str, str] = model["scc_by_block"]
    blocks_by_scc: dict[str, list[str]] = model["blocks_by_scc"]
    scc_succs: dict[str, set[str]] = model["scc_succs"]
    scc_nodes: set[str] = model["scc_nodes"]
    cyclic_sccs: set[str] = model["cyclic_sccs"]
    pdom: dict[str, set[str]] = model["pdom"]

    block_order = {str(block.get("id")): idx for idx, block in enumerate(blocks)}
    entry_blocks_by_scc: dict[str, list[str]] = {}
    for scc_id, nodes in blocks_by_scc.items():
        node_set = set(nodes)
        entry_blocks_by_scc[scc_id] = _stable_sorted([
            block for block in nodes
            if any(pred not in node_set for pred in preds.get(block, set())) or block == blocks[0].get("id")
        ])

    edge_lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in proven_edges:
        edge_lookup[(str(edge.get("source")), str(edge.get("target")))].append(edge)

    conditional_contexts = _canonical_conditional_contexts(
        blocks,
        out_edges_by_source,
        reachable,
        scc_by_block=scc_by_block,
        cyclic_sccs=cyclic_sccs,
    )
    canonical_conditional_headers: set[str] = conditional_contexts["canonical_headers"]
    conditional_alias_to_canonical: dict[str, str] = conditional_contexts["alias_to_canonical"]
    conditional_aliases_by_canonical: dict[str, list[str]] = conditional_contexts["aliases_by_canonical"]

    branches: list[VMLiftedBranch] = []
    conditional_branch_count = 0
    conditional_two_successor_count = 0
    conditional_two_edge_count = 0
    conditional_same_scc_successors_count = 0
    conditional_cross_scc_count = 0
    function_exit_branch_count = 0
    real_join_branch_count = 0
    lifted_cross_scc_branch_count = 0
    lifted_cyclic_scc_branch_count = 0
    cyclic_local_join_branch_count = 0
    cyclic_feedback_join_branch_count = 0
    cyclic_boundary_join_branch_count = 0
    cyclic_multi_exit_branch_count = 0
    conditional_alias_context_count = int(conditional_contexts.get("alias_count", 0) or 0)
    conditional_alias_group_count = int(conditional_contexts.get("alias_group_count", 0) or 0)
    cyclic_feedback_arm_count = 0
    arm_scc_count_hist: Counter[str] = Counter()
    arm_role_hist: Counter[str] = Counter()
    distinct_successor_count_hist: Counter[str] = Counter()
    conditional_edge_count_hist: Counter[str] = Counter()
    lifted_branch_count = 0

    for block in blocks:
        header = str(block.get("id"))
        if header not in reachable:
            continue
        terminator = block.get("terminator") or {}
        semantic = (terminator.get("semantic") or {}) if isinstance(terminator, dict) else {}
        if semantic.get("branch_kind") != "conditional_branch":
            continue
        if header in conditional_alias_to_canonical:
            continue
        if header not in canonical_conditional_headers:
            continue
        conditional_branch_count += 1
        conditional_edges = conditional_contexts["edges_by_header"].get(header) or _conditional_out_edges(out_edges_by_source, header, reachable)
        edge_kinds = _conditional_edge_kind_set(conditional_edges)
        edge_targets = _conditional_edge_targets(conditional_edges)
        successors = _stable_sorted(set(edge_targets))
        conditional_edge_count_hist[str(len(conditional_edges))] += 1
        distinct_successor_count_hist[str(len(successors))] += 1
        if len(conditional_edges) != 2 or edge_kinds != set(REQUIRED_CONDITIONAL_EDGE_KINDS):
            raise ValueError(
                "canonical conditional branch must have exactly one taken and one fallthrough proven edge"
            )
        conditional_two_edge_count += 1

        header_scc = scc_by_block[header]
        if len(successors) != 2:
            raise ValueError("canonical conditional branch must have two distinct successor blocks")

        conditional_two_successor_count += 1
        arm_inputs = []
        for successor in successors:
            successor_scc = scc_by_block[successor]
            arm_inputs.append((successor, edge_lookup.get((header, successor), []), successor_scc))
        successor_sccs = _stable_sorted({scc for _succ, _edges, scc in arm_inputs})
        if len(successor_sccs) == 1 and successor_sccs[0] == header_scc:
            conditional_same_scc_successors_count += 1
            cyclic_model = _cyclic_branch_model(
                header=header,
                successors=successors,
                conditional_edges=conditional_edges,
                header_scc=header_scc,
                proven_edges=proven_edges,
                scc_by_block=scc_by_block,
                blocks_by_scc=blocks_by_scc,
                cyclic_sccs=cyclic_sccs,
                entry_blocks_by_scc=entry_blocks_by_scc,
                block_order=block_order,
            )
            lifted_branch_count += 1
            lifted_cyclic_scc_branch_count += 1
            local_join_kind = str(cyclic_model.get("local_join_kind"))
            if local_join_kind == "cyclic_local_join":
                cyclic_local_join_branch_count += 1
            elif local_join_kind == "cyclic_feedback_join":
                cyclic_feedback_join_branch_count += 1
            elif local_join_kind == "cyclic_boundary_join":
                cyclic_boundary_join_branch_count += 1
            elif local_join_kind == "cyclic_multi_exit":
                cyclic_multi_exit_branch_count += 1
            else:
                raise ValueError(f"unsupported cyclic branch join kind: {local_join_kind}")
            feedback_successors = set(cyclic_model.get("feedback_successors") or [])
            cyclic_feedback_arm_count += len(feedback_successors)
            if not include_branches:
                continue

            arms: list[VMBranchArm] = []
            for arm_index, (successor, entry_edges, successor_scc) in enumerate(sorted(arm_inputs, key=_arm_sort_key)):
                arm = _build_cyclic_arm(
                    arm_id=f"arm{arm_index}",
                    successor=successor,
                    successor_scc=successor_scc,
                    entry_edges=entry_edges,
                    cyclic_model=cyclic_model,
                )
                arms.append(arm)
                arm_scc_count_hist[str(len(arm.sccs))] += 1
                arm_role_hist[arm.role] += 1

            local_join = cyclic_model.get("local_join")
            exit_text = str(local_join) if local_join is not None else CYCLIC_MULTI_EXIT
            branch_nodes = {header, *successors}
            if local_join is not None:
                branch_nodes.add(str(local_join))
            for frontier in cyclic_model.get("arm_exit_frontier_by_successor", {}).values():
                branch_nodes.update(frontier)
            for arm in arms:
                branch_nodes.update(arm.nodes)
            hierarchy_parent = f"scc:{header_scc}"
            branches.append(
                VMLiftedBranch(
                    id=f"branch{len(branches)}",
                    kind="conditional_cyclic_scc_branch",
                    header=header,
                    header_scc=header_scc,
                    exit=exit_text,
                    exit_kind=local_join_kind,
                    join_scc=header_scc,
                    arms=arms,
                    nodes=_stable_sorted(branch_nodes),
                    confidence=1.0,
                    evidence={
                        "region_contract": REGION_CONTRACT_VERSION,
                        "rule": "header and both conditional successors stay inside one cyclic SCC; classify the local projection as forward-join, boundary-join, multi-exit, or open diagnostic",
                        "successors": [succ for succ, _edges, _scc in sorted(arm_inputs, key=_arm_sort_key)],
                        "successor_sccs": successor_sccs,
                        "local_join": local_join,
                        "local_join_kind": local_join_kind,
                        "local_join_candidates": cyclic_model.get("local_join_candidates", [])[:8],
                        "boundary_join_candidates": cyclic_model.get("boundary_join_candidates", [])[:8],
                        "exit_frontier": cyclic_model.get("exit_frontier", [])[:16],
                        "feedback_successors": cyclic_model.get("feedback_successors", []),
                        "cut_edge_count": cyclic_model.get("cut_edge_count", 0),
                        "boundary_edge_count": cyclic_model.get("boundary_edge_count", 0),
                        "predicate_polarity": semantic.get("predicate_polarity"),
                        "alias_contexts": conditional_aliases_by_canonical.get(header, []),
                        "hierarchy_parent": hierarchy_parent,
                    },
                    region_kind="cyclic_branch_region",
                    hierarchy_parent=hierarchy_parent,
                )
            )
            continue

        if len(successor_sccs) == 1:
            raise ValueError(
                "canonical conditional header is outside the single successor SCC; "
                "expected this shape to be removed by branch-atom canonicalization"
            )

        conditional_cross_scc_count += 1
        join_scc, join_candidates = _nearest_common_postdominator_by_order(
            successor_sccs,
            pdom=pdom,
            exclude={header_scc},
        )
        if join_scc is None:
            raise ValueError(
                "cross-SCC conditional branch has no unique common postdominator on the augmented SCC DAG"
            )

        if join_scc == VIRTUAL_FUNCTION_EXIT:
            exit_kind = "virtual_function_exit"
            region_kind = "conditional_multi_exit_region"
            function_exit_branch_count += 1
            join_nodes: list[str] = []
        else:
            exit_kind = "real_scc_join"
            region_kind = "conditional_region"
            real_join_branch_count += 1
            join_nodes = blocks_by_scc.get(join_scc, [])

        hierarchy_parent = f"scc:{header_scc}" if header_scc in cyclic_sccs else "function"
        lifted_branch_count += 1
        lifted_cross_scc_branch_count += 1
        if not include_branches:
            continue

        arms: list[VMBranchArm] = []
        for arm_index, (successor, entry_edges, successor_scc) in enumerate(sorted(arm_inputs, key=_arm_sort_key)):
            arm = _build_arm(
                arm_id=f"arm{arm_index}",
                successor=successor,
                successor_scc=successor_scc,
                join_scc=join_scc,
                entry_edges=entry_edges,
                scc_succs=scc_succs,
                scc_nodes=scc_nodes,
                blocks_by_scc=blocks_by_scc,
            )
            arms.append(arm)
            arm_scc_count_hist[str(len(arm.sccs))] += 1
            arm_role_hist[arm.role] += 1

        branch_nodes = {header, *join_nodes}
        for arm in arms:
            branch_nodes.update(arm.nodes)
        branches.append(
            VMLiftedBranch(
                id=f"branch{len(branches)}",
                kind="conditional_scc_branch",
                header=header,
                header_scc=header_scc,
                exit=join_scc,
                exit_kind=exit_kind,
                join_scc=join_scc,
                arms=arms,
                nodes=_stable_sorted(branch_nodes),
                confidence=1.0,
                evidence={
                    "region_contract": REGION_CONTRACT_VERSION,
                    "rule": "two proven conditional successors enter distinct SCCs and have a nearest common postdominator on the augmented SCC DAG",
                    "successors": [succ for succ, _edges, _scc in sorted(arm_inputs, key=_arm_sort_key)],
                    "successor_sccs": successor_sccs,
                    "join_candidates": join_candidates[:8],
                    "predicate_polarity": semantic.get("predicate_polarity"),
                        "alias_contexts": conditional_aliases_by_canonical.get(header, []),
                    "hierarchy_parent": hierarchy_parent,
                },
                region_kind=region_kind,
                hierarchy_parent=hierarchy_parent,
            )
        )

    summary = {
        "block_count": len(blocks),
        "proven_edge_count": len(proven_edges),
        "reachable_block_count": len(reachable),
        "unreachable_under_proven_edges_count": max(0, len(blocks) - len(reachable)),
        "scc_count": len(scc_nodes),
        "cyclic_scc_count": len(cyclic_sccs),
        "conditional_branch_count": conditional_branch_count,
        "conditional_branch_context_count": int(conditional_contexts.get("context_count", 0) or 0),
        "conditional_branch_atom_count": int(conditional_contexts.get("atom_count", 0) or 0),
        "conditional_alias_context_count": conditional_alias_context_count,
        "conditional_alias_group_count": conditional_alias_group_count,
        "conditional_alias_group_size_histogram": dict(sorted((conditional_contexts.get("alias_group_size_histogram") or {}).items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "conditional_two_successor_count": conditional_two_successor_count,
        "conditional_two_edge_count": conditional_two_edge_count,
        "conditional_edge_count_histogram": dict(sorted(conditional_edge_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "conditional_distinct_successor_count_histogram": dict(sorted(distinct_successor_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "conditional_same_scc_successors_count": conditional_same_scc_successors_count,
        "conditional_cross_scc_count": conditional_cross_scc_count,
        "lifted_branch_count": lifted_branch_count,
        "lifted_cross_scc_branch_count": lifted_cross_scc_branch_count,
        "lifted_cyclic_scc_branch_count": lifted_cyclic_scc_branch_count,
        "cyclic_local_join_branch_count": cyclic_local_join_branch_count,
        "cyclic_feedback_join_branch_count": cyclic_feedback_join_branch_count,
        "cyclic_boundary_join_branch_count": cyclic_boundary_join_branch_count,
        "cyclic_multi_exit_branch_count": cyclic_multi_exit_branch_count,
        "cyclic_feedback_arm_count": cyclic_feedback_arm_count,
        "real_join_branch_count": real_join_branch_count,
        "function_exit_branch_count": function_exit_branch_count,
        "arm_scc_count_histogram": dict(sorted(arm_scc_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "arm_role_histogram": dict(sorted(arm_role_hist.items())),
        "policy": BRANCH_LIFT_POLICY,
    }
    return VMBranchLiftReport(BRANCH_CONTRACT_VERSION, summary, branches)


def analyze_module_branch_lifts(
    module: MBCModule | str | Path,
    *,
    function: Optional[str] = None,
    limit_functions: Optional[int] = None,
) -> dict[str, Any]:
    mod = module if isinstance(module, MBCModule) else MBCModule(module)
    entries = [mod.get_function_entry(function)] if function else mod.function_entries(include_definitions=True, include_exports=True, dedupe=True)
    if limit_functions is not None:
        entries = entries[: max(0, int(limit_functions))]

    functions: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    arm_role_totals: Counter[str] = Counter()
    arm_scc_count_totals: Counter[str] = Counter()
    for entry in entries:
        raw, selection = select_function_body_vmir(mod, entry)
        words = decode_words(raw)
        span = selection.get("span") or {"start": 0, "end": 0}
        cfg = build_control_graph(words, function_start=int(span.get("start", 0) or 0), raw=raw)
        report = analyze_branch_lifts(cfg).to_dict()
        summary = report["summary"]
        for key in [
            "block_count",
            "proven_edge_count",
            "reachable_block_count",
            "unreachable_under_proven_edges_count",
            "scc_count",
            "cyclic_scc_count",
            "conditional_branch_count",
            "conditional_branch_context_count",
            "conditional_branch_atom_count",
            "conditional_alias_context_count",
            "conditional_alias_group_count",
            "conditional_two_successor_count",
            "conditional_two_edge_count",
            "conditional_same_scc_successors_count",
            "conditional_cross_scc_count",
            "lifted_branch_count",
            "lifted_cross_scc_branch_count",
            "lifted_cyclic_scc_branch_count",
            "cyclic_local_join_branch_count",
            "cyclic_feedback_join_branch_count",
            "cyclic_boundary_join_branch_count",
            "cyclic_multi_exit_branch_count",
            "cyclic_feedback_arm_count",
            "real_join_branch_count",
            "function_exit_branch_count",
        ]:
            totals[key] += int(summary.get(key, 0) or 0)
        arm_role_totals.update(summary.get("arm_role_histogram", {}))
        arm_scc_count_totals.update(summary.get("arm_scc_count_histogram", {}))
        functions.append({
            "name": entry.name,
            "symbol": entry.symbol,
            "span": span,
            "summary": summary,
            "branches": report["branches"] if function else [],
        })
    return {
        "contract": BRANCH_CONTRACT_VERSION,
        "module": str(mod.path),
        "summary": {
            **dict(totals),
            "function_count": len(functions),
            "arm_role_histogram": dict(sorted(arm_role_totals.items())),
            "arm_scc_count_histogram": dict(sorted(arm_scc_count_totals.items(), key=lambda kv: (int(kv[0]), kv[0]))),
            "policy": BRANCH_LIFT_POLICY,
        },
        "functions": functions,
    }


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Lift VM conditional branches from hierarchical SCC regions")
    parser.add_argument("module", type=Path)
    parser.add_argument("--function", default=None)
    parser.add_argument("--limit-functions", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = analyze_module_branch_lifts(args.module, function=args.function, limit_functions=args.limit_functions)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
