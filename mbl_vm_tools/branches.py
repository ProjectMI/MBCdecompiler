from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .control import VMControlGraph, build_control_graph
from .parser import MBCModule
from .ir import select_function_body_vmir
from .vm_spec import decode_words
from .regions import (
    REGION_CONTRACT_VERSION,
    VIRTUAL_FUNCTION_EXIT,
    CYCLIC_FEEDBACK_FRONTIER,
    CYCLIC_FEEDBACK_EXIT_FRONTIER,
    CYCLIC_EXTERNAL_EXIT_FRONTIER,
    _as_dicts,
    _build_structural_projection_model,
    _entry_blocks_by_scc,
    _normalize_cyclic_entry_blocks,
    _conditional_out_edges,
    _cyclic_branch_model,
    _stable_sorted,
    analyze_regions,
)


BRANCH_CONTRACT_VERSION = "vm-branch-v15"
BRANCH_LIFT_POLICY = (
    "Branch lifting consumes region facts from the projected VM CFG and emits "
    "branch envelopes while preserving VM taken/fallthrough arm identity."
)
BRANCH_TOTAL_KEYS = "block_count proven_edge_count raw_block_count raw_proven_edge_count raw_scc_count raw_cyclic_scc_count structural_projection_block_count structural_projection_edge_count structural_projection_alias_block_count structural_projection_alias_context_count structural_projection_skipped_alias_internal_edge_count structural_projection_duplicate_edge_count reachable_block_count unreachable_under_proven_edges_count scc_count cyclic_scc_count raw_multi_entry_cyclic_scc_count raw_cfg_multi_entry_cyclic_scc_count projected_multi_entry_cyclic_scc_count normalized_multi_entry_cyclic_scc_count cyclic_entry_port_count cyclic_entry_port_scc_count conditional_branch_count conditional_branch_context_count conditional_branch_atom_count raw_conditional_branch_context_count raw_conditional_branch_atom_count conditional_alias_context_count conditional_alias_group_count conditional_two_successor_count conditional_two_edge_count conditional_projection_no_split_count conditional_same_scc_successors_count conditional_cross_scc_count lifted_branch_count lifted_cross_scc_branch_count lifted_cyclic_scc_branch_count cyclic_local_join_branch_count cyclic_feedback_join_branch_count cyclic_boundary_join_branch_count cyclic_feedback_frontier_branch_count cyclic_feedback_exit_frontier_branch_count cyclic_external_exit_frontier_branch_count cyclic_feedback_arm_count real_join_branch_count function_exit_branch_count".split()


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


def _edge_sort_key(edge: dict[str, Any]) -> tuple[int, str, str, int]:
    kind = str(edge.get("kind"))
    rank = 0 if kind == "conditional_taken" else 1 if kind == "conditional_fallthrough" else 2
    return (
        rank,
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
    return (0 if "conditional_taken" in kinds else 1 if "conditional_fallthrough" in kinds else 2, successor)


def _arm_role(edge_kind: str) -> str:
    kinds = set(str(edge_kind).split("|"))
    if "conditional_taken" in kinds:
        return "vm_taken"
    if "conditional_fallthrough" in kinds:
        return "vm_fallthrough"
    if "jump" in kinds:
        return "vm_jump"
    return "vm_edge"


def _reachable_sccs_until(start_scc: str, join_scc: str, scc_succs: dict[str, set[str]], scc_nodes: set[str]) -> set[str]:
    if start_scc == join_scc or start_scc == VIRTUAL_FUNCTION_EXIT:
        return set()
    out: set[str] = set()
    q: deque[str] = deque([start_scc])
    while q:
        cur = q.popleft()
        if cur == join_scc or cur == VIRTUAL_FUNCTION_EXIT or cur in out or cur not in scc_nodes:
            continue
        out.add(cur)
        q.extend(nxt for nxt in sorted(scc_succs.get(cur, set())) if nxt != join_scc and nxt not in out)
    return out


def _make_arm(
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
    arm_sccs = _reachable_sccs_until(successor_scc, join_scc, scc_succs, scc_nodes)
    nodes = [node for scc in _stable_sorted(arm_sccs) for node in blocks_by_scc.get(scc, [])]
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
        diagnostics={"direct_to_join": successor_scc == join_scc},
        role=_arm_role(edge_kind),
    )


def _make_cyclic_arm(
    *,
    arm_id: str,
    successor: str,
    successor_scc: str,
    entry_edges: list[dict[str, Any]],
    cyclic_model: dict[str, Any],
) -> VMBranchArm:
    kinds = _stable_sorted(str(e.get("kind")) for e in entry_edges)
    edge_kind = "|".join(kinds) if kinds else "unknown"
    nodes = list((cyclic_model.get("arm_nodes_by_successor") or {}).get(successor) or [])
    return VMBranchArm(
        id=arm_id,
        edge_kind=edge_kind,
        successor=successor,
        successor_scc=successor_scc,
        sccs=[successor_scc],
        nodes=_stable_sorted(nodes),
        entry_edges=[_edge_payload(e) for e in sorted(entry_edges, key=_edge_sort_key)],
        diagnostics={
            "direct_to_join": cyclic_model.get("local_join") is not None and successor == cyclic_model.get("local_join"),
            "local_join_kind": cyclic_model.get("local_join_kind"),
            "cyclic_feedback_arm": successor in set(cyclic_model.get("feedback_successors") or []),
        },
        role=_arm_role(edge_kind),
    )


def _branch_summary_from_regions(
    region_summary: dict[str, Any],
    *,
    arm_role_hist: Counter[str],
    arm_scc_count_hist: Counter[str],
    cyclic_feedback_arm_count: int,
) -> dict[str, Any]:
    cyclic_count = int(region_summary.get("conditional_cyclic_scc_region_count", 0) or 0)
    cross_count = int(region_summary.get("conditional_scc_region_count", 0) or 0)
    summary = {key: int(region_summary.get(key, 0) or 0) for key in BRANCH_TOTAL_KEYS}
    summary.update({
        "lifted_branch_count": cross_count + cyclic_count,
        "lifted_cross_scc_branch_count": cross_count,
        "lifted_cyclic_scc_branch_count": cyclic_count,
        "cyclic_local_join_branch_count": int(region_summary.get("conditional_cyclic_local_join_region_count", 0) or 0),
        "cyclic_feedback_join_branch_count": int(region_summary.get("conditional_cyclic_feedback_join_region_count", 0) or 0),
        "cyclic_boundary_join_branch_count": int(region_summary.get("conditional_cyclic_boundary_join_region_count", 0) or 0),
        "cyclic_feedback_frontier_branch_count": int(region_summary.get("conditional_cyclic_feedback_frontier_region_count", 0) or 0),
        "cyclic_feedback_exit_frontier_branch_count": int(region_summary.get("conditional_cyclic_feedback_exit_frontier_region_count", 0) or 0),
        "cyclic_external_exit_frontier_branch_count": int(region_summary.get("conditional_cyclic_external_exit_frontier_region_count", 0) or 0),
        "cyclic_feedback_arm_count": cyclic_feedback_arm_count,
        "real_join_branch_count": int(region_summary.get("conditional_real_scc_join_region_count", 0) or 0),
        "function_exit_branch_count": int(region_summary.get("conditional_function_exit_scc_region_count", 0) or 0),
        "arm_scc_count_histogram": dict(sorted(arm_scc_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "arm_role_histogram": dict(sorted(arm_role_hist.items())),
        "policy": BRANCH_LIFT_POLICY,
    })
    return summary


def _exit_text(local_join: Any, local_join_kind: str) -> str:
    if local_join is not None:
        return str(local_join)
    if local_join_kind == "cyclic_feedback_frontier":
        return CYCLIC_FEEDBACK_FRONTIER
    if local_join_kind == "cyclic_feedback_exit_frontier":
        return CYCLIC_FEEDBACK_EXIT_FRONTIER
    if local_join_kind == "cyclic_external_exit_frontier":
        return CYCLIC_EXTERNAL_EXIT_FRONTIER
    raise ValueError(f"unsupported cyclic branch join kind: {local_join_kind}")


def analyze_branch_lifts(cfg: VMControlGraph | dict[str, Any], *, include_branches: bool = True) -> VMBranchLiftReport:
    region_report = analyze_regions(cfg)
    region_summary = region_report.summary
    arm_role_hist: Counter[str] = Counter()
    arm_scc_count_hist: Counter[str] = Counter()
    cyclic_feedback_arm_count = 0
    branches: list[VMLiftedBranch] = []

    if include_branches and region_report.facts:
        raw_blocks, raw_edges = _as_dicts(cfg)
        projection_model = _build_structural_projection_model(raw_blocks, raw_edges)
        blocks = projection_model["blocks"]
        model = projection_model["model"]
        proven_edges: list[dict[str, Any]] = model["proven_edges"]
        out_edges_by_source: dict[str, list[dict[str, Any]]] = model["out_edges_by_source"]
        reachable: set[str] = model["reachable"]
        scc_by_block: dict[str, str] = model["scc_by_block"]
        blocks_by_scc: dict[str, list[str]] = model["blocks_by_scc"]
        scc_succs: dict[str, set[str]] = model["scc_succs"]
        scc_nodes: set[str] = model["scc_nodes"]
        cyclic_sccs: set[str] = model["cyclic_sccs"]
        pdom: dict[str, set[str]] = model["pdom"]
        block_order = {str(block.get("id")): idx for idx, block in enumerate(blocks)}
        raw_entry_blocks_by_scc = _entry_blocks_by_scc(blocks=blocks, blocks_by_scc=blocks_by_scc, preds=model["preds"])
        entry_blocks_by_scc, _ports = _normalize_cyclic_entry_blocks(
            proven_edges=proven_edges,
            blocks_by_scc=blocks_by_scc,
            scc_by_block=scc_by_block,
            cyclic_sccs=cyclic_sccs,
            raw_entry_blocks_by_scc=raw_entry_blocks_by_scc,
            block_order=block_order,
        )
        edge_lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for edge in proven_edges:
            edge_lookup[(str(edge.get("source")), str(edge.get("target")))].append(edge)

        for fact in region_report.facts:
            if fact.kind not in {"conditional_scc_region", "conditional_cyclic_scc_region"}:
                continue
            header = fact.header
            evidence = dict(fact.evidence or {})
            header_scc = str(evidence.get("header_scc") or scc_by_block.get(header, ""))
            successors = list(evidence.get("successors") or [])
            arm_inputs = [
                (successor, edge_lookup.get((header, successor), []), scc_by_block[successor])
                for successor in successors
                if successor in scc_by_block
            ]
            if len(arm_inputs) != 2:
                continue

            if fact.kind == "conditional_cyclic_scc_region":
                conditional_edges = _conditional_out_edges(out_edges_by_source, header, reachable)
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
                    pdom=pdom,
                )
                local_join = cyclic_model.get("local_join")
                local_join_kind = str(cyclic_model.get("local_join_kind"))
                arms = [
                    _make_cyclic_arm(
                        arm_id=f"arm{idx}",
                        successor=successor,
                        successor_scc=successor_scc,
                        entry_edges=entry_edges,
                        cyclic_model=cyclic_model,
                    )
                    for idx, (successor, entry_edges, successor_scc) in enumerate(sorted(arm_inputs, key=_arm_sort_key))
                ]
                branch_nodes = {header, *successors}
                if local_join is not None:
                    branch_nodes.add(str(local_join))
                for frontier in (cyclic_model.get("arm_exit_frontier_by_successor") or {}).values():
                    branch_nodes.update(frontier)
                for arm in arms:
                    branch_nodes.update(arm.nodes)
                    arm_role_hist[arm.role] += 1
                    arm_scc_count_hist[str(len(arm.sccs))] += 1
                cyclic_feedback_arm_count += len(cyclic_model.get("feedback_successors") or [])
                branches.append(VMLiftedBranch(
                    id=f"branch{len(branches)}",
                    kind="conditional_cyclic_scc_branch",
                    header=header,
                    header_scc=header_scc,
                    exit=_exit_text(local_join, local_join_kind),
                    exit_kind=local_join_kind,
                    join_scc=header_scc,
                    arms=arms,
                    nodes=_stable_sorted(branch_nodes),
                    evidence={
                        "region_contract": REGION_CONTRACT_VERSION,
                        "successors": successors,
                        "successor_sccs": evidence.get("successor_sccs", [header_scc]),
                        "local_join": local_join,
                        "local_join_kind": local_join_kind,
                        "exit_frontier": cyclic_model.get("exit_frontier", []),
                    },
                    region_kind="cyclic_branch_region",
                    hierarchy_parent=f"scc:{header_scc}",
                ))
                continue

            join_scc = str(evidence.get("join_scc") or fact.exit)
            join_nodes = [] if join_scc == VIRTUAL_FUNCTION_EXIT else blocks_by_scc.get(join_scc, [])
            exit_kind = "virtual_function_exit" if join_scc == VIRTUAL_FUNCTION_EXIT else "real_scc_join"
            region_kind = "conditional_multi_exit_region" if join_scc == VIRTUAL_FUNCTION_EXIT else "conditional_region"
            arms = [
                _make_arm(
                    arm_id=f"arm{idx}",
                    successor=successor,
                    successor_scc=successor_scc,
                    join_scc=join_scc,
                    entry_edges=entry_edges,
                    scc_succs=scc_succs,
                    scc_nodes=scc_nodes,
                    blocks_by_scc=blocks_by_scc,
                )
                for idx, (successor, entry_edges, successor_scc) in enumerate(sorted(arm_inputs, key=_arm_sort_key))
            ]
            branch_nodes = {header, *join_nodes}
            for arm in arms:
                branch_nodes.update(arm.nodes)
                arm_role_hist[arm.role] += 1
                arm_scc_count_hist[str(len(arm.sccs))] += 1
            hierarchy_parent = f"scc:{header_scc}" if header_scc in cyclic_sccs else "function"
            branches.append(VMLiftedBranch(
                id=f"branch{len(branches)}",
                kind="conditional_scc_branch",
                header=header,
                header_scc=header_scc,
                exit=join_scc,
                exit_kind=exit_kind,
                join_scc=join_scc,
                arms=arms,
                nodes=_stable_sorted(branch_nodes),
                evidence={
                    "region_contract": REGION_CONTRACT_VERSION,
                    "successors": successors,
                    "successor_sccs": evidence.get("successor_sccs", []),
                    "hierarchy_parent": hierarchy_parent,
                },
                region_kind=region_kind,
                hierarchy_parent=hierarchy_parent,
            ))

    summary = _branch_summary_from_regions(
        region_summary,
        arm_role_hist=arm_role_hist,
        arm_scc_count_hist=arm_scc_count_hist,
        cyclic_feedback_arm_count=cyclic_feedback_arm_count,
    )
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
        for key in BRANCH_TOTAL_KEYS:
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
