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
    REQUIRED_CONDITIONAL_EDGE_KINDS,
    _as_dicts,
    _build_scc_model,
    _conditional_edge_kind_set,
    _conditional_edge_targets,
    _conditional_out_edges,
    _nearest_common_postdominator_by_order,
    _stable_sorted,
)


BRANCH_CONTRACT_VERSION = "vm-branch-v4"
BRANCH_LIFT_POLICY = (
    "Branch lifting consumes hierarchical vm-region-v4 facts. It lifts only "
    "cross-SCC conditional regions on the augmented SCC DAG, preserving VM "
    "taken/fallthrough edge identity and unresolved predicate polarity. "
    "Same-target taken/fallthrough conditionals are not lifted: both edge "
    "identities exist, but they collapse to one destination, so there is no "
    "branch topology. "
    "Same-SCC conditionals are deferred to cyclic-region structuring, not treated "
    "as branch-lift failures."
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
                "conditional_not_two_edge_count": 0,
                "conditional_same_target_edge_count": 0,
                "conditional_control_degenerate_count": 0,
                "lifted_branch_count": 0,
                "policy": BRANCH_LIFT_POLICY,
            },
            branches=[],
        )

    model = _build_scc_model(blocks, edges)
    proven_edges: list[dict[str, Any]] = model["proven_edges"]
    succs: dict[str, set[str]] = model["succs"]
    out_edges_by_source: dict[str, list[dict[str, Any]]] = model["out_edges_by_source"]
    reachable: set[str] = model["reachable"]
    scc_by_block: dict[str, str] = model["scc_by_block"]
    blocks_by_scc: dict[str, list[str]] = model["blocks_by_scc"]
    scc_succs: dict[str, set[str]] = model["scc_succs"]
    scc_nodes: set[str] = model["scc_nodes"]
    cyclic_sccs: set[str] = model["cyclic_sccs"]
    pdom: dict[str, set[str]] = model["pdom"]

    edge_lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in proven_edges:
        edge_lookup[(str(edge.get("source")), str(edge.get("target")))].append(edge)

    branches: list[VMLiftedBranch] = []
    conditional_branch_count = 0
    conditional_two_successor_count = 0
    conditional_not_two_successors_count = 0
    conditional_two_edge_count = 0
    conditional_not_two_edge_count = 0
    conditional_same_target_edge_count = 0
    conditional_control_degenerate_count = 0
    conditional_same_scc_successors_count = 0
    conditional_cross_scc_count = 0
    conditional_ambiguous_scc_join_count = 0
    conditional_no_scc_join_count = 0
    function_exit_branch_count = 0
    real_join_branch_count = 0
    same_target_branch_count = 0
    arm_scc_count_hist: Counter[str] = Counter()
    successor_count_hist: Counter[str] = Counter()
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
        conditional_branch_count += 1
        conditional_edges = _conditional_out_edges(out_edges_by_source, header, reachable)
        edge_kinds = _conditional_edge_kind_set(conditional_edges)
        edge_targets = _conditional_edge_targets(conditional_edges)
        successors = _stable_sorted(set(edge_targets))
        successor_count_hist[str(len(conditional_edges))] += 1
        conditional_edge_count_hist[str(len(conditional_edges))] += 1
        distinct_successor_count_hist[str(len(successors))] += 1
        if len(conditional_edges) != 2 or edge_kinds != set(REQUIRED_CONDITIONAL_EDGE_KINDS):
            conditional_not_two_edge_count += 1
            conditional_not_two_successors_count += 1
            continue
        conditional_two_edge_count += 1

        header_scc = scc_by_block[header]
        if len(successors) == 1:
            conditional_same_target_edge_count += 1
            conditional_control_degenerate_count += 1
            # Do not lift this as a branch. The VM conditional atom has both
            # edge identities, but the CFG has no topological split.
            continue

        conditional_two_successor_count += 1
        arm_inputs = []
        for successor in successors:
            successor_scc = scc_by_block[successor]
            arm_inputs.append((successor, edge_lookup.get((header, successor), []), successor_scc))
        successor_sccs = _stable_sorted({scc for _succ, _edges, scc in arm_inputs})
        if len(successor_sccs) < 2:
            conditional_same_scc_successors_count += 1
            continue

        conditional_cross_scc_count += 1
        join_scc, join_candidates = _nearest_common_postdominator_by_order(
            successor_sccs,
            pdom=pdom,
            exclude={header_scc},
        )
        if join_scc is None:
            if join_candidates:
                conditional_ambiguous_scc_join_count += 1
            else:
                conditional_no_scc_join_count += 1
            continue

        if join_scc == VIRTUAL_FUNCTION_EXIT:
            exit_kind = "virtual_function_exit"
            function_exit_branch_count += 1
            join_nodes: list[str] = []
        else:
            exit_kind = "real_scc_join"
            real_join_branch_count += 1
            join_nodes = blocks_by_scc.get(join_scc, [])

        lifted_branch_count += 1
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
                    "rule": "two proven conditional successors cross SCC boundaries and have a nearest common postdominator on the augmented SCC DAG",
                    "successors": [succ for succ, _edges, _scc in sorted(arm_inputs, key=_arm_sort_key)],
                    "successor_sccs": successor_sccs,
                    "join_candidates": join_candidates[:8],
                    "predicate_polarity": semantic.get("predicate_polarity"),
                },
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
        "conditional_two_successor_count": conditional_two_successor_count,
        "conditional_not_two_successors_count": conditional_not_two_successors_count,
        "conditional_two_edge_count": conditional_two_edge_count,
        "conditional_not_two_edge_count": conditional_not_two_edge_count,
        "conditional_same_target_edge_count": conditional_same_target_edge_count,
        "conditional_control_degenerate_count": conditional_control_degenerate_count,
        "conditional_successor_count_histogram": dict(sorted(successor_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "conditional_edge_count_histogram": dict(sorted(conditional_edge_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "conditional_distinct_successor_count_histogram": dict(sorted(distinct_successor_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "conditional_same_scc_successors_count": conditional_same_scc_successors_count,
        "conditional_cross_scc_count": conditional_cross_scc_count,
        "conditional_ambiguous_scc_join_count": conditional_ambiguous_scc_join_count,
        "conditional_no_scc_join_count": conditional_no_scc_join_count,
        "lifted_branch_count": lifted_branch_count,
        "lifted_scc_branch_count": conditional_cross_scc_count,
        "real_join_branch_count": real_join_branch_count,
        "function_exit_branch_count": function_exit_branch_count,
        "same_target_branch_count": same_target_branch_count,
        "arm_scc_count_histogram": dict(sorted(arm_scc_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
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
            "conditional_two_successor_count",
            "conditional_not_two_successors_count",
            "conditional_two_edge_count",
            "conditional_not_two_edge_count",
            "conditional_same_target_edge_count",
            "conditional_control_degenerate_count",
            "conditional_same_scc_successors_count",
            "conditional_cross_scc_count",
            "conditional_ambiguous_scc_join_count",
            "conditional_no_scc_join_count",
            "lifted_branch_count",
            "lifted_scc_branch_count",
            "real_join_branch_count",
            "function_exit_branch_count",
            "same_target_branch_count",
        ]:
            totals[key] += int(summary.get(key, 0) or 0)
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
        "summary": {**dict(totals), "function_count": len(functions), "policy": BRANCH_LIFT_POLICY},
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
