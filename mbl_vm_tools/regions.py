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


REGION_CONTRACT_VERSION = "vm-region-v4"
VIRTUAL_FUNCTION_EXIT = "__vm_function_exit__"
REGION_POLICY = (
    "Region analysis is hierarchical. It first collapses proven VM control-flow "
    "into SCCs, then derives branch regions on the acyclic SCC condensation graph "
    "with a virtual function-exit node. Same-target taken/fallthrough conditionals "
    "are control-degenerate conditional atoms: both VM edge identities exist, "
    "but there is no branch topology to lift. Same-SCC conditional "
    "branches are cyclic-region members, not failed split/join branches. The pass does not infer branch polarity "
    "or source-level constructs."
)

CONDITIONAL_EDGE_KINDS = {"conditional_taken", "conditional_fallthrough"}
REQUIRED_CONDITIONAL_EDGE_KINDS = frozenset(CONDITIONAL_EDGE_KINDS)



@dataclass(frozen=True)
class VMSCCRegion:
    id: str
    nodes: list[str]
    cyclic: bool
    entry_blocks: list[str] = field(default_factory=list)
    exit_blocks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMRegionFact:
    id: str
    kind: str
    header: str
    nodes: list[str]
    exit: Optional[str] = None
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMRegionReport:
    contract: str
    summary: dict[str, Any]
    facts: list[VMRegionFact]
    scc_regions: list[VMSCCRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "summary": self.summary,
            "facts": [f.to_dict() for f in self.facts],
            "scc_regions": [s.to_dict() for s in self.scc_regions],
        }


# ---------------------------------------------------------------------------
# Graph helpers


def _as_dicts(cfg: VMControlGraph | dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(cfg, VMControlGraph):
        return [b.to_dict() for b in cfg.blocks], [e.to_dict() for e in cfg.edges]
    return list(cfg.get("blocks") or []), list(cfg.get("edges") or [])


def _stable_sorted(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda x: (len(x), x))


def _conditional_edge_sort_key(edge: dict[str, Any]) -> tuple[int, str, str, int]:
    kind = str(edge.get("kind"))
    rank = 0 if kind == "conditional_taken" else 1 if kind == "conditional_fallthrough" else 2
    return (
        rank,
        str(edge.get("source")),
        str(edge.get("target")),
        int(edge.get("instruction_offset", -1) or -1),
    )


def _conditional_out_edges(
    out_edges_by_source: dict[str, list[dict[str, Any]]],
    header: str,
    reachable: set[str],
) -> list[dict[str, Any]]:
    return sorted(
        [
            edge for edge in out_edges_by_source.get(header, [])
            if str(edge.get("kind")) in CONDITIONAL_EDGE_KINDS
            and str(edge.get("target")) in reachable
        ],
        key=_conditional_edge_sort_key,
    )


def _conditional_edge_kind_set(edges: list[dict[str, Any]]) -> set[str]:
    return {str(edge.get("kind")) for edge in edges}


def _conditional_edge_targets(edges: list[dict[str, Any]]) -> list[str]:
    return [str(edge.get("target")) for edge in edges]


def _reachable(start: str, succs: dict[str, set[str]]) -> set[str]:
    seen = {start}
    q: deque[str] = deque([start])
    while q:
        cur = q.popleft()
        for nxt in sorted(succs.get(cur, set())):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _tarjan_scc(nodes: set[str], succs: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in sorted(succs.get(v, set()) & nodes):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            components.append(_stable_sorted(comp))

    for node in _stable_sorted(nodes):
        if node not in indices:
            strongconnect(node)
    return components


def _topological_order(nodes: set[str], succs: dict[str, set[str]]) -> list[str]:
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for node in nodes:
        for nxt in succs.get(node, set()) & nodes:
            indeg[nxt] += 1
    q: deque[str] = deque(_stable_sorted([n for n, d in indeg.items() if d == 0]))
    out: list[str] = []
    while q:
        cur = q.popleft()
        out.append(cur)
        for nxt in sorted(succs.get(cur, set()) & nodes):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if len(out) != len(nodes):
        # SCC condensation should be acyclic. Keep deterministic diagnostics if
        # malformed input ever violates that invariant.
        for node in _stable_sorted(nodes):
            if node not in out:
                out.append(node)
    return out


def _postdominators_dag(nodes: set[str], succs: dict[str, set[str]]) -> dict[str, set[str]]:
    order = _topological_order(nodes, succs)
    pdom: dict[str, set[str]] = {}
    for node in reversed(order):
        node_succs = [s for s in succs.get(node, set()) if s in nodes]
        if not node_succs:
            pdom[node] = {node}
            continue
        common: Optional[set[str]] = None
        for succ in node_succs:
            common = set(pdom[succ]) if common is None else common & pdom[succ]
        pdom[node] = {node} | (common or set())
    return pdom


def _nearest_common_postdominator_by_order(
    successors: list[str],
    *,
    pdom: dict[str, set[str]],
    exclude: set[str],
) -> tuple[Optional[str], list[str]]:
    if not successors:
        return None, []
    common = set(pdom.get(successors[0], set()))
    for succ in successors[1:]:
        common &= pdom.get(succ, set())
    common -= exclude
    if not common:
        return None, []

    # In a proper postdominator chain, the closest common postdominator is the
    # candidate postdominated by all other common candidates. This is a structural
    # relation, unlike graph-distance tie breaking.
    nearest: list[str] = []
    for candidate in common:
        candidate_pdom = pdom.get(candidate, {candidate})
        if all(other == candidate or other in candidate_pdom for other in common):
            nearest.append(candidate)
    nearest = _stable_sorted(nearest)
    return (nearest[0] if len(nearest) == 1 else None), _stable_sorted(common)


def _build_scc_model(
    blocks: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    block_by_id = {str(b.get("id")): b for b in blocks}
    proven_edges = [e for e in edges if e.get("status") == "proven"]
    succs: dict[str, set[str]] = defaultdict(set)
    preds: dict[str, set[str]] = defaultdict(set)
    out_edges_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in proven_edges:
        src = str(edge.get("source"))
        dst = str(edge.get("target"))
        if src not in block_by_id or dst not in block_by_id:
            continue
        succs[src].add(dst)
        preds[dst].add(src)
        out_edges_by_source[src].append(edge)

    start = str(blocks[0].get("id")) if blocks else ""
    reachable = _reachable(start, succs) if blocks else set()
    components = _tarjan_scc(reachable, succs) if reachable else []
    scc_by_block: dict[str, str] = {}
    blocks_by_scc: dict[str, list[str]] = {}
    for idx, comp in enumerate(components):
        scc_id = f"scc{idx}"
        blocks_by_scc[scc_id] = comp
        for block in comp:
            scc_by_block[block] = scc_id

    scc_succs: dict[str, set[str]] = defaultdict(set)
    scc_preds: dict[str, set[str]] = defaultdict(set)
    for src in reachable:
        src_scc = scc_by_block[src]
        for dst in succs.get(src, set()) & reachable:
            dst_scc = scc_by_block[dst]
            if src_scc == dst_scc:
                continue
            scc_succs[src_scc].add(dst_scc)
            scc_preds[dst_scc].add(src_scc)

    scc_nodes = set(blocks_by_scc)
    raw_exit_sccs = {scc for scc in scc_nodes if not (scc_succs.get(scc, set()) & scc_nodes)}
    augmented_succs: dict[str, set[str]] = {scc: set(scc_succs.get(scc, set())) for scc in scc_nodes}
    augmented_succs[VIRTUAL_FUNCTION_EXIT] = set()
    augmented_nodes = set(scc_nodes) | {VIRTUAL_FUNCTION_EXIT}
    for scc in raw_exit_sccs:
        augmented_succs[scc].add(VIRTUAL_FUNCTION_EXIT)

    pdom = _postdominators_dag(augmented_nodes, augmented_succs)
    cyclic_sccs = {
        scc
        for scc, comp in blocks_by_scc.items()
        if len(comp) > 1 or any(block in succs.get(block, set()) for block in comp)
    }

    return {
        "proven_edges": proven_edges,
        "succs": succs,
        "preds": preds,
        "out_edges_by_source": out_edges_by_source,
        "reachable": reachable,
        "scc_by_block": scc_by_block,
        "blocks_by_scc": blocks_by_scc,
        "scc_succs": scc_succs,
        "scc_preds": scc_preds,
        "scc_nodes": scc_nodes,
        "cyclic_sccs": cyclic_sccs,
        "raw_exit_sccs": raw_exit_sccs,
        "augmented_succs": augmented_succs,
        "augmented_nodes": augmented_nodes,
        "pdom": pdom,
    }


# ---------------------------------------------------------------------------
# Public analysis


def analyze_regions(cfg: VMControlGraph | dict[str, Any]) -> VMRegionReport:
    blocks, edges = _as_dicts(cfg)
    if not blocks:
        return VMRegionReport(
            contract=REGION_CONTRACT_VERSION,
            summary={
                "block_count": 0,
                "proven_edge_count": 0,
                "reachable_block_count": 0,
                "region_fact_count": 0,
                "policy": REGION_POLICY,
            },
            facts=[],
            scc_regions=[],
        )

    model = _build_scc_model(blocks, edges)
    proven_edges: list[dict[str, Any]] = model["proven_edges"]
    succs: dict[str, set[str]] = model["succs"]
    preds: dict[str, set[str]] = model["preds"]
    out_edges_by_source: dict[str, list[dict[str, Any]]] = model["out_edges_by_source"]
    reachable: set[str] = model["reachable"]
    scc_by_block: dict[str, str] = model["scc_by_block"]
    blocks_by_scc: dict[str, list[str]] = model["blocks_by_scc"]
    scc_preds: dict[str, set[str]] = model["scc_preds"]
    cyclic_sccs: set[str] = model["cyclic_sccs"]
    raw_exit_sccs: set[str] = model["raw_exit_sccs"]
    pdom: dict[str, set[str]] = model["pdom"]

    scc_regions: list[VMSCCRegion] = []
    for scc_id in _stable_sorted(blocks_by_scc):
        nodes = blocks_by_scc[scc_id]
        if scc_id not in cyclic_sccs:
            continue
        entries = [
            block for block in nodes
            if any(pred not in nodes for pred in preds.get(block, set())) or block == blocks[0].get("id")
        ]
        exits = [
            block for block in nodes
            if any(dst not in nodes for dst in succs.get(block, set()))
        ]
        scc_regions.append(
            VMSCCRegion(
                id=scc_id,
                nodes=nodes,
                cyclic=True,
                entry_blocks=_stable_sorted(entries),
                exit_blocks=_stable_sorted(exits),
            )
        )

    facts: list[VMRegionFact] = []
    conditional_branch_count = 0
    conditional_two_successor_count = 0
    conditional_not_two_successors_count = 0
    conditional_two_edge_count = 0
    conditional_not_two_edge_count = 0
    conditional_same_target_edge_count = 0
    conditional_control_degenerate_count = 0
    conditional_same_scc_successors_count = 0
    conditional_cross_scc_count = 0
    conditional_scc_region_count = 0
    conditional_real_scc_join_region_count = 0
    conditional_function_exit_scc_region_count = 0
    conditional_ambiguous_scc_join_count = 0
    conditional_no_scc_join_count = 0
    successor_count_hist: Counter[str] = Counter()
    distinct_successor_count_hist: Counter[str] = Counter()
    conditional_edge_count_hist: Counter[str] = Counter()

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
            # Both VM edge identities exist, but they collapse to the same
            # destination block. This is a branch-atom diagnostic, not a region:
            # there is no topological split to lift here.
            continue

        conditional_two_successor_count += 1
        successor_sccs = _stable_sorted({scc_by_block[s] for s in successors})
        if len(successor_sccs) < 2:
            conditional_same_scc_successors_count += 1
            facts.append(
                VMRegionFact(
                    id=f"cyclic_cond{len(facts)}",
                    kind="conditional_same_scc",
                    header=header,
                    nodes=[header, *successors],
                    exit=None,
                    confidence=1.0,
                    evidence={
                        "rule": "both conditional successors are in the same SCC; defer to cyclic-region structuring",
                        "header_scc": header_scc,
                        "successor_sccs": successor_sccs,
                        "successors": successors,
                        "predicate_polarity": semantic.get("predicate_polarity"),
                    },
                )
            )
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

        conditional_scc_region_count += 1
        if join_scc == VIRTUAL_FUNCTION_EXIT:
            conditional_function_exit_scc_region_count += 1
            exit_text = VIRTUAL_FUNCTION_EXIT
            join_kind = "virtual_function_exit"
            join_nodes: list[str] = []
        else:
            conditional_real_scc_join_region_count += 1
            join_nodes = blocks_by_scc.get(join_scc, [])
            exit_text = join_scc
            join_kind = "real_scc_join"

        facts.append(
            VMRegionFact(
                id=f"cond{len(facts)}",
                kind="conditional_scc_region",
                header=header,
                nodes=[header, *successors, *join_nodes],
                exit=exit_text,
                confidence=1.0,
                evidence={
                    "rule": "two proven conditional successors cross SCC boundaries; nearest common postdominator is selected on the augmented SCC DAG",
                    "header_scc": header_scc,
                    "successor_sccs": successor_sccs,
                    "successors": successors,
                    "join_scc": join_scc,
                    "join_kind": join_kind,
                    "join_candidates": join_candidates[:8],
                    "raw_exit_scc_count": len(raw_exit_sccs),
                    "predicate_polarity": semantic.get("predicate_polarity"),
                },
            )
        )

    fact_hist = Counter(f.kind for f in facts)
    summary = {
        "block_count": len(blocks),
        "proven_edge_count": len(proven_edges),
        "reachable_block_count": len(reachable),
        "unreachable_under_proven_edges_count": max(0, len(blocks) - len(reachable)),
        "scc_count": len(blocks_by_scc),
        "cyclic_scc_count": len(cyclic_sccs),
        "raw_exit_scc_count": len(raw_exit_sccs),
        "region_fact_count": len(facts),
        "region_kind_histogram": dict(sorted(fact_hist.items())),
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
        "conditional_scc_region_count": conditional_scc_region_count,
        "conditional_real_scc_join_region_count": conditional_real_scc_join_region_count,
        "conditional_function_exit_scc_region_count": conditional_function_exit_scc_region_count,
        "conditional_ambiguous_scc_join_count": conditional_ambiguous_scc_join_count,
        "conditional_no_scc_join_count": conditional_no_scc_join_count,
        "virtual_function_exit": VIRTUAL_FUNCTION_EXIT,
        "policy": REGION_POLICY,
    }
    return VMRegionReport(REGION_CONTRACT_VERSION, summary, facts, scc_regions)


def analyze_module_regions(
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
        report = analyze_regions(cfg).to_dict()
        summary = report["summary"]
        for key in [
            "block_count",
            "proven_edge_count",
            "reachable_block_count",
            "unreachable_under_proven_edges_count",
            "scc_count",
            "cyclic_scc_count",
            "raw_exit_scc_count",
            "region_fact_count",
            "conditional_branch_count",
            "conditional_two_successor_count",
            "conditional_not_two_successors_count",
            "conditional_two_edge_count",
            "conditional_not_two_edge_count",
            "conditional_same_target_edge_count",
            "conditional_control_degenerate_count",
            "conditional_same_scc_successors_count",
            "conditional_cross_scc_count",
            "conditional_scc_region_count",
            "conditional_real_scc_join_region_count",
            "conditional_function_exit_scc_region_count",
            "conditional_ambiguous_scc_join_count",
            "conditional_no_scc_join_count",
        ]:
            totals[key] += int(summary.get(key, 0) or 0)
        functions.append({
            "name": entry.name,
            "symbol": entry.symbol,
            "span": span,
            "summary": summary,
            "facts": report["facts"] if function else [],
            "scc_regions": report["scc_regions"] if function else [],
        })
    return {
        "contract": REGION_CONTRACT_VERSION,
        "module": str(mod.path),
        "summary": {**dict(totals), "function_count": len(functions), "policy": REGION_POLICY},
        "functions": functions,
    }


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build hierarchical VM region facts from proven control graphs")
    parser.add_argument("module", type=Path)
    parser.add_argument("--function", default=None)
    parser.add_argument("--limit-functions", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = analyze_module_regions(args.module, function=args.function, limit_functions=args.limit_functions)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
