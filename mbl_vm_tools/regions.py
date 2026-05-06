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


REGION_CONTRACT_VERSION = "vm-region-v15"
VIRTUAL_FUNCTION_EXIT = "__vm_function_exit__"
CYCLIC_FEEDBACK_FRONTIER = "__cyclic_feedback_frontier__"
CYCLIC_FEEDBACK_EXIT_FRONTIER = "__cyclic_feedback_exit_frontier__"
CYCLIC_EXTERNAL_EXIT_FRONTIER = "__cyclic_external_exit_frontier__"
REGION_POLICY = (
    "Region analysis works on the projected VM CFG: branch-atom aliases are "
    "canonicalized, cyclic entry ports are normalized, and conditional regions are "
    "lifted only at VM topology level."
)

CONDITIONAL_EDGE_KINDS = {"conditional_taken", "conditional_fallthrough"}
REQUIRED_CONDITIONAL_EDGE_KINDS = frozenset(CONDITIONAL_EDGE_KINDS)

REGION_TOTAL_KEYS = "block_count proven_edge_count raw_block_count raw_proven_edge_count raw_scc_count raw_cyclic_scc_count structural_projection_block_count structural_projection_edge_count structural_projection_alias_block_count structural_projection_alias_context_count structural_projection_skipped_alias_internal_edge_count structural_projection_duplicate_edge_count reachable_block_count unreachable_under_proven_edges_count scc_count cyclic_scc_count raw_multi_entry_cyclic_scc_count raw_cfg_multi_entry_cyclic_scc_count projected_multi_entry_cyclic_scc_count normalized_multi_entry_cyclic_scc_count cyclic_entry_port_count cyclic_entry_port_scc_count raw_exit_scc_count region_fact_count hierarchy_node_count cyclic_edge_role_count conditional_branch_count conditional_branch_context_count conditional_branch_atom_count raw_conditional_branch_context_count raw_conditional_branch_atom_count conditional_alias_context_count conditional_alias_group_count conditional_two_successor_count conditional_two_edge_count conditional_projection_no_split_count conditional_same_scc_successors_count conditional_cross_scc_count conditional_scc_region_count conditional_real_scc_join_region_count conditional_function_exit_scc_region_count conditional_cyclic_scc_region_count conditional_cyclic_local_join_region_count conditional_cyclic_feedback_join_region_count conditional_cyclic_boundary_join_region_count conditional_cyclic_feedback_frontier_region_count conditional_cyclic_feedback_exit_frontier_region_count conditional_cyclic_external_exit_frontier_region_count".split()



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
    hierarchy: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "summary": self.summary,
            "facts": [f.to_dict() for f in self.facts],
            "scc_regions": [s.to_dict() for s in self.scc_regions],
            "hierarchy": list(self.hierarchy),
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


def _conditional_context_signature(edges: list[dict[str, Any]]) -> tuple[Any, ...]:

    if not edges:
        return (None, None, ())
    first = edges[0]
    return (
        first.get("instruction_offset"),
        first.get("word_index"),
        tuple(
            (str(edge.get("kind")), str(edge.get("target")), edge.get("local_target"))
            for edge in sorted(edges, key=_conditional_edge_sort_key)
        ),
    )


def _canonical_conditional_contexts(
    blocks: list[dict[str, Any]],
    out_edges_by_source: dict[str, list[dict[str, Any]]],
    reachable: set[str],
    *,
    scc_by_block: Optional[dict[str, str]] = None,
    cyclic_sccs: Optional[set[str]] = None,
) -> dict[str, Any]:

    block_by_id = {str(block.get("id")): block for block in blocks}
    scc_by_block = scc_by_block or {}
    cyclic_sccs = cyclic_sccs or set()
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    edges_by_header: dict[str, list[dict[str, Any]]] = {}
    signature_by_header: dict[str, tuple[Any, ...]] = {}

    for block in blocks:
        header = str(block.get("id"))
        if header not in reachable:
            continue
        terminator = block.get("terminator") or {}
        semantic = (terminator.get("semantic") or {}) if isinstance(terminator, dict) else {}
        if semantic.get("branch_kind") != "conditional_branch":
            continue
        edges = _conditional_out_edges(out_edges_by_source, header, reachable)
        signature = _conditional_context_signature(edges)
        groups[signature].append(header)
        edges_by_header[header] = edges
        signature_by_header[header] = signature

    canonical_headers: set[str] = set()
    alias_to_canonical: dict[str, str] = {}
    aliases_by_canonical: dict[str, list[str]] = defaultdict(list)

    def context_sort_key(header: str) -> tuple[int, int, int, int, str]:
        block = block_by_id.get(header) or {}
        header_scc = scc_by_block.get(header)
        edges = edges_by_header.get(header) or []
        successor_sccs = {
            scc_by_block.get(str(edge.get("target")))
            for edge in edges
            if str(edge.get("target")) in scc_by_block
        }
        successor_sccs.discard(None)
        same_scc_owner = 1 if header_scc is not None and len(successor_sccs) == 1 and header_scc in successor_sccs else 0
        cyclic_owner = 1 if header_scc in cyclic_sccs else 0
        return (
            same_scc_owner,
            cyclic_owner,
            int(block.get("start_offset", -1) or -1),
            int(block.get("end_offset", -1) or -1),
            header,
        )

    for _signature, headers in groups.items():
        ordered = sorted(headers, key=context_sort_key)
        canonical = ordered[-1]
        canonical_headers.add(canonical)
        for header in ordered:
            if header == canonical:
                continue
            alias_to_canonical[header] = canonical
            aliases_by_canonical[canonical].append(header)

    return {
        "canonical_headers": canonical_headers,
        "alias_to_canonical": alias_to_canonical,
        "aliases_by_canonical": {k: _stable_sorted(v) for k, v in aliases_by_canonical.items()},
        "edges_by_header": edges_by_header,
        "signature_by_header": signature_by_header,
        "context_count": len(edges_by_header),
        "atom_count": len(canonical_headers),
        "alias_count": len(alias_to_canonical),
        "alias_group_count": sum(1 for headers in groups.values() if len(headers) > 1),
        "alias_group_size_histogram": dict(Counter(str(len(headers)) for headers in groups.values() if len(headers) > 1)),
    }


def _representative_closure(alias_to_canonical: dict[str, str]):

    def rep(value: Any) -> str:
        cur = str(value)
        seen: set[str] = set()
        while cur in alias_to_canonical and cur not in seen:
            seen.add(cur)
            cur = str(alias_to_canonical[cur])
        return cur

    return rep


def _project_control_graph(
    blocks: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    alias_to_canonical: dict[str, str],
) -> dict[str, Any]:

    rep = _representative_closure(alias_to_canonical)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    first_order: dict[str, int] = {}
    for idx, block in enumerate(blocks):
        block_id = str(block.get("id"))
        rid = rep(block_id)
        grouped[rid].append(block)
        first_order.setdefault(rid, idx)

    projected_blocks: list[dict[str, Any]] = []
    aliases_by_block: dict[str, list[str]] = {}
    for rid, group in grouped.items():
        canonical = next((block for block in group if str(block.get("id")) == rid), group[0])
        starts = [int(block.get("start_offset", 0) or 0) for block in group]
        ends = [int(block.get("end_offset", 0) or 0) for block in group]
        aliases = _stable_sorted(str(block.get("id")) for block in group if str(block.get("id")) != rid)
        payload = dict(canonical)
        payload["id"] = rid
        if aliases:
            payload["aliases"] = aliases
            payload["projection_span"] = {"min_start": min(starts), "max_end": max(ends)}
        aliases_by_block[rid] = aliases
        projected_blocks.append(payload)
    projected_blocks.sort(key=lambda block: first_order[str(block.get("id"))])

    seen_edges: set[tuple[Any, ...]] = set()
    projected_edges: list[dict[str, Any]] = []
    skipped_alias_internal_edges = 0
    duplicate_projected_edges = 0
    for edge in edges:
        src = str(edge.get("source"))
        dst = str(edge.get("target"))
        projected_src = rep(src)
        projected_dst = rep(dst)
        # Edges that only connect two byte/sub-entry contexts of the same VM
        # branch atom become local alias evidence after contraction.  Preserve
        # true self-loops, because they are real cyclic topology.
        if projected_src == projected_dst and src != dst:
            skipped_alias_internal_edges += 1
            continue
        key = (
            projected_src,
            projected_dst,
            edge.get("kind"),
            edge.get("status"),
            edge.get("word_index"),
            edge.get("instruction_offset"),
            edge.get("local_target"),
        )
        if key in seen_edges:
            duplicate_projected_edges += 1
            continue
        seen_edges.add(key)
        projected = dict(edge)
        projected["source"] = projected_src
        projected["target"] = projected_dst
        if projected_src != src or projected_dst != dst:
            projected["projection"] = {
                "source": src,
                "target": dst,
                "rule": "conditional_context_alias_contraction",
            }
        projected_edges.append(projected)

    return {
        "blocks": projected_blocks,
        "edges": projected_edges,
        "aliases_by_block": aliases_by_block,
        "representative": rep,
        "summary": {
            "structural_projection_block_count": len(projected_blocks),
            "structural_projection_edge_count": len(projected_edges),
            "structural_projection_alias_block_count": sum(1 for aliases in aliases_by_block.values() if aliases),
            "structural_projection_alias_context_count": len(alias_to_canonical),
            "structural_projection_skipped_alias_internal_edge_count": skipped_alias_internal_edges,
            "structural_projection_duplicate_edge_count": duplicate_projected_edges,
        },
    }


def _edge_payload(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        key: edge.get(key)
        for key in ["source", "target", "kind", "status", "word_index", "instruction_offset", "local_target", "formula"]
        if edge.get(key) is not None
    }


def _is_cyclic_feedback_candidate(src: str, dst: str, block_order: dict[str, int]) -> bool:

    return block_order.get(dst, 10**9) <= block_order.get(src, -1)


def _entry_blocks_by_scc(
    *,
    blocks: list[dict[str, Any]],
    blocks_by_scc: dict[str, list[str]],
    preds: dict[str, set[str]],
) -> dict[str, list[str]]:

    start_id = str(blocks[0].get("id")) if blocks else ""
    out: dict[str, list[str]] = {}
    for scc_id, nodes in blocks_by_scc.items():
        node_set = set(nodes)
        out[scc_id] = _stable_sorted([
            block for block in nodes
            if any(pred not in node_set for pred in preds.get(block, set())) or block == start_id
        ])
    return out


def _exit_blocks_by_scc(
    *,
    blocks_by_scc: dict[str, list[str]],
    succs: dict[str, set[str]],
) -> dict[str, list[str]]:

    out: dict[str, list[str]] = {}
    for scc_id, nodes in blocks_by_scc.items():
        node_set = set(nodes)
        out[scc_id] = _stable_sorted([
            block for block in nodes
            if any(dst not in node_set for dst in succs.get(block, set()))
        ])
    return out


def _normalize_cyclic_entry_blocks(
    *,
    proven_edges: list[dict[str, Any]],
    blocks_by_scc: dict[str, list[str]],
    scc_by_block: dict[str, str],
    cyclic_sccs: set[str],
    raw_entry_blocks_by_scc: dict[str, list[str]],
    block_order: dict[str, int],
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:

    in_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    out_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in proven_edges:
        src = str(edge.get("source"))
        dst = str(edge.get("target"))
        if src not in scc_by_block or dst not in scc_by_block:
            continue
        in_by_target[dst].append(edge)
        out_by_source[src].append(edge)

    normalized_by_scc: dict[str, list[str]] = {}
    ports_by_scc: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for scc_id, raw_entries in raw_entry_blocks_by_scc.items():
        if scc_id not in cyclic_sccs:
            normalized_by_scc[scc_id] = list(raw_entries)
            continue

        nodes = set(blocks_by_scc.get(scc_id, []))
        normalized = set(raw_entries)
        changed = True
        while changed:
            changed = False
            for entry in list(_stable_sorted(normalized)):
                external_in = [edge for edge in in_by_target.get(entry, []) if str(edge.get("source")) not in nodes]
                if not external_in:
                    continue
                if not all(str(edge.get("kind")) == "jump" for edge in external_in):
                    continue

                internal_out = [edge for edge in out_by_source.get(entry, []) if str(edge.get("target")) in nodes]
                feedback_out = [
                    edge for edge in internal_out
                    if _is_cyclic_feedback_candidate(entry, str(edge.get("target")), block_order)
                ]
                nonfeedback_out = [edge for edge in internal_out if edge not in feedback_out]
                if not feedback_out or nonfeedback_out:
                    continue

                targets = _stable_sorted(str(edge.get("target")) for edge in feedback_out)
                # The target must already be part of the entry frontier; otherwise
                # the edge is merely a backward edge, not proof that this block is
                # an adapter into an existing cyclic entry boundary.
                if not targets or not all(target in normalized and target != entry for target in targets):
                    continue

                normalized.discard(entry)
                normalized.update(targets)
                changed = True
                ports_by_scc[scc_id].append({
                    "block": entry,
                    "kind": "cyclic_reentry_port",
                    "normalized_targets": targets,
                    "external_edges": [_edge_payload(edge) for edge in sorted(external_in, key=_conditional_edge_sort_key)[:16]],
                    "feedback_edges": [_edge_payload(edge) for edge in sorted(feedback_out, key=_conditional_edge_sort_key)[:16]],
                    "rule": "jump-only external entry routes through internal feedback edge into an existing cyclic entry frontier",
                })

        normalized_by_scc[scc_id] = _stable_sorted(normalized)

    return normalized_by_scc, {scc: ports for scc, ports in ports_by_scc.items()}


def _reachable_in_projected_scc(
    start: str,
    *,
    local_succs: dict[str, set[str]],
    stop_blocks: set[str],
) -> set[str]:
    seen: set[str] = set()
    q: deque[str] = deque([start])
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in sorted(local_succs.get(cur, set())):
            if nxt in stop_blocks:
                continue
            if nxt not in seen:
                q.append(nxt)
    return seen


def _reachable_in_projected_scc_until(
    start: str,
    *,
    join: Optional[str],
    local_succs: dict[str, set[str]],
    stop_blocks: set[str],
) -> set[str]:
    if join is not None and start == join:
        return set()
    seen: set[str] = set()
    q: deque[str] = deque([start])
    while q:
        cur = q.popleft()
        if cur in seen or cur in stop_blocks:
            continue
        if join is not None and cur == join:
            continue
        seen.add(cur)
        for nxt in sorted(local_succs.get(cur, set())):
            if nxt in stop_blocks:
                continue
            if join is not None and nxt == join:
                continue
            if nxt not in seen:
                q.append(nxt)
    return seen


def _cyclic_branch_model(
    *,
    header: str,
    successors: list[str],
    conditional_edges: list[dict[str, Any]],
    header_scc: str,
    proven_edges: list[dict[str, Any]],
    scc_by_block: dict[str, str],
    blocks_by_scc: dict[str, list[str]],
    cyclic_sccs: set[str],
    entry_blocks_by_scc: dict[str, list[str]],
    block_order: dict[str, int],
    pdom: Optional[dict[str, set[str]]] = None,
) -> dict[str, Any]:

    scc_nodes = set(blocks_by_scc.get(header_scc, []))
    successor_set = set(successors)
    local_succs: dict[str, set[str]] = defaultdict(set)
    cut_edges: list[dict[str, Any]] = []
    boundary_edges: list[dict[str, Any]] = []
    internal_edge_count = 0

    for edge in proven_edges:
        src = str(edge.get("source"))
        dst = str(edge.get("target"))
        if src not in scc_nodes:
            continue
        if dst not in scc_nodes:
            payload = _edge_payload(edge)
            payload.update({
                "source_scc": header_scc,
                "target_scc": scc_by_block.get(dst, str(edge.get("target_scc", ""))),
                "boundary_kind": "scc_exit",
            })
            boundary_edges.append(payload)
            continue

        internal_edge_count += 1
        is_feedback = _is_cyclic_feedback_candidate(src, dst, block_order)
        is_split_edge = src == header and dst in successor_set and str(edge.get("kind")) in CONDITIONAL_EDGE_KINDS
        if is_feedback and not is_split_edge:
            payload = _edge_payload(edge)
            payload.update({
                "source_scc": header_scc,
                "target_scc": header_scc,
                "boundary_kind": "feedback",
            })
            cut_edges.append(payload)
            boundary_edges.append(payload)
            continue
        local_succs[src].add(dst)

    arm_reachable: dict[str, set[str]] = {}
    feedback_successors: list[str] = []
    header_order = block_order.get(header, -1)
    for successor in successors:
        successor_order = block_order.get(successor, 10**9)
        if successor_order <= header_order:
            feedback_successors.append(successor)
            arm_reachable[successor] = {successor}
            continue
        arm_reachable[successor] = _reachable_in_projected_scc(
            successor,
            local_succs=local_succs,
            stop_blocks={header},
        )

    common: set[str] = set()
    if arm_reachable:
        values = list(arm_reachable.values())
        common = set(values[0])
        for value in values[1:]:
            common &= value
    common.discard(header)
    local_join_candidates = _stable_sorted(common)
    local_join: Optional[str] = None
    local_join_kind: Optional[str] = None
    boundary_join_candidates: list[str] = []
    exit_frontier: list[str] = []
    combined_boundary_frontier: list[str] = []
    feedback_frontier: list[str] = []
    scc_exit_frontier: list[str] = []
    external_exit_join_scc: Optional[str] = None
    external_exit_join_candidates: list[str] = []
    external_exit_join_nodes: list[str] = []
    arm_exit_frontier: dict[str, list[str]] = {}
    arm_feedback_frontier: dict[str, list[str]] = {}
    arm_scc_exit_frontier: dict[str, list[str]] = {}
    arm_exit_edges: dict[str, list[dict[str, Any]]] = {}
    arm_feedback_edges: dict[str, list[dict[str, Any]]] = {}
    arm_scc_exit_edges: dict[str, list[dict[str, Any]]] = {}

    if local_join_candidates:
        local_join = sorted(local_join_candidates, key=lambda node: (block_order.get(node, 10**9), len(node), node))[0]
        local_join_kind = "cyclic_local_join"
    else:
        for successor in successors:
            reachable_nodes = set(arm_reachable.get(successor, set()))
            arm_boundaries = [edge for edge in boundary_edges if str(edge.get("source")) in reachable_nodes]
            feedback_boundaries = [edge for edge in arm_boundaries if str(edge.get("boundary_kind")) == "feedback"]
            scc_exit_boundaries = [edge for edge in arm_boundaries if str(edge.get("boundary_kind")) == "scc_exit"]
            arm_exit_edges[successor] = sorted(arm_boundaries, key=_conditional_edge_sort_key)
            arm_feedback_edges[successor] = sorted(feedback_boundaries, key=_conditional_edge_sort_key)
            arm_scc_exit_edges[successor] = sorted(scc_exit_boundaries, key=_conditional_edge_sort_key)
            arm_exit_frontier[successor] = _stable_sorted(str(edge.get("target")) for edge in arm_boundaries)
            arm_feedback_frontier[successor] = _stable_sorted(str(edge.get("target")) for edge in feedback_boundaries)
            arm_scc_exit_frontier[successor] = _stable_sorted(str(edge.get("target")) for edge in scc_exit_boundaries)

        common_boundary: Optional[set[str]] = None
        for successor in successors:
            frontier = set(arm_exit_frontier.get(successor, []))
            common_boundary = frontier if common_boundary is None else common_boundary & frontier
        boundary_join_candidates = _stable_sorted(common_boundary or set())

        if boundary_join_candidates:
            local_join = sorted(boundary_join_candidates, key=lambda node: (block_order.get(node, 10**9), len(node), node))[0]
            local_join_kind = "cyclic_feedback_join" if local_join in scc_nodes else "cyclic_boundary_join"
        else:
            frontier_union: set[str] = set()
            feedback_union: set[str] = set()
            scc_exit_union: set[str] = set()
            external_exit_sccs: set[str] = set()
            all_arms_have_frontier = True
            for successor in successors:
                frontier = set(arm_exit_frontier.get(successor, []))
                if not frontier:
                    all_arms_have_frontier = False
                frontier_union |= frontier
                feedback_union |= set(arm_feedback_frontier.get(successor, []))
                scc_exit_union |= set(arm_scc_exit_frontier.get(successor, []))
                for edge in arm_scc_exit_edges.get(successor, []):
                    target_scc = edge.get("target_scc")
                    if target_scc is not None:
                        external_exit_sccs.add(str(target_scc))
            if not (all_arms_have_frontier and frontier_union):
                raise ValueError(
                    "cyclic branch projection has no forward join and no stable boundary frontier"
                )

            combined_boundary_frontier = _stable_sorted(frontier_union)
            feedback_frontier = _stable_sorted(feedback_union)
            scc_exit_frontier = _stable_sorted(scc_exit_union)

            if external_exit_sccs and pdom is not None:
                external_exit_join_scc, external_exit_join_candidates = _nearest_common_postdominator_by_order(
                    _stable_sorted(external_exit_sccs),
                    pdom=pdom,
                    exclude={header_scc},
                )
                if external_exit_join_scc is not None and external_exit_join_scc != VIRTUAL_FUNCTION_EXIT:
                    external_exit_join_nodes = blocks_by_scc.get(external_exit_join_scc, [])

            if not scc_exit_frontier:
                # Multiple feedback targets are distinct VM reentry frontiers, not
                # multiple exits from the cyclic envelope. Preserve them as a
                # feedback-frontier fact rather than treating it as an external exit.
                exit_frontier = feedback_frontier
                local_join_kind = "cyclic_feedback_frontier"
            elif external_exit_join_scc is not None:
                # External SCC exits are normalized through the SCC DAG
                # postdominator. Internal feedback targets remain diagnostics.
                exit_frontier = scc_exit_frontier
                local_join_kind = "cyclic_feedback_exit_frontier" if feedback_frontier else "cyclic_external_exit_frontier"
            else:
                raise ValueError(
                    "cyclic branch has an external SCC-exit frontier without a stable SCC-DAG postdominator"
                )

    arm_nodes: dict[str, list[str]] = {}
    for successor in successors:
        if local_join_kind == "cyclic_local_join" and local_join is not None:
            nodes = _reachable_in_projected_scc_until(
                successor,
                join=local_join,
                local_succs=local_succs,
                stop_blocks={header},
            )
        else:
            nodes = set(arm_reachable.get(successor, set()))
        if local_join_kind == "cyclic_local_join" and local_join is not None:
            nodes.discard(local_join)
        nodes.discard(header)
        arm_nodes[successor] = _stable_sorted(nodes)

    return {
        "header_scc": header_scc,
        "successors": list(successors),
        "successor_sccs": [header_scc],
        "local_join": local_join,
        "local_join_kind": local_join_kind,
        "local_join_candidates": local_join_candidates[:8],
        "boundary_join_candidates": boundary_join_candidates[:8],
        "exit_frontier": exit_frontier[:16],
        "combined_boundary_frontier": combined_boundary_frontier[:16],
        "feedback_frontier": feedback_frontier[:16],
        "scc_exit_frontier": scc_exit_frontier[:16],
        "external_exit_join_scc": external_exit_join_scc,
        "external_exit_join_candidates": external_exit_join_candidates[:8],
        "external_exit_join_nodes": external_exit_join_nodes[:16],
        "arm_exit_frontier_by_successor": arm_exit_frontier,
        "arm_feedback_frontier_by_successor": arm_feedback_frontier,
        "arm_scc_exit_frontier_by_successor": arm_scc_exit_frontier,
        "arm_exit_edges_by_successor": {k: v[:8] for k, v in arm_exit_edges.items()},
        "arm_feedback_edges_by_successor": {k: v[:8] for k, v in arm_feedback_edges.items()},
        "arm_scc_exit_edges_by_successor": {k: v[:8] for k, v in arm_scc_exit_edges.items()},
        "arm_nodes_by_successor": arm_nodes,
        "arm_reachable_by_successor": {k: _stable_sorted(v) for k, v in arm_reachable.items()},
        "feedback_successors": _stable_sorted(feedback_successors),
        "cut_edge_count": len(cut_edges),
        "boundary_edge_count": len(boundary_edges),
        "internal_edge_count": internal_edge_count,
    }

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


def _build_structural_projection_model(
    blocks: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:

    raw_model = _build_scc_model(blocks, edges)
    raw_conditional_contexts = _canonical_conditional_contexts(
        blocks,
        raw_model["out_edges_by_source"],
        raw_model["reachable"],
        scc_by_block=raw_model["scc_by_block"],
        cyclic_sccs=raw_model["cyclic_sccs"],
    )
    projection = _project_control_graph(blocks, edges, raw_conditional_contexts["alias_to_canonical"])
    structural_blocks: list[dict[str, Any]] = projection["blocks"]
    structural_edges: list[dict[str, Any]] = projection["edges"]
    structural_model = _build_scc_model(structural_blocks, structural_edges)
    structural_conditional_contexts = _canonical_conditional_contexts(
        structural_blocks,
        structural_model["out_edges_by_source"],
        structural_model["reachable"],
        scc_by_block=structural_model["scc_by_block"],
        cyclic_sccs=structural_model["cyclic_sccs"],
    )
    return {
        "raw_blocks": blocks,
        "raw_edges": edges,
        "raw_model": raw_model,
        "raw_conditional_contexts": raw_conditional_contexts,
        "blocks": structural_blocks,
        "edges": structural_edges,
        "model": structural_model,
        "conditional_contexts": structural_conditional_contexts,
        "projection": projection,
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
                "raw_block_count": 0,
                "raw_proven_edge_count": 0,
                "structural_projection_block_count": 0,
                "structural_projection_edge_count": 0,
                "reachable_block_count": 0,
                "raw_multi_entry_cyclic_scc_count": 0,
                "raw_cfg_multi_entry_cyclic_scc_count": 0,
                "projected_multi_entry_cyclic_scc_count": 0,
                "normalized_multi_entry_cyclic_scc_count": 0,
                "cyclic_entry_port_count": 0,
                "conditional_projection_no_split_count": 0,
                "region_fact_count": 0,
                "hierarchy_node_count": 0,
                "cyclic_edge_role_count": 0,
                "policy": REGION_POLICY,
            },
            facts=[],
            scc_regions=[],
            hierarchy=[],
        )

    raw_blocks = blocks
    raw_edges = edges
    projection_model = _build_structural_projection_model(raw_blocks, raw_edges)
    raw_model: dict[str, Any] = projection_model["raw_model"]
    raw_conditional_contexts: dict[str, Any] = projection_model["raw_conditional_contexts"]
    projection_summary: dict[str, Any] = projection_model["projection"]["summary"]

    blocks = projection_model["blocks"]
    edges = projection_model["edges"]
    model = projection_model["model"]
    proven_edges: list[dict[str, Any]] = model["proven_edges"]
    succs: dict[str, set[str]] = model["succs"]
    preds: dict[str, set[str]] = model["preds"]
    out_edges_by_source: dict[str, list[dict[str, Any]]] = model["out_edges_by_source"]
    reachable: set[str] = model["reachable"]
    scc_by_block: dict[str, str] = model["scc_by_block"]
    blocks_by_scc: dict[str, list[str]] = model["blocks_by_scc"]
    cyclic_sccs: set[str] = model["cyclic_sccs"]
    raw_exit_sccs: set[str] = model["raw_exit_sccs"]
    pdom: dict[str, set[str]] = model["pdom"]

    block_order = {str(block.get("id")): idx for idx, block in enumerate(blocks)}
    conditional_contexts: dict[str, Any] = projection_model["conditional_contexts"]
    canonical_conditional_headers: set[str] = conditional_contexts["canonical_headers"]
    conditional_alias_to_canonical: dict[str, str] = conditional_contexts["alias_to_canonical"]
    raw_entry_blocks_by_scc = _entry_blocks_by_scc(
        blocks=blocks,
        blocks_by_scc=blocks_by_scc,
        preds=preds,
    )
    entry_blocks_by_scc, cyclic_entry_ports_by_scc = _normalize_cyclic_entry_blocks(
        proven_edges=proven_edges,
        blocks_by_scc=blocks_by_scc,
        scc_by_block=scc_by_block,
        cyclic_sccs=cyclic_sccs,
        raw_entry_blocks_by_scc=raw_entry_blocks_by_scc,
        block_order=block_order,
    )
    exit_blocks_by_scc = _exit_blocks_by_scc(blocks_by_scc=blocks_by_scc, succs=succs)
    scc_regions: list[VMSCCRegion] = []
    for scc_id in _stable_sorted(blocks_by_scc):
        nodes = blocks_by_scc[scc_id]
        if scc_id not in cyclic_sccs:
            continue
        scc_regions.append(
            VMSCCRegion(
                id=scc_id,
                nodes=nodes,
                cyclic=True,
                entry_blocks=entry_blocks_by_scc.get(scc_id, []),
                exit_blocks=exit_blocks_by_scc.get(scc_id, []),
            )
        )

    facts: list[VMRegionFact] = []
    conditional_branch_count = 0
    conditional_two_successor_count = 0
    conditional_two_edge_count = 0
    conditional_projection_no_split_count = 0
    conditional_same_scc_successors_count = 0
    conditional_cross_scc_count = 0
    conditional_scc_region_count = 0
    conditional_real_scc_join_region_count = 0
    conditional_function_exit_scc_region_count = 0
    conditional_cyclic_scc_region_count = 0
    conditional_cyclic_local_join_region_count = 0
    conditional_cyclic_feedback_join_region_count = 0
    conditional_cyclic_boundary_join_region_count = 0
    conditional_cyclic_feedback_frontier_region_count = 0
    conditional_cyclic_feedback_exit_frontier_region_count = 0
    conditional_cyclic_external_exit_frontier_region_count = 0
    conditional_alias_context_count = int(raw_conditional_contexts.get("alias_count", 0) or 0)
    conditional_alias_group_count = int(raw_conditional_contexts.get("alias_group_count", 0) or 0)
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
        if len(successors) == 1:
            conditional_projection_no_split_count += 1
            continue
        if len(successors) != 2:
            raise ValueError("canonical conditional branch must have two distinct successor blocks")

        conditional_two_successor_count += 1
        successor_sccs = _stable_sorted({scc_by_block[s] for s in successors})
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
                pdom=pdom,
            )
            conditional_cyclic_scc_region_count += 1
            local_join_kind = str(cyclic_model.get("local_join_kind"))
            if local_join_kind == "cyclic_local_join":
                conditional_cyclic_local_join_region_count += 1
            elif local_join_kind == "cyclic_feedback_join":
                conditional_cyclic_feedback_join_region_count += 1
            elif local_join_kind == "cyclic_boundary_join":
                conditional_cyclic_boundary_join_region_count += 1
            elif local_join_kind == "cyclic_feedback_frontier":
                conditional_cyclic_feedback_frontier_region_count += 1
            elif local_join_kind == "cyclic_feedback_exit_frontier":
                conditional_cyclic_feedback_exit_frontier_region_count += 1
            elif local_join_kind == "cyclic_external_exit_frontier":
                conditional_cyclic_external_exit_frontier_region_count += 1
            else:
                raise ValueError(f"unsupported cyclic branch join kind: {local_join_kind}")
            local_join = cyclic_model.get("local_join")
            nodes = {header, *successors}
            if local_join is not None:
                nodes.add(str(local_join))
            for arm_nodes in cyclic_model.get("arm_nodes_by_successor", {}).values():
                nodes.update(arm_nodes)
            for frontier in cyclic_model.get("arm_exit_frontier_by_successor", {}).values():
                nodes.update(frontier)
            if local_join is not None:
                exit_text = str(local_join)
            elif local_join_kind == "cyclic_feedback_frontier":
                exit_text = CYCLIC_FEEDBACK_FRONTIER
            elif local_join_kind == "cyclic_feedback_exit_frontier":
                exit_text = CYCLIC_FEEDBACK_EXIT_FRONTIER
            elif local_join_kind == "cyclic_external_exit_frontier":
                exit_text = CYCLIC_EXTERNAL_EXIT_FRONTIER
            else:
                raise ValueError(f"unsupported cyclic branch join kind: {local_join_kind}")
            facts.append(
                VMRegionFact(
                    id=f"cyclic_branch{len(facts)}",
                    kind="conditional_cyclic_scc_region",
                    header=header,
                    nodes=_stable_sorted(nodes),
                    exit=exit_text,
                    confidence=1.0,
                    evidence={
                        "header_scc": header_scc,
                        "successors": successors,
                        "successor_sccs": successor_sccs,
                        "local_join": local_join,
                        "local_join_kind": local_join_kind,
                        "exit_frontier": cyclic_model.get("exit_frontier", []),
                        "predicate_polarity": semantic.get("predicate_polarity"),
                    },
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
                    "header_scc": header_scc,
                    "successor_sccs": successor_sccs,
                    "successors": successors,
                    "join_scc": join_scc,
                    "join_kind": join_kind,
                    "region_kind": "conditional_multi_exit_region" if join_kind == "virtual_function_exit" else "conditional_region",
                    "predicate_polarity": semantic.get("predicate_polarity"),
                },
            )
        )

    fact_hist = Counter(f.kind for f in facts)
    hierarchy: list[dict[str, Any]] = []
    hierarchy_hist: Counter[str] = Counter()
    raw_cfg_entry_blocks_by_scc = _entry_blocks_by_scc(
        blocks=raw_blocks,
        blocks_by_scc=raw_model.get("blocks_by_scc", {}),
        preds=raw_model.get("preds", {}),
    )
    raw_cfg_multi_entry_scc_count = sum(
        1 for scc in raw_model.get("cyclic_sccs", set())
        if len(raw_cfg_entry_blocks_by_scc.get(scc, [])) > 1
    )
    projected_multi_entry_scc_count = sum(
        1 for scc in cyclic_sccs
        if len(raw_entry_blocks_by_scc.get(scc, [])) > 1
    )
    normalized_multi_entry_scc_count = sum(
        1 for scc in cyclic_sccs
        if len(entry_blocks_by_scc.get(scc, [])) > 1
    )
    cyclic_entry_port_count = sum(len(ports) for ports in cyclic_entry_ports_by_scc.values())
    summary = {
        "block_count": len(blocks),
        "proven_edge_count": len(proven_edges),
        "raw_block_count": len(raw_blocks),
        "raw_proven_edge_count": len(raw_model.get("proven_edges", [])),
        "raw_scc_count": len(raw_model.get("blocks_by_scc", {})),
        "raw_cyclic_scc_count": len(raw_model.get("cyclic_sccs", set())),
        **projection_summary,
        "reachable_block_count": len(reachable),
        "unreachable_under_proven_edges_count": max(0, len(blocks) - len(reachable)),
        "scc_count": len(blocks_by_scc),
        "cyclic_scc_count": len(cyclic_sccs),
        "raw_cfg_multi_entry_cyclic_scc_count": raw_cfg_multi_entry_scc_count,
        "projected_multi_entry_cyclic_scc_count": projected_multi_entry_scc_count,
        "raw_multi_entry_cyclic_scc_count": projected_multi_entry_scc_count,
        "normalized_multi_entry_cyclic_scc_count": normalized_multi_entry_scc_count,
        "cyclic_entry_port_count": cyclic_entry_port_count,
        "cyclic_entry_port_scc_count": sum(1 for ports in cyclic_entry_ports_by_scc.values() if ports),
        "raw_exit_scc_count": len(raw_exit_sccs),
        "region_fact_count": len(facts),
        "region_kind_histogram": dict(sorted(fact_hist.items())),
        "hierarchy_node_count": 0,
        "hierarchy_kind_histogram": {},
        "cyclic_edge_role_count": 0,
        "cyclic_edge_role_histogram": {},
        "conditional_branch_count": conditional_branch_count,
        "conditional_branch_context_count": int(conditional_contexts.get("context_count", 0) or 0),
        "conditional_branch_atom_count": int(conditional_contexts.get("atom_count", 0) or 0),
        "raw_conditional_branch_context_count": int(raw_conditional_contexts.get("context_count", 0) or 0),
        "raw_conditional_branch_atom_count": int(raw_conditional_contexts.get("atom_count", 0) or 0),
        "conditional_alias_context_count": conditional_alias_context_count,
        "conditional_alias_group_count": conditional_alias_group_count,
        "conditional_alias_group_size_histogram": dict(sorted((raw_conditional_contexts.get("alias_group_size_histogram") or {}).items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "conditional_two_successor_count": conditional_two_successor_count,
        "conditional_two_edge_count": conditional_two_edge_count,
        "conditional_projection_no_split_count": conditional_projection_no_split_count,
        "conditional_edge_count_histogram": dict(sorted(conditional_edge_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "conditional_distinct_successor_count_histogram": dict(sorted(distinct_successor_count_hist.items(), key=lambda kv: (int(kv[0]), kv[0]))),
        "conditional_same_scc_successors_count": conditional_same_scc_successors_count,
        "conditional_cross_scc_count": conditional_cross_scc_count,
        "conditional_scc_region_count": conditional_scc_region_count,
        "conditional_real_scc_join_region_count": conditional_real_scc_join_region_count,
        "conditional_function_exit_scc_region_count": conditional_function_exit_scc_region_count,
        "conditional_cyclic_scc_region_count": conditional_cyclic_scc_region_count,
        "conditional_cyclic_local_join_region_count": conditional_cyclic_local_join_region_count,
        "conditional_cyclic_feedback_join_region_count": conditional_cyclic_feedback_join_region_count,
        "conditional_cyclic_boundary_join_region_count": conditional_cyclic_boundary_join_region_count,
        "conditional_cyclic_feedback_frontier_region_count": conditional_cyclic_feedback_frontier_region_count,
        "conditional_cyclic_feedback_exit_frontier_region_count": conditional_cyclic_feedback_exit_frontier_region_count,
        "conditional_cyclic_external_exit_frontier_region_count": conditional_cyclic_external_exit_frontier_region_count,
        "virtual_function_exit": VIRTUAL_FUNCTION_EXIT,
        "cyclic_feedback_frontier": CYCLIC_FEEDBACK_FRONTIER,
        "cyclic_feedback_exit_frontier": CYCLIC_FEEDBACK_EXIT_FRONTIER,
        "cyclic_external_exit_frontier": CYCLIC_EXTERNAL_EXIT_FRONTIER,
        "policy": REGION_POLICY,
    }
    return VMRegionReport(REGION_CONTRACT_VERSION, summary, facts, scc_regions, hierarchy)


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
    region_kind_totals: Counter[str] = Counter()
    for entry in entries:
        raw, selection = select_function_body_vmir(mod, entry)
        words = decode_words(raw)
        span = selection.get("span") or {"start": 0, "end": 0}
        cfg = build_control_graph(words, function_start=int(span.get("start", 0) or 0), raw=raw)
        report = analyze_regions(cfg).to_dict()
        summary = report["summary"]
        for key in REGION_TOTAL_KEYS:
            totals[key] += int(summary.get(key, 0) or 0)
        region_kind_totals.update(summary.get("region_kind_histogram", {}))
        functions.append({
            "name": entry.name,
            "symbol": entry.symbol,
            "span": span,
            "summary": summary,
            "facts": report["facts"] if function else [],
            "scc_regions": report["scc_regions"] if function else [],
            "hierarchy": report["hierarchy"] if function else [],
        })
    return {
        "contract": REGION_CONTRACT_VERSION,
        "module": str(mod.path),
        "summary": {
            **dict(totals),
            "function_count": len(functions),
            "region_kind_histogram": dict(sorted(region_kind_totals.items())),
            "policy": REGION_POLICY,
        },
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
