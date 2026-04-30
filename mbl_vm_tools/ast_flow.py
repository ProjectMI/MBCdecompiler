from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from mbl_vm_tools.ast_cfg import AstCFG, external_successors, loopback_edges


@dataclass(frozen=True)
class TransferAction:
    kind: str
    target_id: Optional[str] = None
    target_label: Optional[str] = None
    alias: Any = None
    needs_label: bool = False
    counts_as_goto: bool = False


@dataclass(frozen=True)
class TransferAliasPlan:
    target_id: str
    sources: set[str]
    jump_kind: str
    jump_target: Optional[str]
    source_specific: bool = False
    needs_label: bool = False



@dataclass(frozen=True)
class CyclicRegionPlan:
    entry_id: str
    nodes: set[str]
    end_index: int
    exit_ids: list[str]
    classification: str
    suppress_nested_natural_loops: bool = True


def classify_cyclic_component(component: dict[str, Any], cfg: AstCFG) -> Optional[CyclicRegionPlan]:
    nodes = {str(node) for node in component.get("nodes") or set()}
    if len(nodes) < 2 or not component.get("contiguous"):
        return None
    entry_id = str(component.get("entry") or "")
    if not entry_id or entry_id not in nodes:
        return None
    backward_targets = {str(target) for target in component.get("backward_targets") or set()}
    nonlinear_edges = list(component.get("nonlinear_internal_edges") or [])
    branch_nodes = list(component.get("branch_nodes") or [])
    natural_loop = cfg.loops.get(entry_id)
    is_natural_loop = bool(natural_loop and set(natural_loop.get("nodes") or set()) == nodes and component.get("contiguous"))
    natural_exit_count = len(external_successors(natural_loop, cfg)) if natural_loop else 0
    header_block = cfg.by_id.get(entry_id)
    exit_id = natural_loop.get("exit_succ") if natural_loop else None
    exit_block = cfg.by_id.get(str(exit_id)) if exit_id is not None else None
    has_header_exit_params = bool(exit_block and exit_block.block_params and entry_id in exit_block.incoming_args)
    has_header_side_effects = bool(header_block and header_block.statements)
    is_unsafe_natural_loop = bool(is_natural_loop and not natural_loop.get("safe"))
    is_multi_exit_natural_loop = bool(is_natural_loop and natural_loop.get("safe") and natural_exit_count > 1)
    is_guarded_safe_natural_loop = bool(is_natural_loop and natural_loop.get("safe") and (has_header_side_effects or has_header_exit_params))
    force_cyclic_region = is_unsafe_natural_loop or is_multi_exit_natural_loop or is_guarded_safe_natural_loop

    if not force_cyclic_region:
        if backward_targets and backward_targets <= {entry_id} and len(nonlinear_edges) <= len(branch_nodes) + 1:
            return None
        if len(backward_targets) < 2 and len(nonlinear_edges) < 2:
            return None

    idx0, idx1 = component.get("index_range") or (None, None)
    if not isinstance(idx0, int) or not isinstance(idx1, int):
        return None

    external_entries = [(str(pred), str(node)) for pred, node in (component.get("external_entries") or [])]
    side_entries = [(pred, node) for pred, node in external_entries if node != entry_id]
    if side_entries:
        classification = "state_machine"
    elif force_cyclic_region:
        classification = "cyclic_loop"
    else:
        classification = "cyclic_decision_loop"
    return CyclicRegionPlan(
        entry_id=entry_id,
        nodes=nodes,
        end_index=idx1,
        exit_ids=[str(item) for item in (component.get("exit_ids") or [])],
        classification=classification,
    )


def classify_linear_cyclic_span(
    cfg: AstCFG,
    *,
    idx: int,
    stop: int,
    allowed: Optional[set[str]],
    previous_source_id: Optional[str],
    settled_preds_by_block: dict[str, set[str]],
) -> Optional[CyclicRegionPlan]:
    """Recognize a contiguous cyclic span that is not a natural-loop header.

    This is a data-level cyclic-region planner: it only decides whether the
    linear block span should be represented as a decision_loop.  AST materializes
    the returned plan; no legacy jump aliases are produced here.
    """
    limit = min(stop, len(cfg.blocks))
    if idx >= limit:
        return None
    if allowed is not None and cfg.blocks[idx].id not in allowed:
        return None

    end = idx
    saw_non_linear = False
    saw_backward = False
    changed = True
    while changed:
        changed = False
        for pos in range(idx, end + 1):
            block = cfg.blocks[pos]
            if allowed is not None and block.id not in allowed:
                return None
            for succ in cfg.succs.get(block.id, []):
                succ_idx = cfg.index_by_id.get(succ)
                if succ_idx is None or succ_idx < idx or succ_idx >= limit:
                    continue
                if allowed is not None and succ not in allowed:
                    continue
                if succ_idx != block.index + 1:
                    saw_non_linear = True
                    if succ_idx <= block.index:
                        saw_backward = True
                    if succ_idx > end:
                        end = succ_idx
                        changed = True
        for pos in range(end + 1, limit):
            block = cfg.blocks[pos]
            if allowed is not None and block.id not in allowed:
                break
            reaches_current_span = False
            for succ in cfg.succs.get(block.id, []):
                succ_idx = cfg.index_by_id.get(succ)
                if succ_idx is None or succ_idx < idx or succ_idx > end:
                    continue
                if succ_idx != block.index + 1:
                    reaches_current_span = True
                    saw_non_linear = True
                    saw_backward = True
                    break
            if reaches_current_span:
                end = pos
                changed = True
                break

    if not saw_non_linear or not saw_backward or end <= idx:
        return None

    span_blocks = cfg.blocks[idx : end + 1]
    span_ids = {block.id for block in span_blocks}
    if allowed is not None and any(block.id not in allowed for block in span_blocks):
        return None

    entry_id = cfg.blocks[idx].id
    for block in span_blocks:
        settled = set(settled_preds_by_block.get(block.id, set()))
        for pred in cfg.preds.get(block.id, []):
            if pred in span_ids or pred in settled:
                continue
            if block.id == entry_id and pred == previous_source_id:
                continue
            return None

    next_after_id = cfg.blocks[end + 1].id if end + 1 < limit else None
    if next_after_id is None and end + 1 < len(cfg.blocks) and end + 1 <= stop:
        has_settled_side_entry = False
        for span_block in span_blocks:
            if span_block.id == entry_id:
                continue
            settled = set(settled_preds_by_block.get(span_block.id, set()))
            for pred in cfg.preds.get(span_block.id, []):
                if pred not in span_ids and pred in settled:
                    has_settled_side_entry = True
                    break
            if has_settled_side_entry:
                break
        if has_settled_side_entry:
            next_after_id = cfg.blocks[end + 1].id

    has_internal_jump = False
    for block in span_blocks:
        for succ in cfg.succs.get(block.id, []):
            succ_idx = cfg.index_by_id.get(succ)
            if succ in span_ids:
                if succ_idx is not None and succ_idx != block.index + 1:
                    has_internal_jump = True
                continue
            if next_after_id is not None and succ == next_after_id:
                continue
            return None

    if not has_internal_jump:
        return None

    return CyclicRegionPlan(
        entry_id=entry_id,
        nodes=span_ids,
        end_index=end,
        exit_ids=[next_after_id] if next_after_id is not None else [],
        classification="state_machine",
    )

def transfer_target_label(target_id: Optional[str]) -> Optional[str]:
    return target_id


def alias_for_source(target_id: Optional[str], source_id: Optional[str], aliases: dict[str, Any]) -> Any:
    if target_id is None:
        return None
    alias = aliases.get(target_id)
    if not isinstance(alias, dict):
        return alias
    by_source = alias.get("prelude_by_source") or {}
    if by_source and source_id not in by_source:
        return None
    return alias



def merge_scoped_aliases(parent: dict[str, Any], local: dict[str, Any], local_nodes: set[str]) -> dict[str, Any]:
    merged = dict(parent)
    for target_id, alias in local.items():
        target = str(target_id)
        if target in parent and target not in local_nodes:
            continue
        merged[target] = alias
    return merged

def classify_transfer(
    *,
    source_id: Optional[str],
    source_index: Optional[int],
    target_id: Optional[str],
    target_index: Optional[int],
    aliases: dict[str, Any],
    source_specific_alias: bool = False,
) -> TransferAction:
    if target_id is None:
        return TransferAction(kind="none")

    raw_alias = aliases.get(target_id)
    if source_specific_alias or (isinstance(raw_alias, dict) and raw_alias.get("source_specific")):
        alias = alias_for_source(target_id, source_id, aliases)
    else:
        alias = raw_alias

    if alias is not None:
        return TransferAction(kind="alias", target_id=target_id, alias=alias)

    target_label = transfer_target_label(target_id)
    if source_index is not None and target_index is not None and target_index > source_index:
        return TransferAction(kind="break", target_id=target_id, target_label=target_label, needs_label=True)
    return TransferAction(kind="goto", target_id=target_id, target_label=target_label, needs_label=True, counts_as_goto=True)


def plan_loop_aliases(loop: dict[str, Any], cfg: AstCFG, exit_ids: Optional[list[str]] = None) -> list[TransferAliasPlan]:
    nodes = set(loop.get("nodes") or set())
    header_id = str(loop.get("header"))
    plans: list[TransferAliasPlan] = []

    for target_id, sources in loopback_edges(loop, cfg).items():
        target = str(target_id)
        plans.append(
            TransferAliasPlan(
                target_id=target,
                sources={str(source_id) for source_id in sources},
                jump_kind="continue",
                jump_target=None if target == header_id else target,
                needs_label=target != header_id,
            )
        )

    exits = list(exit_ids) if exit_ids is not None else external_successors(loop, cfg)
    idx0, idx1 = loop.get("index_range") or (None, None)
    next_after = cfg.blocks[idx1 + 1].id if isinstance(idx1, int) and idx1 + 1 < len(cfg.blocks) else None
    if next_after in exits:
        exits.remove(next_after)
        exits.insert(0, next_after)

    for pos, exit_id in enumerate(exits):
        target = str(exit_id)
        sources = {str(source_id) for source_id in nodes if exit_id in cfg.succs.get(source_id, [])}
        plans.append(
            TransferAliasPlan(
                target_id=target,
                sources=sources,
                jump_kind="break",
                jump_target=None if pos == 0 else target,
                needs_label=pos != 0,
            )
        )
    return plans
