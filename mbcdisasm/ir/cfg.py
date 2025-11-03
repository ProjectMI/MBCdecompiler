"""Control-flow graph construction helpers for the normalised IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .model import (
    IRAbiFunctionReport,
    IRAbiMetrics,
    IRBlock,
    IRCallReturn,
    IRControlFlowGraph,
    IRCfgBlock,
    IRCfgEdge,
    IRDispatchCase,
    IRFunctionCfg,
    IRFunctionPrologue,
    IRIf,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
    IRFlagCheck,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
)


@dataclass(frozen=True)
class _FunctionStats:
    """Internal summary used to compute ABI consistency metrics."""

    segment_index: int
    name: str
    entry_offset: int
    block_labels: Tuple[str, ...]
    masks: Tuple[Optional[int], ...]

    @property
    def total_exits(self) -> int:
        return len(self.masks)

    @property
    def masked_exits(self) -> int:
        return sum(mask is not None for mask in self.masks)

    @property
    def distinct_masks(self) -> Tuple[Optional[int], ...]:
        ordered: List[Optional[int]] = []
        for mask in self.masks:
            if mask not in ordered:
                ordered.append(mask)
        return tuple(ordered)


def analyse_segments(
    segments: Sequence[IRSegment],
) -> Tuple[IRControlFlowGraph, IRAbiMetrics]:
    """Construct the CFG for ``segments`` and compute ABI metrics."""

    cfg, stats = _build_cfg_with_stats(segments)
    metrics = _compute_abi_metrics(stats)
    return cfg, metrics


def _build_cfg_with_stats(
    segments: Sequence[IRSegment],
) -> Tuple[IRControlFlowGraph, Tuple[_FunctionStats, ...]]:
    functions: List[IRFunctionCfg] = []
    summaries: List[_FunctionStats] = []

    for segment in segments:
        offset_to_label = {block.start_offset: block.label for block in segment.blocks}
        for name, blocks in _group_blocks(segment.blocks):
            if not blocks:
                continue

            cfg_blocks: List[IRCfgBlock] = []
            masks: List[Optional[int]] = []
            for block in blocks:
                terminator = _find_terminator(block)
                terminator_desc = _describe_terminator(terminator)
                edges = tuple(_edges_from_terminator(terminator, offset_to_label))
                cfg_blocks.append(
                    IRCfgBlock(
                        label=block.label,
                        start_offset=block.start_offset,
                        terminator=terminator_desc,
                        edges=edges,
                    )
                )
                masks.extend(_collect_exit_masks(block))

            summaries.append(
                _FunctionStats(
                    segment_index=segment.index,
                    name=name,
                    entry_offset=blocks[0].start_offset,
                    block_labels=tuple(block.label for block in blocks),
                    masks=tuple(masks),
                )
            )

            functions.append(
                IRFunctionCfg(
                    segment_index=segment.index,
                    name=name,
                    entry_block=blocks[0].label,
                    entry_offset=blocks[0].start_offset,
                    blocks=tuple(cfg_blocks),
                )
            )

    return IRControlFlowGraph(functions=tuple(functions)), tuple(summaries)


def _group_blocks(blocks: Sequence[IRBlock]) -> Iterable[Tuple[str, Tuple[IRBlock, ...]]]:
    groups: List[Tuple[str, Tuple[IRBlock, ...]]] = []
    current_blocks: List[IRBlock] = []
    current_name: Optional[str] = None
    auto_counter = 0

    for block in blocks:
        has_prologue = any(isinstance(node, IRFunctionPrologue) for node in block.nodes)
        if has_prologue or current_name is None:
            if current_blocks:
                groups.append((current_name or f"auto_{auto_counter}", tuple(current_blocks)))
                current_blocks = []
            if has_prologue:
                current_name = f"prologue_{block.start_offset:04X}"
            else:
                current_name = f"auto_{auto_counter}"
                auto_counter += 1
        current_blocks.append(block)

    if current_blocks:
        groups.append((current_name or f"auto_{auto_counter}", tuple(current_blocks)))

    return groups


_TERMINATOR_TYPES = (
    IRReturn,
    IRTailCall,
    IRTailcallReturn,
    IRCallReturn,
    IRIf,
    IRTestSetBranch,
    IRFlagCheck,
    IRFunctionPrologue,
    IRSwitchDispatch,
    IRTerminator,
)


def _find_terminator(block: IRBlock):
    for node in reversed(block.nodes):
        if isinstance(node, _TERMINATOR_TYPES):
            return node
    return None


def _describe_terminator(node) -> str:
    if node is None:
        return ""
    describe = getattr(node, "describe", None)
    if callable(describe):
        return describe()
    return repr(node)


def _edges_from_terminator(node, offset_to_label: Dict[int, str]) -> Iterable[IRCfgEdge]:
    if node is None:
        return tuple()

    edges: List[IRCfgEdge] = []

    if isinstance(node, (IRIf, IRTestSetBranch, IRFlagCheck, IRFunctionPrologue)):
        edges.append(IRCfgEdge("then", _resolve_target(node.then_target, offset_to_label)))
        edges.append(IRCfgEdge("else", _resolve_target(node.else_target, offset_to_label)))

    if isinstance(node, IRSwitchDispatch):
        for case in node.cases:
            edges.append(_case_edge(case, offset_to_label))
        if node.default is not None:
            edges.append(IRCfgEdge("default", _resolve_target(node.default, offset_to_label)))

    predicate = getattr(node, "predicate", None)
    if predicate is not None:
        kind = predicate.kind or "predicate"
        if predicate.then_target is not None:
            edges.append(
                IRCfgEdge(f"{kind}.then", _resolve_target(predicate.then_target, offset_to_label))
            )
        if predicate.else_target is not None:
            edges.append(
                IRCfgEdge(f"{kind}.else", _resolve_target(predicate.else_target, offset_to_label))
            )

    return tuple(edges)


def _case_edge(case: IRDispatchCase, offset_to_label: Dict[int, str]) -> IRCfgEdge:
    label = _resolve_target(case.target, offset_to_label)
    return IRCfgEdge(f"case 0x{case.key:02X}", label)


def _resolve_target(target: int, offset_to_label: Dict[int, str]) -> str:
    label = offset_to_label.get(target)
    if label is not None:
        return label
    return f"0x{target:04X}"


def _collect_exit_masks(block: IRBlock) -> Iterable[Optional[int]]:
    masks: List[Optional[int]] = []
    for node in block.nodes:
        mask = _node_return_mask(node)
        if mask is not None or _is_exit_node(node):
            masks.append(mask)
    return masks


def _is_exit_node(node) -> bool:
    if isinstance(node, IRReturn):
        return not node.varargs
    if isinstance(node, (IRTailCall, IRTailcallReturn, IRCallReturn)):
        return not getattr(node, "varargs", False)
    return False


def _node_return_mask(node) -> Optional[int]:
    if isinstance(node, IRReturn):
        if node.varargs:
            return None
        return node.mask
    if isinstance(node, IRTailCall):
        if getattr(node, "varargs", False):
            return None
        return node.cleanup_mask
    if isinstance(node, IRTailcallReturn):
        if node.varargs:
            return None
        return node.cleanup_mask
    if isinstance(node, IRCallReturn):
        if node.varargs:
            return None
        return node.cleanup_mask
    return None


def _compute_abi_metrics(functions: Sequence[_FunctionStats]) -> IRAbiMetrics:
    total_exits = sum(entry.total_exits for entry in functions)
    masked_exits = sum(entry.masked_exits for entry in functions)

    considered = [entry for entry in functions if entry.total_exits > 0]
    total_functions = len(considered)
    inconsistent = [entry for entry in considered if len(entry.distinct_masks) > 1]

    missing_candidates: List[IRAbiFunctionReport] = []
    for entry in considered:
        if entry.masked_exits < entry.total_exits:
            missing_candidates.append(
                IRAbiFunctionReport(
                    segment_index=entry.segment_index,
                    name=entry.name,
                    entry_offset=entry.entry_offset,
                    total_exits=entry.total_exits,
                    masked_exits=entry.masked_exits,
                    masks=entry.distinct_masks,
                )
            )

    missing_candidates.sort(
        key=lambda report: (
            (report.masked_exits / report.total_exits) if report.total_exits else 1.0,
            report.entry_offset,
        )
    )

    inconsistent_candidates = [
        IRAbiFunctionReport(
            segment_index=entry.segment_index,
            name=entry.name,
            entry_offset=entry.entry_offset,
            total_exits=entry.total_exits,
            masked_exits=entry.masked_exits,
            masks=entry.distinct_masks,
        )
        for entry in inconsistent
    ]
    inconsistent_candidates.sort(
        key=lambda report: (
            -len(report.masks),
            report.entry_offset,
        )
    )

    return IRAbiMetrics(
        total_exits=total_exits,
        masked_exits=masked_exits,
        total_functions=total_functions,
        inconsistent_functions=len(inconsistent),
        missing_mask_functions=tuple(missing_candidates[:10]),
        inconsistent_mask_functions=tuple(inconsistent_candidates[:10]),
    )


__all__ = ["analyse_segments"]
