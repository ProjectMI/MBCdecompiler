"""Control-flow graph utilities for the IR segments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple

from mbcdisasm.ir.model import (
    IRBlock,
    IRCall,
    IRCallReturn,
    IRFlagCheck,
    IRFunctionPrologue,
    IRIf,
    IRReturn,
    IRSwitchDispatch,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
)


@dataclass(frozen=True)
class CFGNode:
    """Single control-flow node corresponding to an IR block."""

    label: str
    block: IRBlock
    successors: Tuple[str, ...]
    predecessors: Tuple[str, ...]


@dataclass(frozen=True)
class ControlFlowGraph:
    """Control-flow graph extracted from a segment."""

    segment_index: int
    entry_label: str
    nodes: Mapping[str, CFGNode]

    def iter_successors(self, label: str) -> Iterable[str]:
        node = self.nodes.get(label)
        if node is None:
            return ()
        return node.successors


class CFGBuilder:
    """Build a :class:`ControlFlowGraph` from IR blocks."""

    def build(self, segment) -> ControlFlowGraph:
        blocks = tuple(segment.blocks)
        if not blocks:
            return ControlFlowGraph(segment.index, "", MappingProxyType({}))

        ordered = sorted(blocks, key=lambda block: block.start_offset)
        label_to_block: Dict[str, IRBlock] = {block.label: block for block in ordered}
        offset_to_label: Dict[int, str] = {
            block.start_offset: block.label for block in ordered
        }
        next_map: Dict[str, str] = {}
        for first, second in zip(ordered, ordered[1:]):
            next_map[first.label] = second.label

        succ_map: Dict[str, Tuple[str, ...]] = {}
        pred_map: Dict[str, Set[str]] = {block.label: set() for block in ordered}

        for block in ordered:
            successors = self._resolve_successors(
                block,
                offset_to_label,
                label_to_block,
                next_map,
            )
            succ_map[block.label] = successors
            for target in successors:
                pred_map.setdefault(target, set()).add(block.label)

        nodes: Dict[str, CFGNode] = {}
        for block in ordered:
            preds = tuple(
                sorted(pred_map.get(block.label, set()), key=lambda label: label_to_block[label].start_offset)
            )
            succs = succ_map.get(block.label, tuple())
            nodes[block.label] = CFGNode(
                label=block.label,
                block=block,
                successors=succs,
                predecessors=preds,
            )

        entry_label = ordered[0].label
        return ControlFlowGraph(
            segment_index=segment.index,
            entry_label=entry_label,
            nodes=MappingProxyType(nodes),
        )

    def _resolve_successors(
        self,
        block: IRBlock,
        offset_to_label: Mapping[int, str],
        label_to_block: Mapping[str, IRBlock],
        next_map: Mapping[str, str],
    ) -> Tuple[str, ...]:
        if not block.nodes:
            return self._fallthrough(block, label_to_block, next_map)

        last = block.nodes[-1]
        explicit = self._explicit_successor_offsets(last)
        if explicit is None:
            return self._fallthrough(block, label_to_block, next_map)

        labels: Dict[str, int] = {}
        for offset in explicit:
            label = offset_to_label.get(offset)
            if label is None:
                continue
            labels[label] = label_to_block[label].start_offset

        ordered = tuple(sorted(labels, key=lambda label: labels[label]))
        return ordered

    def _fallthrough(
        self,
        block: IRBlock,
        label_to_block: Mapping[str, IRBlock],
        next_map: Mapping[str, str],
    ) -> Tuple[str, ...]:
        next_label = next_map.get(block.label)
        if next_label is None:
            return tuple()
        return (next_label,)

    @staticmethod
    def _explicit_successor_offsets(node) -> Sequence[int] | None:
        if isinstance(node, (IRReturn, IRTailCall, IRTailcallReturn, IRCallReturn, IRTerminator)):
            return ()
        if isinstance(node, IRCall) and node.tail:
            return ()
        if isinstance(node, (IRIf, IRTestSetBranch, IRFlagCheck, IRFunctionPrologue)):
            return (node.then_target, node.else_target)
        if isinstance(node, IRSwitchDispatch):
            offsets = [case.target for case in node.cases]
            if node.default is not None:
                offsets.append(node.default)
            return offsets
        return None

