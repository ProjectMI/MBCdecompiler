"""Lightweight control-flow graph structures used during lifting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..ir.model import (
    IRBlock,
    IRFlagCheck,
    IRFunctionPrologue,
    IRIf,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
    IRTestSetBranch,
    IRTerminator,
    IRTailCall,
    IRTailcallReturn,
    IRCall,
    IRProgram,
)


@dataclass
class CFGNode:
    """Single node in the control-flow graph."""

    block: IRBlock
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)

    def add_successor(self, target: Optional[int]) -> None:
        if target is None:
            return
        if target not in self.successors:
            self.successors.append(target)

    def add_predecessor(self, source: int) -> None:
        if source not in self.predecessors:
            self.predecessors.append(source)


@dataclass(frozen=True)
class CFGSegment:
    """Graph representation scoped to a single :class:`IRSegment`."""

    segment: IRSegment
    nodes: Mapping[int, CFGNode]
    entry: int


@dataclass(frozen=True)
class CFGProgram:
    """Control-flow graphs for the entire programme."""

    segments: Tuple[CFGSegment, ...]


class CFGBuilder:
    """Build basic-block level graphs from :class:`IRProgram` instances."""

    def build(self, program: IRProgram) -> CFGProgram:
        segments = tuple(self._build_segment(segment) for segment in program.segments)
        return CFGProgram(segments)

    # ------------------------------------------------------------------
    # segment handling
    # ------------------------------------------------------------------
    def _build_segment(self, segment: IRSegment) -> CFGSegment:
        nodes: Dict[int, CFGNode] = {
            block.start_offset: CFGNode(block) for block in segment.blocks
        }
        entry = min(nodes) if nodes else segment.start
        ordered_offsets = sorted(nodes)
        offset_to_index = {offset: idx for idx, offset in enumerate(ordered_offsets)}

        for offset, node in nodes.items():
            successors = self._collect_successors(node.block, ordered_offsets, offset_to_index)
            for target in successors:
                node.add_successor(target)
                target_node = nodes.get(target)
                if target_node is not None:
                    target_node.add_predecessor(offset)

        return CFGSegment(segment=segment, nodes=nodes, entry=entry)

    # ------------------------------------------------------------------
    # successor analysis
    # ------------------------------------------------------------------
    def _collect_successors(
        self,
        block: IRBlock,
        ordered_offsets: Sequence[int],
        offset_to_index: Mapping[int, int],
    ) -> List[int]:
        successors: List[int] = []
        fallthrough = self._next_offset(block.start_offset, ordered_offsets, offset_to_index)

        for node in reversed(block.nodes):
            if isinstance(node, (IRReturn, IRTerminator, IRTailCall, IRTailcallReturn)):
                return []
            if isinstance(node, IRIf):
                successors.extend(self._resolve_branch_targets(node.then_target, node.else_target, fallthrough))
                return successors
            if isinstance(node, IRTestSetBranch):
                successors.extend(self._resolve_branch_targets(node.then_target, node.else_target, fallthrough))
                return successors
            if isinstance(node, IRFlagCheck):
                successors.extend(self._resolve_branch_targets(node.then_target, node.else_target, fallthrough))
                return successors
            if isinstance(node, IRFunctionPrologue):
                successors.extend(self._resolve_branch_targets(node.then_target, node.else_target, fallthrough))
                return successors
            if isinstance(node, IRSwitchDispatch):
                for case in node.cases:
                    successors.append(case.target)
                if node.default is not None:
                    successors.append(node.default)
                if fallthrough is not None:
                    successors.append(fallthrough)
                return successors
            if isinstance(node, IRCall) and node.predicate is not None:
                successors.extend(
                    self._resolve_branch_targets(
                        node.predicate.then_target,
                        node.predicate.else_target,
                        fallthrough,
                    )
                )
                return successors

        if fallthrough is not None:
            successors.append(fallthrough)
        return successors

    def _resolve_branch_targets(
        self,
        then_target: Optional[int],
        else_target: Optional[int],
        fallthrough: Optional[int],
    ) -> List[int]:
        targets: List[int] = []
        resolved_then = self._resolve_target(then_target, fallthrough)
        resolved_else = self._resolve_target(else_target, fallthrough)
        if resolved_then is not None:
            targets.append(resolved_then)
        if resolved_else is not None and resolved_else not in targets:
            targets.append(resolved_else)
        return targets

    def _resolve_target(self, target: Optional[int], fallthrough: Optional[int]) -> Optional[int]:
        if target is None or target == 0:
            return fallthrough
        return target

    def _next_offset(
        self,
        current: int,
        ordered_offsets: Sequence[int],
        offset_to_index: Mapping[int, int],
    ) -> Optional[int]:
        index = offset_to_index.get(current)
        if index is None:
            return None
        next_index = index + 1
        if next_index >= len(ordered_offsets):
            return None
        return ordered_offsets[next_index]


__all__ = ["CFGBuilder", "CFGProgram", "CFGSegment", "CFGNode"]
