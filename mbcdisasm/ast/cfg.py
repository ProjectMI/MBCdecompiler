"""Control-flow graph construction and procedure discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..ir.model import (
    IRBlock,
    IRCall,
    IRCallReturn,
    IRFunctionPrologue,
    IRIf,
    IRReturn,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRFlagCheck,
    IRTerminator,
)
from ..ir.model import IRSegment
from .model import (
    CFGEdge,
    CFGEdgeKind,
    CFGNode,
    SegmentCFG,
    CallSite,
    Procedure,
)


def _block_successors(block: IRBlock) -> List[Tuple[CFGEdgeKind, int, str]]:
    """Extract successor targets from ``block``."""

    successors: List[Tuple[CFGEdgeKind, int, str]] = []
    for node in block.nodes:
        if isinstance(node, IRIf):
            successors.append((CFGEdgeKind.BRANCH_TRUE, node.then_target, "then"))
            successors.append((CFGEdgeKind.BRANCH_FALSE, node.else_target, "else"))
        elif isinstance(node, IRTestSetBranch):
            successors.append((CFGEdgeKind.BRANCH_TRUE, node.then_target, "then"))
            successors.append((CFGEdgeKind.BRANCH_FALSE, node.else_target, "else"))
        elif isinstance(node, IRFlagCheck):
            successors.append((CFGEdgeKind.BRANCH_TRUE, node.then_target, "flag"))
            successors.append((CFGEdgeKind.BRANCH_FALSE, node.else_target, "fallthrough"))
        elif isinstance(node, IRFunctionPrologue):
            successors.append((CFGEdgeKind.BRANCH_TRUE, node.then_target, "prologue_then"))
            successors.append((CFGEdgeKind.BRANCH_FALSE, node.else_target, "prologue_else"))
        elif isinstance(node, IRCall):
            if node.predicate is not None:
                predicate = node.predicate
                if predicate.then_target is not None:
                    successors.append((CFGEdgeKind.CALL_PREDICATE_TRUE, predicate.then_target, "call_then"))
                if predicate.else_target is not None:
                    successors.append((CFGEdgeKind.CALL_PREDICATE_FALSE, predicate.else_target, "call_else"))
        elif isinstance(node, IRCallReturn):
            if node.predicate is not None:
                predicate = node.predicate
                if predicate.then_target is not None:
                    successors.append((CFGEdgeKind.CALL_PREDICATE_TRUE, predicate.then_target, "call_then"))
                if predicate.else_target is not None:
                    successors.append((CFGEdgeKind.CALL_PREDICATE_FALSE, predicate.else_target, "call_else"))
        elif isinstance(node, IRTailCall):
            if node.predicate is not None:
                predicate = node.predicate
                if predicate.then_target is not None:
                    successors.append((CFGEdgeKind.CALL_PREDICATE_TRUE, predicate.then_target, "tail_then"))
                if predicate.else_target is not None:
                    successors.append((CFGEdgeKind.CALL_PREDICATE_FALSE, predicate.else_target, "tail_else"))
        elif isinstance(node, IRTailcallReturn):
            if node.predicate is not None:
                predicate = node.predicate
                if predicate.then_target is not None:
                    successors.append((CFGEdgeKind.CALL_PREDICATE_TRUE, predicate.then_target, "tail_then"))
                if predicate.else_target is not None:
                    successors.append((CFGEdgeKind.CALL_PREDICATE_FALSE, predicate.else_target, "tail_else"))
    return successors


def _block_terminates(block: IRBlock) -> bool:
    for node in block.nodes:
        if isinstance(node, (IRReturn, IRTailCall, IRTailcallReturn, IRTerminator)):
            return True
    return False


def _gather_call_sites(block: IRBlock) -> List[CallSite]:
    sites: List[CallSite] = []
    for node in block.nodes:
        if isinstance(node, IRCall):
            sites.append(
                CallSite(
                    block_offset=block.start_offset,
                    target=node.target,
                    args=node.args,
                    tail=node.tail,
                    arity=node.arity,
                    convention=node.convention,
                    cleanup_mask=node.cleanup_mask,
                    symbol=node.symbol,
                    predicate=node.predicate,
                )
            )
        elif isinstance(node, IRTailCall):
            sites.append(
                CallSite(
                    block_offset=block.start_offset,
                    target=node.target,
                    args=node.args,
                    tail=True,
                    returns=node.returns,
                    varargs=node.varargs,
                    cleanup_mask=node.cleanup_mask,
                    convention=node.convention,
                    symbol=node.symbol,
                    predicate=node.predicate,
                )
            )
        elif isinstance(node, IRCallReturn):
            sites.append(
                CallSite(
                    block_offset=block.start_offset,
                    target=node.target,
                    args=node.args,
                    tail=node.tail,
                    returns=node.returns,
                    varargs=node.varargs,
                    cleanup_mask=node.cleanup_mask,
                    convention=node.convention,
                    symbol=node.symbol,
                    predicate=node.predicate,
                )
            )
        elif isinstance(node, IRTailcallReturn):
            sites.append(
                CallSite(
                    block_offset=block.start_offset,
                    target=node.target,
                    args=node.args,
                    tail=True,
                    returns=node.returns,
                    varargs=node.varargs,
                    cleanup_mask=node.cleanup_mask,
                    convention=node.convention,
                    symbol=node.symbol,
                    predicate=node.predicate,
                )
            )
    return sites


class CFGBuilder:
    """Build control-flow graphs from IR segments."""

    def build_segment(self, segment: IRSegment) -> SegmentCFG:
        nodes: Dict[int, CFGNode] = {}
        edges: List[CFGEdge] = []

        for block in segment.blocks:
            nodes[block.start_offset] = CFGNode(block=block)

        ordered_offsets = [block.start_offset for block in segment.blocks]
        offset_to_index = {offset: idx for idx, offset in enumerate(ordered_offsets)}

        for block in segment.blocks:
            node = nodes[block.start_offset]
            for call in _gather_call_sites(block):
                node.register_call(call)

            successors = _block_successors(block)
            seen_targets: Set[int] = set()
            for kind, target, label in successors:
                if target not in nodes:
                    continue
                edge = CFGEdge(source=block.start_offset, target=target, kind=kind, label=label)
                node.add_successor(edge)
                nodes[target].add_predecessor(edge)
                edges.append(edge)
                seen_targets.add(target)

            terminates = _block_terminates(block)
            if not terminates:
                index = offset_to_index.get(block.start_offset)
                if index is not None and index + 1 < len(ordered_offsets):
                    fallthrough_target = ordered_offsets[index + 1]
                    if fallthrough_target not in seen_targets:
                        edge = CFGEdge(
                            source=block.start_offset,
                            target=fallthrough_target,
                            kind=CFGEdgeKind.FALLTHROUGH,
                            label="fallthrough",
                        )
                        node.add_successor(edge)
                        nodes[fallthrough_target].add_predecessor(edge)
                        edges.append(edge)
            else:
                node.exits = True

        return SegmentCFG(segment=segment, nodes=nodes, edges=tuple(edges))


@dataclass
class _TraversalState:
    entry: int
    nodes: Dict[int, CFGNode]
    entries: Set[int]

    def traverse(self) -> Tuple[Set[int], Set[int]]:
        visited: Set[int] = set()
        exits: Set[int] = set()
        worklist: List[int] = [self.entry]
        while worklist:
            current = worklist.pop()
            if current in visited:
                continue
            visited.add(current)
            node = self.nodes.get(current)
            if node is None:
                continue
            if node.exits:
                exits.add(current)
                continue
            for edge in node.successors:
                target = edge.target
                if target in self.entries and target != self.entry:
                    continue
                worklist.append(target)
        return visited, exits


class ProcedureResolver:
    """Discover procedures by following CFG structure."""

    def detect(self, cfg: SegmentCFG) -> Tuple[Procedure, ...]:
        entry_candidates: Set[int] = set()
        blocks = cfg.segment.blocks
        if blocks:
            entry_candidates.add(blocks[0].start_offset)
        for node in cfg.nodes.values():
            for ir_node in node.block.nodes:
                if isinstance(ir_node, IRFunctionPrologue):
                    entry_candidates.add(node.block.start_offset)
                if isinstance(ir_node, (IRCall, IRTailCall, IRCallReturn)):
                    entry_candidates.add(ir_node.target)

        entry_candidates &= set(cfg.nodes)
        assigned: Dict[int, int] = {}
        procedures: List[Procedure] = []

        for entry in sorted(entry_candidates):
            if entry in assigned:
                continue
            state = _TraversalState(entry=entry, nodes=cfg.nodes, entries=entry_candidates)
            reachable, exits = state.traverse()
            if not reachable:
                continue
            for offset in reachable:
                assigned[offset] = entry

            call_sites = []
            tail_calls = []
            callees: Set[int] = set()
            prologue_node: Optional[IRFunctionPrologue] = None
            for offset in sorted(reachable):
                node = cfg.nodes[offset]
                for call in node.call_sites:
                    if call.tail:
                        tail_calls.append(call)
                    else:
                        call_sites.append(call)
                    if call.target in cfg.nodes:
                        callees.add(call.target)
                if offset == entry:
                    for ir_node in node.block.nodes:
                        if isinstance(ir_node, IRFunctionPrologue):
                            prologue_node = ir_node
                            break

            procedure = Procedure(
                entry_offset=entry,
                blocks=tuple(sorted(reachable)),
                exits=tuple(sorted(exits)),
                call_sites=tuple(call_sites),
                tail_calls=tuple(tail_calls),
                callees=tuple(sorted(callees)),
                prologue=prologue_node,
            )
            procedures.append(procedure)

        return tuple(procedures)

