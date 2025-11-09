"""Construction helpers for transforming CFGs into structured ASTs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, MutableMapping, MutableSet, Optional, Sequence, Set, Tuple

from ..ir.cfg import analyse_segments
from ..ir.model import (
    IRBlock,
    IRCallReturn,
    IRFunctionCfg,
    IRIf,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
    IRTestSetBranch,
    IRFlagCheck,
    IRFunctionPrologue,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
    IRNode,
)
from .model import (
    ASTBlock,
    ASTDominatorInfo,
    ASTFunction,
    ASTLoop,
    ASTProgram,
    ASTSwitchCase,
    ASTTerminator,
)


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


@dataclass
class _MutableBlock:
    label: str
    start_offset: int
    body: Tuple[IRNode, ...]
    terminator: ASTTerminator
    predecessors: MutableSet[str]
    successors: MutableSet[str]

    def to_ast_block(self) -> ASTBlock:
        return ASTBlock(
            label=self.label,
            start_offset=self.start_offset,
            body=self.body,
            terminator=self.terminator,
            predecessors=tuple(sorted(self.predecessors)),
            successors=tuple(sorted(self.successors)),
        )


class ASTBuilder:
    """Build an :class:`ASTProgram` from a normalised :class:`IRProgram`."""

    def build(self, program) -> ASTProgram:
        cfg, _ = analyse_segments(program.segments)
        block_lookup = self._build_block_lookup(program.segments)
        functions: List[ASTFunction] = []
        for function_cfg in cfg.functions:
            function = self._build_function(function_cfg, block_lookup)
            if function.blocks:
                functions.append(function)
        return ASTProgram(functions=tuple(functions))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_block_lookup(
        self, segments: Sequence[IRSegment]
    ) -> Dict[Tuple[int, str], IRBlock]:
        lookup: Dict[Tuple[int, str], IRBlock] = {}
        for segment in segments:
            for block in segment.blocks:
                lookup[(segment.index, block.label)] = block
        return lookup

    def _build_function(
        self,
        function_cfg: IRFunctionCfg,
        block_lookup: Mapping[Tuple[int, str], IRBlock],
    ) -> ASTFunction:
        segment_index = function_cfg.segment_index
        function_labels = {cfg_block.label for cfg_block in function_cfg.blocks}
        if not function_labels:
            return ASTFunction(
                segment_index=segment_index,
                name=function_cfg.name,
                entry=function_cfg.entry_block,
                blocks=tuple(),
                dominators=tuple(),
                post_dominators=tuple(),
                loops=tuple(),
            )

        offset_map = {
            block_lookup[(segment_index, label)].start_offset: label
            for label in function_labels
        }

        blocks: Dict[str, _MutableBlock] = {}
        for cfg_block in function_cfg.blocks:
            ir_block = block_lookup[(segment_index, cfg_block.label)]
            mutable = self._make_mutable_block(ir_block, offset_map)
            mutable.successors.update(edge.target for edge in cfg_block.edges)
            blocks[mutable.label] = mutable

        for block in blocks.values():
            block.predecessors.clear()
        for block in blocks.values():
            for succ in list(block.successors):
                if succ not in blocks:
                    block.successors.discard(succ)
                    continue
                blocks[succ].predecessors.add(block.label)

        entry = function_cfg.entry_block
        if entry not in blocks:
            return ASTFunction(
                segment_index=segment_index,
                name=function_cfg.name,
                entry=entry,
                blocks=tuple(),
                dominators=tuple(),
                post_dominators=tuple(),
                loops=tuple(),
            )

        self._remove_unreachable(blocks, entry)
        self._split_critical_edges(blocks)
        self._merge_degenerate_blocks(blocks, entry)
        self._fold_redundant_branches(blocks)

        dominators = self._compute_dominators(blocks, entry)
        post_dominators = self._compute_post_dominators(blocks)
        loops = self._identify_loops(blocks, dominators)

        ast_blocks = tuple(
            sorted(
                (block.to_ast_block() for block in blocks.values()),
                key=lambda item: (item.start_offset, item.label),
            )
        )

        dom_info = self._build_dom_info(dominators, entry)
        post_dom_info = self._build_post_dom_info(post_dominators, entry)

        return ASTFunction(
            segment_index=segment_index,
            name=function_cfg.name,
            entry=entry,
            blocks=ast_blocks,
            dominators=dom_info,
            post_dominators=post_dom_info,
            loops=loops,
        )

    def _make_mutable_block(
        self, block: IRBlock, offset_map: Mapping[int, str]
    ) -> _MutableBlock:
        nodes = list(block.nodes)
        terminator_node: Optional[IRNode] = None
        for index in range(len(nodes) - 1, -1, -1):
            if isinstance(nodes[index], _TERMINATOR_TYPES):
                terminator_node = nodes.pop(index)
                break
        body = tuple(nodes)
        terminator = self._build_terminator(terminator_node, offset_map)
        return _MutableBlock(
            label=block.label,
            start_offset=block.start_offset,
            body=body,
            terminator=terminator,
            predecessors=set(),
            successors=set(),
        )

    def _build_terminator(
        self, node: Optional[IRNode], offset_map: Mapping[int, str]
    ) -> ASTTerminator:
        if node is None:
            return ASTTerminator(kind="none")
        if isinstance(node, IRReturn):
            return ASTTerminator(kind="return", origin=node, description=node.describe())
        if isinstance(node, IRTailCall):
            return ASTTerminator(kind="tailcall", origin=node, description=node.describe())
        if isinstance(node, IRTailcallReturn):
            return ASTTerminator(
                kind="tailcall_return",
                origin=node,
                description=node.describe(),
            )
        if isinstance(node, IRCallReturn):
            then_target = None
            else_target = None
            branch_kind = None
            condition = None
            if node.predicate is not None:
                branch_kind = node.predicate.kind
                condition = node.predicate.describe()
                then_target = self._resolve_target(node.predicate.then_target, offset_map)
                else_target = self._resolve_target(node.predicate.else_target, offset_map)
            return ASTTerminator(
                kind="call_return",
                origin=node,
                description=node.describe(),
                then_target=then_target,
                else_target=else_target,
                branch_kind=branch_kind,
                condition=condition,
            )
        if isinstance(node, IRIf):
            return ASTTerminator(
                kind="branch",
                origin=node,
                branch_kind="if",
                condition=node.condition,
                then_target=self._resolve_target(node.then_target, offset_map),
                else_target=self._resolve_target(node.else_target, offset_map),
            )
        if isinstance(node, IRTestSetBranch):
            return ASTTerminator(
                kind="branch",
                origin=node,
                branch_kind="testset",
                condition=f"{node.var}={node.expr}",
                then_target=self._resolve_target(node.then_target, offset_map),
                else_target=self._resolve_target(node.else_target, offset_map),
            )
        if isinstance(node, IRFlagCheck):
            return ASTTerminator(
                kind="branch",
                origin=node,
                branch_kind="flag",
                condition=f"0x{node.flag:04X}",
                then_target=self._resolve_target(node.then_target, offset_map),
                else_target=self._resolve_target(node.else_target, offset_map),
            )
        if isinstance(node, IRFunctionPrologue):
            return ASTTerminator(
                kind="branch",
                origin=node,
                branch_kind="prologue",
                condition=f"{node.var}={node.expr}",
                then_target=self._resolve_target(node.then_target, offset_map),
                else_target=self._resolve_target(node.else_target, offset_map),
            )
        if isinstance(node, IRSwitchDispatch):
            cases = tuple(
                ASTSwitchCase(
                    key=str(case.key),
                    target=self._resolve_target(case.target, offset_map) or f"0x{case.target:04X}",
                )
                for case in node.cases
            )
            default_target = self._resolve_target(node.default, offset_map)
            return ASTTerminator(
                kind="switch",
                origin=node,
                description=node.describe(),
                cases=cases,
                default_target=default_target,
            )
        if isinstance(node, IRTerminator):
            return ASTTerminator(
                kind="terminator",
                origin=node,
                description=node.describe(),
            )
        return ASTTerminator(kind="unknown", origin=node)

    def _resolve_target(self, target: Optional[int], offset_map: Mapping[int, str]) -> Optional[str]:
        if target is None:
            return None
        return offset_map.get(target, f"0x{target:04X}")

    # ------------------------------------------------------------------
    # graph clean-up helpers
    # ------------------------------------------------------------------
    def _remove_unreachable(self, blocks: MutableMapping[str, _MutableBlock], entry: str) -> None:
        if entry not in blocks:
            return
        reachable: Set[str] = set()
        worklist = [entry]
        while worklist:
            label = worklist.pop()
            if label in reachable:
                continue
            reachable.add(label)
            worklist.extend(blocks[label].successors)
        for label in list(blocks.keys()):
            if label in reachable:
                continue
            for block in blocks.values():
                block.predecessors.discard(label)
                block.successors.discard(label)
            del blocks[label]

    def _split_critical_edges(self, blocks: MutableMapping[str, _MutableBlock]) -> None:
        counter = 0
        for block in list(blocks.values()):
            successors = list(block.successors)
            if len(successors) <= 1:
                continue
            for target in successors:
                if target not in blocks:
                    continue
                successor = blocks[target]
                if len(successor.predecessors) <= 1:
                    continue
                new_label = f"{block.label}_split_{counter}"
                counter += 1
                while new_label in blocks:
                    new_label = f"{block.label}_split_{counter}"
                    counter += 1
                splitter = _MutableBlock(
                    label=new_label,
                    start_offset=successor.start_offset,
                    body=tuple(),
                    terminator=ASTTerminator(
                        kind="jump",
                        target=target,
                        description=f"critical_edge {block.label}->{target}",
                    ),
                    predecessors={block.label},
                    successors={target},
                )
                blocks[new_label] = splitter
                block.successors.remove(target)
                block.successors.add(new_label)
                successor.predecessors.remove(block.label)
                successor.predecessors.add(new_label)
                block.terminator = self._redirect_terminator(
                    block.terminator, target, new_label
                )

    def _merge_degenerate_blocks(
        self, blocks: MutableMapping[str, _MutableBlock], entry: str
    ) -> None:
        changed = True
        while changed:
            changed = False
            for label, block in list(blocks.items()):
                if label == entry:
                    continue
                if block.body:
                    continue
                if block.terminator.kind != "jump":
                    continue
                if len(block.successors) != 1:
                    continue
                if len(block.predecessors) != 1:
                    continue
                target = next(iter(block.successors))
                if target not in blocks:
                    continue
                successor = blocks[target]
                if len(successor.predecessors) != 1:
                    continue
                predecessors = list(block.predecessors)
                for pred_label in predecessors:
                    if pred_label not in blocks:
                        continue
                    pred = blocks[pred_label]
                    if label in pred.successors:
                        pred.successors.remove(label)
                        pred.successors.add(target)
                    successor.predecessors.add(pred_label)
                successor.predecessors.discard(label)
                del blocks[label]
                changed = True
                break
        for block in blocks.values():
            block.predecessors.intersection_update(blocks.keys())
            block.successors.intersection_update(blocks.keys())

    def _fold_redundant_branches(self, blocks: MutableMapping[str, _MutableBlock]) -> None:
        for block in blocks.values():
            term = block.terminator
            if term.kind != "branch":
                continue
            current_successors = list(block.successors)
            for successor in current_successors:
                if successor in blocks:
                    blocks[successor].predecessors.discard(block.label)
            then_target = term.then_target
            else_target = term.else_target
            if then_target is not None and then_target == else_target:
                block.terminator = ASTTerminator(
                    kind="jump",
                    origin=term.origin,
                    description=term.description,
                    target=then_target,
                )
                new_successors = {then_target} if then_target is not None else set()
                block.successors = new_successors
            else:
                successors = {
                    target
                    for target in (then_target, else_target)
                    if target is not None and target in blocks
                }
                block.successors = successors
            for successor in block.successors:
                if successor in blocks:
                    blocks[successor].predecessors.add(block.label)

    # ------------------------------------------------------------------
    # dominator analysis
    # ------------------------------------------------------------------
    def _compute_dominators(
        self, blocks: Mapping[str, _MutableBlock], entry: str
    ) -> Mapping[str, Set[str]]:
        dom: Dict[str, Set[str]] = {
            label: set(blocks.keys()) for label in blocks
        }
        dom[entry] = {entry}
        changed = True
        while changed:
            changed = False
            for label, block in blocks.items():
                if label == entry:
                    continue
                preds = [blocks[pred] for pred in block.predecessors if pred in blocks]
                if not preds:
                    new_set = {label}
                else:
                    intersect = set(blocks.keys())
                    for pred in preds:
                        intersect &= dom[pred.label]
                    new_set = {label} | intersect
                if new_set != dom[label]:
                    dom[label] = new_set
                    changed = True
        return dom

    def _compute_post_dominators(
        self, blocks: Mapping[str, _MutableBlock]
    ) -> Mapping[str, Set[str]]:
        labels = set(blocks.keys())
        exits = [label for label, block in blocks.items() if not block.successors]
        if not exits:
            exits = list(labels)
        pdom: Dict[str, Set[str]] = {label: set(labels) for label in labels}
        for exit_label in exits:
            pdom[exit_label] = {exit_label}
        changed = True
        while changed:
            changed = False
            for label, block in blocks.items():
                if label in exits:
                    continue
                succs = [succ for succ in block.successors if succ in blocks]
                if not succs:
                    new_set = {label}
                else:
                    intersect = set(labels)
                    for succ in succs:
                        intersect &= pdom[succ]
                    new_set = {label} | intersect
                if new_set != pdom[label]:
                    pdom[label] = new_set
                    changed = True
        return pdom

    def _identify_loops(
        self,
        blocks: Mapping[str, _MutableBlock],
        dominators: Mapping[str, Set[str]],
    ) -> Tuple[ASTLoop, ...]:
        loops: Dict[str, Dict[str, Set[str]]] = {}
        for label, block in blocks.items():
            for succ in block.successors:
                if succ not in blocks:
                    continue
                if succ not in dominators.get(label, set()):
                    continue
                loop = loops.setdefault(
                    succ, {"nodes": set([succ]), "latches": set()}
                )
                loop["latches"].add(label)
                loop_nodes = self._collect_natural_loop(blocks, succ, label)
                loop["nodes"].update(loop_nodes)
        ast_loops = [
            ASTLoop(
                header=header,
                nodes=tuple(sorted(data["nodes"])),
                latches=tuple(sorted(data["latches"])),
            )
            for header, data in loops.items()
        ]
        ast_loops.sort(key=lambda loop: loop.header)
        return tuple(ast_loops)

    def _collect_natural_loop(
        self,
        blocks: Mapping[str, _MutableBlock],
        header: str,
        latch: str,
    ) -> Set[str]:
        nodes = {header, latch}
        worklist = [latch]
        while worklist:
            current = worklist.pop()
            for pred in blocks[current].predecessors:
                if pred == header or pred in nodes:
                    continue
                nodes.add(pred)
                worklist.append(pred)
        return nodes

    def _build_dom_info(
        self,
        dom: Mapping[str, Set[str]],
        entry: str,
    ) -> Tuple[ASTDominatorInfo, ...]:
        info: List[ASTDominatorInfo] = []
        idom = self._compute_immediate(dom, entry)
        for label, dominators in dom.items():
            info.append(
                ASTDominatorInfo(
                    block=label,
                    dominators=tuple(sorted(dominators)),
                    immediate=idom.get(label),
                )
            )
        info.sort(key=lambda item: item.block)
        return tuple(info)

    def _build_post_dom_info(
        self,
        pdom: Mapping[str, Set[str]],
        entry: str,
    ) -> Tuple[ASTDominatorInfo, ...]:
        # Entry is not special for post-dominators; reuse helper with virtual root
        info: List[ASTDominatorInfo] = []
        ipdom = self._compute_immediate(pdom, None)
        for label, postdoms in pdom.items():
            info.append(
                ASTDominatorInfo(
                    block=label,
                    dominators=tuple(sorted(postdoms)),
                    immediate=ipdom.get(label),
                )
            )
        info.sort(key=lambda item: item.block)
        return tuple(info)

    def _compute_immediate(
        self,
        dom: Mapping[str, Set[str]],
        entry: Optional[str],
    ) -> Mapping[str, Optional[str]]:
        result: Dict[str, Optional[str]] = {}
        for label in dom:
            if entry is not None and label == entry:
                result[label] = None
                continue
            candidates = dom[label] - {label}
            immediate: Optional[str] = None
            for candidate in candidates:
                others = candidates - {candidate}
                if all(candidate not in (dom[other] - {other}) for other in others):
                    if immediate is None:
                        immediate = candidate
                    elif len(dom[candidate]) > len(dom[immediate]):
                        immediate = candidate
            result[label] = immediate
        return result

    def _redirect_terminator(
        self, terminator: ASTTerminator, old: str, new: str
    ) -> ASTTerminator:
        if terminator.kind == "branch":
            if terminator.then_target == old:
                terminator = replace(terminator, then_target=new)
            if terminator.else_target == old:
                terminator = replace(terminator, else_target=new)
            return terminator
        if terminator.kind == "call_return":
            if terminator.then_target == old:
                terminator = replace(terminator, then_target=new)
            if terminator.else_target == old:
                terminator = replace(terminator, else_target=new)
            return terminator
        if terminator.kind == "switch":
            changed = False
            updated_cases = []
            for case in terminator.cases:
                if case.target == old:
                    updated_cases.append(ASTSwitchCase(case.key, new))
                    changed = True
                else:
                    updated_cases.append(case)
            default_target = terminator.default_target
            if default_target == old:
                default_target = new
                changed = True
            if changed:
                terminator = replace(
                    terminator,
                    cases=tuple(updated_cases),
                    default_target=default_target,
                )
            return terminator
        return terminator


__all__ = ["ASTBuilder"]
