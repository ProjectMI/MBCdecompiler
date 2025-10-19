"""Convert CFG oriented AST blocks into structured control flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .model import (
    ASTBlock,
    ASTBranch,
    ASTBreak,
    ASTContinue,
    ASTIf,
    ASTReturn,
    ASTStatement,
    ASTSwitch,
    ASTTailCall,
    ASTWhile,
)


@dataclass(frozen=True)
class LoopInfo:
    """Summary of a natural loop discovered in the CFG."""

    header: int
    nodes: Set[int]
    exits: Set[int]
    entry_targets: Tuple[int, ...]


@dataclass(frozen=True)
class RegionContext:
    """Contextual information while structuring a region."""

    exit_target: Optional[int]
    loop_header: Optional[int] = None
    loop_exit: Optional[int] = None

    def with_exit(self, exit_target: Optional[int]) -> "RegionContext":
        return RegionContext(exit_target=exit_target, loop_header=self.loop_header, loop_exit=self.loop_exit)


class ControlFlowStructuralizer:
    """Convert basic-block CFG into canonical structured statements."""

    def __init__(self, blocks: Sequence[ASTBlock]) -> None:
        self._blocks = {block.start_offset: block for block in blocks}
        self._successors: Dict[int, List[int]] = {}
        self._predecessors: Dict[int, Set[int]] = {}
        self._postdominators: Dict[int, Set[int]] = {}
        self._immediate_postdom: Dict[int, Optional[int]] = {}
        self._loops: Dict[int, LoopInfo] = {}
        self._exit_node = object()
        self._build_graph()
        self._compute_postdominators()
        self._identify_loops()

    def build(self, entry_offset: int) -> Tuple[ASTStatement, ...]:
        if entry_offset not in self._blocks:
            return tuple()
        context = RegionContext(exit_target=None)
        statements = self._build_region(entry_offset, None, context)
        return tuple(statements)

    # ------------------------------------------------------------------
    # graph preparation
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        for offset, block in self._blocks.items():
            successors: List[int] = []
            for successor in block.successors or ():
                successors.append(successor.start_offset)
            self._successors[offset] = successors
            for target in successors:
                self._predecessors.setdefault(target, set()).add(offset)
        for offset in self._blocks:
            self._predecessors.setdefault(offset, set())

    def _compute_postdominators(self) -> None:
        nodes: Set[int] = set(self._blocks.keys())
        postdom: Dict[int, Set[int | object]] = {offset: set(nodes) | {offset} for offset in nodes}
        postdom[self._exit_node] = {self._exit_node}

        def successors(node: int) -> Set[int | object]:
            succ = self._successors.get(node)
            if not succ:
                return {self._exit_node}
            return set(succ)

        changed = True
        while changed:
            changed = False
            for node in nodes:
                succ = successors(node)
                intersection: Set[int | object]
                if not succ:
                    intersection = {self._exit_node}
                else:
                    intersection = set(nodes) | {self._exit_node}
                    for candidate in succ:
                        intersection &= postdom[candidate]
                updated = {node} | intersection
                if updated != postdom[node]:
                    postdom[node] = updated
                    changed = True

        immediate: Dict[int, Optional[int]] = {}
        for node in nodes:
            candidates = postdom[node] - {node}
            immediate_candidate: Optional[int] = None
            for candidate in candidates:
                if candidate is self._exit_node:
                    continue
                dominated = False
                for other in candidates:
                    if other == candidate:
                        continue
                    if candidate in postdom[other]:
                        dominated = True
                        break
                if not dominated:
                    immediate_candidate = candidate  # type: ignore[assignment]
                    break
            immediate[node] = immediate_candidate
            self._postdominators[node] = {
                entry for entry in postdom[node] if entry is not self._exit_node
            }
        self._immediate_postdom = immediate

    def _identify_loops(self) -> None:
        dominators = self._compute_dominators()
        back_edges: List[Tuple[int, int]] = []
        for source, successors in self._successors.items():
            for target in successors:
                if target in dominators.get(source, set()):
                    back_edges.append((source, target))
        loops: Dict[int, Set[int]] = {}
        for source, header in back_edges:
            body = self._collect_natural_loop(source, header)
            if header not in loops:
                loops[header] = body
            else:
                loops[header] |= body
        for header, nodes in loops.items():
            exits: Set[int] = set()
            for node in nodes:
                for succ in self._successors.get(node, []):
                    if succ not in nodes:
                        exits.add(succ)
            entry_targets = tuple(
                target
                for target in self._successors.get(header, [])
                if target in nodes and target != header
            )
            self._loops[header] = LoopInfo(
                header=header,
                nodes=nodes,
                exits=exits,
                entry_targets=entry_targets,
            )

    def _compute_dominators(self) -> Dict[int, Set[int]]:
        if not self._blocks:
            return {}
        entry = min(self._blocks.keys())
        nodes = set(self._blocks.keys())
        dominators: Dict[int, Set[int]] = {entry: {entry}}
        others = nodes - {entry}
        for node in others:
            dominators[node] = set(nodes)

        changed = True
        while changed:
            changed = False
            for node in others:
                preds = self._predecessors.get(node)
                if not preds:
                    continue
                intersection = set(nodes)
                for pred in preds:
                    intersection &= dominators[pred]
                updated = {node} | intersection
                if updated != dominators[node]:
                    dominators[node] = updated
                    changed = True
        return dominators

    def _collect_natural_loop(self, source: int, header: int) -> Set[int]:
        loop_nodes: Set[int] = {header}
        stack: List[int] = [source]
        while stack:
            node = stack.pop()
            if node in loop_nodes:
                continue
            loop_nodes.add(node)
            for pred in self._predecessors.get(node, set()):
                stack.append(pred)
        return loop_nodes

    # ------------------------------------------------------------------
    # region construction
    # ------------------------------------------------------------------

    def _build_region(
        self,
        start: int,
        exit_target: Optional[int],
        context: RegionContext,
    ) -> List[ASTStatement]:
        statements: List[ASTStatement] = []
        current = start
        visited: Set[int] = set()
        while current is not None and current != exit_target:
            if current in visited:
                break
            visited.add(current)
            block = self._blocks.get(current)
            if block is None:
                break
            prefix, terminator = self._split_block(block)
            statements.extend(prefix)
            if terminator is None:
                fallthrough = self._fallthrough(block)
                if fallthrough is None or fallthrough == current:
                    break
                current = fallthrough
                continue
            if isinstance(terminator, (ASTReturn, ASTTailCall)):
                statements.append(terminator)
                break
            if isinstance(terminator, ASTSwitch):
                statements.append(terminator)
                break
            if isinstance(terminator, ASTBranch):
                if current in self._loops:
                    loop_stmt, next_offset = self._build_loop(current, terminator, context)
                    statements.append(loop_stmt)
                    if next_offset is None or next_offset == current:
                        break
                    current = next_offset
                    continue
                if_stmt, next_offset = self._build_if(current, terminator, exit_target, context)
                statements.append(if_stmt)
                if next_offset is None or next_offset == current:
                    break
                current = next_offset
                continue
            fallthrough = self._fallthrough(block)
            if fallthrough is None or fallthrough == current:
                break
            current = fallthrough
        return statements

    def _split_block(self, block: ASTBlock) -> Tuple[List[ASTStatement], Optional[ASTStatement]]:
        if not block.statements:
            return [], None
        statements = list(block.statements)
        terminator = statements[-1]
        if isinstance(terminator, ASTBranch):
            return statements[:-1], terminator
        if isinstance(terminator, (ASTReturn, ASTTailCall, ASTSwitch)):
            return statements[:-1], terminator
        return statements, None

    def _fallthrough(self, block: ASTBlock) -> Optional[int]:
        if not block.successors:
            return None
        return block.successors[0].start_offset

    def _build_loop(
        self,
        header: int,
        branch: ASTBranch,
        outer_context: RegionContext,
    ) -> Tuple[ASTWhile, Optional[int]]:
        loop_info = self._loops[header]
        post_exit = self._immediate_postdom.get(header)
        if post_exit is not None and post_exit not in loop_info.exits:
            loop_exit = post_exit
        elif loop_info.exits:
            loop_exit = min(loop_info.exits)
        else:
            loop_exit = outer_context.exit_target
        entry_targets = loop_info.entry_targets or tuple(
            target for target in self._successors.get(header, []) if target in loop_info.nodes and target != header
        )
        body_entry = entry_targets[0] if entry_targets else header
        body_context = RegionContext(exit_target=header, loop_header=header, loop_exit=loop_exit)
        body_statements = []
        if body_entry != header:
            body_statements = self._build_region(body_entry, header, body_context)
        loop_stmt = ASTWhile(condition=branch.condition, body=tuple(body_statements))
        return loop_stmt, loop_exit

    def _build_if(
        self,
        origin: int,
        branch: ASTBranch,
        exit_target: Optional[int],
        context: RegionContext,
    ) -> Tuple[ASTIf, Optional[int]]:
        then_target, else_target = self._branch_targets(branch)
        join = self._immediate_postdom.get(origin)
        loop_nodes: Set[int] = set()
        if context.loop_header is not None and context.loop_header in self._loops:
            loop_nodes = self._loops[context.loop_header].nodes
        then_body = self._build_branch_body(then_target, join, exit_target, context, loop_nodes)
        else_body = self._build_branch_body(else_target, join, exit_target, context, loop_nodes)
        if_stmt = ASTIf(
            condition=branch.condition,
            then_body=tuple(then_body),
            else_body=tuple(else_body),
        )
        return if_stmt, join

    def _build_branch_body(
        self,
        target: Optional[int],
        join: Optional[int],
        exit_target: Optional[int],
        context: RegionContext,
        loop_nodes: Set[int],
    ) -> List[ASTStatement]:
        if target is None:
            return []
        if join is not None and target == join:
            return []
        if context.loop_header is not None:
            if target == context.loop_header:
                return [ASTContinue()]
            if target not in loop_nodes:
                return [ASTBreak()]
        next_exit = join if join is not None else exit_target
        branch_context = context.with_exit(next_exit)
        return self._build_region(target, next_exit, branch_context)

    def _branch_targets(self, branch: ASTBranch) -> Tuple[Optional[int], Optional[int]]:
        then_target = branch.then_branch.start_offset if branch.then_branch else branch.then_offset
        else_target = branch.else_branch.start_offset if branch.else_branch else branch.else_offset
        return then_target, else_target


__all__ = ["ControlFlowStructuralizer"]
