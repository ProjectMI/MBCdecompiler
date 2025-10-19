"""Convert CFG flavoured procedures into canonical structured bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .model import (
    ASTBlock,
    ASTContinue,
    ASTIf,
    ASTProcedure,
    ASTStatement,
    ASTWhile,
    ASTAssign,
    ASTBranch,
    ASTFlagCheck,
    ASTFlagPredicate,
    ASTFunctionPrologue,
    ASTIdentifier,
    ASTExpression,
    ASTReturn,
    ASTTailCall,
    ASTTestSet,
    ASTUnaryNot,
    ASTUnknown,
)


@dataclass
class _BranchInfo:
    condition: ASTExpression
    then_target: Optional[ASTBlock]
    else_target: Optional[ASTBlock]


class ProcedureStructurer:
    """Derive canonical structured statements for a procedure."""

    def __init__(self, procedure: ASTProcedure):
        self._procedure = procedure
        self._blocks: List[ASTBlock] = list(procedure.blocks)
        self._block_ids: List[int] = [id(block) for block in self._blocks]
        self._id_to_block: Dict[int, ASTBlock] = {
            block_id: block for block_id, block in zip(self._block_ids, self._blocks)
        }
        self._block_order: Dict[int, int] = {
            block_id: index for index, block_id in enumerate(self._block_ids)
        }
        self._label_map: Dict[str, ASTBlock] = {block.label: block for block in self._blocks}
        self._offset_map: Dict[int, ASTBlock] = {block.start_offset: block for block in self._blocks}
        self._body_map: Dict[int, Tuple[ASTStatement, ...]] = {}
        self._branch_map: Dict[int, _BranchInfo] = {}
        self._predecessors: Dict[int, Set[int]] = {block_id: set() for block_id in self._block_ids}
        self._dominators: Dict[int, Set[int]] = {}
        self._postdominators: Dict[object, Set[object]] = {}
        self._exit_node: object = object()
        self._consumed: Set[int] = set()
        self._active_stack: List[int] = []
        self._analyse_blocks()
        self._compute_predecessors()
        self._compute_dominators()
        self._compute_postdominators()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self) -> Tuple[ASTStatement, ...]:
        if not self._blocks:
            return tuple()
        entry_block = self._resolve_entry()
        structured = self._emit(entry_block, self._exit_node)
        return tuple(structured)

    # ------------------------------------------------------------------
    # analysis helpers
    # ------------------------------------------------------------------
    def _analyse_blocks(self) -> None:
        for block in self._blocks:
            block_id = id(block)
            body, branch = self._split_block(block)
            self._body_map[block_id] = tuple(body)
            if branch is not None:
                self._branch_map[block_id] = branch

    def _compute_predecessors(self) -> None:
        for block in self._blocks:
            block_id = id(block)
            for successor in self._successor_ids(block_id):
                if isinstance(successor, int):
                    self._predecessors.setdefault(successor, set()).add(block_id)

    def _compute_dominators(self) -> None:
        if not self._block_ids:
            return
        entry_id = self._resolve_entry_id()
        dominators: Dict[int, Set[int]] = {
            block_id: set(self._block_ids) for block_id in self._block_ids
        }
        dominators[entry_id] = {entry_id}
        changed = True
        while changed:
            changed = False
            for block_id in self._block_ids:
                if block_id == entry_id:
                    continue
                preds = self._predecessors.get(block_id, set())
                if not preds:
                    new = {block_id}
                else:
                    iterator = iter(preds)
                    intersection = set(dominators[next(iterator)])
                    for pred in iterator:
                        intersection &= dominators[pred]
                    new = intersection | {block_id}
                if new != dominators[block_id]:
                    dominators[block_id] = new
                    changed = True
        self._dominators = dominators

    def _compute_postdominators(self) -> None:
        nodes: List[object] = list(self._block_ids) + [self._exit_node]
        postdom: Dict[object, Set[object]] = {node: set(nodes) for node in nodes}
        postdom[self._exit_node] = {self._exit_node}
        changed = True
        while changed:
            changed = False
            for block_id in self._block_ids:
                succs = list(self._successor_ids(block_id))
                if not succs:
                    candidates: Iterable[object] = [self._exit_node]
                else:
                    candidates = succs
                intersection: Set[object] = set(nodes)
                for candidate in candidates:
                    intersection &= postdom[candidate]
                new = intersection | {block_id}
                if new != postdom[block_id]:
                    postdom[block_id] = new
                    changed = True
        self._postdominators = postdom

    # ------------------------------------------------------------------
    # structuring
    # ------------------------------------------------------------------
    def _emit(self, block: Optional[ASTBlock], exit_marker: object) -> List[ASTStatement]:
        if block is None:
            return []
        return self._emit_from_id(id(block), exit_marker)

    def _emit_from_id(self, block_id: Optional[int], exit_marker: object) -> List[ASTStatement]:
        if block_id is None or block_id == exit_marker:
            return []
        if block_id not in self._id_to_block:
            return []
        if block_id in self._consumed:
            return []
        if block_id in self._active_stack:
            return [ASTContinue()]
        block = self._id_to_block[block_id]
        self._active_stack.append(block_id)
        statements: List[ASTStatement] = list(self._body_map.get(block_id, ()))
        branch = self._branch_map.get(block_id)
        if branch is None:
            self._consumed.add(block_id)
            self._active_stack.pop()
            tail = statements[-1] if statements else None
            if isinstance(tail, (ASTReturn, ASTTailCall)):
                return statements
            successors = list(self._successor_ids(block_id))
            if not successors:
                return statements
            continuation = successors[0]
            continuation_id = continuation if isinstance(continuation, int) else None
            statements.extend(self._emit_from_id(continuation_id, exit_marker))
            return statements

        then_target = branch.then_target
        else_target = branch.else_target
        then_id = id(then_target) if then_target is not None else None
        else_id = id(else_target) if else_target is not None else None
        if then_id is not None and then_id not in self._id_to_block:
            then_id = None
        if else_id is not None and else_id not in self._id_to_block:
            else_id = None
        join = self._immediate_postdom(block_id)
        loop_id = self._select_loop_target(block_id, then_id, else_id)
        if loop_id is not None:
            if loop_id == else_id:
                condition = ASTUnaryNot(branch.condition)
                exit_target = then_id if then_id is not None else join
            else:
                condition = branch.condition
                exit_target = else_id if else_id is not None else join
            loop_body = self._emit_from_id(loop_id, block_id)
            statements.append(ASTWhile(condition=condition, body=tuple(loop_body)))
            self._consumed.add(block_id)
            self._active_stack.pop()
            next_block = exit_target if isinstance(exit_target, int) else None
            statements.extend(self._emit_from_id(next_block, exit_marker))
            return statements

        join_id = join if isinstance(join, int) else None
        then_id = then_id if then_id is not None else join_id
        else_id = else_id if else_id is not None else join_id
        then_body = self._emit_from_id(then_id, join)
        else_body = self._emit_from_id(else_id, join) if else_id != join else []
        statements.append(
            ASTIf(
                condition=branch.condition,
                then_body=tuple(then_body),
                else_body=tuple(else_body),
            )
        )
        self._consumed.add(block_id)
        self._active_stack.pop()
        statements.extend(self._emit_from_id(join_id, exit_marker))
        return statements

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def _split_block(self, block: ASTBlock) -> Tuple[List[ASTStatement], Optional[_BranchInfo]]:
        body: List[ASTStatement] = []
        for statement in block.statements:
            if isinstance(statement, (ASTBranch, ASTTestSet, ASTFlagCheck, ASTFunctionPrologue)):
                prelude, branch = self._normalise_branch(block, statement)
                body.extend(prelude)
                return body, branch
            body.append(statement)
        return body, None

    def _normalise_branch(
        self, block: ASTBlock, statement: ASTStatement
    ) -> Tuple[List[ASTStatement], _BranchInfo]:
        prelude: List[ASTStatement] = []
        if isinstance(statement, ASTBranch):
            condition = statement.condition
            then_target = self._resolve_target(block, statement.then_branch, statement.then_hint, statement.then_offset)
            else_target = self._resolve_target(block, statement.else_branch, statement.else_hint, statement.else_offset)
        elif isinstance(statement, ASTTestSet):
            if isinstance(statement.var, ASTIdentifier):
                prelude.append(ASTAssign(statement.var, statement.expr))
                condition = statement.var
            else:
                condition = statement.expr
            then_target = self._resolve_target(block, statement.then_branch, statement.then_hint, statement.then_offset)
            else_target = self._resolve_target(block, statement.else_branch, statement.else_hint, statement.else_offset)
        elif isinstance(statement, ASTFlagCheck):
            condition = ASTFlagPredicate(statement.flag)
            then_target = self._resolve_target(block, statement.then_branch, statement.then_hint, statement.then_offset)
            else_target = self._resolve_target(block, statement.else_branch, statement.else_hint, statement.else_offset)
        elif isinstance(statement, ASTFunctionPrologue):
            if isinstance(statement.var, ASTIdentifier):
                prelude.append(ASTAssign(statement.var, statement.expr))
                condition = statement.var
            else:
                condition = statement.var
            then_target = self._resolve_target(block, statement.then_branch, statement.then_hint, statement.then_offset)
            else_target = self._resolve_target(block, statement.else_branch, statement.else_hint, statement.else_offset)
        else:
            condition = ASTUnknown("branch")
            then_target = self._next_block(block)
            else_target = None
        return prelude, _BranchInfo(condition, then_target, else_target)

    def _resolve_target(
        self,
        block: ASTBlock,
        target: Optional[ASTBlock],
        hint: Optional[str],
        offset: Optional[int],
    ) -> Optional[ASTBlock]:
        if target is not None:
            return target
        if hint == "fallthrough":
            return self._next_block(block)
        if hint and hint in self._label_map:
            return self._label_map[hint]
        if offset is not None and offset in self._offset_map:
            return self._offset_map[offset]
        return None

    def _next_block(self, block: ASTBlock) -> Optional[ASTBlock]:
        index = self._block_order.get(id(block))
        if index is None:
            return None
        next_index = index + 1
        if next_index >= len(self._blocks):
            return None
        return self._blocks[next_index]

    def _successor_ids(self, block_id: int) -> Tuple[object, ...]:
        branch = self._branch_map.get(block_id)
        block = self._id_to_block[block_id]
        if branch is not None:
            successors: List[object] = []
            for target in (branch.then_target, branch.else_target):
                next_block = target or self._next_block(block)
                if next_block is not None:
                    block_id = id(next_block)
                    if block_id in self._id_to_block:
                        successors.append(block_id)
            if not successors:
                fallback = self._next_block(block)
                if fallback is not None:
                    block_id = id(fallback)
                    if block_id in self._id_to_block:
                        successors.append(block_id)
            return tuple(dict.fromkeys(successors))
        if block.successors:
            return tuple(
                id(successor)
                for successor in block.successors
                if id(successor) in self._id_to_block
            )
        next_block = self._next_block(block)
        if next_block is None or id(next_block) not in self._id_to_block:
            return tuple()
        return (id(next_block),)

    def _resolve_entry(self) -> ASTBlock:
        entry_offset = self._procedure.entry_offset
        if entry_offset in self._offset_map:
            return self._offset_map[entry_offset]
        return self._blocks[0]

    def _resolve_entry_id(self) -> int:
        return id(self._resolve_entry())

    def _immediate_postdom(self, block_id: int) -> object:
        candidates = self._postdominators.get(block_id, set()) - {block_id}
        if not candidates:
            return self._exit_node
        filtered = set(candidates)
        for candidate in list(candidates):
            for other in candidates:
                if other is candidate:
                    continue
                if candidate in self._postdominators.get(other, set()):
                    filtered.discard(candidate)
                    break
        if not filtered:
            return self._exit_node
        return min(filtered, key=lambda node: self._block_order.get(node, float("inf")))

    def _select_loop_target(
        self, block_id: int, then_id: Optional[int], else_id: Optional[int]
    ) -> Optional[int]:
        for candidate in (then_id, else_id):
            if candidate is None:
                continue
            if block_id in self._dominators.get(candidate, set()):
                return candidate
        return None
