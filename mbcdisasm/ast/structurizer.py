"""Canonical structural normalisation for AST procedures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .model import (
    ASTBlock,
    ASTBlockStatement,
    ASTBreak,
    ASTContinue,
    ASTExpression,
    ASTIfStatement,
    ASTIntegerLiteral,
    ASTStatement,
    ASTUnknown,
    ASTWhileStatement,
    ASTBranch,
    ASTTestSet,
    ASTFlagCheck,
    ASTFunctionPrologue,
    ASTReturn,
    ASTTailCall,
)


BranchLike = ASTBranch | ASTTestSet | ASTFlagCheck | ASTFunctionPrologue
_BRANCH_TYPES = (ASTBranch, ASTTestSet, ASTFlagCheck, ASTFunctionPrologue)


@dataclass(frozen=True)
class _LoopContext:
    header: int
    exit: Optional[int]


class ASTStructurizer:
    """Build structured statement trees from block based procedures."""

    def __init__(self, blocks: Sequence[ASTBlock]):
        self._block_map: Dict[int, ASTBlock] = {block.start_offset: block for block in blocks}
        self._successors: Dict[int, Tuple[int, ...]] = {
            offset: tuple(succ.start_offset for succ in block.successors or tuple())
            for offset, block in self._block_map.items()
        }
        self._ipdom = self._compute_immediate_postdominators(blocks)
        self._visiting: Set[int] = set()

    def build(self, entry_offset: int) -> ASTBlockStatement:
        return self._build_region(entry_offset, None, None)

    # ------------------------------------------------------------------
    # structural reconstruction
    # ------------------------------------------------------------------

    def _build_region(
        self,
        start: Optional[int],
        stop: Optional[int],
        loop_ctx: Optional[_LoopContext],
    ) -> ASTBlockStatement:
        if start is None:
            return ASTBlockStatement(tuple())
        statements: List[ASTStatement] = []
        offset = start
        local_visiting: Set[int] = set()
        while offset is not None and offset != stop:
            if offset in self._visiting:
                break
            block = self._block_map.get(offset)
            if block is None:
                break
            self._visiting.add(offset)
            local_visiting.add(offset)
            branch = self._extract_branch(block.statements)
            for stmt in block.statements:
                if stmt is branch:
                    break
                statements.append(stmt)
            if branch is not None:
                then_target, else_target = self._branch_targets(branch)
                join = self._ipdom.get(offset)
                if join is None:
                    join = else_target
                if loop_ctx and then_target == loop_ctx.header and else_target == loop_ctx.exit:
                    statements.append(
                        ASTIfStatement(
                            condition=self._branch_condition(branch),
                            then_block=ASTBlockStatement((ASTContinue(),)),
                            else_block=ASTBlockStatement((ASTBreak(),))
                            if loop_ctx.exit is not None
                            else None,
                        )
                    )
                    break
                if (
                    then_target is not None
                    and then_target != offset
                    and self._path_returns_to_header(then_target, offset, join)
                ):
                    body = self._build_region(
                        then_target,
                        offset,
                        _LoopContext(header=offset, exit=join),
                    )
                    statements.append(
                        ASTWhileStatement(
                            condition=self._branch_condition(branch),
                            body=body,
                        )
                    )
                    offset = join
                    continue
                then_block = self._build_region(then_target, join, loop_ctx)
                else_block = (
                    self._build_region(else_target, join, loop_ctx)
                    if else_target is not None and else_target != join
                    else None
                )
                statements.append(
                    ASTIfStatement(
                        condition=self._branch_condition(branch),
                        then_block=then_block,
                        else_block=else_block,
                    )
                )
                offset = join
                continue
            if self._block_has_explicit_exit(block):
                break
            next_offset = self._pick_fallthrough(offset, stop)
            if next_offset is None:
                break
            offset = next_offset
        for recorded in local_visiting:
            self._visiting.discard(recorded)
        return ASTBlockStatement(tuple(statements))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_branch(statements: Sequence[ASTStatement]) -> Optional[BranchLike]:
        for stmt in reversed(statements):
            if isinstance(stmt, _BRANCH_TYPES):
                return stmt
        return None

    @staticmethod
    def _branch_targets(statement: BranchLike) -> Tuple[Optional[int], Optional[int]]:
        then_target = getattr(statement, "then_offset", None)
        else_target = getattr(statement, "else_offset", None)
        if getattr(statement, "then_branch", None) is not None:
            then_target = statement.then_branch.start_offset
        if getattr(statement, "else_branch", None) is not None:
            else_target = statement.else_branch.start_offset
        return then_target, else_target

    @staticmethod
    def _branch_condition(statement: BranchLike) -> ASTExpression:
        if isinstance(statement, ASTBranch):
            return statement.condition
        if isinstance(statement, ASTTestSet):
            return statement.var
        if isinstance(statement, ASTFunctionPrologue):
            return statement.var
        if isinstance(statement, ASTFlagCheck):
            return ASTIntegerLiteral(value=statement.flag, bits=16)
        return ASTUnknown("branch")

    def _path_returns_to_header(
        self, start: Optional[int], header: int, stop: Optional[int]
    ) -> bool:
        if start is None:
            return False
        visited: Set[int] = set()
        stack = [start]
        while stack:
            offset = stack.pop()
            if offset == header:
                return True
            if offset == stop or offset in visited:
                continue
            visited.add(offset)
            for successor in self._successors.get(offset, tuple()):
                stack.append(successor)
        return False

    def _pick_fallthrough(
        self, offset: int, stop: Optional[int]
    ) -> Optional[int]:
        for candidate in self._successors.get(offset, tuple()):
            if candidate == stop:
                return candidate
            if candidate not in self._visiting:
                return candidate
        return None

    @staticmethod
    def _block_has_explicit_exit(block: ASTBlock) -> bool:
        for statement in block.statements:
            if isinstance(statement, (ASTReturn, ASTTailCall)):
                return True
        return False

    def _compute_immediate_postdominators(
        self, blocks: Sequence[ASTBlock]
    ) -> Dict[int, Optional[int]]:
        if not blocks:
            return {}
        exit_marker = object()
        offsets = {block.start_offset for block in blocks}
        postdom: Dict[int | object, Set[int | object]] = {
            offset: set(offsets) | {exit_marker} for offset in offsets
        }
        postdom[exit_marker] = {exit_marker}
        successors: Dict[int, Set[int | object]] = {}
        for block in blocks:
            succ: Set[int | object] = set()
            if block.successors:
                succ.update(succ_block.start_offset for succ_block in block.successors)
            if not succ or self._block_has_explicit_exit(block):
                succ.add(exit_marker)
            successors[block.start_offset] = succ or {exit_marker}
        changed = True
        while changed:
            changed = False
            for offset in offsets:
                succ = successors.get(offset, {exit_marker})
                intersection = set(offsets) | {exit_marker}
                for candidate in succ:
                    intersection &= postdom.get(candidate, {exit_marker})
                updated = {offset} | intersection
                if updated != postdom[offset]:
                    postdom[offset] = updated
                    changed = True
        ipdom: Dict[int, Optional[int]] = {}
        for offset in offsets:
            candidates = postdom[offset] - {offset}
            if not candidates:
                continue
            immediate: Optional[int | object] = None
            for candidate in candidates:
                dominated = False
                for other in candidates:
                    if other == candidate:
                        continue
                    if candidate in postdom.get(other, set()):
                        dominated = True
                        break
                if not dominated:
                    immediate = candidate
                    break
            if immediate is exit_marker:
                ipdom[offset] = None
            elif isinstance(immediate, int):
                ipdom[offset] = immediate
        return ipdom
