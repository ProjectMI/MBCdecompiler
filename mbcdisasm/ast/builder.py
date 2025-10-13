"""Conversion from normalised IR into a lightweight AST."""

from __future__ import annotations

from typing import List, Optional

from ..ir.model import (
    IRAsciiFinalize,
    IRAsciiHeader,
    IRAsciiPreamble,
    IRBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    IRCallCleanup,
    IRCallPreparation,
    IRCallReturn,
    IRConditionMask,
    IRFlagCheck,
    IRFunctionPrologue,
    IRIf,
    IRIndirectLoad,
    IRIndirectStore,
    IRIORead,
    IRIOWrite,
    IRLiteral,
    IRLiteralBlock,
    IRLiteralChunk,
    IRLoad,
    IRNode,
    IRPageRegister,
    IRProgram,
    IRReturn,
    IRSegment,
    IRStackDrop,
    IRStackDuplicate,
    IRStore,
    IRSwitchDispatch,
    IRTailCall,
    IRTailcallFrame,
    IRTailcallReturn,
    IRTableBuilderBegin,
    IRTableBuilderCommit,
    IRTableBuilderEmit,
    IRTablePatch,
    IRTestSetBranch,
    IRTerminator,
)
from .cfg import CFGBuilder, CFGSegment
from .model import (
    ASTAssignment,
    ASTBlock,
    ASTCallExpression,
    ASTCallStatement,
    ASTConditional,
    ASTExpression,
    ASTIdentifier,
    ASTIndirectReference,
    ASTIndirectStore,
    ASTLiteral,
    ASTProgram,
    ASTFunction,
    ASTExpressionStatement,
    ASTReturn,
    ASTSlotReference,
    ASTStore,
    ASTSwitch,
    ASTRawStatement,
)


class ASTBuilder:
    """Lift :class:`IRProgram` objects into :class:`ASTProgram` trees."""

    def __init__(self) -> None:
        self._cfg_builder = CFGBuilder()

    def build(self, program: IRProgram) -> ASTProgram:
        cfg_program = self._cfg_builder.build(program)
        cfg_by_index = {segment.segment.index: segment for segment in cfg_program.segments}
        functions = [
            self._build_function(segment, cfg_by_index.get(segment.index))
            for segment in program.segments
        ]
        return ASTProgram(tuple(functions))

    # ------------------------------------------------------------------
    # function and block handling
    # ------------------------------------------------------------------
    def _build_function(
        self, segment: IRSegment, cfg_segment: Optional[CFGSegment] = None
    ) -> ASTFunction:
        name = f"segment_{segment.index:04d}"
        blocks = tuple(self._lift_block(block) for block in segment.blocks)
        attributes: List[str] = []
        if cfg_segment is not None:
            edge_count = sum(len(node.successors) for node in cfg_segment.nodes.values())
            attributes.append(f"blocks={len(cfg_segment.nodes)}")
            attributes.append(f"edges={edge_count}")
        return ASTFunction(
            name=name,
            entry_offset=segment.start,
            blocks=blocks,
            attributes=tuple(attributes),
        )

    def _lift_block(self, block: IRBlock) -> ASTBlock:
        lifted: List = []
        for node in block.nodes:
            stmt = self._lift_node(node)
            if stmt is None:
                continue
            if isinstance(stmt, list):
                lifted.extend(stmt)
            else:
                lifted.append(stmt)
        return ASTBlock(label=block.label, start_offset=block.start_offset, statements=tuple(lifted))

    # ------------------------------------------------------------------
    # individual node lifting
    # ------------------------------------------------------------------
    def _lift_node(self, node: IRNode):
        if isinstance(node, IRLiteral):
            return ASTExpressionStatement(ASTLiteral(node.value))
        if isinstance(node, IRLoad):
            target = ASTIdentifier(node.target)
            destination = ASTSlotReference(node.slot.space.name.lower(), node.slot.index)
            return ASTAssignment(target=target, value=destination)
        if isinstance(node, IRStore):
            dest = ASTSlotReference(node.slot.space.name.lower(), node.slot.index)
            value = self._as_expression(node.value)
            return ASTStore(destination=dest, value=value)
        if isinstance(node, IRIndirectLoad):
            target = ASTIdentifier(node.target)
            ind_ref = ASTIndirectReference(
                base=node.base,
                offset=node.offset,
                pointer=node.pointer,
                ref=node.ref.describe() if node.ref is not None else None,
                offset_source=node.offset_source,
            )
            return ASTAssignment(target=target, value=ind_ref)
        if isinstance(node, IRIndirectStore):
            destination = ASTIndirectReference(
                base=node.base,
                offset=node.offset,
                pointer=node.pointer,
                ref=node.ref.describe() if node.ref is not None else None,
                offset_source=node.offset_source,
            )
            value = self._as_expression(node.value)
            return ASTIndirectStore(destination=destination, value=value)
        if isinstance(node, IRIORead):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRIOWrite):
            return ASTRawStatement(node.describe())
        if isinstance(node, (IRStackDrop, IRStackDuplicate, IRPageRegister)):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRCall):
            return self._lift_call(node)
        if isinstance(node, IRTailCall):
            call_expr = ASTCallExpression(
                target=node.target,
                args=tuple(self._as_expression(arg) for arg in node.args),
                symbol=node.symbol,
            )
            cleanup = tuple(effect.describe() for effect in node.cleanup)
            return ASTCallStatement(
                call=call_expr,
                tail=True,
                returns=node.returns,
                cleanup=cleanup,
                predicate=node.predicate.describe() if node.predicate else None,
            )
        if isinstance(node, IRCallReturn):
            call_expr = ASTCallExpression(
                target=node.target,
                args=tuple(self._as_expression(arg) for arg in node.args),
                symbol=node.symbol,
            )
            cleanup = tuple(effect.describe() for effect in node.cleanup)
            return [
                ASTCallStatement(call=call_expr, cleanup=cleanup),
                ASTReturn(
                    values=tuple(self._as_expression(value) for value in node.returns),
                    varargs=node.varargs,
                ),
            ]
        if isinstance(node, IRTailcallReturn):
            call_expr = ASTCallExpression(
                target=node.target,
                args=tuple(self._as_expression(arg) for arg in node.args),
                symbol=node.symbol,
            )
            cleanup = tuple(effect.describe() for effect in node.cleanup)
            return ASTCallStatement(
                call=call_expr,
                tail=True,
                returns=tuple(f"ret{i}" for i in range(node.returns)),
                cleanup=cleanup,
                predicate=node.predicate.describe() if node.predicate else None,
            )
        if isinstance(node, IRReturn):
            cleanup = tuple(effect.describe() for effect in node.cleanup)
            return ASTReturn(
                values=tuple(self._as_expression(value) for value in node.values),
                varargs=node.varargs,
                cleanup=cleanup,
                mask=node.mask,
            )
        if isinstance(node, IRIf):
            return ASTConditional(node.condition, node.then_target, node.else_target, kind="if")
        if isinstance(node, IRTestSetBranch):
            condition = f"{node.var}={node.expr}"
            return ASTConditional(condition, node.then_target, node.else_target, kind="testset")
        if isinstance(node, IRFunctionPrologue):
            condition = f"{node.var}={node.expr}"
            return ASTConditional(condition, node.then_target, node.else_target, kind="prologue")
        if isinstance(node, IRFlagCheck):
            condition = f"flag(0x{node.flag:04X})"
            return ASTConditional(condition, node.then_target, node.else_target, kind="flag")
        if isinstance(node, IRSwitchDispatch):
            cases = tuple((case.key, case.symbol) for case in node.cases)
            subject = node.helper_symbol or (f"0x{node.helper:04X}" if node.helper is not None else "dispatch")
            return ASTSwitch(subject=subject, cases=cases, default=node.default)
        if isinstance(node, IRTableBuilderBegin):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRTableBuilderEmit):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRTableBuilderCommit):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRTablePatch):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRLiteralChunk):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRLiteralBlock):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRBuildArray):
            rendered = node.describe()
            return ASTRawStatement(rendered)
        if isinstance(node, IRBuildMap):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRBuildTuple):
            return ASTRawStatement(node.describe())
        if isinstance(node, (IRCallPreparation, IRCallCleanup, IRTailcallFrame, IRConditionMask, IRAsciiHeader, IRAsciiPreamble, IRAsciiFinalize)):
            return ASTRawStatement(node.describe())
        if isinstance(node, IRTerminator):
            return ASTRawStatement(node.describe())
        return ASTRawStatement(getattr(node, "describe", lambda: repr(node))())

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _lift_call(self, node: IRCall) -> ASTCallStatement:
        call_expr = ASTCallExpression(
            target=node.target,
            args=tuple(self._as_expression(arg) for arg in node.args),
            symbol=node.symbol,
        )
        cleanup = tuple(effect.describe() for effect in node.cleanup)
        predicate = node.predicate.describe() if node.predicate else None
        return ASTCallStatement(
            call=call_expr,
            tail=node.tail,
            returns=tuple(),
            cleanup=cleanup,
            predicate=predicate,
        )

    def _as_expression(self, value: str) -> ASTExpression:
        if isinstance(value, str) and value.startswith("0x"):
            try:
                return ASTLiteral(int(value, 16))
            except ValueError:
                pass
        if isinstance(value, str) and value.isdigit():
            return ASTLiteral(int(value))
        return ASTIdentifier(str(value))


__all__ = ["ASTBuilder"]
