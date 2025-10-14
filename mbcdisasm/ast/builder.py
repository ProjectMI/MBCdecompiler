"""Reconstruct an AST and CFG from the normalised IR programme."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..constants import CALL_SHUFFLE_STANDARD
from ..ir.model import (
    IRBlock,
    IRCall,
    IRCallCleanup,
    IRCallReturn,
    IRFunctionPrologue,
    IRIORead,
    IRIOWrite,
    IRIf,
    IRFlagCheck,
    IRIndirectLoad,
    IRIndirectStore,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRStackEffect,
    IRSwitchDispatch,
    IRStore,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    SSAValueKind,
)
from .model import (
    ASTAssign,
    ASTBlock,
    ASTBranch,
    ASTCallExpr,
    ASTCallResult,
    ASTCallStatement,
    ASTComment,
    ASTExpression,
    ASTFlagCheck,
    ASTFrameFinalize,
    ASTFunctionPrologue,
    ASTIORead,
    ASTIOWrite,
    ASTIdentifier,
    ASTIndirectLoadExpr,
    ASTLiteral,
    ASTSwitch,
    ASTSwitchCase,
    ASTMetrics,
    ASTProcedure,
    ASTProgram,
    ASTReturn,
    ASTSegment,
    ASTSlotRef,
    ASTStatement,
    ASTStore,
    ASTTailCall,
    ASTTestSet,
    ASTUnknown,
)


_TAIL_HELPER_ALIASES = {
    "tail_helper_72": "ui.flush",
    "tail_helper_f0": "ui.flush",
    "tail_helper_ed": "ui.flush",
    "tail_helper_13d": "page.set",
    "tail_helper_1ec": "ui.flush",
    "tail_helper_1f1": "ui.flush",
    "tail_helper_32c": "switch.dispatch",
    "tail_helper_3e": "switch.dispatch",
    "tail_helper_bf0": "ui.flush",
    "tail_helper_ff0": "ui.flush",
    "tail_helper_16f0": "ui.flush",
}

_STACK_SHUFFLE_VARIANTS = {CALL_SHUFFLE_STANDARD, 0x3032, 0x7223}


@dataclass
class _BlockAnalysis:
    """Cached information describing a block within a segment."""

    block: IRBlock
    successors: Tuple[int, ...]
    exit_reasons: Tuple[str, ...]


class ASTBuilder:
    """Construct a high level AST with CFG and reconstruction metrics."""

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self, program: IRProgram) -> ASTProgram:
        segments: List[ASTSegment] = []
        metrics = ASTMetrics()
        for segment in program.segments:
            segment_result = self._build_segment(segment, metrics)
            segments.append(segment_result)
        metrics.procedure_count = sum(len(seg.procedures) for seg in segments)
        metrics.block_count = sum(len(proc.blocks) for seg in segments for proc in seg.procedures)
        metrics.edge_count = sum(len(block.successors) for seg in segments for proc in seg.procedures for block in proc.blocks)
        return ASTProgram(segments=tuple(segments), metrics=metrics)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_segment(self, segment: IRSegment, metrics: ASTMetrics) -> ASTSegment:
        block_map: Dict[int, IRBlock] = {block.start_offset: block for block in segment.blocks}
        analyses = self._build_cfg(segment, block_map)
        entry_reasons = self._detect_entries(segment, block_map, analyses)
        procedures = self._group_procedures(segment, analyses, entry_reasons, metrics)
        return ASTSegment(index=segment.index, start=segment.start, length=segment.length, procedures=tuple(procedures))

    def _build_cfg(
        self,
        segment: IRSegment,
        block_map: Mapping[int, IRBlock],
    ) -> Mapping[int, _BlockAnalysis]:
        analyses: Dict[int, _BlockAnalysis] = {}
        offsets = [block.start_offset for block in segment.blocks]
        for idx, block in enumerate(segment.blocks):
            successors: Set[int] = set()
            exit_reasons: List[str] = []
            for node in reversed(block.nodes):
                if isinstance(node, IRReturn):
                    exit_reasons.append("return")
                    successors.clear()
                    break
                if isinstance(node, (IRTailCall, IRTailcallReturn)):
                    exit_reasons.append("tail_call")
                    successors.clear()
                    break
                if isinstance(node, IRIf):
                    successors.update({node.then_target, node.else_target})
                    break
                if isinstance(node, IRTestSetBranch):
                    successors.update({node.then_target, node.else_target})
                    break
                if isinstance(node, IRFunctionPrologue):
                    successors.update({node.then_target, node.else_target})
                    break
            if not successors and idx + 1 < len(offsets):
                successors.add(offsets[idx + 1])
            analyses[block.start_offset] = _BlockAnalysis(block=block, successors=tuple(sorted(successors)), exit_reasons=tuple(exit_reasons))
        return analyses

    def _detect_entries(
        self,
        segment: IRSegment,
        block_map: Mapping[int, IRBlock],
        analyses: Mapping[int, _BlockAnalysis],
    ) -> Mapping[int, Tuple[str, ...]]:
        entry_reasons: Dict[int, Set[str]] = defaultdict(set)
        entry_reasons[segment.start].add("segment_start")
        for offset, analysis in analyses.items():
            block = analysis.block
            for node in block.nodes:
                if isinstance(node, IRFunctionPrologue):
                    entry_reasons[offset].add("prologue")
                if isinstance(node, (IRCall, IRCallReturn, IRTailCall)):
                    target = node.target
                    if target in block_map:
                        reason = "tail_target" if getattr(node, "tail", False) else "call_target"
                        entry_reasons[target].add(reason)
        return {offset: tuple(sorted(reasons)) for offset, reasons in entry_reasons.items()}

    def _group_procedures(
        self,
        segment: IRSegment,
        analyses: Mapping[int, _BlockAnalysis],
        entry_reasons: Mapping[int, Tuple[str, ...]],
        metrics: ASTMetrics,
    ) -> Sequence[ASTProcedure]:
        procedures: List[ASTProcedure] = []
        current_blocks: List[ASTBlock] = []
        current_entry: Optional[int] = None
        current_reasons: Tuple[str, ...] = tuple()
        exit_offsets: Set[int] = set()
        value_state: Dict[str, ASTExpression] = {}
        sorted_offsets = sorted(analyses)
        for offset in sorted_offsets:
            analysis = analyses[offset]
            block = analysis.block
            if offset in entry_reasons:
                if current_blocks:
                    procedures.append(
                        ASTProcedure(
                            name=f"proc_{len(procedures)}",
                            entry_offset=current_entry or current_blocks[0].start_offset,
                            entry_reasons=current_reasons,
                            blocks=tuple(current_blocks),
                            exit_offsets=tuple(sorted(exit_offsets)),
                        )
                    )
                    current_blocks = []
                    exit_offsets = set()
                    value_state = {}
                current_entry = offset
                current_reasons = entry_reasons[offset]
            ast_block = self._convert_block(analysis, value_state, metrics)
            current_blocks.append(ast_block)
            if analysis.exit_reasons:
                exit_offsets.add(offset)
        if current_blocks:
            procedures.append(
                ASTProcedure(
                    name=f"proc_{len(procedures)}",
                    entry_offset=current_entry or current_blocks[0].start_offset,
                    entry_reasons=current_reasons,
                    blocks=tuple(current_blocks),
                    exit_offsets=tuple(sorted(exit_offsets)),
                )
            )
        return procedures

    def _convert_block(
        self,
        analysis: _BlockAnalysis,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> ASTBlock:
        block = analysis.block
        statements: List[ASTStatement] = []
        for node in block.nodes:
            statements.extend(self._convert_node(node, value_state, metrics))
        return ASTBlock(
            label=block.label,
            start_offset=block.start_offset,
            statements=tuple(statements),
            successors=analysis.successors,
        )

    def _convert_node(
        self,
        node,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Iterable[ASTStatement]:
        if isinstance(node, IRLoad):
            target = ASTIdentifier(node.target, self._infer_kind(node.target))
            expr = ASTSlotRef(node.slot)
            value_state[node.target] = expr
            metrics.observe_values(int(not isinstance(expr, ASTUnknown)))
            metrics.observe_load(True)
            return [ASTAssign(target=target, value=expr)]
        if isinstance(node, IRStore):
            target_expr = ASTSlotRef(node.slot)
            value_expr = self._resolve_expr(node.value, value_state)
            metrics.observe_store(not isinstance(value_expr, ASTUnknown))
            return [ASTStore(target=target_expr, value=value_expr)]
        if isinstance(node, IRIORead):
            return [ASTIORead(port=node.port)]
        if isinstance(node, IRIOWrite):
            return [ASTIOWrite(port=node.port, mask=node.mask)]
        if isinstance(node, IRIndirectLoad):
            pointer = self._resolve_expr(node.pointer or node.base, value_state)
            offset_expr = (
                self._resolve_expr(node.offset_source, value_state)
                if node.offset_source
                else ASTLiteral(node.offset)
            )
            expr = ASTIndirectLoadExpr(pointer=pointer, offset=offset_expr, ref=node.ref)
            value_state[node.target] = expr
            metrics.observe_values(int(not isinstance(expr, ASTUnknown)))
            metrics.observe_load(not isinstance(pointer, ASTUnknown) and not isinstance(offset_expr, ASTUnknown))
            return [
                ASTAssign(
                    target=ASTIdentifier(node.target, self._infer_kind(node.target)),
                    value=expr,
                )
            ]
        if isinstance(node, IRIndirectStore):
            pointer = self._resolve_expr(node.pointer or node.base, value_state)
            offset_expr = (
                self._resolve_expr(node.offset_source, value_state)
                if node.offset_source
                else ASTLiteral(node.offset)
            )
            value_expr = self._resolve_expr(node.value, value_state)
            metrics.observe_store(
                not any(isinstance(expr, ASTUnknown) for expr in (pointer, offset_expr, value_expr))
            )
            target = ASTIndirectLoadExpr(pointer=pointer, offset=offset_expr, ref=node.ref)
            return [ASTStore(target=target, value=value_expr)]
        if isinstance(node, IRCall):
            call_expr, _ = self._convert_call(
                node.target,
                node.args,
                node.symbol,
                node.tail,
                node.varargs if hasattr(node, "varargs") else False,
                value_state,
                node.convention,
            )
            metrics.call_sites += 1
            metrics.observe_call_args(
                sum(1 for arg in call_expr.args if not isinstance(arg, ASTUnknown)),
                len(call_expr.args),
            )
            return [ASTCallStatement(call=call_expr)]
        if isinstance(node, IRCallReturn):
            call_expr, returns = self._convert_call(
                node.target,
                node.args,
                node.symbol,
                node.tail,
                node.varargs,
                value_state,
                node.convention,
            )
            statements: List[ASTStatement] = []
            metrics.call_sites += 1
            metrics.observe_call_args(
                sum(1 for arg in call_expr.args if not isinstance(arg, ASTUnknown)),
                len(call_expr.args),
            )
            return_identifiers = []
            for index, name in enumerate(node.returns):
                identifier = ASTIdentifier(name, self._infer_kind(name))
                value_state[name] = ASTCallResult(call_expr, index)
                metrics.observe_values(int(not isinstance(value_state[name], ASTUnknown)))
                return_identifiers.append(identifier)
            statements.append(ASTCallStatement(call=call_expr, returns=tuple(return_identifiers)))
            return statements
        if isinstance(node, IRTailCall):
            call_expr, returns = self._convert_call(
                node.call.target,
                node.call.args,
                node.call.symbol,
                True,
                node.varargs,
                value_state,
                node.call.convention,
            )
            metrics.call_sites += 1
            metrics.observe_call_args(
                sum(1 for arg in call_expr.args if not isinstance(arg, ASTUnknown)),
                len(call_expr.args),
            )
            resolved_returns = tuple(self._resolve_expr(name, value_state) for name in node.returns)
            return [ASTTailCall(call=call_expr, returns=resolved_returns)]
        if isinstance(node, IRReturn):
            statements: List[ASTStatement] = []
            statements.extend(self._frame_finalize(node.cleanup))
            values = tuple(self._resolve_expr(name, value_state) for name in node.values)
            statements.append(ASTReturn(values=values, varargs=node.varargs, mask=node.mask))
            return statements
        if isinstance(node, IRCallCleanup):
            return self._frame_finalize(node.steps)
        if isinstance(node, IRIf):
            condition = self._resolve_expr(node.condition, value_state)
            return [ASTBranch(condition=condition, then_target=node.then_target, else_target=node.else_target)]
        if isinstance(node, IRTestSetBranch):
            var_expr = self._resolve_expr(node.var, value_state)
            expr = self._resolve_expr(node.expr, value_state)
            return [
                ASTTestSet(
                    var=var_expr,
                    expr=expr,
                    then_target=node.then_target,
                    else_target=node.else_target,
                )
            ]
        if isinstance(node, IRFunctionPrologue):
            var_expr = self._resolve_expr(node.var, value_state)
            expr = self._resolve_expr(node.expr, value_state)
            return [
                ASTFunctionPrologue(
                    var=var_expr,
                    expr=expr,
                    then_target=node.then_target,
                    else_target=node.else_target,
                )
            ]
        if isinstance(node, IRFlagCheck):
            return [
                ASTFlagCheck(
                    flag=node.flag,
                    then_target=node.then_target,
                    else_target=node.else_target,
                )
            ]
        if isinstance(node, IRSwitchDispatch):
            cases = tuple(
                ASTSwitchCase(key=case.key, target=case.target, symbol=case.symbol)
                for case in node.cases
            )
            return [
                ASTSwitch(
                    helper=node.helper,
                    helper_symbol=node.helper_symbol,
                    cases=cases,
                    default=node.default,
                )
            ]
        return [ASTComment(getattr(node, "describe", lambda: repr(node))())]

    def _convert_call(
        self,
        target: int,
        args: Sequence[str],
        symbol: Optional[str],
        tail: bool,
        varargs: bool,
        value_state: Mapping[str, ASTExpression],
        convention: Optional[IRStackEffect],
    ) -> Tuple[ASTCallExpr, Tuple[ASTExpression, ...]]:
        ordered_args = self._normalize_call_args(args, convention)
        arg_exprs = tuple(self._resolve_expr(arg, value_state) for arg in ordered_args)
        alias = _TAIL_HELPER_ALIASES.get(symbol or "")
        call_expr = ASTCallExpr(target=target, args=arg_exprs, symbol=alias or symbol, tail=tail, varargs=varargs)
        return call_expr, arg_exprs

    def _normalize_call_args(
        self, args: Sequence[str], convention: Optional[IRStackEffect]
    ) -> Tuple[str, ...]:
        if not args:
            return tuple()
        if convention is None or convention.mnemonic != "stack_shuffle":
            return tuple(args)
        operand = convention.operand or 0
        reordered = list(args)
        if operand in _STACK_SHUFFLE_VARIANTS and len(reordered) >= 2:
            reordered[0], reordered[1] = reordered[1], reordered[0]
        return tuple(reordered)

    def _resolve_expr(self, token: Optional[str], value_state: Mapping[str, ASTExpression]) -> ASTExpression:
        if not token:
            return ASTUnknown("")
        if token in value_state:
            return value_state[token]
        if token.startswith("lit(") and token.endswith(")"):
            literal = token[4:-1]
            try:
                value = int(literal, 16)
            except ValueError:
                return ASTUnknown(token)
            return ASTLiteral(value)
        if token.startswith("slot(") and token.endswith(")"):
            try:
                index = int(token[5:-1], 16)
            except ValueError:
                return ASTUnknown(token)
            return ASTIdentifier(f"slot_{index:04X}", SSAValueKind.POINTER)
        return ASTIdentifier(token, self._infer_kind(token))

    def _infer_kind(self, name: str) -> SSAValueKind:
        lowered = name.lower()
        if lowered.startswith("bool"):
            return SSAValueKind.BOOLEAN
        if lowered.startswith("word"):
            return SSAValueKind.WORD
        if lowered.startswith("byte"):
            return SSAValueKind.BYTE
        if lowered.startswith("ptr"):
            return SSAValueKind.POINTER
        if lowered.startswith("page"):
            return SSAValueKind.PAGE_REGISTER
        if lowered.startswith("io"):
            return SSAValueKind.IO
        if lowered.startswith("id"):
            return SSAValueKind.IDENTIFIER
        return SSAValueKind.UNKNOWN

    def _frame_finalize(self, steps: Sequence[IRStackEffect]) -> List[ASTFrameFinalize]:
        pops = sum(step.pops for step in steps)
        notes: List[str] = []
        for step in steps:
            if step.mnemonic in {"stack_teardown", "call_helpers", "cleanup_call"}:
                continue
            alias = step.operand_alias
            if alias:
                notes.append(str(alias))
                continue
            if step.operand_role and step.operand:
                notes.append(f"{step.operand_role}=0x{step.operand:04X}")
                continue
            if step.operand:
                notes.append(f"0x{step.operand:04X}")
                continue
            notes.append(step.mnemonic)
        if pops == 0 and not notes:
            return []
        return [ASTFrameFinalize(pops=pops, notes=tuple(notes))]


__all__ = ["ASTBuilder"]
