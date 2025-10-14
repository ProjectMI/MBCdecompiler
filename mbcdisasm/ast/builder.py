"""Reconstruct an AST and CFG from the normalised IR programme."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

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
    IRBankedLoad,
    IRBankedStore,
    IRIndirectLoad,
    IRIndirectStore,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRStore,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
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
    ASTFunctionPrologue,
    ASTIORead,
    ASTIOWrite,
    ASTIdentifier,
    ASTIndirectLoadExpr,
    ASTBankedLoadExpr,
    ASTBankedRefExpr,
    ASTLiteral,
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


@dataclass
class _BlockAnalysis:
    """Cached information describing a block within a segment."""

    block: IRBlock
    successors: Tuple[int, ...]
    exit_reasons: Tuple[str, ...]
    fallthrough: Optional[int]


BranchStatement = ASTBranch | ASTTestSet | ASTFlagCheck | ASTFunctionPrologue


@dataclass
class _BranchLink:
    """Pending control-flow link for a branch-like statement."""

    statement: BranchStatement
    then_target: int
    else_target: int
    origin_offset: int


@dataclass
class _PendingBlock:
    """Block with unresolved successor references."""

    label: str
    start_offset: int
    statements: List[ASTStatement]
    successors: Tuple[int, ...]
    branch_links: List[_BranchLink]


class ASTBuilder:
    """Construct a high level AST with CFG and reconstruction metrics."""

    def __init__(self) -> None:
        self._current_analyses: Mapping[int, _BlockAnalysis] = {}
        self._current_entry_reasons: Mapping[int, Tuple[str, ...]] = {}
        self._current_block_labels: Mapping[int, str] = {}
        self._current_exit_hints: Mapping[int, str] = {}

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
        self._current_analyses = analyses
        self._current_entry_reasons = entry_reasons
        self._current_block_labels = {offset: analysis.block.label for offset, analysis in analyses.items()}
        self._current_exit_hints = {
            offset: self._format_exit_hint(analysis.exit_reasons)
            for offset, analysis in analyses.items()
            if analysis.exit_reasons
        }
        procedures = self._group_procedures(segment, analyses, entry_reasons, metrics)
        result = ASTSegment(
            index=segment.index,
            start=segment.start,
            length=segment.length,
            procedures=tuple(procedures),
        )
        self._clear_context()
        return result

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
            fallthrough = offsets[idx + 1] if idx + 1 < len(offsets) else None
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
            if not successors and fallthrough is not None:
                successors.add(fallthrough)
            analyses[block.start_offset] = _BlockAnalysis(
                block=block,
                successors=tuple(sorted(successors)),
                exit_reasons=tuple(exit_reasons),
                fallthrough=fallthrough,
            )
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
        current_blocks: List[_PendingBlock] = []
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
                    self._compress_blocks(
                        current_blocks,
                        current_entry or current_blocks[0].start_offset,
                    )
                    procedures.append(
                        self._finalise_procedure(
                            name=f"proc_{len(procedures)}",
                            entry_offset=current_entry or current_blocks[0].start_offset,
                            entry_reasons=current_reasons,
                            blocks=current_blocks,
                            exit_offsets=exit_offsets,
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
            self._compress_blocks(
                current_blocks, current_entry or current_blocks[0].start_offset
            )
            procedures.append(
                self._finalise_procedure(
                    name=f"proc_{len(procedures)}",
                    entry_offset=current_entry or current_blocks[0].start_offset,
                    entry_reasons=current_reasons,
                    blocks=current_blocks,
                    exit_offsets=exit_offsets,
                )
            )
        return procedures

    def _finalise_procedure(
        self,
        name: str,
        entry_offset: int,
        entry_reasons: Tuple[str, ...],
        blocks: Sequence[_PendingBlock],
        exit_offsets: Set[int],
    ) -> ASTProcedure:
        realised_blocks = self._realise_blocks(blocks)
        return ASTProcedure(
            name=name,
            entry_offset=entry_offset,
            entry_reasons=entry_reasons,
            blocks=realised_blocks,
            exit_offsets=tuple(sorted(exit_offsets)),
        )

    def _compress_blocks(self, blocks: List[_PendingBlock], entry_offset: int) -> None:
        """Merge trivial bridge blocks back into their successors."""

        redirect: Dict[int, int] = {}

        while True:
            updates: Dict[int, int] = {}
            for block in blocks:
                if block.start_offset == entry_offset:
                    continue
                if block.statements or block.branch_links:
                    continue
                if len(block.successors) != 1:
                    continue
                successor = self._resolve_redirect(block.successors[0], redirect)
                if successor == block.start_offset:
                    continue
                updates[block.start_offset] = successor

            if not updates:
                break

            redirect.update(updates)

            def resolve(target: int) -> int:
                return self._resolve_redirect(target, redirect)

            for block in blocks:
                block.successors = tuple(resolve(succ) for succ in block.successors)
                for link in block.branch_links:
                    link.then_target = resolve(link.then_target)
                    link.else_target = resolve(link.else_target)

            blocks[:] = [block for block in blocks if block.start_offset not in updates]

    @staticmethod
    def _resolve_redirect(offset: int, redirects: Mapping[int, int]) -> int:
        while offset in redirects:
            offset = redirects[offset]
        return offset

    def _realise_blocks(self, blocks: Sequence[_PendingBlock]) -> Tuple[ASTBlock, ...]:
        block_map: Dict[int, ASTBlock] = {
            block.start_offset: ASTBlock(
                label=block.label,
                start_offset=block.start_offset,
                statements=tuple(),
                successors=tuple(),
            )
            for block in blocks
        }
        for pending in blocks:
            for link in pending.branch_links:
                then_block = block_map.get(link.then_target)
                if then_block is not None:
                    link.statement.then_branch = then_block
                else:
                    link.statement.then_hint = self._describe_branch_target(
                        link.origin_offset, link.then_target
                    )
                else_block = block_map.get(link.else_target)
                if else_block is not None:
                    link.statement.else_branch = else_block
                else:
                    link.statement.else_hint = self._describe_branch_target(
                        link.origin_offset, link.else_target
                    )
            realised = block_map[pending.start_offset]
            realised.statements = tuple(pending.statements)
            realised.successors = tuple(
                block_map[target]
                for target in pending.successors
                if target in block_map
            )
        return tuple(block_map[block.start_offset] for block in blocks)

    def _convert_block(
        self,
        analysis: _BlockAnalysis,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> _PendingBlock:
        block = analysis.block
        statements: List[ASTStatement] = []
        branch_links: List[_BranchLink] = []
        for node in block.nodes:
            node_statements, node_links = self._convert_node(
                node,
                block.start_offset,
                value_state,
                metrics,
            )
            statements.extend(node_statements)
            branch_links.extend(node_links)
        return _PendingBlock(
            label=block.label,
            start_offset=block.start_offset,
            statements=statements,
            successors=analysis.successors,
            branch_links=branch_links,
        )

    def _convert_node(
        self,
        node,
        origin_offset: int,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[List[ASTStatement], List[_BranchLink]]:
        if isinstance(node, IRLoad):
            target = ASTIdentifier(node.target, self._infer_kind(node.target))
            expr = ASTSlotRef(node.slot)
            value_state[node.target] = expr
            metrics.observe_values(int(not isinstance(expr, ASTUnknown)))
            metrics.observe_load(True)
            return [ASTAssign(target=target, value=expr)], []
        if isinstance(node, IRStore):
            target_expr = ASTSlotRef(node.slot)
            value_expr = self._resolve_expr(node.value, value_state)
            metrics.observe_store(not isinstance(value_expr, ASTUnknown))
            return [ASTStore(target=target_expr, value=value_expr)], []
        if isinstance(node, IRIORead):
            return [ASTIORead(port=node.port)], []
        if isinstance(node, IRIOWrite):
            return [ASTIOWrite(port=node.port, mask=node.mask)], []
        if isinstance(node, IRBankedLoad):
            pointer_expr = (
                self._resolve_expr(node.pointer, value_state) if node.pointer else None
            )
            offset_expr = (
                self._resolve_expr(node.offset_source, value_state)
                if node.offset_source
                else None
            )
            expr = ASTBankedLoadExpr(
                ref=node.ref,
                register=node.register,
                register_value=node.register_value,
                pointer=pointer_expr,
                offset=offset_expr,
            )
            value_state[node.target] = expr
            metrics.observe_values(int(not isinstance(expr, ASTUnknown)))
            pointer_known = pointer_expr is None or not isinstance(pointer_expr, ASTUnknown)
            offset_known = offset_expr is None or not isinstance(offset_expr, ASTUnknown)
            metrics.observe_load(pointer_known and offset_known)
            return [
                ASTAssign(
                    target=ASTIdentifier(node.target, self._infer_kind(node.target)),
                    value=expr,
                )
            ], []
        if isinstance(node, IRBankedStore):
            pointer_expr = (
                self._resolve_expr(node.pointer, value_state) if node.pointer else None
            )
            offset_expr = (
                self._resolve_expr(node.offset_source, value_state)
                if node.offset_source
                else None
            )
            value_expr = self._resolve_expr(node.value, value_state)
            pointer_known = pointer_expr is None or not isinstance(pointer_expr, ASTUnknown)
            offset_known = offset_expr is None or not isinstance(offset_expr, ASTUnknown)
            value_known = not isinstance(value_expr, ASTUnknown)
            metrics.observe_store(pointer_known and offset_known and value_known)
            target = ASTBankedRefExpr(
                ref=node.ref,
                register=node.register,
                register_value=node.register_value,
                pointer=pointer_expr,
                offset=offset_expr,
            )
            return [ASTStore(target=target, value=value_expr)], []
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
            ], []
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
            return [ASTStore(target=target, value=value_expr)], []
        if isinstance(node, IRCall):
            call_expr, _ = self._convert_call(
                node.target,
                node.args,
                node.symbol,
                node.tail,
                node.varargs if hasattr(node, "varargs") else False,
                value_state,
            )
            metrics.call_sites += 1
            metrics.observe_call_args(
                sum(1 for arg in call_expr.args if not isinstance(arg, ASTUnknown)),
                len(call_expr.args),
            )
            return [ASTCallStatement(call=call_expr)], []
        if isinstance(node, IRCallReturn):
            call_expr, returns = self._convert_call(
                node.target,
                node.args,
                node.symbol,
                node.tail,
                node.varargs,
                value_state,
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
            return statements, []
        if isinstance(node, IRTailCall):
            call_expr, returns = self._convert_call(
                node.call.target,
                node.call.args,
                node.call.symbol,
                True,
                node.varargs,
                value_state,
            )
            metrics.call_sites += 1
            metrics.observe_call_args(
                sum(1 for arg in call_expr.args if not isinstance(arg, ASTUnknown)),
                len(call_expr.args),
            )
            resolved_returns = tuple(self._resolve_expr(name, value_state) for name in node.returns)
            return [ASTTailCall(call=call_expr, returns=resolved_returns)], []
        if isinstance(node, IRReturn):
            values = tuple(self._resolve_expr(name, value_state) for name in node.values)
            return [ASTReturn(values=values, varargs=node.varargs)], []
        if isinstance(node, IRCallCleanup):
            return [], []
        if isinstance(node, IRTerminator):
            return [], []
        if isinstance(node, IRIf):
            condition = self._resolve_expr(node.condition, value_state)
            branch = ASTBranch(condition=condition)
            return [branch], [
                _BranchLink(
                    statement=branch,
                    then_target=node.then_target,
                    else_target=node.else_target,
                    origin_offset=origin_offset,
                )
            ]
        if isinstance(node, IRTestSetBranch):
            var_expr = self._resolve_expr(node.var, value_state)
            expr = self._resolve_expr(node.expr, value_state)
            statement = ASTTestSet(var=var_expr, expr=expr)
            return [statement], [
                _BranchLink(
                    statement=statement,
                    then_target=node.then_target,
                    else_target=node.else_target,
                    origin_offset=origin_offset,
                )
            ]
        if isinstance(node, IRFunctionPrologue):
            var_expr = self._resolve_expr(node.var, value_state)
            expr = self._resolve_expr(node.expr, value_state)
            statement = ASTFunctionPrologue(var=var_expr, expr=expr)
            return [statement], [
                _BranchLink(
                    statement=statement,
                    then_target=node.then_target,
                    else_target=node.else_target,
                    origin_offset=origin_offset,
                )
            ]
        if isinstance(node, IRFlagCheck):
            statement = ASTFlagCheck(flag=node.flag)
            return [statement], [
                _BranchLink(
                    statement=statement,
                    then_target=node.then_target,
                    else_target=node.else_target,
                    origin_offset=origin_offset,
                )
            ]
        return [ASTComment(getattr(node, "describe", lambda: repr(node))())], []

    def _convert_call(
        self,
        target: int,
        args: Sequence[str],
        symbol: Optional[str],
        tail: bool,
        varargs: bool,
        value_state: Mapping[str, ASTExpression],
    ) -> Tuple[ASTCallExpr, Tuple[ASTExpression, ...]]:
        arg_exprs = tuple(self._resolve_expr(arg, value_state) for arg in args)
        call_expr = ASTCallExpr(target=target, args=arg_exprs, symbol=symbol, tail=tail, varargs=varargs)
        return call_expr, arg_exprs

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

    def _describe_branch_target(self, origin_offset: int, target_offset: int) -> str:
        if target_offset in self._current_block_labels:
            return self._current_block_labels[target_offset]
        origin_analysis = self._current_analyses.get(origin_offset)
        if origin_analysis and origin_analysis.fallthrough == target_offset:
            return "fallthrough"
        exit_hint = self._current_exit_hints.get(target_offset)
        if exit_hint:
            return exit_hint
        entry_reasons = self._current_entry_reasons.get(target_offset)
        if entry_reasons:
            joined = ",".join(entry_reasons) or "unspecified"
            return f"entry({joined})"
        return "fallthrough"

    def _format_exit_hint(self, exit_reasons: Tuple[str, ...]) -> str:
        if not exit_reasons:
            return ""
        mapping = {"return": "return", "tail_call": "tail_call"}
        return "|".join(mapping.get(reason, reason) for reason in exit_reasons)

    @staticmethod
    def _is_hex_literal(value: str) -> bool:
        try:
            if value.lower().startswith("0x"):
                int(value, 16)
                return True
        except ValueError:
            return False
        return False

    def _clear_context(self) -> None:
        self._current_analyses = {}
        self._current_entry_reasons = {}
        self._current_block_labels = {}
        self._current_exit_hints = {}


__all__ = ["ASTBuilder"]
