"""Build reconstructed AST structures from the normalised IR."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..ir.model import (
    IRBlock,
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
    IRLiteral,
    IRLoad,
    IRPageRegister,
    IRProgram,
    IRRaw,
    IRReturn,
    IRSegment,
    IRStore,
    IRSwitchDispatch,
    IRTableBuilderBegin,
    IRTableBuilderCommit,
    IRTableBuilderEmit,
    IRTablePatch,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRTailcallFrame,
    IRTerminator,
    IRStackDuplicate,
    IRStackDrop,
    IRIORead,
    IRIOWrite,
    IRAsciiHeader,
    IRAsciiFinalize,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRLiteralBlock,
    IRStringConstant,
    IRSlot,
    SSAValueKind,
)
from .model import (
    ASTBlock,
    ASTExpression,
    ASTMetrics,
    ASTProcedure,
    ASTProgram,
    ASTSegment,
    ASTStatement,
    CFGNode,
    ControlFlowGraph,
)

_ARITHMETIC_MNEMONICS = {
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "and",
    "or",
    "xor",
    "shl",
    "shr",
}

_REFERENCE_PREFIX_KIND = {
    "byte": SSAValueKind.BYTE,
    "word": SSAValueKind.WORD,
    "ptr": SSAValueKind.POINTER,
    "bool": SSAValueKind.BOOLEAN,
    "page": SSAValueKind.PAGE_REGISTER,
    "io": SSAValueKind.IO,
    "id": SSAValueKind.IDENTIFIER,
}


class ASTBuilder:
    """Drive AST reconstruction from a normalised IR program."""

    def build(self, program: IRProgram) -> ASTProgram:
        cfg = self._build_cfg(program)
        metrics = ASTMetrics()
        ast_blocks: Dict[str, ASTBlock] = {}
        for segment in program.segments:
            for block in segment.blocks:
                ast_blocks[block.label] = self._build_block(block, metrics)
        segments = self._assemble_segments(program, cfg, ast_blocks, metrics)
        metrics.blocks = sum(len(segment.blocks) for segment in segments)
        metrics.procedures = sum(len(segment.procedures) for segment in segments)
        return ASTProgram(segments=segments, cfg=cfg, metrics=metrics)

    # ------------------------------------------------------------------
    # CFG construction
    # ------------------------------------------------------------------
    def _build_cfg(self, program: IRProgram) -> ControlFlowGraph:
        nodes: Dict[str, CFGNode] = {}
        offset_map: Dict[int, str] = {}
        pending_successors: Dict[str, List[int]] = {}

        for segment in program.segments:
            blocks = segment.blocks
            for index, block in enumerate(blocks):
                offset_map[block.start_offset] = block.label
                next_block = blocks[index + 1] if index + 1 < len(blocks) else None
                pending_successors[block.label] = list(self._block_successors(block, next_block))
                nodes[block.label] = CFGNode(
                    label=block.label,
                    block=block,
                    segment_index=segment.index,
                )

        # resolve successors now that offsets are known
        for label, node in list(nodes.items()):
            resolved: List[str] = []
            for offset in pending_successors.get(label, []):
                successor = offset_map.get(offset)
                if successor is not None:
                    resolved.append(successor)
            nodes[label] = replace(node, successors=tuple(sorted(set(resolved))))

        incoming: Dict[str, Set[str]] = defaultdict(set)
        for node in nodes.values():
            for successor in node.successors:
                incoming[successor].add(node.label)
        for label, preds in incoming.items():
            node = nodes.get(label)
            if node is None:
                continue
            nodes[label] = replace(node, predecessors=tuple(sorted(preds)))

        return ControlFlowGraph(nodes=nodes, offsets=offset_map)

    def _block_successors(self, block: IRBlock, next_block: Optional[IRBlock]) -> Sequence[int]:
        targets: List[int] = []
        terminal = False
        for node in reversed(block.nodes):
            if isinstance(node, (IRReturn, IRTailCall, IRTailcallReturn, IRTerminator)):
                terminal = True
                break
            if isinstance(node, IRCallReturn) and node.tail:
                terminal = True
                break
            if isinstance(node, (IRIf, IRTestSetBranch, IRFlagCheck, IRFunctionPrologue)):
                targets.extend([node.then_target, node.else_target])
                terminal = True
                break
            if isinstance(node, IRSwitchDispatch):
                targets.extend(case.target for case in node.cases)
                if node.fallback is not None:
                    targets.append(node.fallback)
                terminal = True
                break
            if isinstance(node, IRCall) and getattr(node, "tail", False):
                terminal = True
                break
        if not terminal and next_block is not None:
            targets.append(next_block.start_offset)
        return targets

    # ------------------------------------------------------------------
    # AST block reconstruction
    # ------------------------------------------------------------------
    def _build_block(self, block: IRBlock, metrics: ASTMetrics) -> ASTBlock:
        env: Dict[str, ASTExpression] = {}
        statements: List[ASTStatement] = []

        for node in block.nodes:
            handler = self._node_handler(node)
            if handler is not None:
                produced = handler(node, env, metrics)
                if produced:
                    if isinstance(produced, ASTStatement):
                        statements.append(produced)
                        metrics.statements += 1
                    else:
                        for stmt in produced:
                            statements.append(stmt)
                            metrics.statements += 1
        return ASTBlock(label=block.label, start_offset=block.start_offset, statements=tuple(statements))

    def _node_handler(self, node):  # type: ignore[override]
        if isinstance(node, IRLiteral):
            return self._handle_literal
        if isinstance(node, IRLoad):
            return self._handle_load
        if isinstance(node, IRIndirectLoad):
            return self._handle_indirect_load
        if isinstance(node, IRStore):
            return self._handle_store
        if isinstance(node, IRIndirectStore):
            return self._handle_indirect_store
        if isinstance(node, IRCallReturn):
            return self._handle_call_return
        if isinstance(node, IRCall):
            return self._handle_call
        if isinstance(node, IRTailCall):
            return self._handle_tail_call
        if isinstance(node, IRTailcallReturn):
            return self._handle_tailcall_return
        if isinstance(node, IRReturn):
            return self._handle_return
        if isinstance(node, IRIf):
            return self._handle_branch
        if isinstance(node, IRTestSetBranch):
            return self._handle_testset
        if isinstance(node, IRFlagCheck):
            return self._handle_flag_check
        if isinstance(node, IRFunctionPrologue):
            return self._handle_prologue
        if isinstance(node, IRRaw):
            return self._handle_raw
        if isinstance(node, IRStackDuplicate):
            return self._handle_stack_duplicate
        if isinstance(node, IRStackDrop):
            return self._handle_stack_drop
        if isinstance(node, IRPageRegister):
            return self._handle_page_register
        if isinstance(node, IRIORead):
            return self._handle_io_read
        if isinstance(node, IRIOWrite):
            return self._handle_io_write
        if isinstance(node, IRConditionMask):
            return self._handle_condition_mask
        if isinstance(node, IRTablePatch):
            return self._handle_table_patch
        if isinstance(node, IRTableBuilderBegin):
            return self._handle_table_builder_begin
        if isinstance(node, IRTableBuilderEmit):
            return self._handle_table_builder_emit
        if isinstance(node, IRTableBuilderCommit):
            return self._handle_table_builder_commit
        if isinstance(node, IRAsciiHeader):
            return self._handle_ascii_header
        if isinstance(node, IRAsciiFinalize):
            return self._handle_ascii_finalize
        if isinstance(node, IRBuildArray):
            return self._handle_build_array
        if isinstance(node, IRBuildMap):
            return self._handle_build_map
        if isinstance(node, IRBuildTuple):
            return self._handle_build_tuple
        if isinstance(node, IRLiteralBlock):
            return self._handle_literal_block
        if isinstance(node, IRStringConstant):
            return self._handle_string_constant
        if isinstance(node, IRTailcallFrame):
            return self._handle_tailcall_frame
        if isinstance(node, IRCallPreparation):
            return self._handle_call_preparation
        if isinstance(node, IRCallCleanup):
            return self._handle_call_cleanup
        return None

    # Handlers -----------------------------------------------------------------
    def _handle_literal(self, node: IRLiteral, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="literal", value=node.value)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_load(self, node: IRLoad, env, metrics: ASTMetrics):
        alias = node.target if node.target != "stack" else None
        expr = ASTExpression(kind="load", value=node.slot, alias=alias, type=self._infer_kind(alias))
        metrics.observe_expression(expr)
        if alias:
            env[alias] = expr
        return ASTStatement(kind="assign", target=alias, expr=expr)

    def _handle_indirect_load(self, node: IRIndirectLoad, env, metrics: ASTMetrics):
        alias = node.target if node.target != "stack" else None
        base_expr = self._reference(node.base, env, metrics)
        offset_expr = ASTExpression(kind="literal", value=node.offset)
        metrics.observe_expression(offset_expr)
        expr = ASTExpression(kind="indirect_load", operands=(base_expr, offset_expr), alias=alias, type=self._infer_kind(alias))
        metrics.observe_expression(expr)
        if alias:
            env[alias] = expr
        return ASTStatement(kind="assign", target=alias, expr=expr)

    def _handle_store(self, node: IRStore, env, metrics: ASTMetrics):
        value = self._reference(node.value, env, metrics)
        metrics.store_values += 1
        target = self._format_slot(node.slot)
        return ASTStatement(kind="store", target=target, expr=value)

    def _handle_indirect_store(self, node: IRIndirectStore, env, metrics: ASTMetrics):
        base = self._reference(node.base, env, metrics)
        value = self._reference(node.value, env, metrics)
        metrics.store_values += 1
        offset_expr = ASTExpression(kind="literal", value=node.offset)
        metrics.observe_expression(offset_expr)
        expr = ASTExpression(kind="indirect_store", operands=(base, value, offset_expr))
        metrics.observe_expression(expr)
        return ASTStatement(kind="store", target="indirect", expr=expr)

    def _handle_call(self, node: IRCall, env, metrics: ASTMetrics):
        args = [self._reference(name, env, metrics) for name in node.args]
        metrics.call_arguments += len(args)
        info = []
        if node.symbol:
            call_name = node.symbol
        else:
            call_name = f"call_0x{node.target:04X}"
        if node.tail:
            info.append("tail")
        return ASTStatement(kind="call", target=call_name, expr=None, args=tuple(args), info=tuple(info))

    def _handle_call_return(self, node: IRCallReturn, env, metrics: ASTMetrics):
        args = [self._reference(name, env, metrics) for name in node.args]
        metrics.call_arguments += len(args)
        returns = [self._reference(name, env, metrics) for name in node.returns]
        metrics.return_values += len(returns)
        call_name = node.symbol or f"call_0x{node.target:04X}"
        info = []
        if node.tail:
            info.append("tail")
        if node.varargs:
            info.append("varargs")
        return ASTStatement(
            kind="call_return",
            target=call_name,
            expr=None,
            args=tuple(args + returns),
            info=tuple(info),
        )

    def _handle_tail_call(self, node: IRTailCall, env, metrics: ASTMetrics):
        args = [self._reference(name, env, metrics) for name in node.args]
        metrics.call_arguments += len(args)
        call_name = node.symbol or f"call_0x{node.target:04X}"
        info = ["tail"]
        return ASTStatement(kind="tail_call", target=call_name, expr=None, args=tuple(args), info=tuple(info))

    def _handle_tailcall_return(self, node: IRTailcallReturn, env, metrics: ASTMetrics):
        args = [self._reference(name, env, metrics) for name in node.args]
        returns = [self._reference(name, env, metrics) for name in node.returns]
        metrics.call_arguments += len(args)
        metrics.return_values += len(returns)
        call_name = node.symbol or f"call_0x{node.target:04X}"
        info = ["tail"]
        if node.varargs:
            info.append("varargs")
        return ASTStatement(
            kind="call_return",
            target=call_name,
            expr=None,
            args=tuple(args + returns),
            info=tuple(info),
        )

    def _handle_return(self, node: IRReturn, env, metrics: ASTMetrics):
        values = [self._reference(name, env, metrics) for name in node.values]
        metrics.return_values += len(values)
        info: List[str] = []
        if node.varargs:
            info.append("varargs")
        if node.mask is not None:
            info.append(f"mask=0x{node.mask:04X}")
        return ASTStatement(kind="return", target=None, expr=None, args=tuple(values), info=tuple(info))

    def _handle_branch(self, node: IRIf, env, metrics: ASTMetrics):
        expr = self._reference(node.condition, env, metrics)
        info = [f"then=0x{node.then_target:04X}", f"else=0x{node.else_target:04X}"]
        return ASTStatement(kind="branch", target=None, expr=expr, info=tuple(info))

    def _handle_testset(self, node: IRTestSetBranch, env, metrics: ASTMetrics):
        expr = self._reference(node.expr, env, metrics)
        info = [
            f"store={node.var}",
            f"then=0x{node.then_target:04X}",
            f"else=0x{node.else_target:04X}",
        ]
        return ASTStatement(kind="branch", target=None, expr=expr, info=tuple(info))

    def _handle_flag_check(self, node: IRFlagCheck, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="flag", value=node.flag)
        metrics.observe_expression(expr)
        info = [f"then=0x{node.then_target:04X}", f"else=0x{node.else_target:04X}"]
        return ASTStatement(kind="branch", target=None, expr=expr, info=tuple(info))

    def _handle_prologue(self, node: IRFunctionPrologue, env, metrics: ASTMetrics):
        expr = self._reference(node.expr, env, metrics)
        info = [node.var, f"then=0x{node.then_target:04X}", f"else=0x{node.else_target:04X}"]
        return ASTStatement(kind="prologue", target=None, expr=expr, info=tuple(info))

    def _handle_raw(self, node: IRRaw, env, metrics: ASTMetrics):
        mnemonic = node.mnemonic.lower()
        annotations = self._parse_annotations(node.annotations)
        operands = []
        if mnemonic in _ARITHMETIC_MNEMONICS:
            for key in ("lhs", "rhs", "arg0", "arg1"):
                name = annotations.get(key)
                if name:
                    operands.append(self._reference(name, env, metrics))
        expr = ASTExpression(kind=mnemonic, operands=tuple(operands), alias=node.operand_alias, type=self._infer_kind(node.operand_alias))
        metrics.observe_expression(expr)
        if node.operand_role == "target" and node.operand_alias:
            env[node.operand_alias] = expr
            target = node.operand_alias
        else:
            target = annotations.get("target")
            if target:
                env[target] = expr
        info = tuple(sorted(annotations.keys()))
        return ASTStatement(kind="assign", target=target, expr=expr, info=info)

    def _handle_stack_duplicate(self, node: IRStackDuplicate, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="stack_dup", value=node.value)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr, info=(f"copies={node.copies}",))

    def _handle_stack_drop(self, node: IRStackDrop, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="stack_drop", value=node.value)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_page_register(self, node: IRPageRegister, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="page_register", value=node.register)
        metrics.observe_expression(expr)
        info = []
        if node.value:
            info.append(f"value={node.value}")
        if node.literal is not None:
            info.append(f"literal=0x{node.literal:04X}")
        return ASTStatement(kind="assign", target=None, expr=expr, info=tuple(info))

    def _handle_io_read(self, node: IRIORead, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="io_read", value=node.port)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_io_write(self, node: IRIOWrite, env, metrics: ASTMetrics):
        info: List[str] = []
        if node.mask is not None:
            info.append(f"mask=0x{node.mask:04X}")
        expr = ASTExpression(kind="io_write", value=node.port)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr, info=tuple(info))

    def _handle_condition_mask(self, node: IRConditionMask, env, metrics: ASTMetrics):
        source = self._reference(node.source, env, metrics)
        mask_expr = ASTExpression(kind="literal", value=node.mask)
        metrics.observe_expression(mask_expr)
        expr = ASTExpression(kind="condition_mask", operands=(source, mask_expr))
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_table_patch(self, node: IRTablePatch, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="table_patch", value=node.kind)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_table_builder_begin(self, node: IRTableBuilderBegin, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="table_begin", value=node.kind)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_table_builder_emit(self, node: IRTableBuilderEmit, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="table_emit", value=node.kind)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_table_builder_commit(self, node: IRTableBuilderCommit, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="table_commit", value=node.kind)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_ascii_header(self, node: IRAsciiHeader, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="ascii_header", value=node.chunks)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_ascii_finalize(self, node: IRAsciiFinalize, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="ascii_finalize", value=node.suffix)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_build_array(self, node: IRBuildArray, env, metrics: ASTMetrics):
        elements = [self._reference(name, env, metrics) for name in node.elements]
        expr = ASTExpression(kind="array", operands=tuple(elements))
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_build_map(self, node: IRBuildMap, env, metrics: ASTMetrics):
        entries = []
        for key, value in node.entries:
            entries.append(self._reference(key, env, metrics))
            entries.append(self._reference(value, env, metrics))
        expr = ASTExpression(kind="map", operands=tuple(entries))
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_build_tuple(self, node: IRBuildTuple, env, metrics: ASTMetrics):
        elements = [self._reference(name, env, metrics) for name in node.elements]
        expr = ASTExpression(kind="tuple", operands=tuple(elements))
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_literal_block(self, node: IRLiteralBlock, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="literal_block", value=node.triplets)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_string_constant(self, node: IRStringConstant, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="string_const", value=node.name)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_tailcall_frame(self, node: IRTailcallFrame, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="tail_frame", value=node.returns)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_call_preparation(self, node: IRCallPreparation, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="call_prep", value=node.steps)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    def _handle_call_cleanup(self, node: IRCallCleanup, env, metrics: ASTMetrics):
        expr = ASTExpression(kind="call_cleanup", value=node.steps)
        metrics.observe_expression(expr)
        return ASTStatement(kind="assign", target=None, expr=expr)

    # ------------------------------------------------------------------
    # Procedure discovery and segment assembly
    # ------------------------------------------------------------------
    def _assemble_segments(
        self,
        program: IRProgram,
        cfg: ControlFlowGraph,
        ast_blocks: Dict[str, ASTBlock],
        metrics: ASTMetrics,
    ) -> Tuple[ASTSegment, ...]:
        segment_procedures: Dict[int, List[ASTProcedure]] = defaultdict(list)
        block_assignments: Dict[str, str] = {}
        entry_info = self._discover_entries(cfg, program)
        ordered_entries = sorted(entry_info.items(), key=lambda item: item[1]["offset"])

        for entry_label, info in ordered_entries:
            if entry_label in block_assignments:
                continue
            proc_blocks, exits = self._collect_procedure(cfg, entry_label, entry_info, block_assignments)
            name = info.get("symbol") or f"proc_0x{info['offset']:04X}"
            sources = tuple(sorted(info.get("reasons", set())))
            procedure = ASTProcedure(
                name=name,
                entry=entry_label,
                blocks=tuple(proc_blocks),
                exits=tuple(sorted(exits)),
                sources=sources,
            )
            segment_index = cfg.segment_index(entry_label)
            segment_procedures[segment_index].append(procedure)

        # handle unassigned blocks
        for label, node in cfg.nodes.items():
            if label in block_assignments:
                continue
            segment_index = node.segment_index
            procedure = ASTProcedure(
                name=f"proc_{label}",
                entry=label,
                blocks=(label,),
                exits=tuple(sorted(cfg.successors(label))),
                sources=("orphan",),
            )
            block_assignments[label] = procedure.name
            segment_procedures[segment_index].append(procedure)

        segments: List[ASTSegment] = []
        for segment in program.segments:
            procedures = tuple(segment_procedures.get(segment.index, []))
            blocks = tuple(ast_blocks[block.label] for block in segment.blocks)
            segments.append(
                ASTSegment(
                    index=segment.index,
                    start=segment.start,
                    length=segment.length,
                    blocks=blocks,
                    procedures=procedures,
                )
            )
        return tuple(segments)

    def _discover_entries(self, cfg: ControlFlowGraph, program: IRProgram) -> Dict[str, Dict[str, object]]:
        info: Dict[str, Dict[str, object]] = {}
        for segment in program.segments:
            if not segment.blocks:
                continue
            first = segment.blocks[0]
            data = info.setdefault(first.label, {"offset": first.start_offset, "reasons": set()})
            data["reasons"].add("segment")

            offset_map = {block.start_offset: block.label for block in segment.blocks}
            for block in segment.blocks:
                block_info = info.setdefault(block.label, {"offset": block.start_offset, "reasons": set()})
                for node in block.nodes:
                    if isinstance(node, IRFunctionPrologue):
                        block_info["reasons"].add("prologue")
                    if isinstance(node, (IRCall, IRCallReturn, IRTailCall, IRTailcallReturn)):
                        target_label = offset_map.get(node.target)
                        if target_label:
                            target_info = info.setdefault(target_label, {"offset": node.target, "reasons": set()})
                            target_info["reasons"].add("call")
                            if getattr(node, "symbol", None):
                                target_info["symbol"] = node.symbol
        return info

    def _collect_procedure(
        self,
        cfg: ControlFlowGraph,
        entry_label: str,
        entry_info: Dict[str, Dict[str, object]],
        assignments: Dict[str, str],
    ) -> Tuple[List[str], Set[str]]:
        queue = deque([entry_label])
        visited: List[str] = []
        exits: Set[str] = set()
        entry_set = {label for label, data in entry_info.items() if data.get("reasons")}

        while queue:
            label = queue.popleft()
            if label in assignments and assignments[label] != entry_label:
                continue
            if label in visited:
                continue
            visited.append(label)
            assignments[label] = entry_label
            node = cfg.node(label)
            if self._is_exit_block(node.block):
                exits.add(label)
                continue
            for successor in node.successors:
                if successor in entry_set and successor != entry_label:
                    continue
                if successor not in assignments or assignments.get(successor) == entry_label:
                    queue.append(successor)
        if not exits and visited:
            exits.add(visited[-1])
        return visited, exits

    def _is_exit_block(self, block: IRBlock) -> bool:
        for node in block.nodes:
            if isinstance(node, (IRReturn, IRTailCall, IRTailcallReturn)):
                return True
            if isinstance(node, IRCallReturn) and node.tail:
                return True
        return False

    # ------------------------------------------------------------------
    # Expression helpers
    # ------------------------------------------------------------------
    def _reference(self, name: Optional[str], env: Dict[str, ASTExpression], metrics: ASTMetrics) -> ASTExpression:
        if not name:
            expr = ASTExpression(kind="unknown")
            metrics.observe_expression(expr)
            metrics.record_reference(resolved=False)
            return expr
        existing = env.get(name)
        if existing is not None:
            metrics.record_reference(resolved=True)
            return existing
        expr = ASTExpression(kind="ref", alias=name, type=self._infer_kind(name))
        env[name] = expr
        metrics.observe_expression(expr)
        metrics.record_reference(resolved=False)
        return expr

    def _infer_kind(self, alias: Optional[str]) -> SSAValueKind:
        if not alias:
            return SSAValueKind.UNKNOWN
        for prefix, kind in _REFERENCE_PREFIX_KIND.items():
            if alias.startswith(prefix):
                return kind
        return SSAValueKind.UNKNOWN

    def _format_slot(self, slot: IRSlot) -> str:
        return f"{slot.space.name.lower()}[{slot.index}]"

    def _parse_annotations(self, annotations: Sequence[str]) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        for note in annotations:
            if "=" not in note:
                continue
            key, value = note.split("=", 1)
            parsed[key.strip()] = value.strip()
        return parsed


__all__ = ["ASTBuilder"]
