"""Reconstruct an AST and CFG from the normalised IR programme."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..ir.model import (
    IRBankedLoad,
    IRBankedStore,
    IRBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    IRCallCleanup,
    IRCallReturn,
    IRDataMarker,
    IRDispatchCase,
    IRFlagCheck,
    IRFunctionPrologue,
    IRIORead,
    IRIOWrite,
    IRIf,
    IRIndirectLoad,
    IRIndirectStore,
    IRLiteral,
    IRLiteralBlock,
    IRLiteralChunk,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRStore,
    IRStringConstant,
    IRSwitchDispatch,
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
    ASTDispatchTable,
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
    ASTSwitch,
    ASTSwitchCase,
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


@dataclass
class _ProcedureAccumulator:
    """Partial reconstruction state for a single procedure."""

    entry_offset: int
    entry_reasons: Set[str] = field(default_factory=set)
    blocks: Dict[int, _PendingBlock] = field(default_factory=dict)


@dataclass
class _PendingDispatchCall:
    """Call statement awaiting an accompanying dispatch table."""

    helper: int
    index: int
    call: ASTCallExpr


@dataclass
class _PendingDispatchTable:
    """Dispatch table awaiting a matching helper call."""

    dispatch: IRSwitchDispatch
    index: int


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
        analyses, entry_reasons = self._compact_cfg(analyses, entry_reasons)
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
            if not successors and fallthrough is not None and not exit_reasons:
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

    def _compact_cfg(
        self,
        analyses: Mapping[int, _BlockAnalysis],
        entry_reasons: Mapping[int, Tuple[str, ...]],
    ) -> Tuple[Mapping[int, _BlockAnalysis], Mapping[int, Tuple[str, ...]]]:
        """Collapse trivial cleanup/terminator blocks into their successors."""

        trivial_targets: Dict[int, int] = {}
        for offset, analysis in analyses.items():
            if offset in entry_reasons:
                continue
            if analysis.exit_reasons:
                continue
            if len(analysis.successors) != 1:
                continue
            block = analysis.block
            if not block.nodes:
                trivial_targets[offset] = analysis.successors[0]
                continue
            if all(isinstance(node, (IRCallCleanup, IRTerminator)) for node in block.nodes):
                trivial_targets[offset] = analysis.successors[0]

        if not trivial_targets:
            return analyses, entry_reasons

        def resolve(target: int) -> int:
            seen: Set[int] = set()
            while target in trivial_targets and target not in seen:
                seen.add(target)
                target = trivial_targets[target]
            return target

        compacted: Dict[int, _BlockAnalysis] = {}
        for offset, analysis in analyses.items():
            if offset in trivial_targets:
                continue
            successors = tuple(
                sorted({resolve(candidate) for candidate in analysis.successors})
            )
            fallthrough = analysis.fallthrough
            if fallthrough is not None:
                fallthrough = resolve(fallthrough)
                if fallthrough == offset:
                    fallthrough = None
            compacted[offset] = replace(
                analysis,
                successors=successors,
                fallthrough=fallthrough,
            )

        updated_entries = {
            offset: reasons
            for offset, reasons in entry_reasons.items()
            if offset in compacted
        }

        return compacted, updated_entries

    def _group_procedures(
        self,
        segment: IRSegment,
        analyses: Mapping[int, _BlockAnalysis],
        entry_reasons: Mapping[int, Tuple[str, ...]],
        metrics: ASTMetrics,
    ) -> Sequence[ASTProcedure]:
        assigned: Dict[int, int] = {}
        accumulators: Dict[int, _ProcedureAccumulator] = {}
        order: List[int] = []

        for entry in sorted(entry_reasons):
            if entry not in analyses:
                continue
            reachable = self._collect_entry_blocks(entry, analyses, entry_reasons, assigned)
            if not reachable:
                continue
            accumulator = _ProcedureAccumulator(entry_offset=entry)
            accumulator.entry_reasons.update(entry_reasons[entry])
            order.append(entry)
            state: Dict[str, ASTExpression] = {}
            for offset in sorted(reachable):
                assigned[offset] = entry
                analysis = analyses[offset]
                accumulator.entry_reasons.update(entry_reasons.get(offset, ()))
                accumulator.blocks[offset] = self._convert_block(analysis, state, metrics)
            accumulators[entry] = accumulator

        for offset in sorted(analyses):
            if offset in assigned:
                continue
            reachable = self._collect_component(offset, analyses, assigned)
            if not reachable:
                continue
            accumulator = _ProcedureAccumulator(entry_offset=offset)
            order.append(offset)
            state: Dict[str, ASTExpression] = {}
            for node in sorted(reachable):
                assigned[node] = offset
                analysis = analyses[node]
                accumulator.entry_reasons.update(entry_reasons.get(node, ()))
                accumulator.blocks[node] = self._convert_block(analysis, state, metrics)
            accumulators[offset] = accumulator

        procedures: List[ASTProcedure] = []
        for index, entry in enumerate(order):
            accumulator = accumulators[entry]
            pending_blocks = [accumulator.blocks[offset] for offset in sorted(accumulator.blocks)]
            name = f"proc_{accumulator.entry_offset:04X}"
            procedure = self._finalise_procedure(
                name=name,
                entry_offset=accumulator.entry_offset,
                entry_reasons=tuple(sorted(accumulator.entry_reasons)),
                blocks=pending_blocks,
            )
            procedures.append(procedure)
        return procedures

    def _finalise_procedure(
        self,
        name: str,
        entry_offset: int,
        entry_reasons: Tuple[str, ...],
        blocks: Sequence[_PendingBlock],
    ) -> ASTProcedure:
        realised_blocks = self._realise_blocks(entry_offset, blocks)
        exit_offsets = self._compute_exit_offsets(blocks)
        return ASTProcedure(
            name=name,
            entry_offset=entry_offset,
            entry_reasons=entry_reasons,
            blocks=realised_blocks,
            exit_offsets=tuple(sorted(exit_offsets)),
        )

    def _collect_entry_blocks(
        self,
        entry: int,
        analyses: Mapping[int, _BlockAnalysis],
        entry_reasons: Mapping[int, Tuple[str, ...]],
        assigned: Mapping[int, int],
    ) -> Set[int]:
        reachable: Set[int] = set()
        stack: List[int] = [entry]
        while stack:
            offset = stack.pop()
            if offset in reachable:
                continue
            if offset in assigned and assigned[offset] != entry:
                continue
            analysis = analyses.get(offset)
            if analysis is None:
                continue
            reachable.add(offset)
            for successor in analysis.successors:
                if successor not in analyses:
                    continue
                if successor in entry_reasons and successor != entry:
                    continue
                stack.append(successor)
        return reachable

    def _collect_component(
        self,
        start: int,
        analyses: Mapping[int, _BlockAnalysis],
        assigned: Mapping[int, int],
    ) -> Set[int]:
        reachable: Set[int] = set()
        stack: List[int] = [start]
        while stack:
            offset = stack.pop()
            if offset in reachable or offset in assigned:
                continue
            analysis = analyses.get(offset)
            if analysis is None:
                continue
            reachable.add(offset)
            for successor in analysis.successors:
                if successor not in analyses:
                    continue
                stack.append(successor)
        return reachable

    def _compute_exit_offsets(self, blocks: Sequence[_PendingBlock]) -> Tuple[int, ...]:
        if not blocks:
            return tuple()

        exit_node = "__exit__"
        offsets = {block.start_offset for block in blocks}
        nodes = set(offsets)
        nodes.add(exit_node)

        postdom: Dict[int | str, Set[int | str]] = {
            offset: set(nodes) for offset in offsets
        }
        postdom[exit_node] = {exit_node}

        successors: Dict[int, Set[int | str]] = {}
        for block in blocks:
            analysis = self._current_analyses.get(block.start_offset)
            succ: Set[int | str] = set()
            if analysis is not None:
                if not analysis.successors:
                    succ.add(exit_node)
                for candidate in analysis.successors:
                    if candidate in offsets:
                        succ.add(candidate)
                    else:
                        succ.add(exit_node)
                if analysis.exit_reasons:
                    succ.add(exit_node)
            else:
                succ.add(exit_node)
            if not succ:
                succ.add(exit_node)
            successors[block.start_offset] = succ

        changed = True
        while changed:
            changed = False
            for offset in offsets:
                succ = successors.get(offset)
                if not succ:
                    succ = {exit_node}
                intersection = set(nodes)
                for candidate in succ:
                    intersection &= postdom[candidate]
                updated = {offset} | intersection
                if updated != postdom[offset]:
                    postdom[offset] = updated
                    changed = True

        exit_offsets: Set[int] = set()
        for offset in offsets:
            succ = successors.get(offset, {exit_node})
            if exit_node not in succ:
                continue
            candidates = postdom[offset] - {offset}
            if not candidates:
                continue
            immediate: Optional[int | str] = None
            for candidate in candidates:
                dominated = False
                for other in candidates:
                    if other == candidate:
                        continue
                    if candidate in postdom[other]:
                        dominated = True
                        break
                if not dominated:
                    immediate = candidate
                    break
            if immediate == exit_node or (immediate is None and exit_node in candidates):
                exit_offsets.add(offset)
        return tuple(sorted(exit_offsets))

    def _realise_blocks(
        self, entry_offset: int, blocks: Sequence[_PendingBlock]
    ) -> Tuple[ASTBlock, ...]:
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
            filtered = self._filter_statements(pending.statements)
            realised.statements = tuple(filtered)
            realised.successors = tuple(
                block_map[target]
                for target in pending.successors
                if target in block_map
            )

        simplified = self._simplify_ast_cfg(block_map, entry_offset)
        return simplified

    def _filter_statements(self, statements: Sequence[ASTStatement]) -> List[ASTStatement]:
        filtered: List[ASTStatement] = []
        for statement in statements:
            if isinstance(statement, ASTComment):
                text = statement.text
                if self._is_technical_comment(text):
                    continue
            filtered.append(statement)
        return filtered

    def _simplify_ast_cfg(
        self, block_map: Mapping[int, ASTBlock], entry_offset: int
    ) -> Tuple[ASTBlock, ...]:
        entry_block = block_map.get(entry_offset)
        blocks = list(block_map.values())
        if not blocks:
            return tuple()

        def compute_predecessors() -> Dict[int, List[ASTBlock]]:
            preds: Dict[int, List[ASTBlock]] = {id(block): [] for block in blocks}
            for block in blocks:
                for successor in block.successors:
                    preds.setdefault(id(successor), []).append(block)
            return preds

        preds = compute_predecessors()
        removed = True
        while removed:
            removed = False
            for block in list(blocks):
                if block is entry_block:
                    continue
                block_id = id(block)
                if block_id not in preds:
                    continue
                if block.statements:
                    continue
                successors = block.successors
                if len(successors) != 1:
                    continue
                predecessor_list = preds.get(block_id, [])
                if len(predecessor_list) != 1:
                    continue
                predecessor = predecessor_list[0]
                successor = successors[0]
                if successor is block:
                    continue
                updated_successors = [
                    successor if candidate is block else candidate
                    for candidate in predecessor.successors
                ]
                predecessor.successors = tuple(updated_successors)
                self._redirect_branch_references(predecessor, block, successor)
                blocks.remove(block)
                removed = True
                break
            if removed:
                preds = compute_predecessors()

        ordered = sorted(blocks, key=lambda block: block.start_offset)
        return tuple(ordered)

    def _redirect_branch_references(
        self, block: ASTBlock, old: ASTBlock, new: ASTBlock
    ) -> None:
        for statement in block.statements:
            if isinstance(statement, BranchStatement):
                if statement.then_branch is old:
                    statement.then_branch = new
                if statement.else_branch is old:
                    statement.else_branch = new

    @staticmethod
    def _is_technical_comment(text: str) -> bool:
        prefixes = (
            "lit(",
            "marker ",
            "ascii(",
            "ascii_header",
            "str(",
            "tuple(",
            "map(",
            "array(",
            "cleanup_call",
            "prep_call_args",
            "drop ",
            "literal_block",
        )
        return text.startswith(prefixes)

    def _should_drop_stack_top_branch(
        self,
        expr: ASTExpression,
        analysis: _BlockAnalysis,
        targets: Tuple[int, int],
    ) -> bool:
        if not (self._expr_is_stack_top(expr) or isinstance(expr, ASTLiteral)):
            return False
        block_offset = analysis.block.start_offset

        def normalise(candidates: Sequence[int]) -> Set[int]:
            return {
                target
                for target in candidates
                if target and target != block_offset
            }

        unique_targets = normalise(targets)
        if not unique_targets:
            unique_targets = normalise(analysis.successors)
        if len(unique_targets) > 1:
            return False
        return True

    @staticmethod
    def _expr_is_stack_top(expr: ASTExpression) -> bool:
        if isinstance(expr, ASTIdentifier) and expr.name == "stack_top":
            return True
        if isinstance(expr, ASTUnknown) and expr.token == "stack_top":
            return True
        return expr.render() == "stack_top"

    def _convert_block(
        self,
        analysis: _BlockAnalysis,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> _PendingBlock:
        block = analysis.block
        statements: List[ASTStatement] = []
        branch_links: List[_BranchLink] = []
        pending_calls: List[_PendingDispatchCall] = []
        pending_tables: List[_PendingDispatchTable] = []
        for node in block.nodes:
            if isinstance(node, IRCall):
                self._handle_dispatch_call(
                    node,
                    value_state,
                    metrics,
                    statements,
                    pending_calls,
                    pending_tables,
                )
                continue
            if isinstance(node, IRSwitchDispatch):
                self._handle_dispatch_table(
                    node,
                    statements,
                    pending_calls,
                    pending_tables,
                )
                continue
            node_statements, node_links = self._convert_node(
                node,
                block.start_offset,
                value_state,
                metrics,
                analysis,
            )
            statements.extend(node_statements)
            branch_links.extend(node_links)
            if not isinstance(node, (IRLiteral, IRLiteralChunk, IRDataMarker, IRLiteralBlock)):
                value_state.pop("stack_top", None)
        return _PendingBlock(
            label=block.label,
            start_offset=block.start_offset,
            statements=statements,
            successors=analysis.successors,
            branch_links=branch_links,
        )

    def _handle_dispatch_call(
        self,
        node: IRCall,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
        statements: List[ASTStatement],
        pending_calls: List[_PendingDispatchCall],
        pending_tables: List[_PendingDispatchTable],
    ) -> None:
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
        dispatch = self._pop_dispatch_table(node.target, pending_tables)
        if dispatch is not None:
            statements.append(self._build_dispatch_switch(call_expr, dispatch))
            return
        index = len(statements)
        statements.append(ASTCallStatement(call=call_expr))
        pending_calls.append(
            _PendingDispatchCall(helper=node.target, index=index, call=call_expr)
        )

    def _handle_dispatch_table(
        self,
        dispatch: IRSwitchDispatch,
        statements: List[ASTStatement],
        pending_calls: List[_PendingDispatchCall],
        pending_tables: List[_PendingDispatchTable],
    ) -> None:
        table_statement = self._build_dispatch_table(dispatch)
        call_info = self._pop_dispatch_call(dispatch, pending_calls)
        if call_info is not None:
            insert_index = call_info.index
            statements.insert(insert_index, table_statement)
            switch_statement = self._build_dispatch_switch(call_info.call, dispatch)
            statements[insert_index + 1] = switch_statement
            self._adjust_pending_indices(pending_calls, insert_index)
            self._adjust_table_indices(pending_tables, insert_index)
            return
        index = len(statements)
        statements.append(table_statement)
        pending_tables.append(_PendingDispatchTable(dispatch=dispatch, index=index))

    def _dispatch_helper_matches(self, helper: int, dispatch: IRSwitchDispatch) -> bool:
        if dispatch.helper is not None:
            return dispatch.helper == helper
        return False

    def _pop_dispatch_table(
        self, helper: int, pending_tables: List[_PendingDispatchTable]
    ) -> Optional[IRSwitchDispatch]:
        for index in range(len(pending_tables) - 1, -1, -1):
            entry = pending_tables[index]
            if self._dispatch_helper_matches(helper, entry.dispatch):
                pending_tables.pop(index)
                return entry.dispatch
        return None

    def _pop_dispatch_call(
        self, dispatch: IRSwitchDispatch, pending_calls: List[_PendingDispatchCall]
    ) -> Optional[_PendingDispatchCall]:
        helper = dispatch.helper
        if helper is None:
            return None
        for index in range(len(pending_calls) - 1, -1, -1):
            entry = pending_calls[index]
            if entry.helper == helper:
                return pending_calls.pop(index)
        return None

    def _adjust_pending_indices(
        self, pending_calls: List[_PendingDispatchCall], insert_index: int
    ) -> None:
        for entry in pending_calls:
            if entry.index >= insert_index:
                entry.index += 1

    def _adjust_table_indices(
        self, pending_tables: List[_PendingDispatchTable], insert_index: int
    ) -> None:
        for entry in pending_tables:
            if entry.index >= insert_index:
                entry.index += 1

    def _build_dispatch_switch(
        self, call_expr: ASTCallExpr, dispatch: IRSwitchDispatch
    ) -> ASTSwitch:
        cases = self._build_dispatch_cases(dispatch.cases)
        return ASTSwitch(
            selector=call_expr,
            cases=cases,
            helper=dispatch.helper,
            helper_symbol=dispatch.helper_symbol,
            default=dispatch.default,
        )

    def _build_dispatch_table(self, dispatch: IRSwitchDispatch) -> ASTDispatchTable:
        cases = self._build_dispatch_cases(dispatch.cases)
        return ASTDispatchTable(
            cases=cases,
            helper=dispatch.helper,
            helper_symbol=dispatch.helper_symbol,
            default=dispatch.default,
        )

    def _build_dispatch_cases(
        self, cases: Sequence[IRDispatchCase]
    ) -> Tuple[ASTSwitchCase, ...]:
        return tuple(
            ASTSwitchCase(key=case.key, target=case.target, symbol=case.symbol)
            for case in cases
        )

    def _convert_node(
        self,
        node,
        origin_offset: int,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
        analysis: _BlockAnalysis,
    ) -> Tuple[List[ASTStatement], List[_BranchLink]]:
        if isinstance(node, IRLiteral):
            value_state["stack_top"] = ASTLiteral(node.value)
            return [], []
        if isinstance(node, IRLiteralChunk):
            value_state["stack_top"] = ASTUnknown(node.describe())
            return [], []
        if isinstance(node, IRLiteralBlock):
            value_state["stack_top"] = ASTUnknown(node.describe())
            return [], []
        if isinstance(node, IRDataMarker):
            value_state["stack_top"] = ASTUnknown(node.describe())
            return [], []
        if isinstance(node, IRBuildTuple):
            value_state["stack_top"] = ASTUnknown(node.describe())
            return [], []
        if isinstance(node, IRBuildArray):
            value_state["stack_top"] = ASTUnknown(node.describe())
            return [], []
        if isinstance(node, IRBuildMap):
            value_state["stack_top"] = ASTUnknown(node.describe())
            return [], []
        if isinstance(node, IRStringConstant):
            value_state["stack_top"] = ASTUnknown(node.describe())
            return [], []
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
            if self._should_drop_stack_top_branch(
                condition, analysis, (node.then_target, node.else_target)
            ):
                return [], []
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
            value_state[node.var] = expr
            if self._should_drop_stack_top_branch(
                expr, analysis, (node.then_target, node.else_target)
            ):
                return [], []
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
            value_state[node.var] = expr
            if self._should_drop_stack_top_branch(
                expr, analysis, (node.then_target, node.else_target)
            ):
                return [], []
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
