"""Reconstruct an AST and CFG from the normalised IR programme."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from enum import Enum, auto
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
    IRDispatchCase,
    IRBankedLoad,
    IRBankedStore,
    IRIndirectLoad,
    IRIndirectStore,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRStore,
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
            if not any(block.statements or block.successors for block in procedure.blocks):
                continue
            procedures.append(procedure)
        return procedures

    def _finalise_procedure(
        self,
        name: str,
        entry_offset: int,
        entry_reasons: Tuple[str, ...],
        blocks: Sequence[_PendingBlock],
    ) -> ASTProcedure:
        realised_blocks = self._realise_blocks(blocks)
        simplified_blocks = self._simplify_blocks(realised_blocks, entry_offset)
        exit_offsets = self._compute_exit_offsets_from_ast(simplified_blocks)
        return ASTProcedure(
            name=name,
            entry_offset=entry_offset,
            entry_reasons=entry_reasons,
            blocks=simplified_blocks,
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

    def _simplify_blocks(
        self, blocks: Tuple[ASTBlock, ...], entry_offset: int
    ) -> Tuple[ASTBlock, ...]:
        if not blocks:
            return tuple()

        block_order: List[ASTBlock] = list(blocks)
        entry_block = next(
            (block for block in block_order if block.start_offset == entry_offset),
            block_order[0],
        )

        id_map: Dict[int, ASTBlock] = {id(block): block for block in block_order}

        for block in block_order:
            self._simplify_stack_branches(block)

        self._propagate_branch_invariants(block_order, entry_block)

        for block in block_order:
            filtered = tuple(
                statement
                for statement in block.statements
                if not self._is_noise_statement(statement)
            )
            if filtered != block.statements:
                block.statements = filtered

        predecessors: Dict[int, Set[int]] = {block_id: set() for block_id in id_map}
        for block in block_order:
            deduped: List[ASTBlock] = []
            seen_ids: Set[int] = set()
            for successor in block.successors:
                succ_id = id(successor)
                if succ_id not in predecessors:
                    continue
                if succ_id not in seen_ids:
                    deduped.append(successor)
                    seen_ids.add(succ_id)
                predecessors[succ_id].add(id(block))
            block.successors = tuple(deduped)

        changed = True
        while changed:
            changed = False
            for block in list(block_order):
                if block is entry_block:
                    continue
                if block.statements:
                    continue
                if len(block.successors) != 1:
                    continue
                successor = block.successors[0]
                succ_id = id(successor)
                block_id = id(block)
                if succ_id not in predecessors:
                    continue
                for pred_id in list(predecessors.get(block_id, set())):
                    pred = id_map.get(pred_id)
                    if pred is None:
                        continue
                    self._replace_successor(pred, block, successor)
                    predecessors[succ_id].add(pred_id)
                predecessors[succ_id].discard(block_id)
                block_order.remove(block)
                predecessors.pop(block_id, None)
                id_map.pop(block_id, None)
                changed = True
                break

        merged = True
        while merged:
            merged = False
            for block in list(block_order):
                if len(block.successors) != 1:
                    continue
                successor = block.successors[0]
                if successor is block:
                    continue
                block_id = id(block)
                succ_id = id(successor)
                if succ_id not in predecessors:
                    continue
                if len(predecessors[succ_id]) != 1:
                    continue
                if block.statements and isinstance(block.statements[-1], BranchStatement):
                    continue
                if successor in successor.successors:
                    continue
                block.statements = tuple(block.statements + successor.statements)
                block.successors = successor.successors
                for succ in successor.successors:
                    succ_key = id(succ)
                    pred_set = predecessors.setdefault(succ_key, set())
                    pred_set.discard(succ_id)
                    pred_set.add(block_id)
                block_order.remove(successor)
                predecessors.pop(succ_id, None)
                id_map.pop(succ_id, None)
                merged = True
                break

        removed = True
        while removed:
            removed = False
            for block in list(block_order):
                if block is entry_block:
                    continue
                block_id = id(block)
                if predecessors.get(block_id):
                    continue
                for succ in block.successors:
                    predecessors.setdefault(id(succ), set()).discard(block_id)
                block_order.remove(block)
                predecessors.pop(block_id, None)
                id_map.pop(block_id, None)
                removed = True
                break

        return tuple(block_order)

    # ------------------------------------------------------------------
    # branch simplification helpers
    # ------------------------------------------------------------------

    class _BooleanValue(Enum):
        UNKNOWN = auto()
        TRUE = auto()
        FALSE = auto()

        @classmethod
        def from_bool(cls, value: bool) -> "ASTBuilder._BooleanValue":
            return cls.TRUE if value else cls.FALSE

        def as_python(self) -> Optional[bool]:
            if self is ASTBuilder._BooleanValue.TRUE:
                return True
            if self is ASTBuilder._BooleanValue.FALSE:
                return False
            return None

    _BooleanState = Dict[str, "ASTBuilder._BooleanValue"]

    def _propagate_branch_invariants(
        self, blocks: Sequence[ASTBlock], entry_block: ASTBlock
    ) -> None:
        if not blocks:
            return

        state_in: Dict[int, ASTBuilder._BooleanState] = {}
        worklist: deque[ASTBlock] = deque([entry_block])
        decisions: Dict[int, Tuple[Optional[ASTBlock], bool]] = {}
        successor_overrides: Dict[int, Tuple[ASTBlock, ...]] = {}

        while worklist:
            block = worklist.popleft()
            block_id = id(block)
            inbound = state_in.get(block_id, {})
            outbound, decision, succ_override = self._evaluate_block_invariants(
                block, inbound
            )
            if decision:
                decisions.update(decision)
            if succ_override is not None:
                successor_overrides[block_id] = succ_override
                successors: Tuple[ASTBlock, ...] = succ_override
            else:
                successors = block.successors

            for successor in successors:
                successor_id = id(successor)
                merged = self._merge_boolean_states(
                    state_in.get(successor_id), outbound
                )
                if merged is not None:
                    state_in[successor_id] = merged
                    worklist.append(successor)

        if not decisions and not successor_overrides:
            return

        for block in blocks:
            block_id = id(block)
            if block_id in successor_overrides:
                block.successors = successor_overrides[block_id]
            updated_statements: List[ASTStatement] = []
            for statement in block.statements:
                decision = decisions.get(id(statement))
                if decision is None:
                    updated_statements.append(statement)
                    continue
                target, value = decision
                if isinstance(statement, ASTBranch):
                    continue
                if isinstance(statement, (ASTTestSet, ASTFunctionPrologue)):
                    replacement = self._materialise_boolean_store(statement.var, value)
                    if replacement is not None:
                        updated_statements.append(replacement)
                    continue
                updated_statements.append(statement)
            block.statements = tuple(updated_statements)

    def _materialise_boolean_store(
        self, target_expr: ASTExpression, value: bool
    ) -> Optional[ASTStatement]:
        literal = ASTLiteral(1 if value else 0)
        if isinstance(target_expr, ASTIdentifier):
            return ASTAssign(target=target_expr, value=literal)
        if isinstance(target_expr, ASTExpression):
            return ASTStore(target=target_expr, value=literal)
        return None

    def _evaluate_block_invariants(
        self,
        block: ASTBlock,
        inbound: _BooleanState,
    ) -> Tuple[_BooleanState, Dict[int, Tuple[Optional[ASTBlock], bool]], Optional[Tuple[ASTBlock, ...]]]:
        state = dict(inbound)
        decisions: Dict[int, Tuple[Optional[ASTBlock], bool]] = {}
        override: Optional[Tuple[ASTBlock, ...]] = None

        for statement in block.statements:
            if isinstance(statement, ASTComment):
                self._update_state_from_comment(state, statement.text)
                continue
            if isinstance(statement, ASTAssign):
                self._update_assignment_state(state, statement.target, statement.value)
                continue
            if isinstance(statement, ASTStore):
                self._update_store_state(state, statement.target, statement.value)
                continue
            if isinstance(statement, ASTTestSet):
                decision = self._evaluate_boolean_branch(
                    state, statement.expr, statement.then_branch, statement.else_branch
                )
                if decision is not None:
                    chosen, value = decision
                    decisions[id(statement)] = (chosen, value)
                    self._update_store_state(
                        state, statement.var, ASTLiteral(1 if value else 0)
                    )
                    override = () if chosen is None else (chosen,)
                break
            if isinstance(statement, ASTFunctionPrologue):
                decision = self._evaluate_boolean_branch(
                    state, statement.expr, statement.then_branch, statement.else_branch
                )
                if decision is not None:
                    chosen, value = decision
                    decisions[id(statement)] = (chosen, value)
                    self._update_store_state(
                        state, statement.var, ASTLiteral(1 if value else 0)
                    )
                    override = () if chosen is None else (chosen,)
                break
            if isinstance(statement, ASTBranch):
                decision = self._evaluate_boolean_branch(
                    state,
                    statement.condition,
                    statement.then_branch,
                    statement.else_branch,
                )
                if decision is not None:
                    chosen, value = decision
                    decisions[id(statement)] = (chosen, value)
                    override = () if chosen is None else (chosen,)
                break

        return state, decisions, override

    def _evaluate_boolean_branch(
        self,
        state: _BooleanState,
        expr: ASTExpression,
        then_branch: Optional[ASTBlock],
        else_branch: Optional[ASTBlock],
    ) -> Optional[Tuple[Optional[ASTBlock], bool]]:
        value = self._evaluate_expression_bool(state, expr)
        if value is None:
            return None
        target = then_branch if value else else_branch
        return (target, value)

    def _evaluate_expression_bool(
        self, state: _BooleanState, expr: ASTExpression
    ) -> Optional[bool]:
        if isinstance(expr, ASTLiteral):
            return bool(expr.value)
        if isinstance(expr, ASTIdentifier):
            name = expr.name
            if name == "stack_top":
                stored = state.get(name)
                return stored.as_python() if stored else None
            if name in state:
                return state[name].as_python()
            lowered = name.lower()
            if lowered.startswith("bool"):
                stored = state.get(name)
                return stored.as_python() if stored else None
        return None

    def _update_state_from_comment(
        self, state: _BooleanState, text: str
    ) -> None:
        body = text.strip()
        if body.startswith("lit(") and body.endswith(")"):
            try:
                literal = int(body[4:-1], 16)
            except ValueError:
                state.pop("stack_top", None)
                return
            state["stack_top"] = self._BooleanValue.from_bool(bool(literal))
            return
        if body.startswith("drop"):
            state.pop("stack_top", None)
            return
        if "stack_teardown" in body:
            state["stack_top"] = self._BooleanValue.FALSE
            return
        if "stack_setup" in body:
            state.pop("stack_top", None)

    def _update_assignment_state(
        self, state: _BooleanState, target: ASTIdentifier, value: ASTExpression
    ) -> None:
        name = target.name
        if not name.lower().startswith("bool"):
            return
        resolved = self._evaluate_expression_bool(state, value)
        if resolved is None:
            state.pop(name, None)
            return
        state[name] = self._BooleanValue.from_bool(resolved)

    def _update_store_state(
        self, state: _BooleanState, target: ASTExpression, value: ASTExpression
    ) -> None:
        if isinstance(target, ASTIdentifier):
            self._update_assignment_state(state, target, value)

    def _merge_boolean_states(
        self,
        existing: Optional[_BooleanState],
        incoming: _BooleanState,
    ) -> Optional[_BooleanState]:
        if existing is None:
            return dict(incoming)
        updated: Dict[str, ASTBuilder._BooleanValue] = dict(existing)
        changed = False
        keys = set(existing) | set(incoming)
        for key in keys:
            left = existing.get(key, self._BooleanValue.UNKNOWN)
            right = incoming.get(key, self._BooleanValue.UNKNOWN)
            if left == right:
                continue
            if left is self._BooleanValue.UNKNOWN:
                if right is not self._BooleanValue.UNKNOWN:
                    updated[key] = right
                    changed = True
                continue
            if right is self._BooleanValue.UNKNOWN:
                continue
            if left != right:
                if updated.get(key) != self._BooleanValue.UNKNOWN:
                    updated[key] = self._BooleanValue.UNKNOWN
                    changed = True
        return updated if changed else None

    def _simplify_stack_branches(self, block: ASTBlock) -> None:
        stack: List[Tuple[str, int | None]] = []
        for statement in list(block.statements):
            if isinstance(statement, ASTComment):
                effect = self._stack_effect_from_comment(statement.text)
                kind = effect[0]
                if kind == "literal":
                    stack.append(effect)
                elif kind == "marker":
                    stack.append(effect)
                else:
                    stack.clear()
                continue
            if isinstance(statement, ASTReturn):
                stack.clear()
                continue
            if isinstance(statement, ASTBranch):
                if self._is_stack_top_expr(statement.condition) and stack:
                    kind, value = stack[-1]
                    if kind == "literal" and value is not None:
                        target = statement.then_branch if value else statement.else_branch
                        if target is not None:
                            block.statements = tuple(
                                stmt for stmt in block.statements if stmt is not statement
                            )
                            block.successors = (target,)
                            return
                stack.clear()
                continue
            stack.clear()

    @staticmethod
    def _is_stack_top_expr(expr: ASTExpression) -> bool:
        return isinstance(expr, ASTIdentifier) and expr.name == "stack_top"

    @staticmethod
    def _stack_effect_from_comment(text: str) -> Tuple[str, int | None]:
        body = text.strip()
        if body.startswith("lit(") and body.endswith(")"):
            literal = body[4:-1]
            try:
                value = int(literal, 16)
            except ValueError:
                return ("invalidate", None)
            return ("literal", value)
        if body.startswith("marker "):
            return ("marker", None)
        return ("invalidate", None)

    @staticmethod
    def _is_noise_statement(statement: ASTStatement) -> bool:
        if isinstance(statement, ASTComment):
            body = statement.text.strip()
            prefixes = ("lit(", "marker ", "literal_block", "ascii(")
            return body.startswith(prefixes)
        return False

    def _replace_successor(self, block: ASTBlock, old: ASTBlock, new: ASTBlock) -> None:
        block.successors = tuple(new if succ is old else succ for succ in block.successors)
        for statement in block.statements:
            if isinstance(statement, BranchStatement):
                if statement.then_branch is old:
                    statement.then_branch = new
                    statement.then_hint = None
                if statement.else_branch is old:
                    statement.else_branch = new
                    statement.else_hint = None

    def _compute_exit_offsets_from_ast(self, blocks: Sequence[ASTBlock]) -> Tuple[int, ...]:
        if not blocks:
            return tuple()

        exit_node = "__exit__"
        offsets = {block.start_offset for block in blocks}
        nodes: Set[int | str] = set(offsets)
        nodes.add(exit_node)

        postdom: Dict[int | str, Set[int | str]] = {
            offset: set(nodes) for offset in offsets
        }
        postdom[exit_node] = {exit_node}

        successors: Dict[int, Set[int | str]] = {}
        for block in blocks:
            succ: Set[int | str] = set()
            if not block.successors:
                succ.add(exit_node)
            for candidate in block.successors:
                if candidate.start_offset in offsets:
                    succ.add(candidate.start_offset)
            if self._block_has_exit(block):
                succ.add(exit_node)
            if not succ:
                succ.add(exit_node)
            successors[block.start_offset] = succ

        changed = True
        while changed:
            changed = False
            for offset in offsets:
                succ = successors.get(offset, {exit_node})
                if not succ:
                    succ = {exit_node}
                intersection = set(nodes)
                for candidate in succ:
                    intersection &= postdom.get(candidate, {exit_node})
                updated = {offset} | intersection
                if updated != postdom[offset]:
                    postdom[offset] = updated
                    changed = True

        exit_offsets: Set[int] = set()
        for offset in offsets:
            if exit_node not in successors.get(offset, {exit_node}):
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
                    if candidate in postdom.get(other, set()):
                        dominated = True
                        break
                if not dominated:
                    immediate = candidate
                    break
            if immediate == exit_node or (immediate is None and exit_node in candidates):
                exit_offsets.add(offset)
        return tuple(sorted(exit_offsets))

    @staticmethod
    def _block_has_exit(block: ASTBlock) -> bool:
        for statement in block.statements:
            if isinstance(statement, (ASTReturn, ASTTailCall)):
                return True
        return False

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
        index_note = None
        if dispatch.index is not None:
            parts: List[str] = []
            expr = dispatch.index.source or ""
            if dispatch.index.mask is not None:
                mask_text = f"0x{dispatch.index.mask:04X}"
                expr = f"{expr} & {mask_text}" if expr else f"& {mask_text}"
            expr = expr.strip()
            if expr:
                parts.append(expr)
            if dispatch.index.base is not None:
                parts.append(f"base=0x{dispatch.index.base:04X}")
            if parts:
                index_note = " ".join(parts)
        return ASTSwitch(
            selector=call_expr,
            cases=cases,
            helper=dispatch.helper,
            helper_symbol=dispatch.helper_symbol,
            default=dispatch.default,
            index_note=index_note,
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
