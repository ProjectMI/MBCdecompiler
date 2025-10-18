"""Reconstruct an AST and CFG from the normalised IR programme."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..constants import (
    CALL_SHUFFLE_STANDARD,
    FANOUT_FLAGS_A,
    FANOUT_FLAGS_B,
    IO_PORT_NAME,
    IO_SLOT_ALIASES,
    PAGE_REGISTER,
    RET_MASK,
)
from ..ir.model import (
    IRBlock,
    IRCall,
    IRCallCleanup,
    IRCallPreparation,
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
    IRSlot,
    IRProgram,
    IRReturn,
    IRSegment,
    IRStore,
    IRSwitchDispatch,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
    IRAbiEffect,
    IRStackEffect,
    MemSpace,
    SSAValueKind,
)


_DIRECT_EPILOGUE_KIND_MAP = {
    "call_helpers": "helpers.invoke",
    "fanout": "helpers.fanout",
    "page_register": "frame.page_select",
    "stack_teardown": "frame.teardown",
    "op_6C_01": "frame.page_select",
    "op_08_00": "helpers.dispatch",
    "op_72_23": "helpers.wrapper",
    "op_52_06": "io.handshake",
    "op_0B_00": "io.handshake",
    "op_0B_E1": "io.handshake",
    "op_76_41": "io.bridge",
    "op_04_EC": "io.bridge",
    "op_2D_01": "frame.drop",
}

_MASK_STEP_MNEMONICS = {
    "epilogue",
    "op_29_10",
    "op_31_52",
    "op_32_29",
    "op_4B_0B",
    "op_4B_45",
    "op_4B_CC",
    "op_4F_01",
    "op_4F_02",
    "op_4F_03",
    "op_52_05",
    "op_5E_29",
    "op_70_29",
    "op_72_4A",
}

_MASK_OPERANDS = {RET_MASK, FANOUT_FLAGS_A, FANOUT_FLAGS_B}
_MASK_ALIASES = {"RET_MASK", "FANOUT_FLAGS"}

_FRAME_DROP_MNEMONICS = {
    "op_01_0C",
    "op_01_30",
    "op_01_78",
    "op_01_C0",
    "op_2D_01",
}

_IO_BRIDGE_MNEMONICS = {
    "op_3A_01",
    "op_3E_4B",
    "op_E1_4D",
    "op_E8_4B",
    "op_ED_4B",
    "op_F0_4B",
}

_IO_STEP_MNEMONICS = {
    "op_10_08",
    "op_10_0A",
    "op_10_0E",
    "op_10_18",
    "op_10_3C",
    "op_10_69",
    "op_10_C5",
    "op_10_CC",
    "op_10_CD",
    "op_10_E3",
    "op_10_E5",
    "op_15_4A",
    "op_18_2A",
    "op_20_8C",
    "op_21_94",
    "op_21_AC",
    "op_28_6C",
    "op_52_06",
    "op_78_00",
    "op_89_01",
    "op_A0_03",
    "op_C0_00",
    "op_C3_01",
}

_IO_OPCODE_FALLBACK = {0x10, 0x15, 0x18, 0x20, 0x21, 0x28, 0x52, 0x78, 0x89, 0xA0, 0xC0, 0xC3}
_MASK_OPCODE_FALLBACK = {0x29, 0x31, 0x32, 0x4B, 0x4F, 0x52, 0x5E, 0x70, 0x72}
_DROP_OPCODE_FALLBACK = {0x01, 0x2D}
_BRIDGE_OPCODE_FALLBACK = {0x3A, 0x3E, 0x04, 0x76, 0xE1, 0xE8, 0xED, 0xF0}

_CHATOUT_KIND_MAP = {
    "helpers.dispatch": "io.chatout.dispatch",
    "helpers.invoke": "io.chatout.invoke",
    "helpers.wrapper": "io.chatout.wrapper",
    "helpers.fanout": "io.chatout.mask",
    "io.bridge": "io.chatout.flush",
    "io.handshake": "io.chatout.handshake",
    "io.step": "io.chatout.write",
    "frame.cleanup": "io.chatout.effect",
    "frame.effect": "io.chatout.effect",
    "frame.drop": "io.chatout.effect",
    "frame.teardown": "io.chatout.effect",
}

_CHATOUT_ORDER = (
    "io.chatout.route",
    "io.chatout.handshake",
    "io.chatout.dispatch",
    "io.chatout.invoke",
    "io.chatout.wrapper",
    "io.chatout.mask",
    "io.chatout.write",
    "io.chatout.flush",
    "io.chatout.effect",
)
from .model import (
    ASTAssign,
    ASTBlock,
    ASTBranch,
    ASTCallExpr,
    ASTCallResult,
    ASTCallStatement,
    ASTCallFrame,
    ASTCallFrameSlot,
    ASTComment,
    ASTEnumDecl,
    ASTEnumMember,
    ASTExpression,
    ASTFrameEffect,
    ASTFrameProtocol,
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
    ASTFinally,
    ASTFinallyStep,
    ASTSwitch,
    ASTSwitchCase,
    ASTTailCall,
    ASTTestSet,
    ASTTupleExpr,
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
class _FramePolicySummary:
    """Aggregated frame policy derived from cleanup sequences."""

    teardown: int = 0
    drops: int = 0
    masks: List[Tuple[int, Optional[str]]] = field(default_factory=list)

    def add_mask(self, value: Optional[int], alias: Optional[str]) -> None:
        if value is None:
            return
        entry = (value, alias)
        if entry not in self.masks:
            self.masks.append(entry)

    def merge(self, other: "_FramePolicySummary") -> None:
        self.teardown += other.teardown
        self.drops += other.drops
        for mask in other.masks:
            if mask not in self.masks:
                self.masks.append(mask)

    def has_effects(self) -> bool:
        return bool(self.teardown or self.drops or self.masks)


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


@dataclass
class _EnumInfo:
    """Aggregated information about a helper-backed enumeration."""

    decl: ASTEnumDecl
    member_names: Dict[int, str]
    owner_segment: int
    order: int


class ASTBuilder:
    """Construct a high level AST with CFG and reconstruction metrics."""

    def __init__(self) -> None:
        self._current_analyses: Mapping[int, _BlockAnalysis] = {}
        self._current_entry_reasons: Mapping[int, Tuple[str, ...]] = {}
        self._current_block_labels: Mapping[int, str] = {}
        self._current_exit_hints: Mapping[int, str] = {}
        self._current_redirects: Mapping[int, int] = {}
        self._pending_call_frame: List[IRStackEffect] = []
        self._pending_epilogue: List[IRStackEffect] = []
        self._enum_infos: Dict[str, _EnumInfo] = {}
        self._enum_order: List[str] = []
        self._enum_name_usage: Dict[str, str] = {}
        self._segment_declared_enum_keys: List[str] = []
        self._segment_declared_enum_set: Set[str] = set()
        self._current_segment_index: int = -1
        self._call_arg_values: Dict[str, ASTExpression] = {}
        self._expression_lookup: Dict[str, ASTExpression] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self, program: IRProgram) -> ASTProgram:
        self._enum_infos = {}
        self._enum_order = []
        self._enum_name_usage = {}
        self._call_arg_values = {}
        self._expression_lookup = {}
        segments: List[ASTSegment] = []
        metrics = ASTMetrics()
        for segment in program.segments:
            segment_result = self._build_segment(segment, metrics)
            segments.append(segment_result)
        metrics.procedure_count = sum(len(seg.procedures) for seg in segments)
        metrics.block_count = sum(len(proc.blocks) for seg in segments for proc in seg.procedures)
        metrics.edge_count = sum(len(block.successors) for seg in segments for proc in seg.procedures for block in proc.blocks)
        return ASTProgram(
            segments=tuple(segments),
            metrics=metrics,
            enums=tuple(self._enum_infos[key].decl for key in self._enum_order),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_segment(self, segment: IRSegment, metrics: ASTMetrics) -> ASTSegment:
        self._segment_declared_enum_keys = []
        self._segment_declared_enum_set = set()
        self._current_segment_index = segment.index
        block_map: Dict[int, IRBlock] = {block.start_offset: block for block in segment.blocks}
        analyses = self._build_cfg(segment, block_map)
        entry_reasons = self._detect_entries(segment, block_map, analyses)
        analyses, entry_reasons, redirects = self._compact_cfg(analyses, entry_reasons)
        self._current_analyses = analyses
        self._current_entry_reasons = entry_reasons
        self._current_block_labels = {offset: analysis.block.label for offset, analysis in analyses.items()}
        self._current_exit_hints = {
            offset: self._format_exit_hint(analysis.exit_reasons)
            for offset, analysis in analyses.items()
            if analysis.exit_reasons
        }
        self._current_redirects = redirects
        procedures = self._group_procedures(segment, analyses, entry_reasons, metrics)
        result = ASTSegment(
            index=segment.index,
            start=segment.start,
            length=segment.length,
            procedures=tuple(procedures),
            enums=tuple(self._enum_infos[key].decl for key in self._segment_declared_enum_keys),
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
    ) -> Tuple[
        Mapping[int, _BlockAnalysis],
        Mapping[int, Tuple[str, ...]],
        Mapping[int, int],
    ]:
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
            return analyses, entry_reasons, {}

        def resolve(target: int) -> int:
            seen: Set[int] = set()
            while target in trivial_targets and target not in seen:
                seen.add(target)
                target = trivial_targets[target]
            return target

        redirects = {offset: resolve(target) for offset, target in trivial_targets.items()}

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

        return compacted, updated_entries, redirects

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
            if procedure is None:
                continue
            procedures.append(procedure)
        return procedures

    def _finalise_procedure(
        self,
        name: str,
        entry_offset: int,
        entry_reasons: Tuple[str, ...],
        blocks: Sequence[_PendingBlock],
    ) -> Optional[ASTProcedure]:
        realised_blocks = self._realise_blocks(blocks)
        simplified_blocks = self._simplify_blocks(realised_blocks, entry_offset)
        simplified_blocks = tuple(
            block for block in simplified_blocks if block.statements or block.successors
        )
        if not simplified_blocks:
            return None
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

        for block in list(block_order):
            self._simplify_branch_targets(block, block_order)
            self._simplify_stack_branches(block)

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
                if block.statements:
                    continue
                if len(block.successors) != 1:
                    continue
                successor = block.successors[0]
                if successor is block:
                    continue
                succ_id = id(successor)
                block_id = id(block)
                if succ_id not in predecessors:
                    continue
                if block is entry_block:
                    entry_block = successor
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

    def _simplify_branch_targets(
        self, block: ASTBlock, block_order: Sequence[ASTBlock]
    ) -> None:
        analysis = self._current_analyses.get(block.start_offset)
        fallthrough = analysis.fallthrough if analysis else None
        simplified: List[ASTStatement] = []
        for statement in block.statements:
            if isinstance(statement, BranchStatement):
                updated = self._simplify_branch_statement(
                    statement, block_order, fallthrough
                )
                if updated is None:
                    continue
                statement = updated
            simplified.append(statement)
        block.statements = tuple(simplified)

    def _simplify_branch_statement(
        self,
        statement: BranchStatement,
        block_order: Sequence[ASTBlock],
        fallthrough: Optional[int],
    ) -> Optional[BranchStatement]:
        then_target = self._ensure_branch_target(
            statement, "then", block_order, fallthrough
        )
        else_target = self._ensure_branch_target(
            statement, "else", block_order, fallthrough
        )
        if (
            then_target is not None
            and else_target is not None
            and then_target == else_target
        ):
            return None
        return statement

    def _ensure_branch_target(
        self,
        statement: BranchStatement,
        prefix: str,
        block_order: Sequence[ASTBlock],
        fallthrough: Optional[int],
    ) -> Optional[int]:
        branch_attr = f"{prefix}_branch"
        hint_attr = f"{prefix}_hint"
        offset_attr = f"{prefix}_offset"
        branch = getattr(statement, branch_attr)
        offset = getattr(statement, offset_attr)
        if branch is not None:
            offset = branch.start_offset
        elif offset is None:
            hint = getattr(statement, hint_attr)
            if hint == "fallthrough":
                offset = fallthrough
        target_block = self._find_block_by_offset(block_order, offset)
        if target_block is not None:
            setattr(statement, branch_attr, target_block)
            setattr(statement, hint_attr, None)
            setattr(statement, offset_attr, target_block.start_offset)
            return target_block.start_offset
        setattr(statement, offset_attr, offset)
        return offset

    @staticmethod
    def _find_block_by_offset(
        blocks: Sequence[ASTBlock], offset: Optional[int]
    ) -> Optional[ASTBlock]:
        if offset is None:
            return None
        for candidate in blocks:
            if candidate.start_offset == offset:
                return candidate
        return None

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
        self._pending_call_frame.clear()
        self._pending_epilogue.clear()
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
                    value_state,
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
        statements = self._collapse_dispatch_sequences(statements)
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
        call_expr, arg_exprs = self._convert_call(
            node.target,
            node.args,
            node.symbol,
            node.tail,
            node.varargs if hasattr(node, "varargs") else False,
            value_state,
        )
        self._register_expression(node.describe(), call_expr)
        metrics.call_sites += 1
        metrics.observe_call_args(
            sum(1 for arg in call_expr.args if not isinstance(arg, ASTUnknown)),
            len(call_expr.args),
        )
        frame = self._build_call_frame(node, arg_exprs)
        if getattr(node, "cleanup", tuple()):
            self._pending_epilogue.extend(node.cleanup)
        pending_table = self._pop_dispatch_table(node.target, pending_tables)
        if pending_table is not None:
            if frame is not None:
                insert_index = pending_table.index
                statements.insert(insert_index, frame)
                self._adjust_pending_indices(pending_calls, insert_index)
                self._adjust_table_indices(pending_tables, insert_index)
                target_index = insert_index + 1
            else:
                target_index = pending_table.index
            statements[target_index] = self._build_dispatch_switch(
                call_expr, pending_table.dispatch, value_state
            )
            return
        if frame is not None:
            statements.append(frame)
        index = len(statements)
        statements.append(ASTCallStatement(call=call_expr))
        pending_calls.append(
            _PendingDispatchCall(helper=node.target, index=index, call=call_expr)
        )

    def _handle_dispatch_table(
        self,
        dispatch: IRSwitchDispatch,
        value_state: MutableMapping[str, ASTExpression],
        statements: List[ASTStatement],
        pending_calls: List[_PendingDispatchCall],
        pending_tables: List[_PendingDispatchTable],
    ) -> None:
        call_info = self._pop_dispatch_call(dispatch, pending_calls)
        if call_info is not None:
            switch_statement = self._build_dispatch_switch(
                call_info.call, dispatch, value_state
            )
            statements[call_info.index] = switch_statement
            return
        index = len(statements)
        switch_statement = self._build_dispatch_switch(
            None, dispatch, value_state
        )
        statements.append(switch_statement)
        pending_tables.append(_PendingDispatchTable(dispatch=dispatch, index=index))

    def _dispatch_helper_matches(self, helper: int, dispatch: IRSwitchDispatch) -> bool:
        if dispatch.helper is not None:
            return dispatch.helper == helper
        return False

    def _pop_dispatch_table(
        self, helper: int, pending_tables: List[_PendingDispatchTable]
    ) -> Optional[_PendingDispatchTable]:
        for index in range(len(pending_tables) - 1, -1, -1):
            entry = pending_tables[index]
            if self._dispatch_helper_matches(helper, entry.dispatch):
                pending_tables.pop(index)
                return entry
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

    def _collapse_dispatch_sequences(
        self, statements: List[ASTStatement]
    ) -> List[ASTStatement]:
        collapsed: List[ASTStatement] = []
        index = 0
        while index < len(statements):
            current = statements[index]
            replacements = self._simplify_dispatch_statement(current)
            for replacement in replacements:
                collapsed.append(replacement)
            if (
                replacements
                and isinstance(replacements[-1], (ASTSwitch, ASTCallStatement))
                and index + 1 < len(statements)
                and self._is_redundant_dispatch_followup(
                    replacements[-1], statements[index + 1]
                )
            ):
                index += 1
            index += 1
        return collapsed

    def _simplify_dispatch_statement(
        self, statement: ASTStatement
    ) -> List[ASTStatement]:
        if not isinstance(statement, ASTSwitch):
            return [statement]
        if len(statement.cases) != 1:
            return [statement]
        if statement.call is None:
            return [statement]
        if statement.default is not None:
            return [statement]
        if statement.call.tail:
            return [ASTTailCall(call=statement.call, returns=tuple())]
        return [ASTCallStatement(call=statement.call)]

    def _is_redundant_dispatch_followup(
        self, primary: ASTStatement, statement: ASTStatement
    ) -> bool:
        call_expr = self._extract_dispatch_call(primary)
        if call_expr is None:
            return False
        if isinstance(statement, ASTTailCall):
            if statement.returns:
                return False
            return self._calls_match(statement.call, call_expr)
        if isinstance(statement, ASTCallStatement):
            if statement.returns:
                return False
            return self._calls_match(statement.call, call_expr)
        return False

    @staticmethod
    def _extract_dispatch_call(statement: ASTStatement) -> ASTCallExpr | None:
        if isinstance(statement, ASTSwitch):
            return statement.call
        if isinstance(statement, ASTCallStatement):
            return statement.call
        if isinstance(statement, ASTTailCall):
            return statement.call
        return None

    @staticmethod
    def _calls_match(lhs: ASTCallExpr, rhs: ASTCallExpr) -> bool:
        return (
            lhs.target == rhs.target
            and lhs.symbol == rhs.symbol
            and lhs.args == rhs.args
            and lhs.varargs == rhs.varargs
        )

    def _register_expression(self, token: Optional[str], expr: ASTExpression) -> None:
        if not token:
            return
        self._expression_lookup[token] = expr

    def _build_dispatch_switch(
        self,
        call_expr: ASTCallExpr | None,
        dispatch: IRSwitchDispatch,
        value_state: Mapping[str, ASTExpression],
    ) -> ASTSwitch:
        collapse_dispatch = self._should_collapse_dispatch(call_expr, dispatch)
        enum_decl: ASTEnumDecl | None
        alias_lookup: Mapping[int, str]
        enum_name: str | None
        if collapse_dispatch:
            enum_decl = None
            alias_lookup = {}
            enum_name = None
        else:
            enum_decl, alias_lookup = self._ensure_dispatch_enum(
                dispatch, call_expr.symbol if call_expr else None
            )
            enum_name = enum_decl.name if enum_decl else None
        cases = self._build_dispatch_cases(dispatch.cases, enum_name, alias_lookup)
        index_expr, index_mask, index_base = self._resolve_dispatch_index(
            dispatch, value_state
        )
        kind = self._classify_dispatch_kind(dispatch)
        return ASTSwitch(
            call=call_expr,
            cases=cases,
            helper=dispatch.helper,
            helper_symbol=dispatch.helper_symbol,
            default=dispatch.default,
            index_expr=index_expr,
            index_mask=index_mask,
            index_base=index_base,
            kind=kind,
            enum_name=enum_name,
        )

    @staticmethod
    def _should_collapse_dispatch(
        call_expr: ASTCallExpr | None, dispatch: IRSwitchDispatch
    ) -> bool:
        if call_expr is None:
            return False
        if len(dispatch.cases) != 1:
            return False
        if dispatch.default is not None:
            return False
        return True

    def _resolve_dispatch_index(
        self,
        dispatch: IRSwitchDispatch,
        value_state: Mapping[str, ASTExpression],
    ) -> Tuple[ASTExpression | None, int | None, int | None]:
        index_info = dispatch.index
        if index_info is None:
            return None, None, None
        index_expr: ASTExpression | None = None
        if index_info.source:
            index_expr = self._resolve_expr(index_info.source, value_state)
        return index_expr, index_info.mask, index_info.base

    def _classify_dispatch_kind(self, dispatch: IRSwitchDispatch) -> str | None:
        helper = dispatch.helper
        symbol = dispatch.helper_symbol or ""
        if helper is None and not symbol:
            return None
        io_helper_addresses = {0x00F0, 0x0029, 0x002C, 0x0041, 0x0069}
        if helper in io_helper_addresses:
            return "io"
        lowered = symbol.lower()
        if lowered.startswith("io.") or lowered.startswith("scheduler.mask_"):
            return "io"
        return None

    def _build_dispatch_cases(
        self,
        cases: Sequence[IRDispatchCase],
        enum_name: str | None,
        alias_lookup: Mapping[int, str],
    ) -> Tuple[ASTSwitchCase, ...]:
        return tuple(
            ASTSwitchCase(
                key=case.key,
                target=case.target,
                symbol=case.symbol,
                key_alias=(
                    f"{enum_name}.{alias_lookup[case.key]}"
                    if enum_name and case.key in alias_lookup
                    else None
                ),
            )
            for case in cases
        )

    def _ensure_dispatch_enum(
        self, dispatch: IRSwitchDispatch, call_symbol: str | None
    ) -> Tuple[ASTEnumDecl | None, Mapping[int, str]]:
        key = self._dispatch_helper_key(dispatch, call_symbol)
        if key is None:
            return None, {}
        info = self._enum_infos.get(key)
        if info is None:
            name = self._allocate_enum_name(dispatch, key, call_symbol)
            enum_decl = ASTEnumDecl(name=name, members=tuple())
            info = _EnumInfo(
                decl=enum_decl,
                member_names={},
                owner_segment=self._current_segment_index,
                order=len(self._enum_order),
            )
            self._enum_infos[key] = info
            self._enum_order.append(key)
            self._register_segment_enum(key, info)
        else:
            if info.owner_segment == self._current_segment_index:
                self._register_segment_enum(key, info)
        self._update_enum_members(info, dispatch, call_symbol)
        return info.decl, info.member_names

    def _register_segment_enum(self, key: str, info: _EnumInfo) -> None:
        if info.owner_segment != self._current_segment_index:
            return
        if key in self._segment_declared_enum_set:
            return
        self._segment_declared_enum_set.add(key)
        self._segment_declared_enum_keys.append(key)

    def _update_enum_members(
        self, info: _EnumInfo, dispatch: IRSwitchDispatch, call_symbol: str | None
    ) -> None:
        aliases = self._resolve_enum_aliases(dispatch, call_symbol)
        used_names = set(info.member_names.values())
        changed = False
        for case in sorted(dispatch.cases, key=lambda item: item.key):
            if case.key in info.member_names:
                continue
            name = aliases.get(case.key)
            if not name:
                name = self._format_enum_member_name(case.key)
            name = self._ensure_unique_member_name(name, used_names)
            used_names.add(name)
            info.member_names[case.key] = name
            changed = True
        if changed:
            members = tuple(
                ASTEnumMember(name=name, value=value)
                for value, name in sorted(info.member_names.items())
            )
            info.decl.members = members

    def _dispatch_helper_key(
        self, dispatch: IRSwitchDispatch, call_symbol: str | None
    ) -> str | None:
        symbol = dispatch.helper_symbol or call_symbol
        if symbol:
            return f"symbol:{symbol}"
        helper = dispatch.helper
        if helper is not None:
            return f"helper:0x{helper:04X}"
        return None

    def _allocate_enum_name(
        self, dispatch: IRSwitchDispatch, key: str, call_symbol: str | None
    ) -> str:
        base = self._normalise_enum_name(
            dispatch.helper_symbol or call_symbol, dispatch.helper
        )
        name = base
        existing = self._enum_name_usage.get(name)
        if existing is not None and existing != key:
            helper = dispatch.helper
            if helper is not None:
                name = f"{base}_{helper:04X}"
            else:
                suffix = 2
                candidate = f"{base}_{suffix}"
                while candidate in self._enum_name_usage:
                    suffix += 1
                    candidate = f"{base}_{suffix}"
                name = candidate
        self._enum_name_usage[name] = key
        return name

    def _normalise_enum_name(
        self, symbol: str | None, helper: int | None
    ) -> str:
        if symbol:
            lowered = symbol.lower()
            if lowered.startswith("scheduler.mask_"):
                base = "SchedulerMask"
            elif lowered.startswith("io.") and lowered.endswith("write"):
                parts = [
                    piece.capitalize()
                    for part in symbol.split(".")
                    for piece in part.split("_")
                    if piece
                ]
                base = "".join(parts) or "Dispatch"
                if not base.endswith("Dispatch"):
                    base += "Dispatch"
            else:
                parts = [
                    piece.capitalize()
                    for part in symbol.replace("/", ".").split(".")
                    for piece in part.split("_")
                    if piece
                ]
                base = "".join(parts) or "Dispatch"
            return base
        if helper is not None:
            return f"Dispatch_0x{helper:04X}"
        return "Dispatch"

    def _resolve_enum_aliases(
        self, dispatch: IRSwitchDispatch, call_symbol: str | None
    ) -> Mapping[int, str]:
        symbol = (dispatch.helper_symbol or call_symbol or "").lower()
        if "mask" in symbol:
            return self._build_mask_aliases(dispatch.cases)
        return {}

    def _build_mask_aliases(
        self, cases: Sequence[IRDispatchCase]
    ) -> Mapping[int, str]:
        aliases: Dict[int, str] = {}
        for case in cases:
            value = case.key
            if value == 0:
                aliases[value] = "MaskClear"
                continue
            if value & (value - 1) == 0:
                bit_index = value.bit_length() - 1
                aliases[value] = f"MaskBit{bit_index:02d}"
            else:
                aliases[value] = f"MaskValue_{value:04X}"
        return aliases

    @staticmethod
    def _format_enum_member_name(value: int) -> str:
        return f"K_{value:04X}"

    @staticmethod
    def _ensure_unique_member_name(name: str, used: Set[str]) -> str:
        if name not in used:
            return name
        index = 2
        while True:
            candidate = f"{name}_{index}"
            if candidate not in used:
                return candidate
            index += 1

    def _convert_node(
        self,
        node,
        origin_offset: int,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[List[ASTStatement], List[_BranchLink]]:
        if isinstance(node, IRCallPreparation):
            for mnemonic, operand in node.steps:
                self._pending_call_frame.append(IRStackEffect(mnemonic=mnemonic, operand=operand))
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
            call_expr, arg_exprs = self._convert_call(
                node.target,
                node.args,
                node.symbol,
                node.tail,
                node.varargs,
                value_state,
            )
            self._register_expression(node.describe(), call_expr)
            statements: List[ASTStatement] = []
            metrics.call_sites += 1
            metrics.observe_call_args(
                sum(1 for arg in call_expr.args if not isinstance(arg, ASTUnknown)),
                len(call_expr.args),
            )
            frame = self._build_call_frame(node, arg_exprs)
            if frame is not None:
                statements.append(frame)
            if node.cleanup:
                self._pending_epilogue.extend(node.cleanup)
            return_identifiers = []
            for index, name in enumerate(node.returns):
                identifier = ASTIdentifier(name, self._infer_kind(name))
                value_state[name] = ASTCallResult(call_expr, index)
                metrics.observe_values(int(not isinstance(value_state[name], ASTUnknown)))
                return_identifiers.append(identifier)
            statements.append(ASTCallStatement(call=call_expr, returns=tuple(return_identifiers)))
            return statements, []
        if isinstance(node, IRTailCall):
            call_expr, arg_exprs = self._convert_call(
                node.call.target,
                node.call.args,
                node.call.symbol,
                True,
                node.varargs,
                value_state,
            )
            self._register_expression(node.call.describe(), call_expr)
            self._register_expression(node.describe(), call_expr)
            statements: List[ASTStatement] = []
            metrics.call_sites += 1
            metrics.observe_call_args(
                sum(1 for arg in call_expr.args if not isinstance(arg, ASTUnknown)),
                len(call_expr.args),
            )
            frame = self._build_call_frame(node, arg_exprs)
            if frame is not None:
                statements.append(frame)
            resolved_returns = tuple(self._resolve_expr(name, value_state) for name in node.returns)
            statements.append(ASTTailCall(call=call_expr, returns=resolved_returns))
            self._pending_epilogue.clear()
            return statements, []
        if isinstance(node, IRReturn):
            value = self._build_return_value(node, value_state)
            protocol_stmt, finally_branch = self._build_finally(node.cleanup, node.abi_effects)
            statements: List[ASTStatement] = []
            if protocol_stmt is not None:
                statements.append(protocol_stmt)
            statements.append(
                ASTReturn(value=value, varargs=node.varargs, finally_branch=finally_branch)
            )
            return statements, []
        if isinstance(node, IRCallCleanup):
            if any(step.mnemonic == "stack_shuffle" for step in node.steps):
                self._pending_call_frame.extend(node.steps)
            else:
                self._pending_epilogue.extend(node.steps)
            return [], []
        if isinstance(node, IRTerminator):
            return [], []
        if isinstance(node, IRIf):
            condition = self._resolve_expr(node.condition, value_state)
            then_target = self._resolve_target(node.then_target)
            else_target = self._resolve_target(node.else_target)
            branch = ASTBranch(
                condition=condition,
                then_offset=then_target,
                else_offset=else_target,
            )
            return [branch], [
                _BranchLink(
                    statement=branch,
                    then_target=then_target,
                    else_target=else_target,
                    origin_offset=origin_offset,
                )
            ]
        if isinstance(node, IRTestSetBranch):
            var_expr = self._resolve_expr(node.var, value_state)
            expr = self._resolve_expr(node.expr, value_state)
            then_target = self._resolve_target(node.then_target)
            else_target = self._resolve_target(node.else_target)
            statement = ASTTestSet(
                var=var_expr,
                expr=expr,
                then_offset=then_target,
                else_offset=else_target,
            )
            return [statement], [
                _BranchLink(
                    statement=statement,
                    then_target=then_target,
                    else_target=else_target,
                    origin_offset=origin_offset,
                )
            ]
        if isinstance(node, IRFunctionPrologue):
            var_expr = self._resolve_expr(node.var, value_state)
            expr = self._resolve_expr(node.expr, value_state)
            then_target = self._resolve_target(node.then_target)
            else_target = self._resolve_target(node.else_target)
            statement = ASTFunctionPrologue(
                var=var_expr,
                expr=expr,
                then_offset=then_target,
                else_offset=else_target,
            )
            return [statement], [
                _BranchLink(
                    statement=statement,
                    then_target=then_target,
                    else_target=else_target,
                    origin_offset=origin_offset,
                )
            ]
        if isinstance(node, IRFlagCheck):
            then_target = self._resolve_target(node.then_target)
            else_target = self._resolve_target(node.else_target)
            statement = ASTFlagCheck(
                flag=node.flag,
                then_offset=then_target,
                else_offset=else_target,
            )
            return [statement], [
                _BranchLink(
                    statement=statement,
                    then_target=then_target,
                    else_target=else_target,
                    origin_offset=origin_offset,
                )
            ]
        return [ASTComment(getattr(node, "describe", lambda: repr(node))())], []

    @staticmethod
    def _stack_effect_operand(step: IRStackEffect) -> Optional[int]:
        include_operand = bool(step.operand_role or step.operand_alias)
        if not include_operand:
            include_operand = bool(step.operand) or step.mnemonic not in {"stack_teardown"}
        return step.operand if include_operand else None

    @staticmethod
    def _mnemonic_opcode(mnemonic: str) -> Optional[int]:
        if not mnemonic.startswith("op_"):
            return None
        try:
            return int(mnemonic[3:5], 16)
        except (ValueError, IndexError):
            return None

    def _epilogue_step_kind(self, step: IRStackEffect) -> str:
        mnemonic = step.mnemonic
        alias = step.operand_alias
        operand = step.operand

        direct = _DIRECT_EPILOGUE_KIND_MAP.get(mnemonic)
        if direct is not None:
            return direct

        alias_text = str(alias) if alias is not None else None
        if (
            mnemonic in _MASK_STEP_MNEMONICS
            or operand in _MASK_OPERANDS
            or (alias_text is not None and alias_text in _MASK_ALIASES)
        ):
            return "frame.return_mask"

        opcode = self._mnemonic_opcode(mnemonic)
        if opcode in _MASK_OPCODE_FALLBACK:
            return "frame.return_mask"

        if mnemonic in _FRAME_DROP_MNEMONICS or opcode in _DROP_OPCODE_FALLBACK:
            if step.pops:
                return "frame.drop"

        if mnemonic in _IO_BRIDGE_MNEMONICS or opcode in _BRIDGE_OPCODE_FALLBACK:
            return "io.bridge"

        if (
            mnemonic in _IO_STEP_MNEMONICS
            or opcode in _IO_OPCODE_FALLBACK
            or (alias_text is not None and alias_text == "ChatOut")
        ):
            return "io.step"

        if step.pops and step.mnemonic != "stack_teardown":
            return "frame.drop"

        return "frame.effect"

    def _convert_frame_effect(self, step: IRStackEffect) -> ASTFrameEffect:
        operand = self._stack_effect_operand(step)
        alias = str(step.operand_alias) if step.operand_alias is not None else None
        return ASTFrameEffect(kind=step.mnemonic, operand=operand, alias=alias, pops=step.pops)

    def _convert_epilogue_step(self, step: IRStackEffect) -> ASTFinallyStep:
        operand = self._stack_effect_operand(step)
        alias = str(step.operand_alias) if step.operand_alias is not None else None
        kind = self._epilogue_step_kind(step)
        return ASTFinallyStep(kind=kind, operand=operand, alias=alias, pops=step.pops)

    def _convert_abi_effect(self, effect: IRAbiEffect) -> ASTFinallyStep:
        operand = effect.operand
        alias = str(effect.alias) if effect.alias is not None else None
        if effect.kind == "return_mask":
            return ASTFinallyStep(kind="frame.return_mask", operand=operand, alias=alias)
        return ASTFinallyStep(kind=f"abi.{effect.kind}", operand=operand, alias=alias)

    def _build_call_frame(
        self,
        node: Any,
        arg_exprs: Sequence[ASTExpression],
    ) -> Optional[ASTCallFrame]:
        steps = list(self._pending_call_frame)
        self._pending_call_frame.clear()
        effects = [
            self._convert_frame_effect(step)
            for step in steps
            if step.mnemonic != "stack_shuffle"
        ]
        arity = getattr(node, "arity", None)
        slot_count = arity or len(arg_exprs)
        slots: List[ASTCallFrameSlot] = []
        if slot_count:
            values = list(arg_exprs)
            if len(values) < slot_count:
                deficit = slot_count - len(values)
                values.extend(ASTUnknown(f"slot_{index}") for index in range(len(values), len(values) + deficit))
            for index in range(slot_count):
                slots.append(ASTCallFrameSlot(index=index, value=values[index]))
                token = f"slot_{index}"
                value = values[index]
                if not isinstance(value, ASTUnknown):
                    self._call_arg_values[token] = value
        live_mask = getattr(node, "cleanup_mask", None)
        if not slots and not effects and live_mask is None:
            return None
        return ASTCallFrame(slots=tuple(slots), effects=tuple(effects), live_mask=live_mask)

    def _build_return_value(
        self,
        node: IRReturn,
        value_state: Mapping[str, ASTExpression],
    ) -> Optional[ASTExpression]:
        values = [self._resolve_expr(name, value_state) for name in node.values]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return ASTTupleExpr(tuple(values))

    @staticmethod
    def _deduplicate_finally_steps(steps: Sequence[ASTFinallyStep]) -> List[ASTFinallyStep]:
        if not steps:
            return []
        aggregated: List[ASTFinallyStep] = []
        index_map: Dict[Tuple[str, Optional[int], Optional[str]], int] = {}
        for step in steps:
            key = (step.kind, step.operand, step.alias)
            if key in index_map:
                idx = index_map[key]
                existing = aggregated[idx]
                aggregated[idx] = ASTFinallyStep(
                    kind=existing.kind,
                    operand=existing.operand,
                    alias=existing.alias,
                    pops=existing.pops + step.pops,
                )
            else:
                index_map[key] = len(aggregated)
                aggregated.append(step)
        return aggregated

    def _normalize_finally_step(self, step: ASTFinallyStep) -> Optional[ASTFinallyStep]:
        operand = step.operand
        alias = step.alias
        kind = step.kind

        if kind.startswith("io.chatout"):
            return step

        if kind == "frame.page_select" and operand == PAGE_REGISTER:
            return ASTFinallyStep(kind="io.chatout.route", operand=None, alias=IO_PORT_NAME, pops=step.pops)

        if alias is None and operand is not None and operand in IO_SLOT_ALIASES:
            alias = IO_PORT_NAME

        if alias == IO_PORT_NAME:
            mapped_kind = _CHATOUT_KIND_MAP.get(kind, "io.chatout.effect")
            return ASTFinallyStep(kind=mapped_kind, operand=None, alias=IO_PORT_NAME, pops=step.pops)

        if operand == CALL_SHUFFLE_STANDARD or alias == "CALL_SHUFFLE_STD":
            mapped_kind = _CHATOUT_KIND_MAP.get(kind)
            if mapped_kind is None:
                mapped_kind = "io.chatout.flush" if kind == "io.bridge" else "io.chatout.effect"
            return ASTFinallyStep(kind=mapped_kind, operand=None, alias=IO_PORT_NAME, pops=step.pops)

        return step

    @staticmethod
    def _is_chatout_step(step: ASTFinallyStep) -> bool:
        if step.kind.startswith("io.chatout"):
            return True
        if step.alias == IO_PORT_NAME:
            return True
        operand = step.operand
        if operand is not None and operand in IO_SLOT_ALIASES:
            return True
        if operand == CALL_SHUFFLE_STANDARD:
            return True
        return False

    def _summarize_chatout_steps(self, steps: Sequence[ASTFinallyStep]) -> List[ASTFinallyStep]:
        if not steps:
            return []

        merged: Dict[str, ASTFinallyStep] = {}
        for step in steps:
            rewritten = self._normalize_finally_step(step)
            if rewritten is None:
                continue
            if not rewritten.kind.startswith("io.chatout"):
                # Re-run classification with chatout defaults.
                rewritten = ASTFinallyStep(
                    kind=_CHATOUT_KIND_MAP.get(rewritten.kind, "io.chatout.effect"),
                    operand=None,
                    alias=IO_PORT_NAME,
                    pops=rewritten.pops,
                )
            existing = merged.get(rewritten.kind)
            if existing is None:
                merged[rewritten.kind] = rewritten
            else:
                merged[rewritten.kind] = ASTFinallyStep(
                    kind=existing.kind,
                    operand=existing.operand,
                    alias=existing.alias,
                    pops=existing.pops + rewritten.pops,
                )

        ordered: List[ASTFinallyStep] = []
        for kind in _CHATOUT_ORDER:
            entry = merged.pop(kind, None)
            if entry is not None:
                ordered.append(entry)
        ordered.extend(sorted(merged.values(), key=lambda step: step.kind))
        return ordered

    def _aggregate_finally_steps(self, steps: Sequence[ASTFinallyStep]) -> List[ASTFinallyStep]:
        if not steps:
            return []

        normalized: List[ASTFinallyStep] = []
        chatout: List[ASTFinallyStep] = []
        for step in steps:
            rewritten = self._normalize_finally_step(step)
            if rewritten is None:
                continue
            if self._is_chatout_step(rewritten):
                chatout.append(rewritten)
            else:
                normalized.append(rewritten)

        normalized.extend(self._summarize_chatout_steps(chatout))
        return self._deduplicate_finally_steps(normalized)

    @staticmethod
    def _build_frame_protocol(policy: _FramePolicySummary) -> ASTFrameProtocol:
        return ASTFrameProtocol(
            masks=tuple(policy.masks),
            teardown=policy.teardown,
            drops=policy.drops,
        )

    def _build_finally(
        self,
        cleanup_steps: Sequence[IRStackEffect],
        abi_effects: Sequence[IRAbiEffect],
    ) -> Tuple[Optional[ASTFrameProtocol], Optional[ASTFinally]]:
        combined = list(self._pending_epilogue)
        combined.extend(cleanup_steps)
        self._pending_epilogue = []

        policy = _FramePolicySummary()
        effect_steps: List[ASTFinallyStep] = []

        for step in combined:
            kind = self._epilogue_step_kind(step)
            operand = self._stack_effect_operand(step)
            alias = str(step.operand_alias) if step.operand_alias is not None else None
            if kind == "frame.teardown":
                policy.teardown += step.pops
                continue
            if kind == "frame.drop":
                policy.drops += step.pops or 1
                continue
            if kind == "frame.return_mask":
                policy.add_mask(operand, alias)
                continue
            effect_steps.append(ASTFinallyStep(kind=kind, operand=operand, alias=alias, pops=step.pops))

        for effect in abi_effects:
            if effect.kind == "return_mask":
                alias = str(effect.alias) if effect.alias is not None else None
                policy.add_mask(effect.operand, alias)
                continue
            effect_steps.append(self._convert_abi_effect(effect))

        aggregated_steps = self._aggregate_finally_steps(effect_steps)
        protocol_stmt = self._build_frame_protocol(policy) if policy.has_effects() else None
        finally_stmt = ASTFinally(steps=tuple(aggregated_steps)) if aggregated_steps else None
        return protocol_stmt, finally_stmt

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
        for token, expr in zip(args, arg_exprs):
            if token and not isinstance(expr, ASTUnknown):
                self._call_arg_values[token] = expr
        return call_expr, arg_exprs

    def _resolve_expr(self, token: Optional[str], value_state: Mapping[str, ASTExpression]) -> ASTExpression:
        if not token:
            return ASTUnknown("")
        if token in value_state:
            return value_state[token]
        if token in self._call_arg_values:
            return self._call_arg_values[token]
        if token in self._expression_lookup:
            return self._expression_lookup[token]
        if " & " in token:
            head, mask = token.rsplit(" & ", 1)
            if self._is_hex_literal(mask.strip()):
                return self._resolve_expr(head.strip(), value_state)
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
            return ASTSlotRef(self._build_slot(index))
        return ASTIdentifier(token, self._infer_kind(token))

    @staticmethod
    def _build_slot(index: int) -> IRSlot:
        if index < 0x1000:
            space = MemSpace.FRAME
        elif index < 0x8000:
            space = MemSpace.GLOBAL
        else:
            space = MemSpace.CONST
        return IRSlot(space=space, index=index)

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

    def _resolve_target(self, target: int) -> int:
        redirects = self._current_redirects
        seen: Set[int] = set()
        while target in redirects and target not in seen:
            seen.add(target)
            target = redirects[target]
        return target

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
        self._current_redirects = {}
        self._pending_call_frame = []
        self._pending_epilogue = []
        self._segment_declared_enum_keys = []
        self._segment_declared_enum_set = set()
        self._current_segment_index = -1
        self._call_arg_values = {}
        self._expression_lookup = {}


__all__ = ["ASTBuilder"]
