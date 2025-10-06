"""Normalisation pipeline that converts raw instructions into IR nodes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from ..constants import (
    CALL_SHUFFLE_STANDARD,
    IO_PORT_NAME,
    IO_SLOT,
    MEMORY_BANK_ALIASES,
    MEMORY_PAGE_ALIASES,
    RET_MASK,
)
from ..analyzer.instruction_profile import InstructionKind, InstructionProfile
from ..analyzer.stack import StackEvent, StackTracker, StackValueType
from ..instruction import read_instructions
from ..knowledge import CallSignature, CallSignatureEffect, CallSignaturePattern, KnowledgeBase
from ..mbc import MbcContainer, Segment
from .model import (
    IRAsciiFinalize,
    IRAsciiHeader,
    IRAsciiPreamble,
    IRAsciiWrapperCall,
    IRBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    CallPredicate,
    IRCallCleanup,
    IRCallPreparation,
    IRCallReturn,
    IRTailcallReturn,
    IRConditionMask,
    IRFlagCheck,
    IRFunctionPrologue,
    IRLiteral,
    IRLiteralBlock,
    IRLiteralChunk,
    IRLoad,
    IRNode,
    IRProgram,
    IRRaw,
    IRReturn,
    IRSegment,
    IRSlot,
    MemRef,
    IRStackEffect,
    IRStore,
    IRStackDuplicate,
    IRStackDrop,
    IRTailcallAscii,
    IRTablePatch,
    IRTailcallFrame,
    IRTestSetBranch,
    IRIf,
    IRIndirectLoad,
    IRIndirectStore,
    IRIORead,
    IRIOWrite,
    MemSpace,
    SSAValueKind,
    NormalizerMetrics,
)


ANNOTATION_MNEMONICS = {"literal_marker"}
RETURN_NIBBLE_MODES = {0x29, 0x2C, 0x32, 0x41, 0x65, 0x69, 0x6C}


CALL_PREPARATION_PREFIXES = {"stack_shuffle", "fanout"}
CALL_CLEANUP_MNEMONICS = {"call_helpers", "op_32_29", "op_52_05", "op_05_00", "stack_shuffle", "fanout"}
CALL_CLEANUP_PREFIXES = ("stack_teardown_", "op_4A_")
CALL_PREDICATE_SKIP_MNEMONICS = {"op_29_10", "op_70_29", "op_0B_29", "op_06_66"}


LITERAL_MARKER_HINTS: Dict[int, str] = {
    0x0067: "literal_hint",
    0x6704: "literal_hint",
    0x0400: "literal_hint",
    0x0110: "literal_hint",
}


IO_READ_MNEMONICS = {"op_10_38"}
IO_WRITE_MNEMONICS = {"op_10_24", "op_10_48"}
IO_ACCEPTED_OPERANDS = {0, IO_SLOT}
IO_HANDSHAKE_MNEMONICS = {"op_3D_30", "op_31_30"}
IO_BRIDGE_MNEMONICS = {"op_01_3D", "op_F1_3D", "op_38_00", "op_4C_00"}
IO_BRIDGE_NODE_TYPES = (
    IRLiteral,
    IRLiteralChunk,
    IRCall,
    IRLoad,
    IRStore,
    IRIndirectLoad,
    IRIndirectStore,
    IRTablePatch,
)


@dataclass(frozen=True)
class RawInstruction:
    """Wrapper that couples a profile with stack tracking details."""

    profile: InstructionProfile
    event: StackEvent
    annotations: Tuple[str, ...]
    ssa_values: Tuple[str, ...]
    ssa_kinds: Tuple[SSAValueKind, ...]

    @property
    def mnemonic(self) -> str:
        return self.profile.mnemonic

    @property
    def operand(self) -> int:
        return self.profile.word.operand

    @property
    def offset(self) -> int:
        return self.profile.word.offset

    def pushes_value(self) -> bool:
        return bool(self.event.pushed_types)

    def describe_source(self) -> str:
        return f"{self.profile.label}@0x{self.offset:06X}"


@dataclass(frozen=True)
class RawBlock:
    """Sequence of executable instructions terminated by control flow."""

    index: int
    start_offset: int
    instructions: Tuple[RawInstruction, ...]


class _ItemList:
    """Mutable wrapper around a block during normalisation passes."""

    def __init__(self, items: Sequence[Union[RawInstruction, IRNode]]):
        self._items: List[Union[RawInstruction, IRNode]] = list(items)

    def __iter__(self) -> Iterator[Union[RawInstruction, IRNode]]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Union[RawInstruction, IRNode]:
        return self._items[index]

    def replace_slice(
        self, start: int, end: int, replacement: Sequence[Union[RawInstruction, IRNode]]
    ) -> None:
        self._items[start:end] = list(replacement)

    def insert(self, index: int, value: Union[RawInstruction, IRNode]) -> None:
        self._items.insert(index, value)

    def pop(self, index: int) -> Union[RawInstruction, IRNode]:
        return self._items.pop(index)

    def to_tuple(self) -> Tuple[Union[RawInstruction, IRNode], ...]:
        return tuple(self._items)


class IRNormalizer:
    """Drive the multi-pass IR normalisation pipeline."""

    _SSA_PREFIX = {
        SSAValueKind.UNKNOWN: "ssa",
        SSAValueKind.INTEGER: "int",
        SSAValueKind.ADDRESS: "addr",
        SSAValueKind.POINTER: "ptr",
        SSAValueKind.BOOLEAN: "bool",
        SSAValueKind.IDENTIFIER: "id",
    }

    _SSA_PRIORITY = {
        SSAValueKind.UNKNOWN: 0,
        SSAValueKind.INTEGER: 1,
        SSAValueKind.IDENTIFIER: 1,
        SSAValueKind.ADDRESS: 2,
        SSAValueKind.POINTER: 3,
        SSAValueKind.BOOLEAN: 4,
    }

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge
        self._annotation_offsets: Set[int] = set()
        self._ssa_bindings: Dict[int, Tuple[str, ...]] = {}
        self._ssa_types: Dict[str, SSAValueKind] = {}
        self._ssa_aliases: Dict[str, str] = {}
        self._ssa_counters: Dict[str, int] = defaultdict(int)
        self._memref_regions: Dict[int, str] = {}
        self._memref_symbols: Dict[Tuple[str, Optional[int], Optional[int], Optional[int]], str] = {}
        self._memref_symbol_counters: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def normalise_container(
        self,
        container: MbcContainer,
        *,
        segment_indices: Optional[Sequence[int]] = None,
    ) -> IRProgram:
        segments: List[IRSegment] = []
        aggregate_metrics = NormalizerMetrics()
        selection = set(segment_indices or [])

        for segment in container.segments():
            if selection and segment.index not in selection:
                continue
            normalised = self.normalise_segment(segment)
            segments.append(normalised)
            aggregate_metrics.observe(normalised.metrics)

        return IRProgram(segments=tuple(segments), metrics=aggregate_metrics)

    def normalise_segment(self, segment: Segment) -> IRSegment:
        raw_blocks = self._parse_segment(segment)
        blocks: List[IRBlock] = []
        metrics = NormalizerMetrics()

        for block in raw_blocks:
            ir_block, block_metrics = self._normalise_block(block)
            blocks.append(ir_block)
            metrics.observe(block_metrics)

        return IRSegment(
            index=segment.index,
            start=segment.start,
            length=segment.length,
            blocks=tuple(blocks),
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # parsing helpers
    # ------------------------------------------------------------------
    def _parse_segment(self, segment: Segment) -> Tuple[RawBlock, ...]:
        instructions, _ = read_instructions(segment.data, segment.start)
        if not instructions:
            return tuple()

        profiles = [InstructionProfile.from_word(word, self.knowledge) for word in instructions]
        executable: List[InstructionProfile] = []
        annotations: Dict[int, List[str]] = {}
        pending: List[str] = []

        for profile in profiles:
            if profile.mnemonic in ANNOTATION_MNEMONICS or (
                profile.is_literal_marker() and profile.mnemonic.startswith("op_")
            ):
                pending.append(profile.mnemonic)
                continue
            index = len(executable)
            if pending:
                annotations.setdefault(index, []).extend(pending)
                pending.clear()
            executable.append(profile)

        tracker = StackTracker()
        events = tracker.process_sequence(executable)

        raw_instructions: List[RawInstruction] = []
        stack_names: List[str] = []
        next_ssa = 0
        forced_push_kinds = {InstructionKind.REDUCE}
        for index, (profile, event) in enumerate(zip(executable, events)):
            notes = tuple(annotations.get(index, ()))
            pops = len(event.popped_types)
            if pops:
                del stack_names[-pops:]
            pushes = len(event.pushed_types)
            if pushes == 0 and profile.kind in forced_push_kinds:
                baseline = max(event.depth_before - pops, 0)
                inferred = event.depth_after - baseline
                pushes = max(inferred, 1)
            pushed_names: List[str] = []
            pushed_kinds: List[SSAValueKind] = []
            for _ in range(pushes):
                name = f"ssa{next_ssa}"
                next_ssa += 1
                pushed_names.append(name)
            if event.pushed_types:
                for value_type in event.pushed_types[:pushes]:
                    pushed_kinds.append(self._map_stack_type(value_type))
            while len(pushed_kinds) < len(pushed_names):
                pushed_kinds.append(SSAValueKind.UNKNOWN)
            if pushed_names:
                stack_names.extend(pushed_names)
            raw_instructions.append(
                RawInstruction(
                    profile=profile,
                    event=event,
                    annotations=notes,
                    ssa_values=tuple(pushed_names),
                    ssa_kinds=tuple(pushed_kinds),
                )
            )

        blocks: List[RawBlock] = []
        current: List[RawInstruction] = []
        block_index = 0
        block_start = raw_instructions[0].offset if raw_instructions else segment.start

        for instruction in raw_instructions:
            if not current:
                block_start = instruction.offset
            current.append(instruction)
            if self._is_block_terminator(instruction):
                blocks.append(
                    RawBlock(index=block_index, start_offset=block_start, instructions=tuple(current))
                )
                block_index += 1
                current = []

        if current:
            blocks.append(
                RawBlock(index=block_index, start_offset=block_start, instructions=tuple(current))
            )

        return tuple(blocks)

    @staticmethod
    def _is_block_terminator(instruction: RawInstruction) -> bool:
        profile = instruction.profile
        if profile.kind in {
            InstructionKind.BRANCH,
            InstructionKind.RETURN,
            InstructionKind.TERMINATOR,
            InstructionKind.TAILCALL,
        }:
            return True
        control = (profile.control_flow or "").lower()
        return any(token in control for token in {"return", "jump", "stop"})

    # ------------------------------------------------------------------
    # normalisation passes
    # ------------------------------------------------------------------
    def _normalise_block(self, block: RawBlock) -> Tuple[IRBlock, NormalizerMetrics]:
        self._annotation_offsets.clear()
        self._ssa_bindings.clear()
        self._ssa_types.clear()
        self._ssa_aliases.clear()
        self._ssa_counters.clear()
        items = _ItemList(block.instructions)
        metrics = NormalizerMetrics()

        self._pass_literals(items, metrics)
        self._pass_ascii_runs(items, metrics)
        self._pass_stack_manipulation(items, metrics)
        self._pass_calls_and_returns(items, metrics)
        self._pass_aggregates(items, metrics)
        self._pass_literal_blocks(items)
        self._pass_literal_block_reducers(items, metrics)
        self._pass_ascii_preamble(items)
        self._pass_call_preparation(items)
        self._pass_call_cleanup(items)
        self._pass_io_operations(items)
        self._pass_call_conventions(items)
        self._pass_tailcall_frames(items)
        self._pass_table_patches(items)
        self._pass_ascii_finalize(items)
        self._pass_assign_ssa_names(items)
        self._pass_branches(items, metrics)
        self._pass_flag_checks(items)
        self._pass_function_prologues(items)
        self._pass_ascii_wrappers(items)
        self._pass_ascii_headers(items)
        self._pass_call_contracts(items)
        self._pass_call_return_templates(items)
        self._pass_condition_masks(items)
        self._pass_call_predicates(items)
        self._pass_indirect_access(items, metrics)

        nodes: List[IRNode] = []
        block_annotations: List[str] = []
        for item in items:
            if isinstance(item, RawInstruction):
                if self._is_annotation_only(item):
                    annotation = self._format_annotation(item)
                    if annotation:
                        block_annotations.append(annotation)
                    continue
                metrics.raw_remaining += 1
                nodes.append(
                    IRRaw(
                        mnemonic=item.mnemonic,
                        operand=item.operand,
                        operand_role=item.profile.operand_role(),
                        operand_alias=item.profile.operand_alias(),
                        annotations=item.annotations,
                    )
                )
            else:
                nodes.append(item)

        ir_block = IRBlock(
            label=f"block_{block.index}",
            start_offset=block.start_offset,
            nodes=tuple(nodes),
            annotations=tuple(block_annotations),
        )
        return ir_block, metrics

    # ------------------------------------------------------------------
    # SSA helpers
    # ------------------------------------------------------------------
    def _record_ssa(
        self,
        item: Union[RawInstruction, IRNode],
        values: Sequence[str],
        *,
        kinds: Optional[Sequence[SSAValueKind]] = None,
    ) -> None:
        if values:
            stored = tuple(values)
            self._ssa_bindings[id(item)] = stored
            if kinds:
                for name, kind in zip(stored, kinds):
                    self._set_ssa_kind(name, kind)
        else:
            self._ssa_bindings.pop(id(item), None)

    def _transfer_ssa(
        self,
        source: Union[RawInstruction, IRNode],
        target: Union[RawInstruction, IRNode],
    ) -> None:
        values: Sequence[str] = ()
        if isinstance(source, RawInstruction):
            values = source.ssa_values
        values = self._ssa_bindings.get(id(source), values)
        if values:
            self._record_ssa(target, values)
        self._ssa_bindings.pop(id(source), None)

    def _ssa_value(
        self,
        item: Union[RawInstruction, IRNode],
        index: int = -1,
        *,
        raw: bool = False,
    ) -> Optional[str]:
        values = self._ssa_bindings.get(id(item))
        if not values:
            return None
        if index < 0:
            index += len(values)
        if not (0 <= index < len(values)):
            return None
        name = values[index]
        if raw:
            return name
        return self._render_ssa(name)

    def _render_ssa(self, name: str) -> str:
        kind = self._ssa_types.get(name, SSAValueKind.UNKNOWN)
        prefix = self._SSA_PREFIX.get(kind, "ssa")
        alias = self._ssa_aliases.get(name)
        if alias and alias.startswith(prefix):
            return alias
        alias = f"{prefix}{self._ssa_counters[prefix]}"
        self._ssa_counters[prefix] += 1
        self._ssa_aliases[name] = alias
        return alias

    def _set_ssa_kind(self, name: str, kind: SSAValueKind) -> None:
        current = self._ssa_types.get(name)
        if current is None or self._SSA_PRIORITY.get(kind, 0) > self._SSA_PRIORITY.get(current, 0):
            self._ssa_types[name] = kind
            self._ssa_aliases.pop(name, None)

    def _promote_ssa_kind(self, name: Optional[str], kind: SSAValueKind) -> None:
        if not name:
            return
        current = self._ssa_types.get(name)
        if current is None or self._SSA_PRIORITY.get(kind, 0) > self._SSA_PRIORITY.get(current, 0):
            self._ssa_types[name] = kind
            self._ssa_aliases.pop(name, None)

    def _map_stack_type(self, value_type: StackValueType) -> SSAValueKind:
        if value_type is StackValueType.SLOT:
            return SSAValueKind.POINTER
        if value_type is StackValueType.NUMBER:
            return SSAValueKind.INTEGER
        if value_type is StackValueType.IDENTIFIER:
            return SSAValueKind.IDENTIFIER
        return SSAValueKind.UNKNOWN

    def _stack_sources(
        self, items: _ItemList, index: int, count: int
    ) -> List[Tuple[str, Union[RawInstruction, IRNode]]]:
        sources: List[Tuple[str, Union[RawInstruction, IRNode]]] = []
        scan = index - 1
        while scan >= 0 and len(sources) < count:
            candidate = items[scan]
            values = self._ssa_bindings.get(id(candidate))
            if values:
                for name in reversed(values):
                    sources.append((name, candidate))
                    if len(sources) == count:
                        break
            scan -= 1
        return sources

    # ------------------------------------------------------------------
    # individual pass implementations
    # ------------------------------------------------------------------
    def _pass_literals(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            literal = self._literal_from_instruction(item)
            if literal is None:
                index += 1
                continue

            if isinstance(literal, IRLiteral):
                metrics.literals += 1
            elif isinstance(literal, IRLiteralChunk):
                metrics.literal_chunks += 1

            self._transfer_ssa(item, literal)
            items.replace_slice(index, index + 1, [literal])
            index += 1

    def _pass_stack_manipulation(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            mnemonic = item.mnemonic
            if mnemonic == "op_02_66":
                value = self._describe_stack_top(items, index)
                copies = max(1, item.event.depth_after - item.event.depth_before)
                node = IRStackDuplicate(value=value, copies=copies + 1)
                self._transfer_ssa(item, node)
                items.replace_slice(index, index + 1, [node])
                continue

            if mnemonic == "op_01_66":
                value = self._describe_stack_top(items, index)
                node = IRStackDrop(value=value)
                items.replace_slice(index, index + 1, [node])
                continue

            if mnemonic == "op_03_66":
                slot = self._classify_slot(item.operand)
                node = IRLoad(slot=slot)
                self._transfer_ssa(item, node)
                items.replace_slice(index, index + 1, [node])
                metrics.loads += 1
                continue

            index += 1

    def _literal_from_instruction(self, instruction: RawInstruction) -> Optional[IRNode]:
        profile = instruction.profile

        if profile.is_literal_marker():
            self._annotation_offsets.add(instruction.offset)
            hint = LITERAL_MARKER_HINTS.get(instruction.operand)
            if hint is not None:
                return IRLiteral(
                    value=instruction.operand,
                    mode=profile.mode,
                    source=hint,
                    annotations=instruction.annotations,
                )
            return None

        if profile.kind is InstructionKind.ASCII_CHUNK or profile.mnemonic.startswith(
            "inline_ascii_chunk"
        ):
            data = instruction.profile.word.raw.to_bytes(4, "big")
            return IRLiteralChunk(
                data=data,
                source=profile.mnemonic,
                annotations=instruction.annotations,
            )

        if profile.kind is InstructionKind.LITERAL and instruction.pushes_value():
            return IRLiteral(
                value=instruction.operand,
                mode=profile.mode,
                source=profile.mnemonic,
                annotations=instruction.annotations,
            )

        return None

    def _pass_calls_and_returns(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            mnemonic = item.mnemonic
            if mnemonic in {"call_dispatch", "tailcall_dispatch"}:
                args, start = self._collect_call_arguments(items, index)
                symbol = self.knowledge.lookup_address(item.operand)
                call = IRCall(
                    target=item.operand,
                    args=tuple(args),
                    tail=mnemonic == "tailcall_dispatch",
                    symbol=symbol,
                )
                metrics.calls += 1
                if call.tail:
                    metrics.tail_calls += 1
                items.replace_slice(start, index + 1, [call])
                index = start
                if call.tail:
                    self._collapse_tail_return(items, index, metrics)
                continue

            if mnemonic == "return_values":
                count, varargs = self._resolve_return_signature(items, index)
                values = tuple(f"ret{i}" for i in range(count)) if count else tuple()
                if varargs and not values:
                    values = ("ret*",)
                items.replace_slice(index, index + 1, [IRReturn(values=values, varargs=varargs)])
                metrics.returns += 1
                continue

            index += 1

    def _pass_ascii_runs(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            start = index
            data_parts: List[bytes] = []
            annotations: List[str] = []

            while index < len(items):
                candidate = items[index]
                if isinstance(candidate, IRLiteralChunk):
                    data_parts.append(candidate.data)
                    annotations.extend(candidate.annotations)
                    index += 1
                    continue

                break

            if not data_parts:
                index += 1
                continue

            if len(data_parts) <= 1:
                index = start + 1
                continue

            chunk = IRLiteralChunk(
                data=b"".join(data_parts),
                source="ascii_run",
                annotations=tuple(annotations),
            )
            items.replace_slice(start, index, [chunk])
            index = start + 1

    def _collect_call_arguments(
        self, items: _ItemList, call_index: int
    ) -> Tuple[List[str], int]:
        args: List[str] = []
        start = call_index
        scan = call_index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, (IRLiteral, IRLiteralChunk)):
                name = self._ssa_value(candidate)
                args.append(name or candidate.describe())
                scan -= 1
                continue
            if isinstance(candidate, IRStackDuplicate):
                args.append(candidate.value)
                scan -= 1
                continue
            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    args.append(self._describe_value(candidate))
                    scan -= 1
                    continue
                break
            else:
                break
        args.reverse()
        start = scan + 1
        return args, start

    def _collapse_tail_return(self, items: _ItemList, call_index: int, metrics: NormalizerMetrics) -> None:
        index = call_index + 1
        collected_cleanup: List[IRStackEffect] = []
        while index < len(items):
            item = items[index]
            if isinstance(item, RawInstruction) and item.mnemonic == "return_values":
                count, varargs = self._resolve_return_signature(items, index)
                values = tuple(f"ret{i}" for i in range(count)) if count else tuple()
                if varargs and not values:
                    values = ("ret*",)
                items.replace_slice(
                    index,
                    index + 1,
                    [IRReturn(values=values, varargs=varargs, cleanup=tuple(collected_cleanup))],
                )
                metrics.returns += 1
                return
            if isinstance(item, RawInstruction) and item.profile.kind in {
                InstructionKind.STACK_TEARDOWN,
                InstructionKind.META,
            }:
                if item.profile.kind is InstructionKind.STACK_TEARDOWN:
                    collected_cleanup.append(self._call_cleanup_effect(item))
                items.pop(index)
                continue
            break

    def _resolve_return_signature(self, items: _ItemList, index: int) -> Tuple[int, bool]:
        instruction = items[index]
        assert isinstance(instruction, RawInstruction)

        operand = instruction.operand
        lo = operand & 0xFF
        hi = (operand >> 8) & 0xFF
        mode = instruction.profile.mode
        nibble = lo & 0x0F

        if mode in RETURN_NIBBLE_MODES:
            if nibble:
                return nibble, False
            hint = self._stack_teardown_hint(items, index)
            if hint is not None:
                return hint, False
            base = hi & 0x1F
            if base:
                return base, False
            return 0, True

        if lo:
            if lo > 0x3F:
                narrowed = lo & 0x0F
                if narrowed:
                    return narrowed, False
            return lo, False
        if hi:
            return hi, False

        hint = self._stack_teardown_hint(items, index)
        if hint is not None:
            return hint, False

        return 1, False

    def _stack_teardown_hint(self, items: _ItemList, index: int) -> Optional[int]:
        scan = index - 1
        total = 0
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                kind = candidate.profile.kind
                if kind is InstructionKind.STACK_TEARDOWN:
                    delta = -candidate.event.delta
                    if delta > 0:
                        total += delta
                    scan -= 1
                    continue
                if kind in {InstructionKind.RETURN, InstructionKind.BRANCH, InstructionKind.TERMINATOR}:
                    break
            else:
                break
            scan -= 1
        if total:
            return total
        return None

    def _pass_aggregates(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            literal = self._literal_at(items, index)
            if literal is None:
                index += 1
                continue

            literal_nodes: List[IRLiteral] = [literal]
            reducers: List[RawInstruction] = []
            spacers: List[RawInstruction] = []
            scan = index + 1
            added_literal_since_reduce = True

            while scan < len(items):
                candidate_literal = self._literal_at(items, scan)
                if candidate_literal is not None:
                    literal_nodes.append(candidate_literal)
                    scan += 1
                    added_literal_since_reduce = True
                    continue

                candidate = items[scan]
                if isinstance(candidate, RawInstruction) and self._is_annotation_only(candidate):
                    spacers.append(candidate)
                    scan += 1
                    continue

                if isinstance(candidate, RawInstruction) and candidate.mnemonic.startswith("reduce"):
                    if not added_literal_since_reduce:
                        self._annotation_offsets.add(candidate.offset)
                        spacers.append(candidate)
                        scan += 1
                        continue
                    reducers.append(candidate)
                    scan += 1
                    added_literal_since_reduce = False
                    continue

                break

            if not reducers:
                index += 1
                continue

            literals = [self._literal_repr(node) for node in literal_nodes]
            replacement: IRNode
            if len(literals) >= 2 and len(literals) == 2 * len(reducers):
                entries = []
                for pos in range(0, len(literals), 2):
                    entries.append((literals[pos], literals[pos + 1]))
                replacement = IRBuildMap(entries=tuple(entries))
            elif len(literals) == len(reducers) + 1:
                replacement = IRBuildArray(elements=tuple(literals))
            else:
                replacement = IRBuildTuple(elements=tuple(literals))

            metrics.aggregates += 1
            metrics.reduce_replaced += len(reducers)
            replacement_sequence: List[Union[RawInstruction, IRNode]] = [replacement]
            if reducers:
                self._transfer_ssa(reducers[-1], replacement)
            if spacers:
                replacement_sequence.extend(spacers)
            items.replace_slice(index, scan, replacement_sequence)
            index += 1

    def _pass_literal_blocks(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRBuildTuple):
                values = self._parse_literal_list(item.elements)
                block = self._literal_block_from_values(values)
                if block is not None:
                    items.replace_slice(index, index + 1, [block])
                    continue
            if isinstance(item, IRLiteral):
                values, end = self._collect_literal_block_literals(items, index)
                block = self._literal_block_from_values(values)
                if block is not None:
                    items.replace_slice(index, end, [block])
                    continue
            index += 1

    def _literal_block_from_values(
        self,
        values: Sequence[int],
        *,
        reducer: Optional[str] = None,
        operand: Optional[int] = None,
    ) -> Optional[IRLiteralBlock]:
        if not values:
            return None
        if self._is_literal_block(values):
            triplets = tuple(
                tuple(values[pos : pos + 3]) for pos in range(0, len(values), 3)
            )
            return IRLiteralBlock(
                triplets=triplets, reducer=reducer, reducer_operand=operand
            )

        if len(values) < 6:
            return None

        trip_count = len(values) // 3
        if trip_count < 2:
            return None

        triplets = [tuple(values[pos : pos + 3]) for pos in range(0, trip_count * 3, 3)]
        prefix = triplets[0][:2]
        if not all(chunk[:2] == prefix for chunk in triplets):
            return None

        tail = tuple(values[trip_count * 3 :])
        return IRLiteralBlock(
            triplets=tuple(triplets),
            reducer=reducer,
            reducer_operand=operand,
            tail=tail,
        )

    def _pass_literal_block_reducers(
        self, items: _ItemList, metrics: NormalizerMetrics
    ) -> None:
        index = 0
        while index < len(items) - 1:
            first = items[index]
            second = items[index + 1]
            if isinstance(first, RawInstruction) and first.mnemonic.startswith("reduce"):
                if isinstance(second, IRLiteralBlock):
                    if second.reducer is None:
                        updated = IRLiteralBlock(
                            triplets=second.triplets,
                            reducer=first.mnemonic,
                            reducer_operand=first.operand,
                            tail=second.tail,
                        )
                        self._transfer_ssa(first, updated)
                        items.replace_slice(index, index + 2, [updated])
                        metrics.reduce_replaced += 1
                        continue
                if isinstance(second, IRBuildTuple):
                    values = self._parse_literal_list(second.elements)
                    block = self._literal_block_from_values(
                        values, reducer=first.mnemonic, operand=first.operand
                    )
                    if block is not None:
                        self._transfer_ssa(first, block)
                        items.replace_slice(index, index + 2, [block])
                        metrics.reduce_replaced += 1
                        continue
                if isinstance(second, IRLiteral):
                    values, end = self._collect_literal_block_literals(items, index + 1)
                    block = self._literal_block_from_values(
                        values, reducer=first.mnemonic, operand=first.operand
                    )
                    if block is not None:
                        self._transfer_ssa(first, block)
                        items.replace_slice(index, end, [block])
                        metrics.reduce_replaced += 1
                        continue
            index += 1

    def _pass_ascii_preamble(self, items: _ItemList) -> None:
        index = 0
        while index <= len(items) - 3:
            first, second, third = items[index : index + 3]
            if (
                isinstance(first, RawInstruction)
                and isinstance(second, RawInstruction)
                and isinstance(third, RawInstruction)
                and first.mnemonic == "op_72_23"
                and second.mnemonic == "op_31_30"
                and third.mnemonic == "stack_shuffle"
                and third.operand == CALL_SHUFFLE_STANDARD
            ):
                node = IRAsciiPreamble(
                    loader_operand=first.operand,
                    mode_operand=second.operand,
                    shuffle_operand=third.operand,
                )
                items.replace_slice(index, index + 3, [node])
                continue
            index += 1

    def _pass_call_preparation(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, IRCall):
                index += 1
                continue

            start = index
            steps: List[Tuple[str, int]] = []
            scan = index - 1
            while scan >= 0:
                candidate = items[scan]
                if isinstance(candidate, IRCallPreparation):
                    steps = list(candidate.steps) + steps
                    start = scan
                    break
                if isinstance(candidate, RawInstruction) and self._is_call_preparation_instruction(candidate):
                    steps.insert(0, self._call_preparation_step(candidate))
                    start = scan
                    scan -= 1
                    continue
                break

            if steps:
                items.replace_slice(start, index, [IRCallPreparation(steps=tuple(steps))])
                index = start + 2
                continue

            index += 1

    def _pass_call_cleanup(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction) or not self._is_call_cleanup_instruction(item):
                index += 1
                continue

            steps: List[IRStackEffect] = []
            start = index
            end = index
            scan = index
            while scan < len(items):
                candidate = items[scan]
                if isinstance(candidate, RawInstruction) and self._is_call_cleanup_instruction(candidate):
                    steps.append(self._call_cleanup_effect(candidate))
                    end = scan + 1
                    scan += 1
                    continue
                break

            if not steps:
                index += 1
                continue

            prev_index = start - 1
            while prev_index >= 0 and isinstance(items[prev_index], (IRLiteral, IRLiteralChunk)):
                prev_index -= 1

            if prev_index >= 0 and isinstance(items[prev_index], IRCall):
                items.replace_slice(start, end, [IRCallCleanup(steps=tuple(steps))])
                index = prev_index + 2
                continue

            next_index = end
            while next_index < len(items) and isinstance(items[next_index], (IRLiteral, IRLiteralChunk)):
                next_index += 1

            if next_index < len(items) and isinstance(items[next_index], IRReturn):
                return_node = items[next_index]
                assert isinstance(return_node, IRReturn)
                combined = return_node.cleanup + tuple(steps)
                updated = IRReturn(
                    values=return_node.values,
                    varargs=return_node.varargs,
                    cleanup=combined,
                )
                self._transfer_ssa(return_node, updated)
                items.replace_slice(start, end, [])
                next_index -= len(steps)
                items.replace_slice(next_index, next_index + 1, [updated])
                index = next_index + 1
                continue

            items.replace_slice(start, end, [IRCallCleanup(steps=tuple(steps))])
            index = start + 1

    def _pass_io_operations(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not self._is_io_handshake(item):
                index += 1
                continue

            candidate_index = self._find_io_candidate(items, index)
            if candidate_index is None:
                index += 1
                continue

            candidate = items[candidate_index]
            assert isinstance(candidate, RawInstruction)

            node = self._build_io_node(items, candidate_index, candidate)
            if node is None:
                index += 1
                continue

            self._transfer_ssa(candidate, node)
            self._transfer_ssa(item, node)
            new_index = min(index, candidate_index)
            if index <= candidate_index:
                items.replace_slice(candidate_index, candidate_index + 1, [node])
                items.replace_slice(index, index + 1, [])
            else:
                items.replace_slice(index, index + 1, [])
                items.replace_slice(candidate_index, candidate_index + 1, [node])
            index = new_index

    def _find_io_candidate(self, items: _ItemList, handshake_index: int) -> Optional[int]:
        for direction in (-1, 1):
            scan = handshake_index + direction
            steps = 0
            while 0 <= scan < len(items) and steps < 12:
                node = items[scan]
                if isinstance(node, RawInstruction):
                    if self._is_io_handshake(node) or node.mnemonic in IO_BRIDGE_MNEMONICS:
                        scan += direction
                        steps += 1
                        continue
                    if (
                        node.mnemonic.startswith("op_10_")
                        and node.operand in IO_ACCEPTED_OPERANDS
                    ):
                        return scan
                    break
                if self._is_io_bridge_node(node):
                    scan += direction
                    steps += 1
                    continue
                break
        return None

    def _build_io_node(
        self, items: _ItemList, index: int, instruction: RawInstruction
    ) -> Optional[IRNode]:
        mnemonic = instruction.mnemonic
        if mnemonic in IO_READ_MNEMONICS:
            return IRIORead(port=IO_PORT_NAME)
        if mnemonic in IO_WRITE_MNEMONICS:
            mask = self._io_mask_value(items, index)
            if mask is None and instruction.operand not in IO_ACCEPTED_OPERANDS:
                mask = instruction.operand
            return IRIOWrite(mask=mask, port=IO_PORT_NAME)
        return None

    def _io_mask_value(self, items: _ItemList, index: int) -> Optional[int]:
        scan = index - 1
        steps = 0
        while scan >= 0 and steps < 8:
            node = items[scan]
            if isinstance(node, IRLiteral):
                return node.value
            if isinstance(node, RawInstruction):
                if (
                    node.mnemonic in IO_BRIDGE_MNEMONICS
                    or self._is_io_handshake(node)
                ):
                    scan -= 1
                    steps += 1
                    continue
                if node.mnemonic.startswith("op_10_"):
                    break
            elif isinstance(node, IRLiteralChunk):
                scan -= 1
                steps += 1
                continue
            elif self._is_io_bridge_node(node):
                scan -= 1
                steps += 1
                continue
            else:
                break
            scan -= 1
            steps += 1
        return None

    @staticmethod
    def _is_io_handshake(item: Union[RawInstruction, IRNode]) -> bool:
        return (
            isinstance(item, RawInstruction)
            and item.mnemonic in IO_HANDSHAKE_MNEMONICS
            and item.operand == IO_SLOT
        )

    @staticmethod
    def _is_io_bridge_node(node: Union[RawInstruction, IRNode]) -> bool:
        return isinstance(node, IO_BRIDGE_NODE_TYPES)

    def _pass_call_conventions(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, IRCall):
                index += 1
                continue

            shuffle = item.shuffle
            arity = item.arity
            cleanup_mask = item.cleanup_mask
            cleanup_steps = item.cleanup

            consumed: List[int] = []
            scan = index - 1
            while scan >= 0:
                candidate = items[scan]
                if isinstance(candidate, IRCallPreparation):
                    for mnemonic, operand in candidate.steps:
                        if mnemonic == "stack_shuffle":
                            shuffle = operand
                    consumed.append(scan)
                    scan -= 1
                    continue
                if isinstance(candidate, IRLiteral) and arity is None:
                    decoded = self._decode_call_arity(candidate.value)
                    if decoded is not None:
                        arity = decoded
                        consumed.append(scan)
                        scan -= 1
                        continue
                if isinstance(candidate, IRCallCleanup):
                    shuffle_operand = self._extract_call_shuffle(candidate)
                    if shuffle_operand is not None:
                        shuffle = shuffle_operand
                        consumed.append(scan)
                        scan -= 1
                        continue
                break

            post_index = index + 1
            while post_index < len(items) and isinstance(items[post_index], IRLiteralChunk):
                post_index += 1

            cleanup_index: Optional[int] = None
            if post_index < len(items) and isinstance(items[post_index], IRCallCleanup):
                cleanup_index = post_index
                cleanup_node = items[cleanup_index]
                cleanup_steps = cleanup_steps + cleanup_node.steps
                if cleanup_mask is None:
                    cleanup_mask = self._extract_cleanup_mask(cleanup_node.steps)

            if cleanup_index is not None:
                items.pop(cleanup_index)
                if cleanup_index < index:
                    index -= 1

            for position in sorted(consumed, reverse=True):
                items.pop(position)
                if position < index:
                    index -= 1

            call = items[index]
            assert isinstance(call, IRCall)

            tail = call.tail
            if not tail and cleanup_mask == RET_MASK:
                tail = True

            updated = IRCall(
                target=call.target,
                args=call.args,
                tail=tail,
                arity=arity,
                shuffle=shuffle,
                cleanup_mask=cleanup_mask,
                cleanup=cleanup_steps,
                symbol=call.symbol,
            )
            self._transfer_ssa(call, updated)
            items.replace_slice(index, index + 1, [updated])
            index += 1

    def _pass_tailcall_frames(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction) or item.mnemonic != "op_3D_30":
                index += 1
                continue

            steps: List[Tuple[str, int]] = [(item.mnemonic, item.operand)]
            scan = index + 1
            end = index + 1
            while scan < len(items) and scan - index <= 6:
                candidate = items[scan]
                if not isinstance(candidate, RawInstruction):
                    break
                steps.append((candidate.mnemonic, candidate.operand))
                if candidate.mnemonic == "op_F0_4B":
                    end = scan + 1
                    break
                scan += 1

            if steps[-1][0] == "op_F0_4B":
                items.replace_slice(index, end, [IRTailcallFrame(steps=tuple(steps))])
                continue

            index += 1

    def _pass_table_patches(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not (
                isinstance(item, RawInstruction)
                and item.mnemonic.startswith("op_2C_")
                and 0x6600 <= item.operand <= 0x66FF
            ):
                index += 1
                continue

            operations: List[Tuple[str, int]] = [(item.mnemonic, item.operand)]
            scan = index + 1
            while scan < len(items):
                candidate = items[scan]
                if isinstance(candidate, RawInstruction):
                    if (
                        candidate.mnemonic.startswith("op_2C_")
                        and 0x6600 <= candidate.operand <= 0x66FF
                    ):
                        operations.append((candidate.mnemonic, candidate.operand))
                        scan += 1
                        continue
                    if candidate.mnemonic in {"fanout", "stack_teardown_4", "stack_teardown_5"}:
                        operations.append((candidate.mnemonic, candidate.operand))
                        scan += 1
                        continue
                break

            items.replace_slice(index, scan, [IRTablePatch(operations=tuple(operations))])
            index += 1

    def _pass_ascii_finalize(self, items: _ItemList) -> None:
        index = 0
        ascii_helpers = {0xF172, 0x7223, 0x3D30}
        while index < len(items):
            item = items[index]
            if not (
                isinstance(item, RawInstruction)
                and item.mnemonic == "call_helpers"
                and item.operand in ascii_helpers
            ):
                index += 1
                continue

            summary = ""
            if index > 0:
                previous = items[index - 1]
                if isinstance(previous, IRLiteralChunk):
                    summary = previous.describe()
                elif isinstance(previous, IRLiteral):
                    summary = previous.describe()
                elif isinstance(previous, IRAsciiPreamble):
                    summary = previous.describe()
            node = IRAsciiFinalize(helper=item.operand, summary=summary or "ascii")
            items.replace_slice(index, index + 1, [node])
            index += 1

    def _pass_flag_checks(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRIf):
                flag = self._flag_literal_value(item.condition)
                if flag is not None:
                    node = IRFlagCheck(
                        flag=flag,
                        then_target=item.then_target,
                        else_target=item.else_target,
                    )
                    items.replace_slice(index, index + 1, [node])
                    index += 1
                    continue
            index += 1

    def _pass_function_prologues(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRTestSetBranch):
                if index == 0 or all(
                    self._is_prologue_prefix(items[pos]) for pos in range(index)
                ):
                    if item.var.startswith("slot("):
                        node = IRFunctionPrologue(
                            var=item.var,
                            expr=item.expr,
                            then_target=item.then_target,
                            else_target=item.else_target,
                        )
                        items.replace_slice(index, index + 1, [node])
                        index += 1
                        continue
            index += 1

    @staticmethod
    def _is_prologue_prefix(item: Union[RawInstruction, IRNode]) -> bool:
        return isinstance(
            item,
            (
                IRAsciiHeader,
                IRLiteralChunk,
                IRLiteral,
                IRCallPreparation,
                IRCallCleanup,
            ),
        )

    def _pass_ascii_wrappers(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, IRCall):
                index += 1
                continue

            ascii_chunks: List[str] = []
            scan = index + 1
            while scan < len(items):
                candidate = items[scan]
                if isinstance(candidate, IRLiteralChunk):
                    ascii_chunks.append(candidate.describe())
                    scan += 1
                    continue
                break

            if not ascii_chunks:
                index += 1
                continue

            branch: Optional[IRIf] = None
            if scan < len(items) and isinstance(items[scan], IRIf):
                candidate = items[scan]
                first_chunk = ascii_chunks[0]
                if (
                    candidate.condition in {first_chunk, "stack_top"}
                    or candidate.condition.startswith("ssa")
                ) and item.tail:
                    branch = candidate
                    scan += 1

            ascii_tuple = tuple(ascii_chunks)
            if branch is not None:
                node = IRTailcallAscii(
                    target=item.target,
                    args=item.args,
                    ascii_chunks=ascii_tuple,
                    condition=branch.condition,
                    then_target=branch.then_target,
                    else_target=branch.else_target,
                    arity=item.arity,
                    shuffle=item.shuffle,
                    cleanup_mask=item.cleanup_mask,
                    cleanup=item.cleanup,
                    symbol=item.symbol,
                )
                items.replace_slice(index, scan, [node])
                continue

            node = IRAsciiWrapperCall(
                target=item.target,
                args=item.args,
                ascii_chunks=ascii_tuple,
                tail=item.tail,
                arity=item.arity,
                shuffle=item.shuffle,
                cleanup_mask=item.cleanup_mask,
                cleanup=item.cleanup,
                symbol=item.symbol,
            )
            end = index + 1 + len(ascii_chunks)
            items.replace_slice(index, end, [node])
            index += 1

    def _pass_ascii_headers(self, items: _ItemList) -> None:
        if not items:
            return
        chunks: List[str] = []
        index = 0
        while index < len(items) and isinstance(items[index], IRLiteralChunk):
            chunks.append(items[index].describe())
            index += 1
        if len(chunks) >= 2:
            items.replace_slice(0, index, [IRAsciiHeader(chunks=tuple(chunks))])
            return

        if len(chunks) == 1:
            literal = items[0]
            if isinstance(literal, IRLiteralChunk) and len(literal.data) >= 8:
                if len(literal.data) % 4 == 0:
                    parts = []
                    for pos in range(0, len(literal.data), 4):
                        segment = literal.data[pos : pos + 4]
                        piece = IRLiteralChunk(data=segment, source=literal.source)
                        parts.append(piece.describe())
                    if len(parts) >= 2:
                        items.replace_slice(0, 1, [IRAsciiHeader(chunks=tuple(parts))])

    def _pass_call_return_templates(self, items: _ItemList) -> None:
        index = 0
        while index < len(items) - 1:
            call = items[index]
            if isinstance(call, IRCall):
                offset = index + 1
                cleanup_steps: List[IRStackEffect] = list(call.cleanup)
                cleanup_mask = call.cleanup_mask
                base_tail = call.tail
                consumed = 0

                while offset < len(items):
                    candidate = items[offset]
                    if isinstance(candidate, IRCallCleanup):
                        cleanup_steps.extend(candidate.steps)
                        offset += 1
                        consumed += 1
                        continue
                    if (
                        isinstance(candidate, RawInstruction)
                        and candidate.mnemonic == "op_29_10"
                        and candidate.operand == RET_MASK
                    ):
                        cleanup_steps.append(self._call_cleanup_effect(candidate))
                        cleanup_mask = RET_MASK
                        offset += 1
                        consumed += 1
                        continue
                    break

                tail_hint = base_tail
                if cleanup_mask == RET_MASK:
                    tail_hint = True

                if offset < len(items) and isinstance(items[offset], IRReturn):
                    return_node = items[offset]
                    combined_cleanup = tuple(cleanup_steps + list(return_node.cleanup))
                    varargs = return_node.varargs
                    return_count = len(return_node.values)
                    should_bundle = tail_hint and (
                        base_tail or (return_count == 0 and not varargs)
                    )

                    if should_bundle:
                        node = IRTailcallReturn(
                            target=call.target,
                            args=call.args,
                            returns=return_count,
                            varargs=varargs,
                            cleanup=combined_cleanup,
                            tail=True,
                            arity=call.arity,
                            shuffle=call.shuffle,
                            cleanup_mask=cleanup_mask,
                            symbol=call.symbol,
                            predicate=call.predicate,
                        )
                    else:
                        tail = base_tail and consumed == 0 and not return_node.cleanup
                        if not tail and cleanup_mask == RET_MASK and consumed == 0 and not return_node.cleanup:
                            tail = True
                        node = IRCallReturn(
                            target=call.target,
                            args=call.args,
                            tail=tail,
                            returns=return_node.values,
                            varargs=varargs,
                            cleanup=combined_cleanup,
                            arity=call.arity,
                            shuffle=call.shuffle,
                            cleanup_mask=cleanup_mask,
                            symbol=call.symbol,
                            predicate=call.predicate,
                        )
                    self._transfer_ssa(call, node)
                    end = offset + 1
                    items.replace_slice(index, end, [node])
                    continue
            index += 1

    def _pass_condition_masks(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRCallCleanup) and len(item.steps) == 1:
                step = item.steps[0]
                if step.mnemonic == "fanout" and step.operand == RET_MASK:
                    node = IRConditionMask(source="fanout", mask=step.operand)
                    items.replace_slice(index, index + 1, [node])
                    continue
            if isinstance(item, RawInstruction):
                if item.operand == RET_MASK and item.mnemonic in {"terminator", "op_29_10"}:
                    node = IRConditionMask(source=item.mnemonic, mask=item.operand)
                    items.replace_slice(index, index + 1, [node])
                    continue
            index += 1

    def _pass_call_predicates(self, items: _ItemList) -> None:
        index = 0
        while index < len(items) - 1:
            call = items[index]
            if not isinstance(
                call,
                (
                    IRCall,
                    IRCallReturn,
                    IRAsciiWrapperCall,
                    IRTailcallAscii,
                    IRTailcallReturn,
                ),
            ):
                index += 1
                continue

            if getattr(call, "predicate", None) is not None:
                index += 1
                continue

            extracted = self._extract_call_predicate(items, index)
            if extracted is None:
                index += 1
                continue

            predicate, branch_index, alias = extracted
            updated = self._call_with_predicate(call, predicate)
            self._transfer_ssa(call, updated)
            items.replace_slice(index, index + 1, [updated])

            branch = items[branch_index]
            if isinstance(branch, IRTestSetBranch) and branch.expr != alias:
                updated_branch = IRTestSetBranch(
                    var=branch.var,
                    expr=alias,
                    then_target=branch.then_target,
                    else_target=branch.else_target,
                )
                items.replace_slice(branch_index, branch_index + 1, [updated_branch])
            elif isinstance(branch, IRIf) and branch.condition != alias:
                updated_branch = IRIf(
                    condition=alias,
                    then_target=branch.then_target,
                    else_target=branch.else_target,
                )
                items.replace_slice(branch_index, branch_index + 1, [updated_branch])

            index += 1

    def _extract_call_predicate(
        self, items: _ItemList, index: int
    ) -> Optional[Tuple[CallPredicate, int, str]]:
        scan = index + 1
        while scan < len(items):
            candidate = items[scan]
            if isinstance(candidate, (IRCallCleanup, IRConditionMask)):
                scan += 1
                continue
            if isinstance(candidate, RawInstruction) and candidate.mnemonic in CALL_PREDICATE_SKIP_MNEMONICS:
                scan += 1
                continue
            break

        if scan >= len(items):
            return None

        branch = items[scan]
        call = items[index]
        alias = self._ssa_value(call)
        if not alias or alias == "stack_top":
            raw_name = f"call_predicate_{id(call)}"
            self._record_ssa(call, (raw_name,))
            self._promote_ssa_kind(raw_name, SSAValueKind.BOOLEAN)
            alias = self._render_ssa(raw_name)

        if isinstance(branch, IRIf):
            predicate = CallPredicate(
                kind="if",
                expr=alias,
                then_target=branch.then_target,
                else_target=branch.else_target,
            )
            return predicate, scan, alias

        if isinstance(branch, IRTestSetBranch):
            predicate = CallPredicate(
                kind="testset",
                var=branch.var,
                expr=alias,
                then_target=branch.then_target,
                else_target=branch.else_target,
            )
            return predicate, scan, alias

        return None

    def _call_with_predicate(
        self,
        node: Union[
            IRCall,
            IRCallReturn,
            IRAsciiWrapperCall,
            IRTailcallAscii,
            IRTailcallReturn,
        ],
        predicate: CallPredicate,
    ) -> Union[
        IRCall,
        IRCallReturn,
        IRAsciiWrapperCall,
        IRTailcallAscii,
        IRTailcallReturn,
    ]:
        if isinstance(node, IRCall):
            return IRCall(
                target=node.target,
                args=node.args,
                tail=node.tail,
                arity=node.arity,
                shuffle=node.shuffle,
                cleanup_mask=node.cleanup_mask,
                cleanup=node.cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRCallReturn):
            return IRCallReturn(
                target=node.target,
                args=node.args,
                tail=node.tail,
                returns=node.returns,
                varargs=node.varargs,
                cleanup=node.cleanup,
                arity=node.arity,
                shuffle=node.shuffle,
                cleanup_mask=node.cleanup_mask,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRAsciiWrapperCall):
            return IRAsciiWrapperCall(
                target=node.target,
                args=node.args,
                ascii_chunks=node.ascii_chunks,
                tail=node.tail,
                arity=node.arity,
                shuffle=node.shuffle,
                cleanup_mask=node.cleanup_mask,
                cleanup=node.cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRTailcallAscii):
            return IRTailcallAscii(
                target=node.target,
                args=node.args,
                ascii_chunks=node.ascii_chunks,
                condition=node.condition,
                then_target=node.then_target,
                else_target=node.else_target,
                arity=node.arity,
                shuffle=node.shuffle,
                cleanup_mask=node.cleanup_mask,
                cleanup=node.cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRTailcallReturn):
            return IRTailcallReturn(
                target=node.target,
                args=node.args,
                returns=node.returns,
                varargs=node.varargs,
                cleanup=node.cleanup,
                tail=node.tail,
                arity=node.arity,
                shuffle=node.shuffle,
                cleanup_mask=node.cleanup_mask,
                symbol=node.symbol,
                predicate=predicate,
            )

        return node

    def _pass_call_contracts(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            node = items[index]
            if not isinstance(
                node,
                (
                    IRCall,
                    IRCallReturn,
                    IRAsciiWrapperCall,
                    IRTailcallAscii,
                    IRTailcallReturn,
                ),
            ):
                index += 1
                continue

            signature = self.knowledge.call_signature(getattr(node, "target", -1))
            if signature is None:
                index += 1
                continue

            updated_index = self._apply_call_signature(items, index, node, signature)
            index = updated_index + 1

    def _apply_call_signature(
        self,
        items: _ItemList,
        index: int,
        node: Union[
            IRCall,
            IRCallReturn,
            IRAsciiWrapperCall,
            IRTailcallAscii,
            IRTailcallReturn,
        ],
        signature: CallSignature,
    ) -> int:
        current_index = index
        target = getattr(node, "target", -1)

        tail = getattr(node, "tail", False)
        arity = getattr(node, "arity", None)
        shuffle = getattr(node, "shuffle", None)
        cleanup_mask = getattr(node, "cleanup_mask", None)
        predicate = getattr(node, "predicate", None)
        existing_cleanup = list(getattr(node, "cleanup", tuple()))

        if signature.tail is not None:
            tail = signature.tail or tail
        if signature.arity is not None:
            arity = signature.arity
        if signature.cleanup_mask is not None and cleanup_mask is None:
            cleanup_mask = signature.cleanup_mask
        if signature.shuffle is not None:
            shuffle = signature.shuffle
        elif signature.shuffle_options:
            if shuffle is None or shuffle not in signature.shuffle_options:
                shuffle = signature.shuffle_options[0]

        prefix_effects: List[IRStackEffect] = []
        suffix_effects: List[IRStackEffect] = []

        for pattern in signature.prelude:
            matched, effects, mask, tail_flag, predicate_update, consumed = self._consume_call_pattern_before(
                items, current_index, pattern
            )
            if not matched:
                if pattern.optional:
                    continue
            else:
                prefix_effects = effects + prefix_effects
                if mask is not None:
                    cleanup_mask = mask
                if tail_flag:
                    tail = True
                if predicate_update is not None:
                    predicate = predicate_update
                current_index -= consumed
                continue
            # required pattern not matched: keep scanning without consuming

        node = items[current_index]

        for pattern in signature.postlude:
            matched, effects, mask, tail_flag, predicate_update = self._consume_call_pattern_after(
                items, current_index, pattern
            )
            if not matched:
                if pattern.optional:
                    continue
            else:
                suffix_effects.extend(effects)
                if mask is not None:
                    cleanup_mask = mask
                if tail_flag:
                    tail = True
                if predicate_update is not None:
                    predicate = predicate_update
                continue
            # required pattern not matched: leave remaining nodes untouched

        if cleanup_mask is None:
            cleanup_mask = signature.cleanup_mask

        signature_cleanup = self._convert_signature_effects(signature.cleanup)
        combined_cleanup = tuple(prefix_effects + signature_cleanup + existing_cleanup + suffix_effects)

        updated = self._rebuild_call_node(
            node,
            tail=tail,
            arity=arity,
            shuffle=shuffle,
            cleanup_mask=cleanup_mask,
            cleanup=combined_cleanup,
            predicate=predicate,
            target=target,
            signature=signature,
        )
        self._transfer_ssa(node, updated)
        items.replace_slice(current_index, current_index + 1, [updated])
        return current_index

    def _consume_call_pattern_before(
        self,
        items: _ItemList,
        index: int,
        pattern: CallSignaturePattern,
    ) -> Tuple[bool, List[IRStackEffect], Optional[int], bool, Optional[CallPredicate], int]:
        position = index - 1
        if position < 0:
            return False, [], None, False, None, 0

        candidate = items[position]
        if pattern.kind == "raw" and isinstance(candidate, RawInstruction):
            if pattern.mnemonic and candidate.mnemonic != pattern.mnemonic:
                return False, [], None, False, None, 0
            if pattern.operand is not None and candidate.operand != pattern.operand:
                return False, [], None, False, None, 0
            effects: List[IRStackEffect] = []
            if pattern.effect is not None:
                effects.append(self._stack_effect_from_signature(pattern.effect, candidate))
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            predicate: Optional[CallPredicate] = None
            items.pop(position)
            return True, effects, cleanup_mask, tail, predicate, 1

        return False, [], None, False, None, 0

    def _consume_call_pattern_after(
        self,
        items: _ItemList,
        index: int,
        pattern: CallSignaturePattern,
    ) -> Tuple[bool, List[IRStackEffect], Optional[int], bool, Optional[CallPredicate]]:
        if index + 1 >= len(items):
            return False, [], None, False, None

        candidate = items[index + 1]

        if pattern.kind == "raw" and isinstance(candidate, RawInstruction):
            if pattern.mnemonic and candidate.mnemonic != pattern.mnemonic:
                return False, [], None, False, None
            if pattern.operand is not None and candidate.operand != pattern.operand:
                return False, [], None, False, None
            effects: List[IRStackEffect] = []
            if pattern.effect is not None:
                effects.append(self._stack_effect_from_signature(pattern.effect, candidate))
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            predicate: Optional[CallPredicate] = None
            items.pop(index + 1)
            return True, effects, cleanup_mask, tail, predicate

        if pattern.kind == "testset" and isinstance(candidate, IRTestSetBranch):
            predicate = CallPredicate(
                kind="testset",
                var=candidate.var,
                expr=candidate.expr,
                then_target=candidate.then_target,
                else_target=candidate.else_target,
            )
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            items.pop(index + 1)
            return True, [], cleanup_mask, tail, predicate

        if pattern.kind == "flag" and isinstance(candidate, IRFlagCheck):
            predicate = CallPredicate(
                kind="flag",
                flag=candidate.flag,
                then_target=candidate.then_target,
                else_target=candidate.else_target,
            )
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            items.pop(index + 1)
            return True, [], cleanup_mask, tail, predicate

        if pattern.kind == "if" and isinstance(candidate, IRIf):
            predicate = CallPredicate(
                kind="if",
                expr=candidate.condition,
                then_target=candidate.then_target,
                else_target=candidate.else_target,
            )
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            items.pop(index + 1)
            return True, [], cleanup_mask, tail, predicate

        return False, [], None, False, None

    def _convert_signature_effects(
        self, specs: Sequence[CallSignatureEffect]
    ) -> List[IRStackEffect]:
        effects: List[IRStackEffect] = []
        for spec in specs:
            effects.append(self._stack_effect_from_signature(spec, None))
        return effects

    def _stack_effect_from_signature(
        self, spec: CallSignatureEffect, instruction: Optional[RawInstruction]
    ) -> IRStackEffect:
        operand = spec.operand
        operand_role = spec.operand_role
        operand_alias = spec.operand_alias
        if instruction is not None:
            if operand is None or spec.inherit_operand:
                operand = instruction.operand
            if operand_role is None:
                operand_role = instruction.profile.operand_role()
            if spec.inherit_alias or operand_alias is None:
                alias = instruction.profile.operand_alias()
                if alias is not None:
                    operand_alias = alias
        if operand is None:
            operand = 0

        pops = spec.pops
        if instruction is not None and pops == 0:
            if instruction.profile.kind is InstructionKind.STACK_TEARDOWN:
                delta = -instruction.event.delta
                if delta > 0:
                    pops = delta

        return IRStackEffect(
            mnemonic=spec.mnemonic,
            operand=operand,
            pops=pops,
            operand_role=operand_role,
            operand_alias=operand_alias,
        )

    def _rebuild_call_node(
        self,
        node: Union[
            IRCall,
            IRCallReturn,
            IRAsciiWrapperCall,
            IRTailcallAscii,
            IRTailcallReturn,
        ],
        *,
        tail: bool,
        arity: Optional[int],
        shuffle: Optional[int],
        cleanup_mask: Optional[int],
        cleanup: Tuple[IRStackEffect, ...],
        predicate: Optional[CallPredicate],
        target: int,
        signature: CallSignature,
    ) -> Union[
        IRCall,
        IRCallReturn,
        IRAsciiWrapperCall,
        IRTailcallAscii,
        IRTailcallReturn,
    ]:
        if isinstance(node, IRCall):
            return IRCall(
                target=target,
                args=node.args,
                tail=tail,
                arity=arity,
                shuffle=shuffle,
                cleanup_mask=cleanup_mask,
                cleanup=cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRAsciiWrapperCall):
            return IRAsciiWrapperCall(
                target=target,
                args=node.args,
                ascii_chunks=node.ascii_chunks,
                tail=tail,
                arity=arity,
                shuffle=shuffle,
                cleanup_mask=cleanup_mask,
                cleanup=cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRTailcallAscii):
            return IRTailcallAscii(
                target=target,
                args=node.args,
                ascii_chunks=node.ascii_chunks,
                condition=node.condition,
                then_target=node.then_target,
                else_target=node.else_target,
                arity=arity,
                shuffle=shuffle,
                cleanup_mask=cleanup_mask,
                cleanup=cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRCallReturn):
            returns = node.returns
            if signature.returns is not None and not node.varargs:
                count = max(signature.returns, 0)
                returns = tuple(f"ret{i}" for i in range(count))
            return IRCallReturn(
                target=target,
                args=node.args,
                tail=tail,
                returns=returns,
                varargs=node.varargs,
                cleanup=cleanup,
                arity=arity,
                shuffle=shuffle,
                cleanup_mask=cleanup_mask,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRTailcallReturn):
            returns = node.returns
            if signature.returns is not None and not node.varargs:
                returns = max(signature.returns, 0)
            return IRTailcallReturn(
                target=target,
                args=node.args,
                returns=returns,
                varargs=node.varargs,
                cleanup=cleanup,
                tail=tail,
                arity=arity,
                shuffle=shuffle,
                cleanup_mask=cleanup_mask,
                symbol=node.symbol,
                predicate=predicate,
            )

        return node

    @staticmethod
    def _is_call_cleanup_instruction(instruction: RawInstruction) -> bool:
        mnemonic = instruction.mnemonic
        if mnemonic in CALL_CLEANUP_MNEMONICS:
            return True
        return any(mnemonic.startswith(prefix) for prefix in CALL_CLEANUP_PREFIXES)

    @staticmethod
    def _call_preparation_step(instruction: RawInstruction) -> Tuple[str, int]:
        mnemonic = instruction.mnemonic
        if instruction.profile.kind is InstructionKind.STACK_TEARDOWN:
            pops = -instruction.event.delta
            if pops > 0:
                return ("stack_teardown", pops)
        return (mnemonic, instruction.operand)

    @staticmethod
    def _is_call_preparation_instruction(instruction: RawInstruction) -> bool:
        mnemonic = instruction.mnemonic
        if mnemonic == "op_4A_05":
            return True
        if instruction.profile.kind is InstructionKind.STACK_TEARDOWN:
            return True
        return any(mnemonic.startswith(prefix) for prefix in CALL_PREPARATION_PREFIXES)

    def _call_cleanup_effect(self, instruction: RawInstruction) -> IRStackEffect:
        mnemonic = instruction.mnemonic
        operand = instruction.operand
        pops = 0
        if instruction.profile.kind is InstructionKind.STACK_TEARDOWN:
            pops = -instruction.event.delta
            if mnemonic.startswith("stack_teardown"):
                mnemonic = "stack_teardown"
        return IRStackEffect(
            mnemonic=mnemonic,
            operand=operand,
            pops=pops,
            operand_role=instruction.profile.operand_role(),
            operand_alias=instruction.profile.operand_alias(),
        )

    @staticmethod
    def _extract_call_shuffle(cleanup: IRCallCleanup) -> Optional[int]:
        if len(cleanup.steps) != 1:
            return None
        step = cleanup.steps[0]
        if step.mnemonic != "stack_shuffle":
            return None
        return step.operand

    @staticmethod
    def _decode_call_arity(value: int) -> Optional[int]:
        if value <= 0:
            return None
        high = (value >> 8) & 0xFF
        low = value & 0xFF
        if high and low == 0:
            return high
        if high:
            return high
        if value <= 0x3F:
            return value
        return None

    @staticmethod
    def _extract_cleanup_mask(steps: Sequence[IRStackEffect]) -> Optional[int]:
        for mnemonic in ("op_52_05", "op_32_29", "fanout"):
            for step in steps:
                if step.mnemonic == mnemonic:
                    return step.operand
        return None

    def _literal_at(self, items: _ItemList, index: int) -> Optional[IRLiteral]:
        item = items[index]
        if isinstance(item, IRLiteral):
            return item
        if isinstance(item, RawInstruction):
            if item.profile.kind is InstructionKind.LITERAL and item.pushes_value():
                literal = IRLiteral(
                    value=item.operand,
                    mode=item.profile.mode,
                    source=item.profile.mnemonic,
                    annotations=item.annotations,
                )
                items.replace_slice(index, index + 1, [literal])
                return literal
        return None

    @staticmethod
    def _literal_repr(node: IRLiteral) -> str:
        return node.describe()

    def _parse_literal_list(self, elements: Sequence[str]) -> Tuple[int, ...]:
        values: List[int] = []
        for text in elements:
            value = self._parse_literal_text(text)
            if value is None:
                return tuple()
            values.append(value)
        return tuple(values)

    @staticmethod
    def _parse_literal_text(text: str) -> Optional[int]:
        if text.startswith("lit(0x") and text.endswith(")"):
            try:
                return int(text[4:-1], 16)
            except ValueError:
                return None
        return None

    def _collect_literal_block_literals(
        self, items: _ItemList, start: int
    ) -> Tuple[Tuple[int, ...], int]:
        values: List[int] = []
        index = start
        while index < len(items):
            item = items[index]
            if isinstance(item, IRLiteral):
                values.append(item.value)
                index += 1
                continue
            break
        sequence = tuple(values)
        if sequence:
            return sequence, index
        return tuple(), start

    @staticmethod
    def _is_literal_block(values: Sequence[int]) -> bool:
        if not values or len(values) % 3:
            return False
        for pos in range(0, len(values), 3):
            chunk = values[pos : pos + 3]
            if len(chunk) < 3:
                return False
            if chunk[0] != 0x6704 or chunk[1] != 0x0067:
                return False
            if chunk[2] not in {0x0400, 0x0110}:
                return False
        return True

    def _pass_assign_ssa_names(self, items: _ItemList) -> None:
        for item in items:
            if isinstance(item, RawInstruction) and item.ssa_values:
                self._record_ssa(item, item.ssa_values, kinds=item.ssa_kinds)

    def _pass_branches(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            if item.mnemonic == "testset_branch":
                expr = self._describe_condition(items, index, skip_literals=True)
                node = IRTestSetBranch(
                    var=self._format_testset_var(item),
                    expr=expr,
                    then_target=self._branch_target(item),
                    else_target=self._fallthrough_target(item),
                )
                items.replace_slice(index, index + 1, [node])
                metrics.testset_branches += 1
                continue

            if item.profile.kind is InstructionKind.BRANCH:
                node = IRIf(
                    condition=self._describe_condition(items, index),
                    then_target=self._branch_target(item),
                    else_target=self._fallthrough_target(item),
                )
                items.replace_slice(index, index + 1, [node])
                metrics.if_branches += 1
                continue

            index += 1

    def _pass_indirect_access(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            kind = item.event.kind
            if kind in {InstructionKind.INDIRECT_LOAD, InstructionKind.INDIRECT}:
                base_sources = self._stack_sources(items, index, 1)
                base_slot: Optional[IRSlot] = None
                base_alias = "stack"
                if base_sources:
                    base_name, base_node = base_sources[0]
                    self._promote_ssa_kind(base_name, SSAValueKind.POINTER)
                    base_alias = self._render_ssa(base_name)
                    if isinstance(base_node, IRLoad):
                        base_slot = base_node.slot
                target_name = item.ssa_values[0] if item.ssa_values else None
                target_alias = self._render_ssa(target_name) if target_name else "stack"
                base_name = base_sources[0][0] if base_sources else None
                memref, index = self._collect_memref(items, index, base_slot, base_name, item)
                if base_name is not None:
                    base_alias = self._render_ssa(base_name)
                node = IRIndirectLoad(
                    base=base_alias,
                    offset=item.operand,
                    target=target_alias,
                    base_slot=base_slot,
                    ref=memref,
                )
                self._transfer_ssa(item, node)
                items.replace_slice(index, index + 1, [node])
                metrics.loads += 1
                continue
            if kind is InstructionKind.INDIRECT_STORE:
                base_sources = self._stack_sources(items, index, 2)
                if base_sources:
                    base_name, base_node = base_sources[0]
                    value_name = base_sources[1][0] if len(base_sources) > 1 else None
                else:
                    base_name = None
                    base_node = None
                    value_name = None
                base_slot = base_node.slot if isinstance(base_node, IRLoad) else None
                if base_name:
                    self._promote_ssa_kind(base_name, SSAValueKind.POINTER)
                base_alias = self._render_ssa(base_name) if base_name else "stack"
                value_alias = self._render_ssa(value_name) if value_name else "stack"
                memref, index = self._collect_memref(items, index, base_slot, base_name, item)
                if base_name:
                    base_alias = self._render_ssa(base_name)
                node = IRIndirectStore(
                    base=base_alias,
                    value=value_alias,
                    offset=item.operand,
                    base_slot=base_slot,
                    ref=memref,
                )
                items.replace_slice(index, index + 1, [node])
                metrics.stores += 1
                continue

            index += 1

    # ------------------------------------------------------------------
    # description helpers
    # ------------------------------------------------------------------
    def _describe_stack_top(self, items: _ItemList, index: int) -> str:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    name = self._ssa_value(candidate)
                    if name:
                        return name
                    return self._describe_value(candidate)
            elif isinstance(candidate, IRLiteral):
                name = self._ssa_value(candidate)
                if name:
                    return name
                return candidate.describe()
            elif isinstance(candidate, IRLiteralChunk):
                name = self._ssa_value(candidate)
                if name:
                    return name
                return candidate.describe()
            elif isinstance(candidate, IRStackDuplicate):
                return candidate.value
            elif isinstance(candidate, IRNode):
                name = self._ssa_value(candidate)
                if name:
                    return name
                describe = getattr(candidate, "describe", None)
                if callable(describe):
                    return describe()
            scan -= 1
        return "stack_top"

    def _describe_value(self, instruction: RawInstruction) -> str:
        mapped = self._ssa_value(instruction)
        if mapped is not None:
            return mapped
        mnemonic = instruction.mnemonic
        operand = instruction.operand
        if mnemonic == "push_literal":
            return f"lit(0x{operand:04X})"
        if mnemonic.startswith("reduce"):
            return f"reduce(0x{operand:04X})"
        if "slot" in mnemonic:
            return f"slot(0x{operand:04X})"
        if instruction.profile.kind is InstructionKind.LITERAL:
            return f"lit(0x{operand:04X})"
        return instruction.describe_source()

    def _describe_condition(self, items: _ItemList, index: int, *, skip_literals: bool = False) -> str:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    if skip_literals and candidate.mnemonic == "push_literal":
                        mapped = self._ssa_value(candidate, raw=True)
                        if mapped is not None:
                            self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                            return self._render_ssa(mapped)
                        scan -= 1
                        continue
                    mapped = self._ssa_value(candidate, raw=True)
                    if mapped is not None:
                        self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                        return self._render_ssa(mapped)
                    return self._describe_value(candidate)
                if skip_literals:
                    mapped = self._ssa_value(candidate, raw=True)
                    if mapped is not None:
                        self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                        return self._render_ssa(mapped)
                    return self._describe_value(candidate)
            if isinstance(candidate, IRLiteral):
                mapped = self._ssa_value(candidate, raw=True)
                if mapped is not None:
                    self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                    return self._render_ssa(mapped)
                if skip_literals:
                    scan -= 1
                    continue
                return candidate.describe()
            if isinstance(candidate, IRLiteralChunk):
                mapped = self._ssa_value(candidate, raw=True)
                if mapped is not None:
                    self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                    return self._render_ssa(mapped)
                if skip_literals:
                    scan -= 1
                    continue
                return candidate.describe()
            if isinstance(candidate, IRStackDuplicate):
                mapped = self._ssa_value(candidate, raw=True)
                if mapped is not None:
                    self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                    return self._render_ssa(mapped)
                return candidate.value
            if isinstance(candidate, IRNode):
                mapped = self._ssa_value(candidate, raw=True)
                if mapped is not None:
                    self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                    return self._render_ssa(mapped)
                return getattr(candidate, "describe", lambda: "expr()")()
            scan -= 1
        return "stack_top"

    @staticmethod
    def _branch_target(instruction: RawInstruction) -> int:
        return instruction.operand

    @staticmethod
    def _fallthrough_target(instruction: RawInstruction) -> int:
        return instruction.offset + 4

    @staticmethod
    def _classify_slot(operand: int) -> IRSlot:
        if operand < 0x1000:
            space = MemSpace.FRAME
        elif operand < 0x8000:
            space = MemSpace.GLOBAL
        else:
            space = MemSpace.CONST
        return IRSlot(space=space, index=operand)

    def _collect_memref(
        self,
        items: _ItemList,
        index: int,
        base_slot: Optional[IRSlot],
        base_name: Optional[str],
        instruction: RawInstruction,
    ) -> Tuple[Optional[MemRef], int]:
        components: List[Tuple[int, int]] = []
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                value = self._memref_component(candidate)
                if value is None:
                    break
                components.append((scan, value))
                scan -= 1
                continue
            if self._is_memref_bridge(candidate):
                scan -= 1
                continue
            break

        if not components:
            return None, index

        removal_indexes = [position for position, _ in components]
        removal_indexes.sort(reverse=True)
        for position in removal_indexes:
            removed = items.pop(position)
            self._ssa_bindings.pop(id(removed), None)
        index -= len(removal_indexes)

        components.reverse()
        bank = components[0][1] if components else None
        base_value = components[1][1] if len(components) > 1 else None
        page = instruction.operand >> 8
        offset = instruction.operand & 0xFF
        region, page_alias = self._memref_region(base_slot, bank, page)
        symbol = self._memref_symbol(region, bank, page, offset)
        memref = MemRef(
            region=region,
            bank=bank,
            base=base_value,
            page=page,
            offset=offset,
            symbol=symbol,
            page_alias=page_alias,
        )

        if base_name and symbol:
            self._ssa_aliases[base_name] = symbol

        return memref, index

    @staticmethod
    def _is_memref_bridge(node: Union[RawInstruction, IRNode]) -> bool:
        return isinstance(node, (IRLoad, IRStackDuplicate))

    def _memref_component(self, instruction: RawInstruction) -> Optional[int]:
        if instruction.event.delta != 0:
            return None
        if instruction.event.popped_types or instruction.event.pushed_types:
            return None
        mnemonic = instruction.mnemonic
        if not mnemonic.startswith("op_"):
            return None
        parts = mnemonic.split("_")
        if len(parts) != 3:
            return None
        try:
            high = int(parts[1], 16)
            low = int(parts[2], 16)
        except ValueError:
            return None
        return (high << 8) | low

    def _memref_region(
        self, base_slot: Optional[IRSlot], bank: Optional[int], page: Optional[int]
    ) -> Tuple[str, Optional[str]]:
        if base_slot is not None:
            return base_slot.space.name.lower(), None
        if bank is None:
            return "mem", None

        normalized = bank & 0xFFF0
        alias = MEMORY_BANK_ALIASES.get(normalized)
        page_alias = self._memref_page_alias(normalized, page)
        if alias is not None:
            return alias, page_alias

        label = self._memref_regions.get(bank)
        if label is None:
            label = f"bank_{bank:04X}"
            self._memref_regions[bank] = label
        return label, page_alias

    def _memref_page_alias(
        self, bank: Optional[int], page: Optional[int]
    ) -> Optional[str]:
        if page is None:
            return None
        if bank is not None:
            alias = MEMORY_PAGE_ALIASES.get((bank, page))
            if alias is not None:
                return alias
            alias = MEMORY_PAGE_ALIASES.get((bank & 0xFFF0, page))
            if alias is not None:
                return alias
        return MEMORY_PAGE_ALIASES.get((None, page))

    def _memref_symbol(
        self, region: str, bank: Optional[int], page: Optional[int], offset: Optional[int]
    ) -> Optional[str]:
        if bank is None:
            return None
        key = (region, bank, page, offset)
        alias = self._memref_symbols.get(key)
        if alias is not None:
            return alias
        prefix = region[:1].upper() if region else "M"
        if not prefix.isalpha():
            prefix = "M"
        alias_base = f"{prefix}_{bank:04X}"
        if page is not None:
            alias_base += f"_{page:02X}"
        if offset is not None:
            alias_base += f"_{offset:02X}"
        counter = self._memref_symbol_counters[alias_base]
        self._memref_symbol_counters[alias_base] += 1
        alias = alias_base if counter == 0 else f"{alias_base}_{counter}"
        self._memref_symbols[key] = alias
        return alias

    @staticmethod
    def _flag_literal_value(text: str) -> Optional[int]:
        if text.startswith("lit(0x") and text.endswith(")"):
            try:
                value = int(text[4:-1], 16)
            except ValueError:
                return None
            if value in {0x0166, 0x0266}:
                return value
        return None

    # ------------------------------------------------------------------
    # annotation helpers
    # ------------------------------------------------------------------
    def _is_annotation_only(self, instruction: RawInstruction) -> bool:
        if instruction.offset in self._annotation_offsets:
            return True
        if instruction.profile.is_literal_marker():
            return True
        if not instruction.annotations:
            return False
        for note in instruction.annotations:
            if note in ANNOTATION_MNEMONICS:
                return True
            if note.startswith("literal_marker") or note.startswith("inline_ascii_chunk"):
                return True
            if note.startswith("op_") and instruction.profile.kind is InstructionKind.LITERAL:
                return True
        return False

    def _format_annotation(self, instruction: RawInstruction) -> str:
        parts = [f"0x{instruction.offset:06X}"]
        if instruction.annotations:
            parts.extend(instruction.annotations)
        else:
            parts.append(instruction.mnemonic)
        return " ".join(parts)

    @staticmethod
    def _format_testset_var(instruction: RawInstruction) -> str:
        operand = instruction.operand
        return f"slot(0x{operand:04X})"


__all__ = ["IRNormalizer", "RawInstruction", "RawBlock"]
