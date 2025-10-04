"""Normalisation pipeline that converts raw instructions into IR nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from ..analyzer.instruction_profile import InstructionKind, InstructionProfile
from ..analyzer.stack import StackEvent, StackTracker
from ..instruction import read_instructions
from ..knowledge import KnowledgeBase
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
    IRCallPreparation,
    IRCallReturn,
    IRFlagCheck,
    IRFunctionPrologue,
    IRLiteral,
    IRLiteralBlock,
    IRLiteralChunk,
    IRLoad,
    IRNode,
    IRPredicateAssign,
    IRProgram,
    IRRaw,
    IRReturn,
    IRSegment,
    IRSlot,
    IRStore,
    IRStackDuplicate,
    IRStackDrop,
    IRStackValueAssign,
    IRTailcallAscii,
    IRTablePatch,
    IRTailcallFrame,
    IRTestSetBranch,
    IRIf,
    MemSpace,
    NormalizerMetrics,
)


ANNOTATION_MNEMONICS = {"literal_marker"}
RETURN_NIBBLE_MODES = {0x29, 0x2C, 0x32, 0x41, 0x65, 0x69, 0x6C}


LITERAL_MARKER_HINTS: Dict[int, str] = {
    0x0067: "literal_hint",
    0x6704: "literal_hint",
    0x0400: "literal_hint",
    0x0110: "literal_hint",
}


@dataclass(frozen=True)
class RawInstruction:
    """Wrapper that couples a profile with stack tracking details."""

    profile: InstructionProfile
    event: StackEvent
    annotations: Tuple[str, ...]

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


@dataclass(frozen=True)
class _StackValueDescriptor:
    """Details about a value currently residing on the VM stack."""

    name: str
    expression: str
    source_offset: Optional[int]
    producer_index: Optional[int]


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

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge
        self._annotation_offsets: Set[int] = set()
        self._stack_name_counters: Dict[int, int] = {}
        self._value_name_counters: Dict[int, int] = {}
        self._value_names: Dict[Union[RawInstruction, IRNode], str] = {}
        self._value_definitions: Dict[str, str] = {}

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
        for index, (profile, event) in enumerate(zip(executable, events)):
            notes = tuple(annotations.get(index, ()))
            raw_instructions.append(RawInstruction(profile=profile, event=event, annotations=notes))

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
        self._stack_name_counters.clear()
        self._value_name_counters.clear()
        self._value_names.clear()
        self._value_definitions.clear()
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
        self._pass_tailcall_frames(items)
        self._pass_table_patches(items)
        self._pass_ascii_finalize(items)
        self._pass_branches(items, metrics)
        self._pass_branch_predicates(items)
        self._pass_flag_checks(items)
        self._pass_function_prologues(items)
        self._pass_ascii_wrappers(items)
        self._pass_ascii_headers(items)
        self._pass_call_return_templates(items)
        self._pass_indirect_access(items, metrics)
        self._verify_branch_conditions(items)

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
                node = IRStackDuplicate(
                    value=value, copies=copies + 1, source_offset=item.offset
                )
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
                    source_offset=instruction.offset,
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
                source_offset=instruction.offset,
            )

        if profile.kind is InstructionKind.LITERAL and instruction.pushes_value():
            return IRLiteral(
                value=instruction.operand,
                mode=profile.mode,
                source=profile.mnemonic,
                annotations=instruction.annotations,
                source_offset=instruction.offset,
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
                call = IRCall(target=item.operand, args=tuple(args), tail=mnemonic == "tailcall_dispatch")
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
                args.append(candidate.describe())
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
        while index < len(items):
            item = items[index]
            if isinstance(item, RawInstruction) and item.mnemonic == "return_values":
                count, varargs = self._resolve_return_signature(items, index)
                values = tuple(f"ret{i}" for i in range(count)) if count else tuple()
                if varargs and not values:
                    values = ("ret*",)
                items.replace_slice(index, index + 1, [IRReturn(values=values, varargs=varargs)])
                metrics.returns += 1
                return
            if isinstance(item, RawInstruction) and item.profile.kind in {
                InstructionKind.STACK_TEARDOWN,
                InstructionKind.META,
            }:
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
                        items.replace_slice(index, index + 2, [updated])
                        metrics.reduce_replaced += 1
                        continue
                if isinstance(second, IRBuildTuple):
                    values = self._parse_literal_list(second.elements)
                    block = self._literal_block_from_values(
                        values, reducer=first.mnemonic, operand=first.operand
                    )
                    if block is not None:
                        items.replace_slice(index, index + 2, [block])
                        metrics.reduce_replaced += 1
                        continue
                if isinstance(second, IRLiteral):
                    values, end = self._collect_literal_block_literals(items, index + 1)
                    block = self._literal_block_from_values(
                        values, reducer=first.mnemonic, operand=first.operand
                    )
                    if block is not None:
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
                and third.operand == 0x4B08
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
        allowed_prefix = {"stack_shuffle", "fanout"}
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction) or item.mnemonic != "op_4A_05":
                index += 1
                continue

            steps: List[Tuple[str, int]] = [(item.mnemonic, item.operand)]
            start = index
            scan = index - 1
            while scan >= 0:
                candidate = items[scan]
                if isinstance(candidate, (IRLiteral, IRLiteralChunk)):
                    scan -= 1
                    continue
                if isinstance(candidate, RawInstruction):
                    mnemonic = candidate.mnemonic
                    if mnemonic in allowed_prefix or mnemonic.startswith("stack_teardown"):
                        steps.insert(0, (mnemonic, candidate.operand))
                        start = scan
                        scan -= 1
                        continue
                break

            if start < index:
                items.replace_slice(start, index + 1, [IRCallPreparation(steps=tuple(steps))])
                index = start
                continue

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
                    isinstance(items[pos], IRAsciiHeader) for pos in range(index)
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

            stack_value_node: Optional[IRStackValueAssign] = None
            predicate_node: Optional[IRPredicateAssign] = None
            branch: Optional[IRIf] = None
            branch_index = scan
            if branch_index < len(items) and isinstance(items[branch_index], IRStackValueAssign):
                stack_value_node = items[branch_index]
                branch_index += 1
            if branch_index < len(items) and isinstance(items[branch_index], IRPredicateAssign):
                predicate_node = items[branch_index]
                branch_index += 1
            if branch_index < len(items) and isinstance(items[branch_index], IRIf):
                candidate = items[branch_index]
                first_chunk = ascii_chunks[0]
                cond_matches = candidate.condition in {first_chunk, "stack_top"}
                if not cond_matches and predicate_node is not None:
                    if (
                        predicate_node.name == candidate.condition
                        and predicate_node.operands
                    ):
                        operand = predicate_node.operands[0]
                        cond_matches = operand == first_chunk
                        if not cond_matches:
                            operand_definition = self._resolve_value_definition(operand)
                            cond_matches = operand_definition == first_chunk
                        if (
                            not cond_matches
                            and stack_value_node is not None
                            and operand == stack_value_node.name
                        ):
                            cond_matches = (
                                self._resolve_value_definition(stack_value_node.value)
                                == first_chunk
                            )
                if cond_matches and item.tail:
                    branch = candidate
                else:
                    stack_value_node = None
                    predicate_node = None
            else:
                stack_value_node = None
                predicate_node = None

            ascii_tuple = tuple(ascii_chunks)
            if branch is not None:
                node = IRTailcallAscii(
                    target=item.target,
                    args=item.args,
                    ascii_chunks=ascii_tuple,
                    condition=branch.condition,
                    then_target=branch.then_target,
                    else_target=branch.else_target,
                )
                replacement: List[IRNode] = [node]
                if predicate_node is not None:
                    if stack_value_node is not None:
                        replacement.insert(0, stack_value_node)
                    replacement.insert(len(replacement) - 1, predicate_node)
                items.replace_slice(index, branch_index + 1, replacement)
                continue

            node = IRAsciiWrapperCall(
                target=item.target,
                args=item.args,
                ascii_chunks=ascii_tuple,
                tail=item.tail,
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
                        piece = IRLiteralChunk(
                            data=segment,
                            source=literal.source,
                            source_offset=literal.source_offset,
                        )
                        parts.append(piece.describe())
                    if len(parts) >= 2:
                        items.replace_slice(0, 1, [IRAsciiHeader(chunks=tuple(parts))])

    def _pass_call_return_templates(self, items: _ItemList) -> None:
        index = 0
        while index < len(items) - 1:
            call = items[index]
            if isinstance(call, IRCall):
                nxt = items[index + 1]
                if isinstance(nxt, IRReturn):
                    node = IRCallReturn(
                        target=call.target,
                        args=call.args,
                        tail=call.tail,
                        returns=nxt.values,
                        varargs=nxt.varargs,
                    )
                    items.replace_slice(index, index + 2, [node])
                    continue
            index += 1

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
                    source_offset=item.offset,
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
                    source_offset=item.offset,
                )
                items.replace_slice(index, index + 1, [node])
                metrics.if_branches += 1
                continue

            index += 1

    def _pass_branch_predicates(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, IRIf):
                index += 1
                continue

            if not self._condition_requires_fix(item.condition):
                index += 1
                continue

            predicate_name, branch_index = self._ensure_branch_predicate(items, index, item)
            branch = items[branch_index]
            rewritten = IRIf(
                condition=predicate_name,
                then_target=branch.then_target,
                else_target=branch.else_target,
                source_offset=branch.source_offset,
            )
            items.replace_slice(branch_index, branch_index + 1, [rewritten])
            index = branch_index + 1

    @staticmethod
    def _condition_requires_fix(condition: str) -> bool:
        if not condition:
            return True
        lowered = condition.lower()
        if lowered.startswith(("pred@", "test@", "cmp@", "bool@")):
            return False
        if condition.startswith("slot("):
            return False
        invalid_prefixes = (
            "lit(",
            "tuple(",
            "array(",
            "map(",
            "ascii(",
            "literal_block",
            "dup ",
        )
        if condition == "stack_top":
            return True
        return condition.startswith(invalid_prefixes)

    @staticmethod
    def _make_predicate_name(branch: IRIf) -> str:
        return IRNormalizer._make_predicate_name_for_offset(branch.source_offset)

    @staticmethod
    def _make_predicate_name_for_offset(offset: Optional[int]) -> str:
        if offset is None:
            return "pred@unknown"
        return f"pred@0x{offset:04X}"

    def _make_stack_value_name(self, offset: Optional[int]) -> str:
        if offset is None:
            key = -1
            base = "stack@unknown"
        else:
            key = offset
            base = f"stack@0x{offset:04X}"
        counter = self._stack_name_counters.get(key, 0)
        self._stack_name_counters[key] = counter + 1
        if counter:
            return f"{base}#{counter}"
        return base

    def _make_value_name_for_offset(self, offset: Optional[int]) -> str:
        if offset is None:
            key = -1
            base = "value@unknown"
        else:
            key = offset
            base = f"value@0x{offset:04X}"
        counter = self._value_name_counters.get(key, 0)
        self._value_name_counters[key] = counter + 1
        if counter:
            return f"{base}#{counter}"
        return base

    def _register_value_descriptor(
        self,
        node: Union[RawInstruction, IRNode],
        *,
        expression: str,
        offset: Optional[int],
        producer_index: Optional[int],
    ) -> _StackValueDescriptor:
        name = self._value_names.get(node)
        if name is None:
            name = self._make_value_name_for_offset(offset)
            self._value_names[node] = name
        self._value_definitions.setdefault(name, expression)
        return _StackValueDescriptor(
            name=name,
            expression=expression,
            source_offset=offset,
            producer_index=producer_index,
        )

    def _resolve_value_definition(self, name: str) -> Optional[str]:
        return self._value_definitions.get(name)

    def _ensure_branch_predicate(
        self, items: _ItemList, branch_index: int, branch: IRIf
    ) -> Tuple[str, int]:
        adjacent = self._predicate_adjacent_to_branch(items, branch_index)
        if adjacent is not None:
            return adjacent.name, branch_index

        test_index, operator = self._find_test_instruction(items, branch_index)
        if test_index is not None and operator is not None:
            predicate_name, inserted = self._materialise_test_instruction(
                items, test_index, operator
            )
            if test_index < branch_index:
                branch_index += inserted
            return predicate_name, branch_index

        predicate_name, inserted = self._synthesise_truthy_predicate(
            items, branch_index, branch
        )
        branch_index += inserted
        return predicate_name, branch_index

    def _predicate_adjacent_to_branch(
        self, items: _ItemList, branch_index: int
    ) -> Optional[IRPredicateAssign]:
        if branch_index > 0:
            previous = items[branch_index - 1]
            if isinstance(previous, IRPredicateAssign):
                return previous
        if branch_index > 1:
            stack_candidate = items[branch_index - 1]
            predicate_candidate = items[branch_index - 2]
            if isinstance(stack_candidate, IRStackValueAssign) and isinstance(
                predicate_candidate, IRPredicateAssign
            ):
                return predicate_candidate
        return None

    def _find_test_instruction(
        self, items: _ItemList, branch_index: int
    ) -> Tuple[Optional[int], Optional[str]]:
        limit = max(0, branch_index - 12)
        scan = branch_index - 1
        while scan >= limit:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                operator = self._classify_predicate_instruction(candidate)
                if operator is not None:
                    return scan, operator
            scan -= 1
        return None, None

    def _classify_predicate_instruction(self, instruction: RawInstruction) -> Optional[str]:
        mnemonic = instruction.mnemonic.lower()
        if mnemonic == "test_branch":
            return "test"
        if mnemonic == "testset_branch":
            return "testset"
        if "test" in mnemonic:
            return "test"
        summary = (instruction.profile.summary or "").lower()
        if "compare" in summary or "cmp" in mnemonic:
            return "compare"
        return None

    def _materialise_test_instruction(
        self, items: _ItemList, index: int, operator: str
    ) -> Tuple[str, int]:
        instruction = items[index]
        assert isinstance(instruction, RawInstruction)
        predicate_name = self._make_predicate_name_for_offset(instruction.offset)

        insert_at = index + 1
        inserted = 0

        if operator == "testset":
            if (
                insert_at < len(items)
                and isinstance(items[insert_at], IRPredicateAssign)
                and items[insert_at].name == predicate_name
            ):
                return predicate_name, inserted
            assignment = IRPredicateAssign(
                name=predicate_name,
                operator=operator,
                operands=(self._format_testset_var(instruction),),
                source_offset=instruction.offset,
                synthetic=False,
            )
            items.insert(insert_at, assignment)
            inserted += 1
            return predicate_name, inserted

        descriptor = self._stack_value_descriptor(items, index)
        stack_name = self._make_stack_value_name(instruction.offset)
        stack_node = IRStackValueAssign(
            name=stack_name,
            value=descriptor.name,
            source_offset=instruction.offset,
            synthetic=False,
        )
        self._value_names[stack_node] = stack_name
        self._value_definitions[stack_name] = f"pop({descriptor.name})"
        predicate_node = IRPredicateAssign(
            name=predicate_name,
            operator=operator,
            operands=(stack_name,),
            source_offset=instruction.offset,
            synthetic=False,
        )

        if insert_at < len(items) and isinstance(items[insert_at], IRStackValueAssign):
            existing_stack = items[insert_at]
            if existing_stack.name == stack_name and existing_stack.value == descriptor.name:
                insert_at += 1
                if (
                    insert_at < len(items)
                    and isinstance(items[insert_at], IRPredicateAssign)
                    and items[insert_at].name == predicate_name
                ):
                    return predicate_name, inserted

        items.insert(insert_at, stack_node)
        items.insert(insert_at + 1, predicate_node)
        inserted += 2
        return predicate_name, inserted

    def _synthesise_truthy_predicate(
        self, items: _ItemList, index: int, branch: IRIf
    ) -> Tuple[str, int]:
        predicate_name = self._make_predicate_name(branch)
        descriptor = self._stack_value_descriptor(items, index)
        stack_name = self._make_stack_value_name(branch.source_offset)
        stack_node = IRStackValueAssign(
            name=stack_name,
            value=descriptor.name,
            source_offset=branch.source_offset,
            synthetic=True,
        )
        self._value_names[stack_node] = stack_name
        self._value_definitions[stack_name] = f"pop({descriptor.name})"
        predicate_node = IRPredicateAssign(
            name=predicate_name,
            operator="truthy",
            operands=(stack_name,),
            source_offset=branch.source_offset,
            synthetic=True,
        )
        items.insert(index, stack_node)
        items.insert(index + 1, predicate_node)
        return predicate_name, 2

    def _verify_branch_conditions(self, items: _ItemList) -> None:
        problems: List[str] = []
        for node in items:
            if isinstance(node, IRIf) and self._condition_requires_fix(node.condition):
                location = f"0x{node.source_offset:04X}" if node.source_offset is not None else "unknown"
                problems.append(f"invalid predicate {node.condition} at {location}")
        if problems:
            details = "; ".join(problems)
            raise ValueError(f"IR branch predicate verification failed: {details}")

    def _pass_indirect_access(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            kind = item.event.kind
            if kind in {InstructionKind.INDIRECT_LOAD, InstructionKind.INDIRECT}:
                slot = self._classify_slot(item.operand)
                node = IRLoad(slot=slot)
                items.replace_slice(index, index + 1, [node])
                metrics.loads += 1
                continue
            if kind is InstructionKind.INDIRECT_STORE:
                slot = self._classify_slot(item.operand)
                node = IRStore(slot=slot)
                items.replace_slice(index, index + 1, [node])
                metrics.stores += 1
                continue

            index += 1

    # ------------------------------------------------------------------
    # description helpers
    # ------------------------------------------------------------------
    def _stack_value_descriptor(self, items: _ItemList, index: int) -> _StackValueDescriptor:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            mapped = self._value_names.get(candidate)
            if mapped is not None:
                expression = self._value_definitions.get(mapped)
                if expression is None:
                    describe = getattr(candidate, "describe", None)
                    if callable(describe):
                        expression = describe()
                    else:
                        expression = mapped
                    self._value_definitions[mapped] = expression
                offset = getattr(candidate, "source_offset", None)
                return _StackValueDescriptor(
                    name=mapped,
                    expression=expression,
                    source_offset=offset,
                    producer_index=scan,
                )

            if isinstance(candidate, IRPredicateAssign):
                scan -= 1
                continue

            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    expression = self._describe_value(candidate)
                    return self._register_value_descriptor(
                        candidate,
                        expression=expression,
                        offset=candidate.offset,
                        producer_index=scan,
                    )

            elif isinstance(candidate, IRLiteral):
                return self._register_value_descriptor(
                    candidate,
                    expression=candidate.describe(),
                    offset=candidate.source_offset,
                    producer_index=scan,
                )

            elif isinstance(candidate, IRLiteralChunk):
                return self._register_value_descriptor(
                    candidate,
                    expression=candidate.describe(),
                    offset=candidate.source_offset,
                    producer_index=scan,
                )

            elif isinstance(candidate, IRStackDuplicate):
                return self._register_value_descriptor(
                    candidate,
                    expression=candidate.describe(),
                    offset=candidate.source_offset,
                    producer_index=scan,
                )

            elif isinstance(candidate, IRStackValueAssign):
                self._value_definitions.setdefault(
                    candidate.name, f"pop({candidate.value})"
                )
                return _StackValueDescriptor(
                    name=candidate.name,
                    expression=self._value_definitions[candidate.name],
                    source_offset=candidate.source_offset,
                    producer_index=scan,
                )

            elif isinstance(candidate, IRNode):
                describe = getattr(candidate, "describe", None)
                expression = describe() if callable(describe) else candidate.__class__.__name__
                offset = getattr(candidate, "source_offset", None)
                return self._register_value_descriptor(
                    candidate,
                    expression=expression,
                    offset=offset,
                    producer_index=scan,
                )

            scan -= 1

        name = self._make_value_name_for_offset(None)
        self._value_definitions.setdefault(name, "stack_top")
        return _StackValueDescriptor(
            name=name,
            expression="stack_top",
            source_offset=None,
            producer_index=None,
        )

    def _describe_stack_top(self, items: _ItemList, index: int) -> str:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            mapped = self._value_names.get(candidate)
            if mapped is not None:
                return mapped
            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    return self._describe_value(candidate)
            elif isinstance(candidate, IRLiteral):
                return candidate.describe()
            elif isinstance(candidate, IRLiteralChunk):
                return candidate.describe()
            elif isinstance(candidate, IRStackDuplicate):
                return candidate.value
            elif isinstance(candidate, IRStackValueAssign):
                return candidate.name
            elif isinstance(candidate, IRNode):
                describe = getattr(candidate, "describe", None)
                if callable(describe):
                    return describe()
            scan -= 1
        return "stack_top"

    def _describe_value(self, instruction: RawInstruction) -> str:
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
                        scan -= 1
                        continue
                    return self._describe_value(candidate)
                if skip_literals:
                    return self._describe_value(candidate)
            if isinstance(candidate, IRLiteral) and skip_literals:
                scan -= 1
                continue
            if isinstance(candidate, IRLiteralChunk) and skip_literals:
                scan -= 1
                continue
            if isinstance(candidate, IRStackDuplicate):
                return candidate.value
            if isinstance(candidate, IRNode):
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
