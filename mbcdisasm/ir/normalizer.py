"""Normalisation pipeline that converts raw instructions into IR nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union, cast

from ..analyzer.instruction_profile import InstructionKind, InstructionProfile
from ..analyzer.stack import StackEvent, StackTracker
from ..instruction import read_instructions
from ..knowledge import KnowledgeBase
from ..mbc import MbcContainer, Segment
from .model import (
    IRAsciiFinalize,
    IRAsciiPreamble,
    IRBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    IRCallSetup,
    IRIf,
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
    IRStore,
    IRStackDuplicate,
    IRStackDrop,
    IRTablePatch,
    IRTailCallSetup,
    IRTestSetBranch,
    MemSpace,
    NormalizerMetrics,
)


ANNOTATION_MNEMONICS = {"literal_marker"}
RETURN_NIBBLE_MODES = {0x29, 0x2C, 0x32, 0x41, 0x65, 0x69, 0x6C}


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
        items = _ItemList(block.instructions)
        metrics = NormalizerMetrics()

        self._pass_literals(items, metrics)
        self._pass_literal_preambles(items, metrics)
        self._pass_stack_manipulation(items, metrics)
        self._pass_ascii_preamble(items, metrics)
        self._pass_tailcall_setup(items, metrics)
        self._pass_call_setup_sequences(items, metrics)
        self._pass_calls_and_returns(items, metrics)
        self._pass_aggregates(items, metrics)
        self._pass_table_patches(items, metrics)
        self._pass_ascii_helpers(items, metrics)
        self._pass_branches(items, metrics)
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

    def _pass_literal_preambles(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        pattern = (0x6704, 0x0067, 0x0400)
        index = 0
        while index <= len(items) - len(pattern):
            segment = items[index : index + len(pattern)]
            if not all(
                isinstance(node, IRLiteral) and node.value == expected
                for node, expected in zip(segment, pattern)
            ):
                index += 1
                continue

            start = index
            values: List[int] = []
            while index <= len(items) - len(pattern):
                window = items[index : index + len(pattern)]
                if all(
                    isinstance(node, IRLiteral) and node.value == expected
                    for node, expected in zip(window, pattern)
                ):
                    values.extend(cast(IRLiteral, lit).value for lit in window)
                    index += len(pattern)
                    continue
                break

            reduced = False
            if index < len(items):
                next_item = items[index]
                if (
                    isinstance(next_item, RawInstruction)
                    and next_item.mnemonic == "reduce_pair"
                    and next_item.operand == 0
                ):
                    reduced = True
                    index += 1

            block = IRLiteralBlock(values=tuple(values), reduced=reduced)
            items.replace_slice(start, index, [block])
            index = start + 1

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

    def _pass_ascii_preamble(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index + 2 < len(items):
            first = items[index]
            second = items[index + 1]
            third = items[index + 2]

            if not (
                isinstance(first, RawInstruction)
                and isinstance(second, RawInstruction)
                and isinstance(third, RawInstruction)
            ):
                index += 1
                continue

            if not (
                self._matches_opcode(first, 0x72, 0x23)
                and first.operand == 0x4F00
                and self._matches_opcode(second, 0x31, 0x30)
                and second.operand == 0x2C00
            ):
                index += 1
                continue

            if not self._is_ascii_marker_tail(third):
                index += 1
                continue

            items.replace_slice(index, index + 3, [IRAsciiPreamble()])
            continue

    def _is_ascii_marker_tail(self, instruction: RawInstruction) -> bool:
        if instruction.mnemonic == "stack_shuffle" and instruction.operand == 0x4B08:
            return True
        if self._matches_opcode(instruction, 0x66) and instruction.operand == 0x4B08:
            return True
        if self._matches_opcode(instruction, 0x4B, 0x08):
            return True
        return False

    def _pass_tailcall_setup(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            if index + 3 >= len(items):
                break

            first = items[index]
            second = items[index + 1]
            third = items[index + 2]
            fourth = items[index + 3]

            if not (
                isinstance(first, RawInstruction)
                and isinstance(second, RawInstruction)
                and isinstance(third, RawInstruction)
                and isinstance(fourth, RawInstruction)
            ):
                index += 1
                continue

            if not (
                self._matches_opcode(first, 0x3D, 0x30)
                and self._matches_opcode(second, 0x32, 0x29)
                and third.profile.word.opcode == 0x4B
                and self._matches_opcode(fourth, 0xF0, 0x4B)
            ):
                index += 1
                continue

            setup = IRTailCallSetup(
                frame_operand=first.operand,
                layout_operand=second.operand,
                shuffle_operand=third.operand,
                helper_operand=fourth.operand,
            )
            items.replace_slice(index, index + 4, [setup])
            index += 1

    def _pass_call_setup_sequences(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction) or not self._is_call_setup_instruction(item):
                index += 1
                continue

            start = index
            operations: List[Tuple[str, int]] = []
            anchor = False

            while index < len(items):
                candidate = items[index]
                if isinstance(candidate, RawInstruction) and self._is_call_setup_instruction(candidate):
                    operations.append((candidate.mnemonic, candidate.operand))
                    if candidate.profile.word.opcode == 0x4A or candidate.mnemonic.startswith("op_4A_"):
                        anchor = True
                    index += 1
                    continue
                break

            if anchor and operations:
                items.replace_slice(start, index, [IRCallSetup(operations=tuple(operations))])
                index = start + 1
                continue

            index = start + 1

    def _literal_from_instruction(self, instruction: RawInstruction) -> Optional[IRNode]:
        profile = instruction.profile

        if profile.is_literal_marker():
            self._annotation_offsets.add(instruction.offset)
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

    def _pass_table_patches(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction) or item.profile.word.opcode != 0x2C:
                index += 1
                continue

            start = index
            operations: List[str] = []

            while index < len(items):
                candidate = items[index]
                if isinstance(candidate, RawInstruction):
                    if candidate.profile.word.opcode == 0x2C:
                        operations.append(self._format_table_operation(candidate))
                        index += 1
                        continue
                    if candidate.mnemonic.startswith("fanout"):
                        operations.append(self._format_table_operation(candidate))
                        index += 1
                        continue
                break

            if len(operations) >= 2:
                items.replace_slice(start, index, [IRTablePatch(operations=tuple(operations))])
                index = start + 1
                continue

            index = start + 1

    def _pass_ascii_helpers(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        ascii_helpers = {0xF172, 0x7223, 0x3D30}
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, RawInstruction) and item.mnemonic == "call_helpers" and item.operand in ascii_helpers:
                items.replace_slice(index, index + 1, [IRAsciiFinalize(helper=item.operand)])
                index += 1
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
                )
                items.replace_slice(index, index + 1, [literal])
                return literal
        return None

    @staticmethod
    def _literal_repr(node: IRLiteral) -> str:
        return node.describe()

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
    @staticmethod
    def _matches_opcode(instruction: RawInstruction, opcode: int, mode: Optional[int] = None) -> bool:
        if instruction.profile.word.opcode != opcode:
            return False
        if mode is not None and instruction.profile.word.mode != mode:
            return False
        return True

    @staticmethod
    def _is_call_setup_instruction(instruction: RawInstruction) -> bool:
        mnemonic = instruction.mnemonic
        if mnemonic.startswith("stack_teardown"):
            return True
        if mnemonic.startswith("stack_shuffle"):
            return True
        if mnemonic.startswith("fanout"):
            return True
        if mnemonic.startswith("op_4A_") or instruction.profile.word.opcode == 0x4A:
            return True
        if mnemonic.startswith("op_4B_") or instruction.profile.word.opcode == 0x4B:
            return True
        return False

    @staticmethod
    def _format_table_operation(instruction: RawInstruction) -> str:
        return f"{instruction.mnemonic}(0x{instruction.operand:04X})"

    @staticmethod
    def _flag_name(value: int) -> Optional[str]:
        mapping = {0x0166: "FLAG_0166", 0x0266: "FLAG_0266"}
        return mapping.get(value)

    def _describe_stack_top(self, items: _ItemList, index: int) -> str:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    return self._describe_value(candidate)
            elif isinstance(candidate, IRLiteral):
                return candidate.describe()
            elif isinstance(candidate, IRLiteralChunk):
                return candidate.describe()
            elif isinstance(candidate, IRStackDuplicate):
                return candidate.value
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
                    flag = self._flag_name(candidate.operand)
                    if flag is not None:
                        return f"check_flag({flag})"
                    if skip_literals and candidate.mnemonic == "push_literal":
                        scan -= 1
                        continue
                    return self._describe_value(candidate)
                if skip_literals:
                    flag = self._flag_name(candidate.operand)
                    if flag is not None:
                        return f"check_flag({flag})"
                    return self._describe_value(candidate)
            if isinstance(candidate, IRLiteral):
                flag = self._flag_name(candidate.value)
                if flag is not None:
                    return f"check_flag({flag})"
                if skip_literals:
                    scan -= 1
                    continue
                return candidate.describe()
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
