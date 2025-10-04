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
    IRAsciiBlock,
    IRAsciiPrologue,
    IRBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    IRCallArgPrep,
    IRCheckFlag,
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
    IRTableChunk,
    IRTailcallPrep,
    IRTestSetBranch,
    IRIf,
    MemSpace,
    NormalizerMetrics,
)


ANNOTATION_MNEMONICS = {"literal_marker"}
RETURN_NIBBLE_MODES = {0x29, 0x2C, 0x32, 0x41, 0x65, 0x69, 0x6C}
LITERAL_BLOCK_VALUES = {0x0067, 0x6704}
ASCII_PROLOGUE_FIRST = ("op_72_23", 0x4F00)
ASCII_PROLOGUE_SECOND = ("op_31_30", 0x2C00)
ASCII_PROLOGUE_THIRD_MNEMONICS = {"op_4B_08", "stack_shuffle"}
ASCII_HELPER_OPERANDS = {0xF172, 0x7223, 0x3D30}
FLAG_LITERAL_MAP = {0x0166: "FLAG_0166", 0x0266: "FLAG_0266"}


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
        self._pass_ascii_prologues(items)
        self._pass_call_arg_preparation(items)
        self._pass_stack_manipulation(items, metrics)
        self._pass_ascii_helpers(items)
        self._pass_calls_and_returns(items, metrics)
        self._pass_table_chunks(items)
        self._pass_aggregates(items, metrics)
        self._pass_branches(items, metrics)
        self._pass_flag_checks(items)
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

    def _pass_ascii_prologues(self, items: _ItemList) -> None:
        index = 0
        while index + 2 < len(items):
            first, second, third = items[index : index + 3]
            if not (
                isinstance(first, RawInstruction)
                and isinstance(second, RawInstruction)
                and isinstance(third, RawInstruction)
            ):
                index += 1
                continue

            if (first.mnemonic, first.operand) != ASCII_PROLOGUE_FIRST:
                index += 1
                continue
            if (second.mnemonic, second.operand) != ASCII_PROLOGUE_SECOND:
                index += 1
                continue
            if third.operand != 0x4B08 or third.mnemonic not in ASCII_PROLOGUE_THIRD_MNEMONICS:
                index += 1
                continue

            node = IRAsciiPrologue(
                marker_operand=first.operand,
                layout_operand=second.operand,
                shuffle_operand=third.operand,
            )
            items.replace_slice(index, index + 3, [node])
            continue

    def _pass_call_arg_preparation(self, items: _ItemList) -> None:
        index = 0
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
                if isinstance(candidate, RawInstruction) and self._is_call_prep_component(candidate):
                    steps.append((candidate.mnemonic, candidate.operand))
                    start = scan
                    scan -= 1
                    continue
                break

            if len(steps) > 1:
                steps.reverse()
                node = IRCallArgPrep(steps=tuple(steps))
                items.replace_slice(start, index + 1, [node])
                index = start
                continue

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

    def _pass_ascii_helpers(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not (isinstance(item, RawInstruction) and item.mnemonic == "call_helpers"):
                index += 1
                continue

            if item.operand not in ASCII_HELPER_OPERANDS:
                index += 1
                continue

            collected = self._collect_ascii_block(items, index)
            if collected is None:
                index += 1
                continue

            start, data = collected
            node = IRAsciiBlock(
                data=data,
                helper_operand=item.operand,
                annotations=item.annotations,
            )
            items.replace_slice(start, index + 1, [node])
            index = start
            continue

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
                if mnemonic == "tailcall_dispatch":
                    prep = self._extract_tailcall_prep(items, index)
                    if prep is not None:
                        start_prep, node = prep
                        items.replace_slice(start_prep, index, [node])
                        index = start_prep + 1
                        item = items[index]
                        assert isinstance(item, RawInstruction)
                        mnemonic = item.mnemonic

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

    def _pass_table_chunks(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            first = items[index]
            if not (isinstance(first, RawInstruction) and first.mnemonic == "op_2C_00"):
                index += 1
                continue

            if index + 2 >= len(items):
                index += 1
                continue

            second = items[index + 1]
            third = items[index + 2]
            if not (
                isinstance(second, RawInstruction)
                and isinstance(third, RawInstruction)
                and second.mnemonic == "op_2C_02"
                and third.mnemonic == "op_2C_03"
            ):
                index += 1
                continue

            operands = (first.operand, second.operand, third.operand)
            if not all(0x6600 <= operand <= 0x66FF for operand in operands):
                index += 1
                continue

            extras: List[Tuple[str, int]] = []
            end = index + 3
            while end < len(items):
                candidate = items[end]
                if isinstance(candidate, RawInstruction) and self._is_table_chunk_extra(candidate):
                    extras.append((candidate.mnemonic, candidate.operand))
                    end += 1
                    continue
                break

            node = IRTableChunk(
                base_operand=operands[0],
                key_operand=operands[1],
                value_operand=operands[2],
                extras=tuple(extras),
            )
            items.replace_slice(index, end, [node])
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

            literal_block = self._recognize_literal_block(literal_nodes, reducers)
            if literal_block is not None:
                metrics.aggregates += 1
                metrics.reduce_replaced += len(reducers)
                replacement_sequence: List[Union[RawInstruction, IRNode]] = [literal_block]
                if spacers:
                    replacement_sequence.extend(spacers)
                items.replace_slice(index, scan, replacement_sequence)
                index += 1
                continue

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

    def _recognize_literal_block(
        self, literals: Sequence[IRLiteral], reducers: Sequence[RawInstruction]
    ) -> Optional[IRLiteralBlock]:
        if len(literals) < 3 or len(literals) % 3 != 0:
            return None

        if len(literals) // 3 < 2:
            return None

        values = [literal.value for literal in literals]
        chunks = [values[pos : pos + 3] for pos in range(0, len(values), 3)]

        first_chunk = chunks[0]
        if first_chunk[2] != 0x0400:
            return None

        base_pair = (first_chunk[0], first_chunk[1])
        if set(base_pair) != LITERAL_BLOCK_VALUES:
            return None

        mirrored = False
        for chunk in chunks:
            if len(chunk) != 3:
                return None
            a, b, terminator = chunk
            if terminator != first_chunk[2]:
                return None
            if {a, b} != LITERAL_BLOCK_VALUES:
                return None
            if (a, b) != base_pair:
                mirrored = True

        high = max(base_pair)
        low = min(base_pair)
        canonical_pair = (high, low)
        return IRLiteralBlock(
            pair=canonical_pair,
            terminator=first_chunk[2],
            count=len(chunks),
            reducers=len(reducers),
            mirrored=mirrored,
        )

    @staticmethod
    def _literal_repr(node: IRLiteral) -> str:
        return node.describe()

    def _match_flag_literal(self, items: _ItemList, index: int) -> Optional[Tuple[int, int]]:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, IRLiteral):
                flag_name = FLAG_LITERAL_MAP.get(candidate.value)
                if flag_name is not None:
                    return scan, candidate.value
                return None
            if isinstance(candidate, RawInstruction):
                if (
                    candidate.profile.kind is InstructionKind.LITERAL
                    and candidate.operand in FLAG_LITERAL_MAP
                ):
                    literal = IRLiteral(
                        value=candidate.operand,
                        mode=candidate.profile.mode,
                        source=candidate.mnemonic,
                        annotations=candidate.annotations,
                    )
                    items.replace_slice(scan, scan + 1, [literal])
                    return scan, candidate.operand
                break
            if isinstance(candidate, IRNode):
                break
            scan -= 1
        return None

    @staticmethod
    def _parse_flag_condition(condition: str) -> Optional[int]:
        prefix = "lit(0x"
        suffix = ")"
        if condition.startswith(prefix) and condition.endswith(suffix):
            token = condition[len(prefix) : -len(suffix)]
            try:
                value = int(token, 16)
            except ValueError:
                return None
            if value in FLAG_LITERAL_MAP:
                return value
        return None

    @staticmethod
    def _is_call_prep_component(instruction: RawInstruction) -> bool:
        mnemonic = instruction.mnemonic
        if mnemonic.startswith("op_4B_"):
            return True
        if mnemonic == "stack_shuffle":
            return True
        if mnemonic.startswith("stack_teardown"):
            return True
        if mnemonic.startswith("fanout"):
            return True
        return False

    def _collect_ascii_block(self, items: _ItemList, index: int) -> Optional[Tuple[int, bytes]]:
        ascii_chunks: List[IRLiteralChunk] = []
        start = index
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, IRLiteralChunk):
                ascii_chunks.append(candidate)
                start = scan
                scan -= 1
                continue
            if isinstance(candidate, RawInstruction) and self._is_annotation_only(candidate):
                scan -= 1
                continue
            break

        if not ascii_chunks:
            return None

        ascii_chunks.reverse()
        data = b"".join(chunk.data for chunk in ascii_chunks)
        return start, data

    @staticmethod
    def _is_table_chunk_extra(instruction: RawInstruction) -> bool:
        mnemonic = instruction.mnemonic
        if mnemonic.startswith("fanout"):
            return True
        if mnemonic == "stack_shuffle":
            return True
        if mnemonic.startswith("stack_teardown"):
            return True
        if mnemonic.startswith("op_4B_"):
            return True
        return False

    def _extract_tailcall_prep(self, items: _ItemList, index: int) -> Optional[Tuple[int, IRTailcallPrep]]:
        if index < 4:
            return None

        window = items[index - 4 : index]
        if not all(isinstance(entry, RawInstruction) for entry in window):
            return None

        first = cast(RawInstruction, window[0])
        second = cast(RawInstruction, window[1])
        third = cast(RawInstruction, window[2])
        fourth = cast(RawInstruction, window[3])
        if first.mnemonic != "op_3D_30":
            return None
        if second.mnemonic != "op_32_29":
            return None
        if third.mnemonic not in {"stack_shuffle"} and not third.mnemonic.startswith("op_4B_"):
            return None
        if fourth.mnemonic != "op_F0_4B":
            return None

        steps = (
            (first.mnemonic, first.operand),
            (second.mnemonic, second.operand),
            (third.mnemonic, third.operand),
            (fourth.mnemonic, fourth.operand),
        )
        return index - 4, IRTailcallPrep(steps=steps)


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

    def _pass_flag_checks(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRIf):
                match = self._match_flag_literal(items, index)
                if match is not None:
                    start, flag_value = match
                else:
                    flag_value = self._parse_flag_condition(item.condition)
                    if flag_value is None:
                        index += 1
                        continue
                    start = index

                node = IRCheckFlag(
                    flag=flag_value,
                    then_target=item.then_target,
                    else_target=item.else_target,
                )
                items.replace_slice(start, index + 1, [node])
                index = start
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
