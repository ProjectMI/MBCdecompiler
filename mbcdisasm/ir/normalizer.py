"""Normalisation pipeline that converts raw instructions into IR nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from ..analyzer.instruction_profile import InstructionKind, InstructionProfile
from ..analyzer.stack import StackEvent, StackTracker
from ..instruction import read_instructions
from ..knowledge import KnowledgeBase
from ..mbc import MbcContainer, Segment
from .model import (
    IRBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    IRLoad,
    IRNode,
    IRProgram,
    IRRaw,
    IRReturn,
    IRSegment,
    IRSlot,
    IRStore,
    IRTestSetBranch,
    IRIf,
    MemSpace,
    NormalizerMetrics,
)


ANNOTATION_MNEMONICS = {"literal_marker", "inline_ascii_chunk"}


@dataclass(frozen=True)
class RawInstruction:
    """Wrapper that couples a profile with stack tracking details."""

    profile: InstructionProfile
    event: StackEvent
    annotations: Tuple[str, ...]
    annotation_offsets: Tuple[int, ...] = tuple()
    metadata_only: bool = False

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
    annotations: Tuple[Tuple[int, Tuple[str, ...]], ...]


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
        annotation_offsets: Dict[int, List[int]] = {}
        pending_labels: List[str] = []
        pending_offsets: List[int] = []
        trailing_annotations: List[Tuple[int, str]] = []

        for profile in profiles:
            if profile.mnemonic in ANNOTATION_MNEMONICS or (
                profile.is_literal_marker()
                and (profile.mnemonic.startswith("op_") or profile.mnemonic == "literal_marker")
            ):
                pending_labels.append(self._annotation_label(profile))
                pending_offsets.append(profile.word.offset)
                continue
            index = len(executable)
            if pending_labels:
                annotations.setdefault(index, []).extend(pending_labels)
                annotation_offsets.setdefault(index, []).extend(pending_offsets)
                pending_labels.clear()
                pending_offsets.clear()
            executable.append(profile)

        if pending_labels:
            trailing_annotations.extend(zip(pending_offsets, pending_labels))

        tracker = StackTracker()
        events = tracker.process_sequence(executable)

        raw_instructions: List[RawInstruction] = []
        for index, (profile, event) in enumerate(zip(executable, events)):
            labels = annotations.get(index, ())
            offsets = annotation_offsets.get(index, ())
            raw_instructions.append(
                RawInstruction(
                    profile=profile,
                    event=event,
                    annotations=tuple(labels),
                    annotation_offsets=tuple(offsets),
                )
            )

        blocks: List[RawBlock] = []
        current: List[RawInstruction] = []
        current_annotations: List[Tuple[int, Tuple[str, ...]]] = []
        block_index = 0
        block_start = raw_instructions[0].offset if raw_instructions else segment.start

        for instruction in raw_instructions:
            if not current:
                block_start = instruction.offset
            current.append(instruction)
            if instruction.annotation_offsets:
                grouped = [
                    (offset, (label,))
                    for offset, label in zip(instruction.annotation_offsets, instruction.annotations)
                ]
                current_annotations.extend(grouped)
            if self._is_block_terminator(instruction):
                blocks.append(
                    RawBlock(
                        index=block_index,
                        start_offset=block_start,
                        instructions=tuple(current),
                        annotations=tuple(current_annotations),
                    )
                )
                block_index += 1
                current = []
                current_annotations = []

        if current:
            if trailing_annotations:
                current_annotations.extend((offset, (label,)) for offset, label in trailing_annotations)
            blocks.append(
                RawBlock(
                    index=block_index,
                    start_offset=block_start,
                    instructions=tuple(current),
                    annotations=tuple(current_annotations),
                )
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

    @staticmethod
    def _annotation_label(profile: InstructionProfile) -> str:
        if profile.mnemonic in ANNOTATION_MNEMONICS:
            return profile.mnemonic
        if profile.is_literal_marker():
            return "literal_marker"
        return profile.mnemonic

    # ------------------------------------------------------------------
    # normalisation passes
    # ------------------------------------------------------------------
    def _normalise_block(self, block: RawBlock) -> Tuple[IRBlock, NormalizerMetrics]:
        items = _ItemList(block.instructions)
        metrics = NormalizerMetrics()
        annotations: List[Tuple[int, Tuple[str, ...]]] = list(block.annotations)

        self._pass_calls_and_returns(items, metrics)
        self._pass_aggregates(items, metrics)
        self._pass_branches(items, metrics)
        self._pass_indirect_access(items, metrics)

        nodes: List[IRNode] = []
        for item in items:
            if isinstance(item, RawInstruction):
                if self._is_metadata_instruction(item):
                    annotations.append(self._metadata_entry(item))
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
            annotations=tuple(annotations),
        )
        return ir_block, metrics

    # ------------------------------------------------------------------
    # individual pass implementations
    # ------------------------------------------------------------------
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
                arity = self._return_arity(item, items, index)
                values = tuple(f"ret{i}" for i in range(arity))
                items.replace_slice(index, index + 1, [IRReturn(values=values)])
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
            if not isinstance(candidate, RawInstruction):
                break
            if candidate.pushes_value():
                args.append(self._describe_value(candidate))
                scan -= 1
                continue
            break
        args.reverse()
        start = scan + 1
        return args, start

    def _collapse_tail_return(self, items: _ItemList, call_index: int, metrics: NormalizerMetrics) -> None:
        index = call_index + 1
        while index < len(items):
            item = items[index]
            if isinstance(item, RawInstruction) and item.mnemonic == "return_values":
                arity = self._return_arity(item, items, index)
                values = tuple(f"ret{i}" for i in range(arity))
                items.replace_slice(index, index + 1, [IRReturn(values=values)])
                metrics.returns += 1
                return
            if isinstance(item, RawInstruction) and item.profile.kind in {
                InstructionKind.STACK_TEARDOWN,
                InstructionKind.META,
            }:
                items.pop(index)
                continue
            break

    def _return_arity(
        self,
        instruction: RawInstruction,
        items: _ItemList | None = None,
        index: int | None = None,
    ) -> int:
        operand = instruction.operand
        lo = operand & 0xFF
        hi = (operand >> 8) & 0xFF

        if 0 < lo <= 16:
            return lo
        if 0 < hi <= 16:
            return hi

        teardown = self._teardown_hint(items, index)
        if teardown:
            return teardown

        mode_hint = instruction.profile.mode >> 4
        if mode_hint:
            return mode_hint

        depth = instruction.event.depth_before
        if depth:
            return max(1, min(depth, 16))

        return 1

    def _pass_aggregates(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction) or not self._is_literal_instruction(item):
                index += 1
                continue

            literal_instructions: List[RawInstruction] = []
            reducers: List[RawInstruction] = []
            metadata: List[RawInstruction] = []
            scan = index
            while scan < len(items):
                candidate = items[scan]
                if isinstance(candidate, RawInstruction) and self._is_literal_instruction(candidate):
                    literal_instructions.append(candidate)
                    scan += 1
                    continue
                if isinstance(candidate, RawInstruction) and self._is_reduce_instruction(candidate):
                    reducers.append(candidate)
                    scan += 1
                    continue
                if isinstance(candidate, RawInstruction) and self._is_metadata_instruction(candidate):
                    metadata.append(candidate)
                    scan += 1
                    continue
                break

            if not reducers:
                index += 1
                continue

            literals = [self._describe_value(instr) for instr in literal_instructions]
            expected_pairs = len(literals) // 2
            if len(reducers) > expected_pairs:
                excess = reducers[expected_pairs:]
                metadata.extend(replace(instr, metadata_only=True) for instr in excess)
                reducers = reducers[:expected_pairs]
            if not reducers:
                index += len(metadata) + 1
                continue
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
            items.replace_slice(index, scan, [*metadata, replacement])
            index += len(metadata) + 1

    def _pass_branches(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            if item.mnemonic == "testset_branch":
                expr = self._describe_condition(items, index, prefer_expression=True)
                node = IRTestSetBranch(
                    var=f"t{index}",
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
    def _is_literal_instruction(self, instruction: RawInstruction) -> bool:
        if instruction.mnemonic == "push_literal":
            return True
        if (
            instruction.mnemonic.startswith("op_")
            and instruction.annotations
            and any(note in ANNOTATION_MNEMONICS for note in instruction.annotations)
        ):
            return False
        if instruction.profile.kind in {InstructionKind.LITERAL, InstructionKind.PUSH}:
            return instruction.pushes_value()
        return False

    @staticmethod
    def _is_reduce_instruction(instruction: RawInstruction) -> bool:
        if instruction.mnemonic.startswith("reduce"):
            return True
        return instruction.profile.kind is InstructionKind.REDUCE

    def _is_metadata_instruction(self, instruction: RawInstruction) -> bool:
        profile = instruction.profile
        if instruction.metadata_only:
            return True
        if instruction.pushes_value() or instruction.event.popped_types:
            if (
                instruction.annotations
                and any(note in ANNOTATION_MNEMONICS for note in instruction.annotations)
                and instruction.mnemonic.startswith("op_")
            ):
                pass
            else:
                return False
        if profile.mnemonic in ANNOTATION_MNEMONICS:
            return True
        if profile.is_literal_marker() and (
            profile.mnemonic.startswith("op_") or profile.mnemonic == "literal_marker"
        ):
            return True
        if instruction.annotations and any(note in ANNOTATION_MNEMONICS for note in instruction.annotations):
            return True
        if profile.kind is InstructionKind.ASCII_CHUNK:
            return True
        return False

    def _metadata_entry(self, instruction: RawInstruction) -> Tuple[int, Tuple[str, ...]]:
        labels: List[str] = []
        if instruction.annotations:
            labels.extend(instruction.annotations)
        else:
            labels.append(self._annotation_label(instruction.profile))
        return instruction.offset, tuple(labels)

    def _teardown_hint(self, items: _ItemList | None, index: int | None) -> int:
        if items is None or index is None:
            return 0
        total = 0
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if not isinstance(candidate, RawInstruction):
                break
            if self._is_metadata_instruction(candidate):
                scan -= 1
                continue
            if candidate.profile.kind is InstructionKind.STACK_TEARDOWN:
                total += -candidate.event.delta
                scan -= 1
                continue
            break
        return total

    def _describe_value(self, instruction: RawInstruction) -> str:
        mnemonic = instruction.mnemonic
        operand = instruction.operand
        if mnemonic == "push_literal":
            return f"lit(0x{operand:04X})"
        if mnemonic.startswith("reduce"):
            return f"reduce(0x{operand:04X})"
        if "slot" in mnemonic:
            return f"slot(0x{operand:04X})"
        return instruction.describe_source()

    def _describe_condition(
        self,
        items: _ItemList,
        index: int,
        *,
        prefer_expression: bool = False,
    ) -> str:
        scan = index - 1
        fallback: Optional[str] = None
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    value = self._describe_value(candidate)
                    if prefer_expression and self._looks_like_literal(value):
                        if fallback is None:
                            fallback = value
                        scan -= 1
                        continue
                    return value
                if prefer_expression and candidate.profile.kind in {
                    InstructionKind.CALL,
                    InstructionKind.TAILCALL,
                }:
                    return candidate.describe_source()
            if isinstance(candidate, IRNode):
                value = getattr(candidate, "describe", lambda: "expr()")()
                if prefer_expression and self._looks_like_literal(value):
                    if fallback is None:
                        fallback = value
                    scan -= 1
                    continue
                return value
            scan -= 1
        if fallback is not None:
            return fallback
        return "stack_top"

    @staticmethod
    def _looks_like_literal(description: str) -> bool:
        return description.startswith("lit(")

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


__all__ = ["IRNormalizer", "RawInstruction", "RawBlock"]
