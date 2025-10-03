"""Normalisation pipeline that converts raw instructions into IR nodes."""

from __future__ import annotations

from dataclasses import dataclass
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
        items = _ItemList(block.instructions)
        metrics = NormalizerMetrics()

        self._pass_calls_and_returns(items, metrics)
        self._pass_aggregates(items, metrics)
        self._pass_branches(items, metrics)
        self._pass_indirect_access(items, metrics)

        nodes: List[IRNode] = []
        for item in items:
            if isinstance(item, RawInstruction):
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

        ir_block = IRBlock(label=f"block_{block.index}", start_offset=block.start_offset, nodes=tuple(nodes))
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
                arity = self._return_arity(item)
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
                arity = self._return_arity(item)
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

    def _return_arity(self, instruction: RawInstruction) -> int:
        operand = instruction.operand
        lo = operand & 0xFF
        hi = (operand >> 8) & 0xFF
        if lo:
            return lo
        if hi:
            return hi
        return 1

    def _pass_aggregates(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction) or item.mnemonic != "push_literal":
                index += 1
                continue

            literal_instructions: List[RawInstruction] = []
            reducers: List[RawInstruction] = []
            scan = index
            while scan < len(items):
                candidate = items[scan]
                if isinstance(candidate, RawInstruction) and candidate.mnemonic == "push_literal":
                    literal_instructions.append(candidate)
                    scan += 1
                    continue
                if isinstance(candidate, RawInstruction) and candidate.mnemonic.startswith("reduce"):
                    reducers.append(candidate)
                    scan += 1
                    continue
                break

            if not reducers:
                index += 1
                continue

            literals = [self._describe_value(instr) for instr in literal_instructions]
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
            items.replace_slice(index, scan, [replacement])
            index += 1

    def _pass_branches(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            if item.mnemonic == "testset_branch":
                expr = self._describe_condition(items, index)
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

    def _describe_condition(self, items: _ItemList, index: int) -> str:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction) and candidate.pushes_value():
                return self._describe_value(candidate)
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


__all__ = ["IRNormalizer", "RawInstruction", "RawBlock"]
