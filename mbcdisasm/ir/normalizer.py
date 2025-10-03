"""Normalisation pipeline turning raw instruction blocks into IR blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from ..analyzer.instruction_profile import InstructionKind
from .model import (
    IRBasicBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    IRIf,
    IRLiteral,
    IRLoad,
    IRReturn,
    IRSlot,
    IRStore,
    IRTemp,
    IRTestSetBranch,
    MemSpace,
    NormalizerMetrics,
)
from .raw import RawBasicBlock, RawInstruction


@dataclass
class _Event:
    kind: str
    instruction: RawInstruction
    data: dict


class BlockNormaliser:
    """Convert :class:`RawBasicBlock` instances into :class:`IRBasicBlock`."""

    def __init__(self, block: RawBasicBlock) -> None:
        self.block = block
        self.metrics = NormalizerMetrics()
        self.events: List[_Event] = []
        self._temp_counter = 0

    def run(self) -> IRBasicBlock:
        self._build_events()
        self._fold_literal_windows()
        operations: List = []
        stack: List[object] = []

        for event in self.events:
            handler = getattr(self, f"_handle_{event.kind}", None)
            if handler is None:
                self.metrics.raw_remaining += 1
                continue
            produced = handler(event, stack)
            if produced is None:
                continue
            if isinstance(produced, list):
                operations.extend(produced)
            else:
                operations.append(produced)

        return IRBasicBlock(label=self.block.label, operations=operations)

    def _build_events(self) -> None:
        instructions = self.block.instructions
        consumed: set[int] = set()
        for idx, instruction in enumerate(instructions):
            if idx in consumed:
                continue
            mnemonic = instruction.mnemonic
            profile = instruction.profile
            if mnemonic == "push_literal":
                self.events.append(_Event("literal", instruction, {"value": instruction.profile.operand}))
                continue
            if mnemonic == "reduce_pair":
                self.events.append(_Event("reduce", instruction, {}))
                continue
            if mnemonic == "tailcall_dispatch":
                tail_data = {"target": instruction.profile.operand, "stack_delta": instruction.stack_delta, "return_count": None}
                lookahead = idx + 1
                while lookahead < len(instructions):
                    candidate = instructions[lookahead]
                    if candidate.mnemonic == "return_values":
                        tail_data["return_count"] = candidate.profile.operand & 0xFF
                        consumed.add(lookahead)
                        break
                    if candidate.stack_delta != 0:
                        break
                    lookahead += 1
                self.events.append(_Event("tailcall", instruction, tail_data))
                continue
            if profile.kind is InstructionKind.CALL:
                payload = {"target": instruction.profile.operand, "stack_delta": instruction.stack_delta}
                self.events.append(_Event("call", instruction, payload))
                continue
            if mnemonic == "return_values" or profile.kind is InstructionKind.RETURN:
                count = instruction.profile.operand & 0xFF
                self.events.append(_Event("return", instruction, {"count": count}))
                continue
            if mnemonic == "testset_branch":
                operand = instruction.profile.operand
                self.events.append(
                    _Event(
                        "testset",
                        instruction,
                        {"operand": operand, "successors": self.block.successors},
                    )
                )
                continue
            if mnemonic == "test_branch" or profile.kind is InstructionKind.BRANCH:
                operand = instruction.profile.operand
                self.events.append(
                    _Event(
                        "if_branch",
                        instruction,
                        {"operand": operand, "successors": self.block.successors},
                    )
                )
                continue
            if profile.kind in {InstructionKind.INDIRECT, InstructionKind.INDIRECT_LOAD, InstructionKind.INDIRECT_STORE}:
                self.events.append(
                    _Event(
                        "indirect",
                        instruction,
                        {"operand": instruction.profile.operand, "delta": instruction.stack_delta},
                    )
                )
                continue
            self.events.append(_Event("raw", instruction, {}))

    def _fold_literal_windows(self) -> None:
        folded: List[_Event] = []
        idx = 0
        while idx < len(self.events):
            event = self.events[idx]
            if event.kind != "literal":
                folded.append(event)
                idx += 1
                continue
            literals: List[_Event] = []
            j = idx
            while j < len(self.events) and self.events[j].kind == "literal":
                literals.append(self.events[j])
                j += 1
            k = j
            reduce_count = 0
            while k < len(self.events) and self.events[k].kind == "reduce":
                reduce_count += 1
                k += 1
            aggregate_event = self._try_fold_literals(literals, reduce_count)
            if aggregate_event:
                folded.append(aggregate_event)
                idx = j + reduce_count
                self.metrics.aggregates += 1
                self.metrics.reduce_replaced += reduce_count
            else:
                folded.extend(literals)
                idx = j
        self.events = folded

    def _try_fold_literals(self, literals: Sequence[_Event], reduce_count: int) -> Optional[_Event]:
        if not literals or reduce_count == 0:
            return None
        values = [IRLiteral(evt.data["value"]) for evt in literals]
        if reduce_count * 2 == len(literals):
            items = []
            for idx in range(0, len(values), 2):
                items.append((values[idx], values[idx + 1]))
            temp = self._new_temp()
            return _Event("aggregate_map", literals[0].instruction, {"items": items, "temp": temp})
        if reduce_count == len(literals) - 1:
            temp = self._new_temp()
            return _Event("aggregate_array", literals[0].instruction, {"values": values, "temp": temp})
        return None

    def _new_temp(self) -> IRTemp:
        name = f"t{self._temp_counter}"
        self._temp_counter += 1
        return IRTemp(name)

    def _handle_literal(self, event: _Event, stack: List[object]):
        literal = IRLiteral(event.data["value"])
        stack.append(literal)
        return None

    def _handle_aggregate_map(self, event: _Event, stack: List[object]):
        items = event.data["items"]
        result = event.data["temp"]
        stack.append(result)
        return IRBuildMap(result=result, items=items)

    def _handle_aggregate_array(self, event: _Event, stack: List[object]):
        values = event.data["values"]
        result = event.data["temp"]
        stack.append(result)
        return IRBuildArray(result=result, elements=values)

    def _handle_tailcall(self, event: _Event, stack: List[object]):
        arg_count = max(0, -event.data.get("stack_delta", 0))
        args = _pop_args(stack, arg_count)
        result = None
        self.metrics.tail_calls += 1
        self.metrics.calls += 1
        call_op = IRCall(target=event.data["target"], args=args, result=result, tail=True)
        operations = [call_op]
        return_count = event.data.get("return_count")
        if return_count is not None:
            returns = _pop_args(stack, return_count)
            self.metrics.returns += 1
            operations.append(IRReturn(values=returns))
        return operations

    def _handle_call(self, event: _Event, stack: List[object]):
        arg_count = max(0, -event.data.get("stack_delta", 0))
        args = _pop_args(stack, arg_count)
        result = self._new_temp()
        stack.append(result)
        self.metrics.calls += 1
        return IRCall(target=event.data["target"], args=args, result=result, tail=False)

    def _handle_return(self, event: _Event, stack: List[object]):
        count = event.data.get("count", 0)
        values = _pop_args(stack, count)
        self.metrics.returns += 1
        return IRReturn(values=values)

    def _handle_testset(self, event: _Event, stack: List[object]):
        if stack:
            expr = stack.pop()
        else:
            expr = IRLiteral(0)
        temp = self._new_temp()
        then_target, else_target = _branch_targets(self.block.successors)
        node = IRTestSetBranch(target=temp, expression=expr, then_target=then_target, else_target=else_target)
        stack.append(temp)
        self.metrics.testset_branches += 1
        return node

    def _handle_if_branch(self, event: _Event, stack: List[object]):
        predicate = stack.pop() if stack else IRLiteral(0)
        then_target, else_target = _branch_targets(self.block.successors)
        self.metrics.if_branches += 1
        return IRIf(predicate=predicate, then_target=then_target, else_target=else_target)

    def _handle_indirect(self, event: _Event, stack: List[object]):
        operand = event.data["operand"]
        slot = IRSlot(space=_classify_space(operand), index=operand)
        delta = event.data["delta"]
        if delta >= 0:
            temp = self._new_temp()
            stack.append(temp)
            self.metrics.loads += 1
            return IRLoad(result=temp, slot=slot)
        value = stack.pop() if stack else IRLiteral(0)
        self.metrics.stores += 1
        return IRStore(slot=slot, value=value)

    def _handle_raw(self, event: _Event, stack: List[object]):
        self.metrics.raw_remaining += 1
        return None


def _pop_args(stack: List[object], count: int) -> List[object]:
    if count <= 0:
        return []
    popped: List[object] = []
    for _ in range(count):
        if stack:
            popped.append(stack.pop())
        else:
            popped.append(IRLiteral(0))
    popped.reverse()
    return popped


def _branch_targets(successors: Sequence[str]) -> Tuple[str, str]:
    if not successors:
        return ("fallthrough", "fallthrough")
    if len(successors) == 1:
        return (successors[0], "fallthrough")
    return successors[0], successors[1]


def _classify_space(operand: int) -> MemSpace:
    if operand < 0x1000:
        return MemSpace.FRAME
    if operand < 0x8000:
        return MemSpace.GLOBAL
    return MemSpace.CONST
