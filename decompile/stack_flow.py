"""Block-local stack lifting with explicit edge values.

This pass deliberately keeps edge assignments visible. Textual short-circuit
folding must not erase a value that is live on only one predecessor edge.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InputState:
    types: tuple[int | None, ...]
    argc: int | None
    frame_bases: tuple[int, ...] = (0,)


def build_stack_flow(builder: Any, instructions: list[Any]) -> list[Any]:
    from .vm_ast import AstStatement, VMSlot, VMStackMachine, label_for_offset

    by_offset = {ins.offset: ins for ins in instructions}
    branch_ops = {71, 74, 73, 75, 76, 77}
    terminal_ops = {114, 116, 35, 72, 103}
    leaders = {instructions[0].offset}
    for ins in instructions:
        if ins.opcode in branch_ops:
            target = ins.operands.get("target")
            if target in by_offset:
                leaders.add(target)
        if ins.opcode in branch_ops | terminal_ops | {124} and ins.offset + ins.length in by_offset:
            leaders.add(ins.offset + ins.length)
        if ins.mnemonic == "stack_frame_reset":
            leaders.add(ins.offset)
        if ins.mnemonic == "program_prologue":
            builder._bind_program_prologue(ins.operands)
    blocks: dict[int, list[Any]] = {}
    current = instructions[0].offset
    for ins in instructions:
        if ins.offset in leaders:
            current = ins.offset
            blocks[current] = []
        blocks[current].append(ins)

    def edge_targets(block: list[Any]) -> list[tuple[int, bool]]:
        last = block[-1]
        fallthrough = last.offset + last.length
        if last.opcode in terminal_ops:
            return []
        if last.opcode in {71, 74}:
            return [(last.operands["target"], True)]
        if last.opcode in {73, 75, 76, 77}:
            return [(last.operands["target"], True), (fallthrough, False)]
        return [(fallthrough, False)] if fallthrough in blocks else []

    states = {instructions[0].offset: InputState((), None)}
    work = deque(states)
    outputs: dict[int, tuple[list[Any], list[tuple[int, list[Any]]]]] = {}
    underflows = 0
    diagnostics = set()
    visits = 0
    while work:
        start = work.popleft()
        visits += 1
        if visits > max(100, len(blocks) * 32):
            raise ValueError("Symbolic stack dataflow did not converge")
        state = states[start]
        builder.vm = VMStackMachine(memory=builder.memory)
        builder.vm.stack = [VMSlot(f"merge_{start:08X}_{i}", type_id=typ) for i, typ in enumerate(state.types)]
        builder.vm.frame_bases = list(state.frame_bases)
        builder.pending_arg_count = state.argc
        builder.statements = []
        block = blocks[start]
        condition = None
        for ins in block:
            if ins.opcode in {76, 77} and builder.vm.stack:
                condition = builder.vm.stack[-1].clone(type_id=16)
            builder._visit(ins)
        underflows = max(underflows, builder.vm.underflows)
        edges = []
        for target, taken in edge_targets(block):
            if target not in blocks:
                continue
            slots = [slot.clone() for slot in builder.vm.stack]
            if taken and block[-1].opcode in {76, 77} and condition is not None:
                slots.append(condition)
            frames = tuple(builder.vm.frame_bases)
            if blocks[target][0].mnemonic == "stack_frame_reset":
                # Values above the current expression frame are dead on this
                # edge. Different discarded heights are not conflicting live phis.
                slots = slots[:frames[-1]]
            incoming = InputState(tuple(slot.type_id for slot in slots), builder.pending_arg_count, frames)
            previous = states.get(target)
            if previous is not None:
                if len(previous.types) != len(incoming.types):
                    diagnostics.add(f"stack height differs at 0x{target:X}: {len(previous.types)} / {len(incoming.types)}")
                    # Retain every live incoming value and report the ambiguous
                    # state; never silently select one predecessor's expression.
                    width = max(len(previous.types), len(incoming.types))
                    old = previous.types + (None,) * (width - len(previous.types))
                    new = incoming.types + (None,) * (width - len(incoming.types))
                else:
                    old, new = previous.types, incoming.types
                if previous.frame_bases != incoming.frame_bases:
                    diagnostics.add(f"expression frames differ at 0x{target:X}")
                incoming = InputState(tuple(a if a == b else None for a, b in zip(old, new)),
                                      previous.argc if previous.argc == incoming.argc else None,
                                      previous.frame_bases)
            if incoming != previous:
                states[target] = incoming
                if target not in work:
                    work.append(target)
            edges.append((target, slots))
        outputs[start] = (list(builder.statements), edges)

    def statement(offset: int, kind: str, text: str, **operands: Any) -> Any:
        return AstStatement(offset, offset + 32, kind, text, -1, "stack_flow", operands)

    result = []
    for start, state in sorted(states.items()):
        for index, typ in enumerate(state.types):
            result.append(statement(instructions[0].offset, "decl", f"vm_value merge_{start:08X}_{index};", type_id=typ))
    for start in sorted(outputs):
        body, edges = outputs[start]
        last = blocks[start][-1]
        transfers = []
        for target, slots in edges:
            for index, slot in enumerate(slots):
                transfers.append(statement(last.offset, "assign", f"merge_{target:08X}_{index} = {slot.render()};", edge=target))
        branch_index = next((i for i in range(len(body) - 1, -1, -1)
                             if body[i].kind in {"goto", "if_goto", "yield"}), len(body))
        body[branch_index:branch_index] = transfers
        result.append(statement(start, "label", f"{label_for_offset(start)}:", target=start))
        result.extend(body)
    for diagnostic in sorted(diagnostics):
        result.append(statement(instructions[0].offset, "warning", f"// {diagnostic}"))
    builder.vm = VMStackMachine(memory=builder.memory)
    builder.vm.underflows = underflows
    builder.flow_diagnostics = sorted(diagnostics)
    return result
