"""Virtual machine reconstruction primitives.

This module interprets the :class:`InstructionSemantics` metadata produced by
the manual annotation analyser and expands it into a rich, reusable description
of how each instruction interacts with the virtual machine stack.  The
resulting data structures power both the pseudo-Lua emitter and the higher level
reconstruction layers so they no longer need to duplicate stack and call-site
heuristics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .manual_semantics import InstructionSemantics


# ---------------------------------------------------------------------------
# Stack metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VMStackValue:
    """Represents a value currently materialised on the VM stack."""

    name: str
    origin: str
    comment: Optional[str]
    offset: Optional[int]

    def describe(self) -> str:
        details = [self.origin]
        if self.comment:
            details.append(self.comment)
        if self.offset is not None:
            details.append(f"@0x{self.offset:06X}")
        return " ".join(details)


@dataclass(frozen=True)
class VMOperation:
    """Semantic reconstruction for a single instruction."""

    offset: int
    semantics: InstructionSemantics
    inputs: Tuple[VMStackValue, ...]
    outputs: Tuple[VMStackValue, ...]
    call_expression: str
    operand_literal: Optional[str]
    comment: Optional[str]
    warnings: Tuple[str, ...]

    def has_outputs(self) -> bool:
        return bool(self.outputs)

    def describe_inputs(self) -> List[str]:
        return [value.describe() for value in self.inputs]

    def describe_outputs(self) -> List[str]:
        return [value.describe() for value in self.outputs]


@dataclass(frozen=True)
class VMInstructionState:
    """Snapshot of the stack immediately before and after an instruction."""

    before: Tuple[VMStackValue, ...]
    after: Tuple[VMStackValue, ...]
    depth_before: int
    depth_after: int
    delta: int


@dataclass(frozen=True)
class VMInstructionTrace:
    """Trace information for a single VM instruction."""

    operation: VMOperation
    state: VMInstructionState


@dataclass
class VMBlockTrace:
    """Trace describing how a basic block manipulates the VM stack."""

    start: int
    instructions: List[VMInstructionTrace]
    entry_stack: Tuple[VMStackValue, ...]
    exit_stack: Tuple[VMStackValue, ...]

    @property
    def operations(self) -> List[VMOperation]:
        return [trace.operation for trace in self.instructions]

    def max_depth(self) -> int:
        depths = [trace.state.depth_after for trace in self.instructions]
        return max([len(self.entry_stack), *depths, len(self.exit_stack)])

    def min_depth(self) -> int:
        depths = [trace.state.depth_before for trace in self.instructions]
        return min([len(self.entry_stack), *depths, len(self.exit_stack)])


@dataclass
class VMProgramTrace:
    """Aggregated VM traces for an entire IR program."""

    segment_index: int
    blocks: Dict[int, VMBlockTrace]

    def block_order(self) -> List[VMBlockTrace]:
        return [self.blocks[offset] for offset in sorted(self.blocks)]

    def total_instructions(self) -> int:
        return sum(len(block.instructions) for block in self.blocks.values())

    def max_depth(self) -> int:
        return max((block.max_depth() for block in self.blocks.values()), default=0)

    def min_depth(self) -> int:
        return min((block.min_depth() for block in self.blocks.values()), default=0)


@dataclass(frozen=True)
class VMLifetime:
    """Describes how long a stack value persists within a trace."""

    value: VMStackValue
    created_offset: Optional[int]
    consumed_offsets: Tuple[int, ...]
    survives: bool

    def describe(self) -> str:
        created = (
            f"0x{self.created_offset:08X}" if self.created_offset is not None else "entry"
        )
        consumers = ", ".join(f"0x{offset:08X}" for offset in self.consumed_offsets)
        if not consumers:
            consumers = "<never>"
        return (
            f"{self.value.name} created={created} consumed=[{consumers}]"
            f" survives={self.survives}"
        )


class VMStackState:
    """Mutable stack model used when reconstructing VM behaviour."""

    def __init__(self, *, counter_start: int = 0) -> None:
        self._stack: List[VMStackValue] = []
        self._counter = counter_start
        self._placeholder_counter = 0

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def counter(self) -> int:
        return self._counter

    def snapshot(self) -> Tuple[VMStackValue, ...]:
        return tuple(self._stack)

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------
    def pop_values(self, count: int) -> Tuple[List[VMStackValue], List[str]]:
        values: List[VMStackValue] = []
        warnings: List[str] = []
        for _ in range(count):
            if self._stack:
                values.append(self._stack.pop())
            else:
                placeholder = self._new_placeholder()
                values.append(placeholder)
                warnings.append("underflow")
        values.reverse()
        return values, warnings

    def push_values(
        self,
        semantics: InstructionSemantics,
        count: int,
        *,
        offset: Optional[int],
    ) -> List[VMStackValue]:
        values: List[VMStackValue] = []
        if count <= 0:
            return values
        base_name = _value_base_name(semantics)
        primary_origin = "literal" if semantics.has_tag("literal") else "result"
        for index in range(count):
            origin = primary_origin if index == 0 else f"{primary_origin}-extra"
            value = VMStackValue(
                name=f"{base_name}_{self._counter}",
                origin=origin,
                comment=semantics.summary,
                offset=offset,
            )
            self._counter += 1
            self._stack.append(value)
            values.append(value)
        return values

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _new_placeholder(self) -> VMStackValue:
        name = f"missing_{self._placeholder_counter}"
        self._placeholder_counter += 1
        return VMStackValue(
            name=name,
            origin="placeholder",
            comment="synthetic underflow placeholder",
            offset=None,
        )


# ---------------------------------------------------------------------------
# VM reconstruction
# ---------------------------------------------------------------------------


class VirtualMachineAnalyzer:
    """Derive VM stack operations from instruction semantics."""

    def __init__(self, *, counter_start: int = 0) -> None:
        self._value_counter = counter_start

    def trace_block(self, block) -> VMBlockTrace:
        state = VMStackState(counter_start=self._value_counter)
        entry_stack = state.snapshot()
        traces: List[VMInstructionTrace] = []
        for instruction in block.instructions:
            before = state.snapshot()
            operation = self._trace_instruction(state, instruction)
            after = state.snapshot()
            trace_state = VMInstructionState(
                before=before,
                after=after,
                depth_before=len(before),
                depth_after=len(after),
                delta=len(after) - len(before),
            )
            traces.append(VMInstructionTrace(operation=operation, state=trace_state))
        self._value_counter = state.counter
        return VMBlockTrace(
            start=block.start,
            instructions=traces,
            entry_stack=entry_stack,
            exit_stack=state.snapshot(),
        )

    def trace_program(self, program) -> "VMProgramTrace":
        blocks: Dict[int, VMBlockTrace] = {}
        for start in sorted(program.blocks):
            blocks[start] = self.trace_block(program.blocks[start])
        return VMProgramTrace(program.segment_index, blocks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _trace_instruction(self, state: VMStackState, instruction) -> VMOperation:
        semantics: InstructionSemantics = instruction.semantics
        inputs_count, outputs_count = estimate_stack_io(semantics)
        inputs, warnings = state.pop_values(inputs_count)
        outputs = state.push_values(
            semantics,
            outputs_count,
            offset=instruction.offset,
        )
        operand_literal: Optional[str] = None
        if semantics.uses_operand:
            operand_literal = _format_operand(instruction.operand)
        call_expression = _format_call(semantics, inputs, operand_literal)
        comment = _format_comment(semantics)
        return VMOperation(
            offset=getattr(instruction, "offset", None) or 0,
            semantics=semantics,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            call_expression=call_expression,
            operand_literal=operand_literal,
            comment=comment,
            warnings=tuple(sorted(set(warnings))),
        )


# ---------------------------------------------------------------------------
# Shared heuristics
# ---------------------------------------------------------------------------


def estimate_stack_io(semantics: InstructionSemantics) -> Tuple[int, int]:
    """Best-effort estimation of stack inputs/outputs for an instruction."""

    inputs = semantics.stack_inputs if semantics.stack_inputs is not None else 0
    outputs = semantics.stack_outputs if semantics.stack_outputs is not None else 0
    delta = semantics.stack_effect.delta if semantics.stack_effect else semantics.stack_delta

    if semantics.has_tag("comparison"):
        inputs = max(inputs, 2)
        outputs = max(outputs, 1)
    elif semantics.control_flow == "branch":
        inputs = max(inputs, 1)
    elif semantics.control_flow == "return":
        inputs = max(inputs, 1)
    elif semantics.has_tag("literal"):
        outputs = max(outputs, 1)
    elif delta is not None:
        rounded = int(round(delta))
        if rounded < 0:
            inputs = max(inputs, abs(rounded))
        elif rounded > 0:
            outputs = max(outputs, rounded)
    return max(inputs, 0), max(outputs, 0)


def analyze_block_lifetimes(trace: VMBlockTrace) -> Dict[str, VMLifetime]:
    """Compute the lifetimes for values manipulated within ``trace``."""

    known_values: Dict[str, VMStackValue] = {}
    creation_offsets: Dict[str, Optional[int]] = {}
    consumers: Dict[str, List[int]] = {}

    for value in trace.entry_stack:
        known_values[value.name] = value
        creation_offsets[value.name] = None

    for instruction in trace.instructions:
        operation = instruction.operation
        for value in operation.inputs:
            consumers.setdefault(value.name, []).append(operation.offset)
            known_values.setdefault(value.name, value)
            creation_offsets.setdefault(value.name, None)
        for value in operation.outputs:
            known_values[value.name] = value
            creation_offsets[value.name] = operation.offset

    exit_names = {value.name for value in trace.exit_stack}

    lifetimes: Dict[str, VMLifetime] = {}
    for name, value in known_values.items():
        consumed = tuple(consumers.get(name, ()))
        lifetimes[name] = VMLifetime(
            value=value,
            created_offset=creation_offsets.get(name),
            consumed_offsets=consumed,
            survives=name in exit_names,
        )
    return lifetimes


def analyze_program_lifetimes(trace: VMProgramTrace) -> Dict[int, Dict[str, VMLifetime]]:
    """Compute value lifetimes for every block in the program trace."""

    return {block.start: analyze_block_lifetimes(block) for block in trace.block_order()}


def render_value_lifetimes(lifetimes: Dict[str, VMLifetime]) -> List[str]:
    lines = []
    for name in sorted(lifetimes):
        lines.append(lifetimes[name].describe())
    return lines


def _format_operand(operand: int) -> str:
    signed = operand if operand < 0x8000 else operand - 0x10000
    if -9 <= signed <= 9:
        return str(signed)
    text = _ascii_candidate(operand)
    if text is not None:
        return _lua_string(text)
    return f"0x{operand:04X}"


def _format_call(
    semantics: InstructionSemantics,
    inputs: Sequence[VMStackValue],
    operand_literal: Optional[str],
) -> str:
    args = [value.name for value in inputs]
    if operand_literal is not None:
        args.append(operand_literal)
    joined = ", ".join(args)
    method = semantics.vm_method or semantics.mnemonic
    style = semantics.vm_call_style or "method"
    if style == "function":
        return f"{method}({joined})" if joined else f"{method}()"
    if style == "literal":
        return f"vm:{method}({joined})" if joined else f"vm:{method}()"
    return f"vm:{method}({joined})" if joined else f"vm:{method}()"


def _format_comment(semantics: InstructionSemantics) -> Optional[str]:
    parts: List[str] = []
    if semantics.manual_name:
        parts.append(semantics.manual_name)
    if semantics.summary:
        parts.append(semantics.summary)
    if semantics.control_flow:
        parts.append(semantics.control_flow)
    if not parts:
        return None
    return ", ".join(parts)


def _value_base_name(semantics: InstructionSemantics) -> str:
    if semantics.has_tag("literal"):
        base = "literal"
    elif semantics.has_tag("comparison"):
        base = "cmp"
    elif semantics.control_flow == "branch":
        base = "cond"
    else:
        base = semantics.vm_method or semantics.manual_name or semantics.mnemonic
    return _sanitize_identifier(base or "value")


def _sanitize_identifier(name: str) -> str:
    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in name]
    if not cleaned:
        return "value"
    identifier = "".join(cleaned)
    while identifier.startswith("_"):
        identifier = identifier[1:]
    identifier = identifier.strip("_")
    if not identifier:
        identifier = "value"
    if identifier[0].isdigit():
        identifier = "v_" + identifier
    return identifier


def _ascii_candidate(operand: int) -> Optional[str]:
    raw = operand.to_bytes(2, "little")
    if all(32 <= byte <= 126 for byte in raw):
        return raw.decode("ascii")
    if raw[1] == 0 and 32 <= raw[0] <= 126:
        return chr(raw[0])
    return None


def _lua_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class LuaLiteralFormatter:
    """Utility class mirroring the operand formatting used by the VM layer."""

    def format_operand(self, operand: int) -> str:
        return _format_operand(operand)


def format_vm_block_trace(trace: VMBlockTrace) -> List[str]:
    """Generate a textual description of a :class:`VMBlockTrace`."""

    lines = [
        (
            f"block 0x{trace.start:06X} stack depth"
            f" {len(trace.entry_stack)} -> {len(trace.exit_stack)}"
        )
    ]
    for instruction in trace.instructions:
        operation = instruction.operation
        state = instruction.state
        inputs = ", ".join(value.name for value in operation.inputs) or "-"
        outputs = ", ".join(value.name for value in operation.outputs) or "-"
        comment = operation.comment or ""
        warn = f" warnings={','.join(operation.warnings)}" if operation.warnings else ""
        depth = f"depth {state.depth_before}->{state.depth_after}"
        lines.append(
            f"  {operation.offset:08X} {operation.semantics.manual_name}: {depth} {inputs} -> {outputs} :"
            f" {operation.call_expression}{warn}"
            + (f"  -- {comment}" if comment else "")
        )
    if trace.exit_stack:
        lines.append(
            "  exit stack: "
            + ", ".join(value.name for value in trace.exit_stack)
        )
    return lines


def render_vm_program(trace: VMProgramTrace) -> str:
    header = (
        f"segment {trace.segment_index} vm-trace depth"
        f" {trace.min_depth()}..{trace.max_depth()}"
        f" instructions={trace.total_instructions()}"
    )
    sections: List[str] = [header]
    for block in trace.block_order():
        sections.extend(format_vm_block_trace(block))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def render_vm_traces(traces: Iterable[VMBlockTrace] | VMProgramTrace) -> str:
    if isinstance(traces, VMProgramTrace):
        return render_vm_program(traces)
    sections = ["\n".join(format_vm_block_trace(trace)) for trace in traces]
    return "\n\n".join(section.rstrip() for section in sections if section) + "\n"


def vm_block_trace_to_dict(trace: VMBlockTrace) -> dict:
    return {
        "start": trace.start,
        "entry_depth": len(trace.entry_stack),
        "exit_depth": len(trace.exit_stack),
        "instructions": [
            {
                "offset": instruction.operation.offset,
                "mnemonic": instruction.operation.semantics.manual_name,
                "call": instruction.operation.call_expression,
                "inputs": [value.name for value in instruction.operation.inputs],
                "outputs": [value.name for value in instruction.operation.outputs],
                "warnings": list(instruction.operation.warnings),
                "depth_before": instruction.state.depth_before,
                "depth_after": instruction.state.depth_after,
            }
            for instruction in trace.instructions
        ],
    }


def vm_program_trace_to_dict(trace: VMProgramTrace) -> dict:
    return {
        "segment_index": trace.segment_index,
        "total_instructions": trace.total_instructions(),
        "min_depth": trace.min_depth(),
        "max_depth": trace.max_depth(),
        "blocks": {
            f"0x{block.start:06X}": vm_block_trace_to_dict(block)
            for block in trace.block_order()
        },
    }


def lifetimes_to_dict(lifetimes: Dict[str, VMLifetime]) -> Dict[str, dict]:
    return {
        name: {
            "created_offset": lifetime.created_offset,
            "consumed_offsets": list(lifetime.consumed_offsets),
            "survives": lifetime.survives,
            "origin": lifetime.value.origin,
        }
        for name, lifetime in lifetimes.items()
    }


def vm_block_trace_to_json(trace: VMBlockTrace, *, indent: int = 2) -> str:
    return json.dumps(vm_block_trace_to_dict(trace), indent=indent)


def vm_program_trace_to_json(trace: VMProgramTrace, *, indent: int = 2) -> str:
    return json.dumps(vm_program_trace_to_dict(trace), indent=indent)


def lifetimes_to_json(lifetimes: Dict[str, VMLifetime], *, indent: int = 2) -> str:
    return json.dumps(lifetimes_to_dict(lifetimes), indent=indent)


def summarise_program(trace: VMProgramTrace) -> str:
    return (
        f"segment {trace.segment_index}: blocks={len(trace.blocks)} "
        f"instructions={trace.total_instructions()} depth={trace.min_depth()}..{trace.max_depth()}"
    )


def count_operations(trace: VMBlockTrace | VMProgramTrace) -> int:
    if isinstance(trace, VMProgramTrace):
        return trace.total_instructions()
    return len(trace.instructions)

