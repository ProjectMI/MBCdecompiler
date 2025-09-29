"""Derive lightweight stack behaviour summaries for IR programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ir import IRProgram
from .vm_analysis import (
    VMBlockTrace,
    VMOperation,
    VMProgramTrace,
    VirtualMachineAnalyzer,
    analyze_block_lifetimes,
)


@dataclass
class BlockStackSummary:
    """Condensed view of the stack behaviour for a single block."""

    start: int
    instruction_count: int
    entry_depth: int
    exit_depth: int
    min_depth: int
    max_depth: int
    underflows: int

    def summary_line(self) -> str:
        return (
            f"block 0x{self.start:06X}: instructions={self.instruction_count} "
            f"depth={self.entry_depth}->{self.exit_depth} "
            f"range=[{self.min_depth},{self.max_depth}] "
            f"underflows={self.underflows}"
        )


@dataclass
class StackSummary:
    """Aggregated stack statistics for an entire program."""

    instruction_count: int
    min_depth: int
    max_depth: int
    total_underflows: int
    placeholder_values: int
    blocks: List[BlockStackSummary] = field(default_factory=list)
    anomalies: List["StackAnomaly"] = field(default_factory=list)
    long_lived: List[str] = field(default_factory=list)
    operations: List["InstructionStackSummary"] = field(default_factory=list)

    def summary_lines(self, *, block_limit: int = 6) -> List[str]:
        lines = ["stack summary:"]
        lines.append(f"- instructions analysed: {self.instruction_count}")
        lines.append(f"- stack depth range: {self.min_depth}..{self.max_depth}")
        lines.append(f"- underflow warnings: {self.total_underflows}")
        lines.append(f"- synthetic placeholders: {self.placeholder_values}")
        if not self.blocks:
            return lines
        lines.append("- block overview:")
        for block in self.blocks[:block_limit]:
            lines.append(f"  - {block.summary_line()}")
        remaining = len(self.blocks) - block_limit
        if remaining > 0:
            lines.append(f"  - ... ({remaining} additional blocks)")
        return lines

    def anomaly_lines(self, *, limit: int = 6) -> List[str]:
        if not self.anomalies:
            return []
        lines = ["stack anomalies:"]
        for anomaly in self.anomalies[:limit]:
            lines.append(f"- {anomaly.summary_line()}")
        remaining = len(self.anomalies) - limit
        if remaining > 0:
            lines.append(f"- ... ({remaining} additional anomalies)")
        return lines

    def lifetime_lines(self, *, limit: int = 4) -> List[str]:
        if not self.long_lived:
            return []
        lines = ["stack lifetimes:"]
        for entry in self.long_lived[:limit]:
            lines.append(f"- {entry}")
        remaining = len(self.long_lived) - limit
        if remaining > 0:
            lines.append(f"- ... ({remaining} additional values)")
        return lines

    def operation_lines(self, *, limit: int = 8) -> List[str]:
        if not self.operations:
            return []
        lines = ["stack operations:"]
        for entry in self.operations[:limit]:
            lines.append(f"- {entry.summary_line()}")
        remaining = len(self.operations) - limit
        if remaining > 0:
            lines.append(f"- ... ({remaining} additional instructions)")
        return lines


def summarise_stack_behaviour(
    program: IRProgram,
    analyzer: Optional[VirtualMachineAnalyzer] = None,
    trace: Optional[VMProgramTrace] = None,
) -> StackSummary:
    """Return a :class:`StackSummary` for ``program`` using ``analyzer``."""

    tracer = analyzer or VirtualMachineAnalyzer()
    program_trace = trace or tracer.trace_program(program)
    instruction_total = program_trace.total_instructions()
    block_summaries: List[BlockStackSummary] = []
    anomalies: List[StackAnomaly] = []
    long_lived: List[str] = []
    operations: List[InstructionStackSummary] = []
    underflows = 0
    placeholders = 0
    min_depth = 0
    max_depth = 0

    for block in program_trace.block_order():
        (
            block_summary,
            placeholder_delta,
            block_anomalies,
            lifetimes,
            instruction_summaries,
        ) = _summarise_block(block)
        block_summaries.append(block_summary)
        underflows += block_summary.underflows
        placeholders += placeholder_delta
        min_depth = min(min_depth, block_summary.min_depth)
        max_depth = max(max_depth, block_summary.max_depth)
        anomalies.extend(block_anomalies)
        long_lived.extend(lifetimes)
        operations.extend(instruction_summaries)

    return StackSummary(
        instruction_count=instruction_total,
        min_depth=min_depth,
        max_depth=max_depth,
        total_underflows=underflows,
        placeholder_values=placeholders,
        blocks=block_summaries,
        anomalies=sorted(anomalies, key=lambda item: (item.block_start, item.offset)),
        long_lived=long_lived,
        operations=operations,
    )


def _summarise_block(
    block: VMBlockTrace,
) -> tuple[
    BlockStackSummary,
    int,
    List["StackAnomaly"],
    List[str],
    List["InstructionStackSummary"],
]:
    underflows = 0
    placeholders = 0
    anomalies: List[StackAnomaly] = []
    operation_summaries: List[InstructionStackSummary] = []
    for trace in block.instructions:
        operation = trace.operation
        underflows += _count_underflow(operation)
        placeholders += _count_placeholders(operation)
        anomalies.extend(_collect_anomalies(block.start, operation))
        operation_summaries.append(
            InstructionStackSummary(
                block_start=block.start,
                offset=operation.offset,
                mnemonic=operation.semantics.manual_name,
                depth_before=trace.state.depth_before,
                depth_after=trace.state.depth_after,
                warnings=list(operation.warnings),
            )
        )

    summary = BlockStackSummary(
        start=block.start,
        instruction_count=len(block.instructions),
        entry_depth=len(block.entry_stack),
        exit_depth=len(block.exit_stack),
        min_depth=block.min_depth(),
        max_depth=block.max_depth(),
        underflows=underflows,
    )
    lifetimes = _describe_lifetimes(block)
    return summary, placeholders, anomalies, lifetimes, operation_summaries


def _count_underflow(operation: VMOperation) -> int:
    return sum(1 for warning in operation.warnings if warning == "underflow")


def _count_placeholders(operation: VMOperation) -> int:
    return sum(1 for value in operation.inputs if value.origin == "placeholder")


def _collect_anomalies(block_start: int, operation: VMOperation) -> List["StackAnomaly"]:
    anomalies: List[StackAnomaly] = []
    if operation.warnings:
        description = f"{operation.semantics.manual_name} underflow"
        anomalies.append(
            StackAnomaly(
                block_start=block_start,
                offset=operation.offset,
                description=description,
                count=len(operation.warnings),
            )
        )
    placeholder_inputs = [value for value in operation.inputs if value.origin == "placeholder"]
    if placeholder_inputs:
        description = f"{operation.semantics.manual_name} placeholder inputs"
        anomalies.append(
            StackAnomaly(
                block_start=block_start,
                offset=operation.offset,
                description=description,
                count=len(placeholder_inputs),
            )
        )
    return anomalies


def _describe_lifetimes(block: VMBlockTrace) -> List[str]:
    lifetimes = analyze_block_lifetimes(block)
    descriptions: List[str] = []
    for lifetime in lifetimes.values():
        if lifetime.value.origin == "placeholder":
            continue
        if not lifetime.consumed_offsets and not lifetime.survives:
            continue
        consumers = ", ".join(f"0x{offset:06X}" for offset in lifetime.consumed_offsets)
        if not consumers:
            consumers = "<none>"
        created = (
            f"0x{lifetime.created_offset:06X}"
            if lifetime.created_offset is not None
            else "entry"
        )
        descriptions.append(
            f"value {lifetime.value.name} created {created} consumed {consumers}"
        )
    descriptions.sort()
    return descriptions


@dataclass(order=True)
class StackAnomaly:
    """Describe a suspicious stack interaction encountered while tracing."""

    block_start: int
    offset: int
    description: str
    count: int = 1

    def summary_line(self) -> str:
        suffix = "" if self.count == 1 else f" (x{self.count})"
        return f"0x{self.block_start:06X}/0x{self.offset:06X}: {self.description}{suffix}"


@dataclass(order=True)
class InstructionStackSummary:
    """Describe a single instruction's effect on the stack."""

    block_start: int
    offset: int
    mnemonic: str
    depth_before: int
    depth_after: int
    warnings: List[str] = field(default_factory=list)

    def summary_line(self) -> str:
        delta = self.depth_after - self.depth_before
        descriptor = f"{self.mnemonic} depth {self.depth_before}->{self.depth_after}"
        if delta > 0:
            descriptor += f" (+{delta})"
        elif delta < 0:
            descriptor += f" ({delta})"
        if self.warnings:
            descriptor += f" warnings={','.join(self.warnings)}"
        return f"0x{self.block_start:06X}/0x{self.offset:06X}: {descriptor}"


__all__ = [
    "BlockStackSummary",
    "StackSummary",
    "StackAnomaly",
    "InstructionStackSummary",
    "summarise_stack_behaviour",
]
