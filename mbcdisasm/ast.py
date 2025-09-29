"""Pseudo-Lua reconstruction helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .branch_analysis import BranchDescriptor, BranchGraph, describe_branches
from .ir import IRBlock, IRProgram
from .vm_analysis import VMBlockTrace, VMOperation, VirtualMachineAnalyzer


@dataclass
class LuaStatement:
    """Base class for pseudo-Lua statements."""

    def render(self) -> List[str]:
        raise NotImplementedError

    def metadata(self) -> Optional[dict]:
        return None


@dataclass
class LuaRawStatement(LuaStatement):
    text: str

    def render(self) -> List[str]:
        return [self.text]

    def metadata(self) -> Optional[dict]:
        return {"type": "raw", "text": self.text}


@dataclass
class LuaBlankLine(LuaStatement):
    def render(self) -> List[str]:
        return [""]

    def metadata(self) -> Optional[dict]:
        return None


@dataclass
class LuaLabel(LuaStatement):
    label: str

    def render(self) -> List[str]:
        return [f"::{self.label}::"]

    def metadata(self) -> Optional[dict]:
        return {"type": "label", "label": self.label}


@dataclass
class LuaVMOperation(LuaStatement):
    operation: VMOperation

    def render(self) -> List[str]:
        lines: List[str] = []
        for value in self.operation.inputs:
            comment = _stack_comment("pop", value)
            line = "stack[#stack] = nil"
            if comment:
                line += f"  -- {comment}"
            lines.append(line)

        call_line = self._render_call()
        if call_line:
            lines.append(call_line)

        for value in self.operation.outputs:
            comment = _stack_comment("push", value)
            line = f"stack[#stack + 1] = {value.name}"
            if comment:
                line += f"  -- {comment}"
            lines.append(line)
        return lines

    def _render_call(self) -> Optional[str]:
        call = self.operation.call_expression
        if not call:
            return None
        if self.operation.outputs:
            names = ", ".join(value.name for value in self.operation.outputs)
            line = f"local {names} = {call}"
        else:
            line = call

        comment_parts: List[str] = []
        if self.operation.comment:
            comment_parts.append(self.operation.comment)
        if self.operation.warnings:
            comment_parts.append("warnings: " + ", ".join(self.operation.warnings))
        if comment_parts:
            line += "  -- " + " | ".join(comment_parts)
        return line

    def metadata(self) -> Optional[dict]:
        operation = self.operation
        return {
            "type": "operation",
            "offset": operation.offset,
            "mnemonic": operation.semantics.manual_name,
            "call": operation.call_expression,
            "inputs": [value.name for value in operation.inputs],
            "outputs": [value.name for value in operation.outputs],
            "warnings": list(operation.warnings),
            "comment": operation.comment,
        }


@dataclass
class LuaBranchSummary(LuaStatement):
    """Summarises the control flow decisions taken by a branch instruction."""

    descriptor: BranchDescriptor

    def render(self) -> List[str]:
        descriptor = self.descriptor
        header = f"-- branch {descriptor.mnemonic} [{descriptor.classification}]"
        if descriptor.comment:
            header += f" :: {descriptor.comment}"
        lines = [header]
        for edge in descriptor.edges:
            target = "exit" if edge.target is None else f"0x{edge.target:06X}"
            qualifiers: List[str] = []
            if edge.is_backward:
                qualifiers.append("back")
            if edge.is_fallthrough:
                qualifiers.append("fallthrough")
            if edge.comment:
                qualifiers.append(edge.comment)
            suffix = f" ({', '.join(qualifiers)})" if qualifiers else ""
            lines.append(f"--   -> {edge.role}: {target}{suffix}")
        return lines

    def metadata(self) -> Optional[dict]:
        descriptor = self.descriptor
        return {
            "type": "branch",
            "offset": descriptor.instruction_offset,
            "mnemonic": descriptor.mnemonic,
            "classification": descriptor.classification,
            "comment": descriptor.comment,
            "edges": [
                {
                    "role": edge.role,
                    "target": edge.target,
                    "is_backward": edge.is_backward,
                    "is_fallthrough": edge.is_fallthrough,
                    **({"comment": edge.comment} if edge.comment else {}),
                }
                for edge in descriptor.edges
            ],
        }


@dataclass
class LuaFunction:
    name: str
    statements: List[LuaStatement]

    def render(self) -> str:
        lines = [f"function {self.name}()"]
        for statement in self.statements:
            for line in statement.render():
                if line:
                    lines.append("  " + line)
                else:
                    lines.append("")
        lines.append("end")
        return "\n".join(lines) + "\n"

    def metadata(self) -> List[dict]:
        records: List[dict] = []
        for statement in self.statements:
            record = statement.metadata()
            if record is not None:
                records.append(record)
        return records

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps({"name": self.name, "body": self.metadata()}, indent=indent)

    def operation_count(self) -> int:
        return sum(isinstance(statement, LuaVMOperation) for statement in self.statements)

    def warning_count(self) -> int:
        count = 0
        for statement in self.statements:
            if isinstance(statement, LuaVMOperation):
                count += len(statement.operation.warnings)
        return count

    def summary(self) -> dict:
        return {
            "name": self.name,
            "operations": self.operation_count(),
            "warnings": self.warning_count(),
            "labels": self.labels(),
            "has_warnings": self.has_warnings(),
        }

    def labels(self) -> List[str]:
        return [statement.label for statement in self.statements if isinstance(statement, LuaLabel)]

    def has_warnings(self) -> bool:
        return self.warning_count() > 0


class LuaReconstructor:
    """Transform IR blocks into a Lua-flavoured listing with stack effects."""

    def __init__(self, analyzer: Optional[VirtualMachineAnalyzer] = None) -> None:
        self._analyzer = analyzer or VirtualMachineAnalyzer()
        self._branch_graph: Optional[BranchGraph] = None
        self._branch_by_block: Dict[int, BranchDescriptor] = {}

    def from_ir(self, segment_index: int, program: IRProgram) -> LuaFunction:
        branch_graph = describe_branches(program)
        self._branch_graph = branch_graph
        self._branch_by_block = {descriptor.block_start: descriptor for descriptor in branch_graph}
        statements: List[LuaStatement] = [
            LuaRawStatement("local stack = {}"),
            LuaRawStatement("local vm = {}"),
        ]
        for idx, block in enumerate(self._iter_blocks(program)):
            trace = self._analyzer.trace_block(block)
            if idx > 0:
                statements.append(LuaBlankLine())
            statements.extend(self._emit_block(trace))
            descriptor = self._branch_by_block.get(trace.start)
            if descriptor is not None:
                statements.append(LuaBranchSummary(descriptor))
        name = f"segment_{segment_index:03d}"
        return LuaFunction(name=name, statements=statements)

    def render(self, function: LuaFunction) -> str:
        return function.render()

    @property
    def branch_graph(self) -> Optional[BranchGraph]:
        return self._branch_graph

    def _emit_block(self, trace: VMBlockTrace) -> List[LuaStatement]:
        statements: List[LuaStatement] = [LuaLabel(f"block_{trace.start:06X}")]
        for operation in trace.operations:
            statements.append(LuaVMOperation(operation))
        return statements

    @staticmethod
    def _iter_blocks(program: IRProgram) -> Iterable[IRBlock]:
        for start in sorted(program.blocks):
            yield program.blocks[start]


def _stack_comment(action: str, value) -> Optional[str]:
    parts: List[str] = [action, value.name]
    if value.origin and value.origin not in value.name:
        parts.append(value.origin)
    if value.comment:
        parts.append(value.comment)
    if value.offset is not None:
        parts.append(f"@0x{value.offset:06X}")
    return " | ".join(parts) if len(parts) > 2 else " ".join(parts)
