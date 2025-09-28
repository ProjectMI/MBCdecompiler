"""Intermediate representation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .cfg import ControlFlowGraph
from .knowledge import KnowledgeBase
from .manual_semantics import AnnotatedInstruction, InstructionSemantics
from .vm_analysis import estimate_stack_io


@dataclass
class IRInstruction:
    offset: int
    key: str
    mnemonic: str
    operand: int
    stack_delta: Optional[float]
    control_flow: Optional[str]
    semantics: InstructionSemantics
    stack_inputs: int
    stack_outputs: int

    def to_text(self) -> str:
        stack = "" if self.stack_delta is None else f" stackΔ={self.stack_delta:+.1f}"
        cf = f" [{self.control_flow}]" if self.control_flow else ""
        io = f" inputs={self.stack_inputs} outputs={self.stack_outputs}"
        return (
            f"{self.offset:08X}: {self.mnemonic} operand=0x{self.operand:04X}"
            f" ({self.semantics.manual_name})" + stack + io + cf
        )


@dataclass
class IRBlock:
    start: int
    instructions: List[IRInstruction]
    successors: List[int]

    def to_text(self) -> List[str]:
        lines = [f"block 0x{self.start:06X} -> {[hex(s) for s in self.successors]}"]
        for instr in self.instructions:
            lines.append("  " + instr.to_text())
        return lines


@dataclass
class IRProgram:
    segment_index: int
    blocks: Dict[int, IRBlock]

    def render_text(self) -> str:
        lines: List[str] = [f"segment {self.segment_index} IR"]
        for start in sorted(self.blocks):
            lines.extend(self.blocks[start].to_text())
        return "\n".join(lines) + "\n"


class IRBuilder:
    """Translate CFGs into a lightweight intermediate representation."""

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge

    def from_cfg(self, segment, graph: ControlFlowGraph) -> IRProgram:
        blocks: Dict[int, IRBlock] = {}
        for start, block in graph.blocks.items():
            ir_instructions = [self._lower_instruction(instr) for instr in block.instructions]
            blocks[start] = IRBlock(
                start=start,
                instructions=ir_instructions,
                successors=sorted(block.successors),
            )
        return IRProgram(segment.index, blocks)

    def _lower_instruction(self, instr: AnnotatedInstruction) -> IRInstruction:
        word = instr.word
        key = word.label()
        semantics = instr.semantics
        inputs, outputs = estimate_stack_io(semantics)
        return IRInstruction(
            offset=word.offset,
            key=key,
            mnemonic=semantics.mnemonic,
            operand=word.operand,
            stack_delta=semantics.stack_delta,
            control_flow=semantics.control_flow,
            semantics=semantics,
            stack_inputs=inputs,
            stack_outputs=outputs,
        )


def render_ir_programs(programs: Iterable[IRProgram]) -> str:
    lines: List[str] = []
    for program in programs:
        lines.append(program.render_text().rstrip())
    return "\n\n".join(lines) + "\n"


def write_ir_programs(
    programs: Iterable[IRProgram], path: Path, *, encoding: str = "utf-8"
) -> None:
    """Render the provided IR programs and write them to ``path``.

    The helper mirrors :func:`render_ir_programs` but persists the result to disk,
    making it convenient for tests and scripts to capture the IR output without
    duplicating file-handling logic.
    """

    path.write_text(render_ir_programs(programs), encoding)
