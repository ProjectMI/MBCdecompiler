"""Intermediate representation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .cfg import ControlFlowGraph, BasicBlock
from .knowledge import KnowledgeBase
from .manual_semantics import AnnotatedInstruction, InstructionSemantics
from .vm_analysis import estimate_stack_io
from .branch_analysis import BranchRegistry, BranchResolver, analyse_branches
from .instruction import InstructionWord


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
    graph: Optional[ControlFlowGraph] = None
    _branch_registry: Optional[BranchRegistry] = field(default=None, init=False, repr=False)

    def render_text(self) -> str:
        lines: List[str] = [f"segment {self.segment_index} IR"]
        for start in sorted(self.blocks):
            lines.extend(self.blocks[start].to_text())
        return "\n".join(lines) + "\n"

    def branch_registry(self, knowledge: KnowledgeBase) -> BranchRegistry:
        """Return a lazily constructed :class:`BranchRegistry`."""

        if self._branch_registry is None:
            if self.graph is not None:
                self._branch_registry = analyse_branches(knowledge, self.graph)
            else:
                resolver = BranchResolver(knowledge)
                pseudo_graph = self._to_control_flow_graph()
                self._branch_registry = resolver.analyse(pseudo_graph)
        return self._branch_registry

    def _to_control_flow_graph(self) -> ControlFlowGraph:
        """Reconstruct a lightweight CFG using stored IR information."""

        blocks: Dict[int, BasicBlock] = {}
        for start, ir_block in self.blocks.items():
            annotated: List[AnnotatedInstruction] = []
            for instr in ir_block.instructions:
                word = _synthetic_word(instr)
                annotated.append(AnnotatedInstruction(word=word, semantics=instr.semantics))
            block = BasicBlock(
                start=start,
                instructions=annotated,
                successors=set(ir_block.successors),
                predecessors=set(),
            )
            blocks[start] = block

        for block in blocks.values():
            for successor in block.successors:
                if successor in blocks:
                    blocks[successor].predecessors.add(block.start)

        segment = self.graph.segment if self.graph is not None else None
        return ControlFlowGraph(segment=segment, blocks=blocks, remainder=0)


def _synthetic_word(instr: IRInstruction) -> InstructionWord:
    try:
        opcode_hex, mode_hex = instr.key.split(":", 1)
        opcode = int(opcode_hex, 16)
        mode = int(mode_hex, 16)
    except ValueError:
        opcode = instr.semantics.mnemonic.__hash__() & 0xFF
        mode = 0
    raw = (opcode << 24) | (mode << 16) | (instr.operand & 0xFFFF)
    return InstructionWord(offset=instr.offset, raw=raw)


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
        return IRProgram(segment.index, blocks, graph=graph)

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
