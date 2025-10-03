"""Raw instruction stream parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..instruction import InstructionWord
from ..knowledge import KnowledgeBase, OpcodeInfo


@dataclass(frozen=True)
class RawInstruction:
    """Parsed opcode description with metadata for normalisation passes."""

    word: InstructionWord
    mnemonic: str
    opcode: int
    mode: int
    operand: int
    info: Optional[OpcodeInfo]
    stack_delta: int
    category: Optional[str]
    control_flow: Optional[str]
    note: Optional[str] = None

    def label(self) -> str:
        return f"{self.opcode:02X}:{self.mode:02X}"


@dataclass(frozen=True)
class RawBlock:
    """Basic block consisting of raw instructions."""

    instructions: Tuple[RawInstruction, ...]

    @classmethod
    def from_instructions(
        cls, instructions: Sequence[RawInstruction]
    ) -> "RawBlock":
        return cls(tuple(instructions))


@dataclass(frozen=True)
class RawProgram:
    """Container for the parsed instruction stream."""

    blocks: Tuple[RawBlock, ...]

    @classmethod
    def from_blocks(cls, blocks: Sequence[RawBlock]) -> "RawProgram":
        return cls(tuple(blocks))


CONTROL_FLOW_TERMINALS = {"return", "tail", "jump"}


def parse_stream(
    words: Sequence[InstructionWord], knowledge: KnowledgeBase
) -> RawProgram:
    """Convert raw instructions to :class:`RawInstruction` entries."""

    raw_instructions: List[RawInstruction] = []
    for word in words:
        label = word.label()
        info = knowledge.lookup(label)
        mnemonic = info.mnemonic if info is not None else f"op_{label}"
        stack_delta = info.stack_effect() if info is not None else 0
        category = info.category if info is not None else None
        control = info.control_flow if info is not None else None
        raw_instructions.append(
            RawInstruction(
                word=word,
                mnemonic=mnemonic,
                opcode=word.opcode,
                mode=word.mode,
                operand=word.operand,
                info=info,
                stack_delta=stack_delta or 0,
                category=category,
                control_flow=control,
            )
        )

    blocks: List[List[RawInstruction]] = [[]]
    for inst in raw_instructions:
        blocks[-1].append(inst)
        if _terminates_block(inst):
            blocks.append([])
    if blocks and not blocks[-1]:
        blocks.pop()

    return RawProgram.from_blocks(RawBlock.from_instructions(block) for block in blocks)


def _terminates_block(inst: RawInstruction) -> bool:
    if inst.control_flow in CONTROL_FLOW_TERMINALS:
        return True
    mnemonic = inst.mnemonic.lower()
    if mnemonic in {"return", "return_values"}:
        return True
    return False
