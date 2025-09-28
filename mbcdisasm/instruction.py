"""Representation utilities for raw instruction words."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


WORD_SIZE = 4


@dataclass(frozen=True)
class InstructionWord:
    offset: int
    raw: int

    @property
    def opcode(self) -> int:
        return (self.raw >> 24) & 0xFF

    @property
    def mode(self) -> int:
        return (self.raw >> 16) & 0xFF

    @property
    def operand(self) -> int:
        return self.raw & 0xFFFF

    def label(self) -> str:
        return f"{self.opcode:02X}:{self.mode:02X}"

    def format(self) -> str:
        return f"{self.offset:08X}: {self.raw:08X}    op={self.opcode:02X} mode={self.mode:02X} operand={self.operand:04X}"


def read_instructions(segment_data: bytes, base_offset: int) -> Tuple[List[InstructionWord], int]:
    """Decode a raw segment into instruction words.

    The historical implementation rejected segments whose length was not a
    multiple of four bytes which caused the disassembler to skip swathes of
    scripts that merely contained a short footer or alignment padding.  Instead
    of bailing out we now decode the aligned portion and return the number of
    bytes that could not be interpreted as a full instruction.  Callers can use
    this remainder to surface a warning while still keeping the useful output.
    """

    remainder = len(segment_data) % WORD_SIZE
    usable = len(segment_data) - remainder
    instructions: List[InstructionWord] = []
    for idx in range(0, usable, WORD_SIZE):
        raw = int.from_bytes(segment_data[idx : idx + WORD_SIZE], "big")
        instructions.append(InstructionWord(base_offset + idx, raw))
    return instructions, remainder
