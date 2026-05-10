from __future__ import annotations

"""Instruction-level MBC bytecode decoder.

This module owns only bytecode decoding and direct edge extraction. It does not
walk program reachability, assemble basic blocks, render DOT, or produce dump
artifacts.
"""

from dataclasses import dataclass
from typing import List, Optional

from .loader import MbcProgram, MbcScript
from .opcodes import (
    CODE_FILE_OFFSET,
    builtin_to_dict,
    decode_opcode,
    opcode_to_dict,
    safe_chr,
)


@dataclass
class Edge:
    kind: str
    src: int
    dst: Optional[int]
    dst_program: Optional[str] = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "src": self.src,
            "src_file": self.src + CODE_FILE_OFFSET,
            "dst": self.dst,
            "dst_file": None if self.dst is None else self.dst + CODE_FILE_OFFSET,
            "dst_program": self.dst_program,
            "note": self.note,
        }


@dataclass
class Instruction:
    offset: int
    file_offset: int
    opcode: int
    mnemonic: str
    length: int
    raw: str
    operands: dict
    edges: List[Edge]
    terminal: bool = False
    known: bool = True

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "file_offset": self.file_offset,
            "opcode": self.opcode,
            "opcode_hex": f"0x{self.opcode:02X}",
            "opcode_chr": safe_chr(self.opcode),
            "mnemonic": self.mnemonic,
            "length": self.length,
            "raw": self.raw,
            "operands": self.operands,
            "terminal": self.terminal,
            "known": self.known,
            "edges": [e.to_dict() for e in self.edges],
        }


class MbcDecoder:
    """Decode a single MBC instruction at a code-section offset.

    PC convention: all offsets are code-section offsets. File offset is
    ``code_offset + 0x20``. The original VM increments PC before invoking a
    handler, so relative branches are based at ``off + 1``.
    """

    def __init__(self, script: MbcScript):
        self.script = script
        self.code = script.code

    def opcode_tables(self) -> dict:
        return {
            "top_opcodes": opcode_to_dict(),
            "builtins": builtin_to_dict(),
        }

    def decode_at(self, off: int, program: Optional[MbcProgram] = None) -> Instruction:
        if not (0 <= off < len(self.code)):
            raise ValueError(f"code offset 0x{off:X} is outside code section")

        decoded = decode_opcode(self.code, off)
        length = max(decoded.length, 1)
        raw = self.code[off:min(len(self.code), off + length)].hex(" ")
        edges = [
            Edge(edge.kind, off, edge.dst, self._program_name_for(edge.dst) if edge.dst is not None else None, edge.note)
            for edge in decoded.edges
        ]

        # Add program references as metadata edges where the operand is a valid
        # program table index. They are useful navigation hints, not fallthrough.
        operands = dict(decoded.operands)
        program_index = operands.get("program_index") if operands else None
        if isinstance(program_index, int) and 0 <= program_index < len(self.script.programs):
            target_program = self.script.programs[program_index]
            operands["program_name"] = target_program.name
            operands["program_start"] = target_program.start
            operands["program_start_file"] = target_program.file_start
            edges.append(Edge("program_ref", off, target_program.start, target_program.name, decoded.mnemonic))

        return Instruction(
            offset=off,
            file_offset=off + CODE_FILE_OFFSET,
            opcode=self.code[off],
            mnemonic=decoded.mnemonic,
            length=length,
            raw=raw,
            operands=operands,
            edges=edges,
            terminal=decoded.terminal,
            known=decoded.known,
        )

    def _program_name_for(self, offset: Optional[int]) -> Optional[str]:
        if offset is None:
            return None
        p = self.script.program_for_offset(offset)
        return None if p is None else p.name
