"""Pseudo-Lua reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .ir import IRProgram


@dataclass
class PseudoStatement:
    text: str


@dataclass
class LuaFunction:
    name: str
    statements: List[PseudoStatement]

    def render(self) -> str:
        lines = [f"function {self.name}()"]
        for stmt in self.statements:
            lines.append("  " + stmt.text)
        lines.append("end")
        return "\n".join(lines) + "\n"


class LuaReconstructor:
    """Transform IR blocks into a comment-rich Lua-like listing."""

    def from_ir(self, segment_index: int, program: IRProgram) -> LuaFunction:
        statements: List[PseudoStatement] = []
        for start in sorted(program.blocks):
            block = program.blocks[start]
            statements.append(PseudoStatement(f"::block_{start:06X}::"))
            for instr in block.instructions:
                invocation = f"{instr.mnemonic}(0x{instr.operand:04X})"
                suffix: List[str] = []
                if instr.stack_delta is not None:
                    suffix.append(f"stackΔ={instr.stack_delta:+.1f}")
                if instr.control_flow:
                    suffix.append(instr.control_flow)
                if suffix:
                    invocation += "  -- " + ", ".join(suffix)
                statements.append(PseudoStatement(invocation))
        name = f"segment_{segment_index:03d}"
        return LuaFunction(name=name, statements=statements)

    def render(self, function: LuaFunction) -> str:
        return function.render()
