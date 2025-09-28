"""Pseudo-Lua reconstruction helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .ir import IRBlock, IRInstruction, IRProgram


@dataclass
class LuaFunction:
    name: str
    statements: List[str]

    def render(self) -> str:
        lines = [f"function {self.name}()"]
        for stmt in self.statements:
            if stmt:
                lines.append("  " + stmt)
            else:
                lines.append("")
        lines.append("end")
        return "\n".join(lines) + "\n"


class LuaLiteralFormatter:
    """Best-effort conversion of operands into Lua literals."""

    def format_operand(self, operand: int) -> str:
        signed = operand if operand < 0x8000 else operand - 0x10000
        if -9 <= signed <= 9:
            return str(signed)

        ascii_text = self._ascii_candidate(operand)
        if ascii_text is not None:
            return self._lua_string(ascii_text)

        return f"0x{operand:04X}"

    @staticmethod
    def _ascii_candidate(operand: int) -> Optional[str]:
        raw = operand.to_bytes(2, "little")
        if all(32 <= byte <= 126 for byte in raw):
            return raw.decode("ascii")
        if raw[1] == 0 and 32 <= raw[0] <= 126:
            return chr(raw[0])
        return None

    @staticmethod
    def _lua_string(text: str) -> str:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'


class LuaEmitter:
    """Translate IR instructions into Lua stack operations."""

    def __init__(self) -> None:
        self.literal_formatter = LuaLiteralFormatter()

    def emit_block(self, block: IRBlock) -> List[str]:
        output: List[str] = [f"::block_{block.start:06X}::"]
        for instruction in block.instructions:
            output.extend(self.emit_instruction(instruction))
        return output

    def emit_instruction(self, instruction: IRInstruction) -> List[str]:
        if instruction.mnemonic.startswith("push_literal"):
            return self._emit_literal_push(instruction)

        lines: List[str] = []
        comment = self._format_comment(instruction, include_mnemonic=False)
        invocation = f"{instruction.mnemonic}(0x{instruction.operand:04X})"
        if comment:
            invocation += f"  -- {comment}"
        lines.append(invocation)

        delta = self._integer_stack_delta(instruction.stack_delta)
        if delta is None or delta == 0:
            return lines
        if delta > 0:
            placeholder = f'"{instruction.mnemonic}_result"'
            for index in range(delta):
                suffix = ""
                if index == 0:
                    suffix = f"  -- result from {instruction.mnemonic}"
                lines.append(f"stack[#stack + 1] = {placeholder}{suffix}")
            return lines

        pops = abs(delta)
        for index in range(pops):
            suffix = ""
            if index == 0:
                suffix = f"  -- pop via {instruction.mnemonic}"
            lines.append(f"stack[#stack] = nil{suffix}")
        return lines

    def _emit_literal_push(self, instruction: IRInstruction) -> List[str]:
        delta = self._integer_stack_delta(instruction.stack_delta)
        count = max(1, delta or 1)
        literal = self.literal_formatter.format_operand(instruction.operand)
        lines: List[str] = []
        for index in range(count):
            components: List[str] = []
            suffix = self._format_comment(instruction, include_mnemonic=True)
            if suffix:
                components.append(suffix)
            if count > 1:
                components.append(f"value {index + 1}/{count}")
            comment = ", ".join(components)
            push_line = f"stack[#stack + 1] = {literal}"
            if comment:
                push_line += f"  -- {comment}"
            lines.append(push_line)
        return lines

    @staticmethod
    def _integer_stack_delta(delta: Optional[float]) -> Optional[int]:
        if delta is None:
            return None
        rounded = int(round(delta))
        if math.isfinite(delta) and abs(delta - rounded) < 1e-6:
            return rounded
        return None

    @staticmethod
    def _format_comment(
        instruction: IRInstruction, *, include_mnemonic: bool
    ) -> str:
        parts: List[str] = []
        if include_mnemonic:
            parts.append(instruction.mnemonic)
        if instruction.stack_delta is not None:
            parts.append(f"stackΔ={instruction.stack_delta:+.1f}")
        if instruction.control_flow:
            parts.append(instruction.control_flow)
        return ", ".join(parts)


class LuaReconstructor:
    """Transform IR blocks into a Lua-flavoured listing with stack effects."""

    def __init__(self) -> None:
        self._emitter = LuaEmitter()

    def from_ir(self, segment_index: int, program: IRProgram) -> LuaFunction:
        statements: List[str] = ["local stack = {}"]
        for block in self._iter_blocks(program):
            statements.extend(self._emitter.emit_block(block))
        name = f"segment_{segment_index:03d}"
        return LuaFunction(name=name, statements=statements)

    def render(self, function: LuaFunction) -> str:
        return function.render()

    @staticmethod
    def _iter_blocks(program: IRProgram) -> Iterable[IRBlock]:
        for start in sorted(program.blocks):
            yield program.blocks[start]
