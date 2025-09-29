"""Helpers for decoding VM literal operands into Lua-friendly objects.

Historically literal operands were formatted directly into strings which made it
hard to distinguish between numeric and textual constants during later passes.
This module exposes a small set of dataclasses that retain the semantic
classification of a literal while still providing formatted Lua source code on
request.  The :class:`LuaLiteralFormatter` mirrors the behaviour previously
embedded in :mod:`mbcdisasm.vm_analysis` but additionally categorises operands
and exposes richer metadata which higher level reconstruction stages can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


_PRINTABLE_LOW = 32
_PRINTABLE_HIGH = 126
_HEX_THRESHOLD = 9


@dataclass(frozen=True)
class LuaLiteral:
    """A decoded literal ready for Lua emission."""

    kind: str
    value: object
    text: str
    comment: Optional[str] = None

    def render(self) -> str:
        return self.text

    def is_string(self) -> bool:
        return self.kind == "string"

    def is_numeric(self) -> bool:
        return self.kind == "number"


def escape_lua_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("\"", "\\\"")
    return f'"{escaped}"'


def _ascii_candidate(operand: int) -> Optional[str]:
    raw = operand.to_bytes(2, "little")
    if all(_PRINTABLE_LOW <= byte <= _PRINTABLE_HIGH for byte in raw):
        return raw.decode("ascii")
    if raw[1] == 0 and _PRINTABLE_LOW <= raw[0] <= _PRINTABLE_HIGH:
        return chr(raw[0])
    return None


def _format_number(value: int) -> str:
    signed = value if value < 0x8000 else value - 0x10000
    if -_HEX_THRESHOLD <= signed <= _HEX_THRESHOLD:
        return str(signed)
    return f"0x{value:04X}"


def _classify_operand(operand: int) -> LuaLiteral:
    text = _ascii_candidate(operand)
    if text is not None:
        return LuaLiteral("string", text, escape_lua_string(text))
    return LuaLiteral("number", operand, _format_number(operand))


def _combine_ascii_chunks(chunks: Sequence[LuaLiteral]) -> LuaLiteral:
    combined = "".join(str(chunk.value) for chunk in chunks)
    return LuaLiteral("string", combined, escape_lua_string(combined))


def merge_adjacent_strings(literals: Sequence[LuaLiteral]) -> List[LuaLiteral]:
    """Merge consecutive string literals into a single literal.

    Several bytecode sequences push multiple ASCII chunks that conceptually form
    a longer string.  The low level representation keeps each chunk separate but
    the high level output benefits from seeing the full text.  This helper walks
    a sequence of literals and collapses neighbouring string entries.
    """

    merged: List[LuaLiteral] = []
    buffer: List[LuaLiteral] = []
    for literal in literals:
        if literal.is_string():
            buffer.append(literal)
            continue
        if buffer:
            merged.append(_combine_ascii_chunks(buffer))
            buffer.clear()
        merged.append(literal)
    if buffer:
        merged.append(_combine_ascii_chunks(buffer))
    return merged


class LuaLiteralFormatter:
    """Utility that converts operands into typed :class:`LuaLiteral` objects."""

    def format_operand(self, operand: int) -> LuaLiteral:
        return _classify_operand(operand)

    def format_operands(self, operands: Iterable[int]) -> List[LuaLiteral]:
        literals = [_classify_operand(operand) for operand in operands]
        return merge_adjacent_strings(literals)


__all__ = [
    "LuaLiteral",
    "LuaLiteralFormatter",
    "escape_lua_string",
    "merge_adjacent_strings",
]
