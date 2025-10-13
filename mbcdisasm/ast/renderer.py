"""Serialisation helpers for the lifted AST."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import ASTBlock, ASTFunction, ASTProgram


class ASTRenderer:
    """Render :class:`ASTProgram` instances to a textual representation."""

    def render(self, program: ASTProgram) -> str:
        lines: List[str] = []
        for function in program.functions:
            lines.extend(self._render_function(function))
        return "\n".join(lines) + "\n"

    def write(self, program: ASTProgram, output_path: Path) -> None:
        output_path.write_text(self.render(program), "utf-8")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _render_function(self, function: ASTFunction) -> Iterable[str]:
        yield f"function {function.name} entry=0x{function.entry_offset:06X}"
        if function.attributes:
            attrs = ", ".join(function.attributes)
            yield f"  ; attrs: {attrs}"
        for block in function.blocks:
            yield from self._render_block(block)
        yield ""

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        yield f"  block {block.label} offset=0x{block.start_offset:06X}"
        for statement in block.statements:
            yield f"    {statement.describe()}"


__all__ = ["ASTRenderer"]
