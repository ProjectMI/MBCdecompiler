"""Serialise the lifted AST into a stable textual representation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import ASTBlock, ASTFunction, ASTProgram


class ASTPrinter:
    """Render :class:`ASTProgram` instances as text."""

    def render(self, program: ASTProgram) -> str:
        lines: List[str] = []
        for function in program.functions:
            lines.extend(self._render_function(function))
        return "\n".join(lines).rstrip() + "\n"

    def write(self, program: ASTProgram, output_path: Path) -> None:
        output_path.write_text(self.render(program), "utf-8")

    # ------------------------------------------------------------------
    def _render_function(self, function: ASTFunction) -> Iterable[str]:
        header = (
            f"function {function.name} segment={function.segment_index} "
            f"entry={function.entry_label} offset=0x{function.start_offset:06X}"
        )
        yield header
        for block in function.blocks:
            yield from self._render_block(block)
        yield ""

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        succ = ", ".join(block.successors)
        suffix = f" succ=[{succ}]" if succ else ""
        yield f"  block {block.label} offset=0x{block.start_offset:06X}{suffix}"
        for statement in block.statements:
            yield f"    {statement.text}"


__all__ = ["ASTPrinter"]

