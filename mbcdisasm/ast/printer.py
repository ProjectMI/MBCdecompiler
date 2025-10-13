"""Serialisation helpers for writing AST structures to text."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import ASTBlock, ASTFunction, ASTProgram, ASTSegment


class ASTTextRenderer:
    """Render :class:`ASTProgram` objects into a stable textual form."""

    def render(self, program: ASTProgram) -> str:
        lines: List[str] = []
        lines.append("; ast metrics: " + program.metrics.describe())
        if program.string_pool:
            lines.append("; string pool")
            for const in program.string_pool:
                lines.append(const.describe())
            lines.append("")

        for segment in program.segments:
            lines.extend(self._render_segment(segment))
        return "\n".join(lines) + "\n"

    def write(self, program: ASTProgram, output_path: Path) -> None:
        output_path.write_text(self.render(program), "utf-8")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _render_segment(self, segment: ASTSegment) -> Iterable[str]:
        header = (
            f"; segment {segment.index} offset=0x{segment.start:06X} length={segment.length}"
        )
        yield header
        yield "; metrics: " + segment.metrics.describe()
        if not segment.functions:
            yield "; functions: none"
            yield ""
            return
        for function in segment.functions:
            yield from self._render_function(function)
        yield ""

    def _render_function(self, function: ASTFunction) -> Iterable[str]:
        yield f"function {function.name} entry=0x{function.entry_offset:06X}"
        for block in function.blocks:
            yield from self._render_block(block)
        yield ""

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        yield f"  block {block.label} offset=0x{block.start_offset:06X}"
        for note in block.annotations:
            yield f"    ; {note}"
        for stmt in block.statements:
            yield f"    {stmt.text}"


__all__ = ["ASTTextRenderer"]
