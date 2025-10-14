"""Text renderer for the reconstructed AST."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import ASTBlock, ASTProcedure, ASTProgram, ASTSegment


class ASTTextRenderer:
    """Render :class:`ASTProgram` instances into a stable textual representation."""

    def render(self, program: ASTProgram) -> str:
        lines: List[str] = []
        lines.append("; ast metrics: " + program.metrics.describe())
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
        entry_text = ", ".join(f"0x{offset:04X}" for offset in segment.entry_offsets) or "<none>"
        yield f"; entries: {entry_text}"
        if segment.dangling_targets:
            dangling = ", ".join(f"0x{target:04X}" for target in segment.dangling_targets)
            yield f"; dangling targets: {dangling}"
        for procedure in segment.procedures:
            yield from self._render_procedure(procedure)
        yield ""

    def _render_procedure(self, procedure: ASTProcedure) -> Iterable[str]:
        header = (
            f"procedure {procedure.name} entry={procedure.entry_label} "
            f"offset=0x{procedure.entry_offset:04X}"
        )
        yield header
        if procedure.exits:
            exits = ", ".join(procedure.exits)
            yield f"  ; exits: {exits}"
        for block in procedure.blocks:
            yield from self._render_block(block)
        yield ""

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        yield f"  block {block.label} offset=0x{block.start_offset:06X} successors=[{block.describe_successors()}]"
        for statement in block.statements:
            yield f"    {statement.text}"


__all__ = ["ASTTextRenderer"]
