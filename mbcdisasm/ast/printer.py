"""Render reconstructed AST programmes into text."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import ASTBlock, ASTProgram, ASTSegment


class ASTTextRenderer:
    """Render :class:`ASTProgram` instances into a textual report."""

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
        procedure_map = {proc.entry: proc for proc in segment.procedures}
        block_map = {block.label: block for block in segment.blocks}
        for procedure in segment.procedures:
            yield from self._render_procedure(procedure, block_map)
        # render orphan blocks not associated with explicit procedures
        for block in segment.blocks:
            if block.label not in procedure_map:
                yield from self._render_block(block)
        yield ""

    def _render_procedure(self, procedure, block_map: dict[str, ASTBlock]) -> Iterable[str]:
        exits = ",".join(procedure.exits) if procedure.exits else "-"
        sources = ",".join(procedure.sources) if procedure.sources else "-"
        yield (
            f"procedure {procedure.name} entry={procedure.entry} blocks={len(procedure.blocks)} "
            f"exits={exits} sources={sources}"
        )
        for label in procedure.blocks:
            block = block_map.get(label)
            if block:
                yield from self._render_block(block)

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        yield f"  block {block.label} offset=0x{block.start_offset:06X}"
        for statement in block.statements:
            yield f"    {statement.describe()}"


__all__ = ["ASTTextRenderer"]
