"""Serialise reconstructed AST structures into a textual report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import ASTBlock, ASTProcedure, ASTProgram, ASTSegment


class ASTTextRenderer:
    """Render :class:`ASTProgram` instances into a stable textual form."""

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
        if not segment.procedures:
            yield ";   no procedures reconstructed"
            yield ""
            return
        for procedure in segment.procedures:
            yield from self._render_procedure(procedure)
        yield ""

    def _render_procedure(self, procedure: ASTProcedure) -> Iterable[str]:
        reasons = ",".join(procedure.entry_reasons) or "unspecified"
        exits = ",".join(f"0x{offset:04X}" for offset in procedure.exit_offsets) or "?"
        yield (
            f"procedure {procedure.name} entry=0x{procedure.entry_offset:04X} "
            f"reasons={reasons} exits=[{exits}]"
        )
        for line in self._render_procedure_metadata(procedure):
            yield line
        for block in procedure.blocks:
            yield from self._render_block(block)
        yield ""

    def _render_procedure_metadata(self, procedure: ASTProcedure) -> Iterable[str]:
        if procedure.parameters:
            rendered = ", ".join(param.render() for param in procedure.parameters)
            yield f"    ; parameters: {rendered}"
        if procedure.return_mask is not None:
            yield f"    ; return_mask=0x{procedure.return_mask:04X}"
        if procedure.frame.slots:
            rendered_slots = ", ".join(slot.describe() for slot in procedure.frame.slots)
            yield f"    ; frame: {rendered_slots}"

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        successors = ", ".join(target.label for target in block.successors)
        yield f"  block {block.label} offset=0x{block.start_offset:04X} succ=[{successors}]"
        for statement in block.statements:
            yield f"    {statement.render()}"


__all__ = ["ASTTextRenderer"]
