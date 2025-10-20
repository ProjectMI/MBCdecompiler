"""Serialise reconstructed AST structures into a textual report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import ASTBlock, ASTEnumDecl, ASTProcedure, ASTProgram, ASTSegment


class ASTTextRenderer:
    """Render :class:`ASTProgram` instances into a stable textual form."""

    def render(self, program: ASTProgram) -> str:
        lines: List[str] = []
        lines.append("; ast metrics: " + program.metrics.describe())
        if program.symbols:
            lines.append("; symbol table:")
            for entry in program.symbols:
                lines.append(f";   {entry.render()}")
        if program.strings:
            lines.append("; string pool:")
            for entry in program.strings:
                lines.append(f";   {entry.render()}")
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
            f"; segment {segment.index} kind={segment.kind} "
            f"offset=0x{segment.start:06X} length={segment.length}"
        )
        yield header
        if segment.enums:
            for enum in segment.enums:
                yield from self._render_enum(enum)
            yield ""
        if not segment.procedures:
            yield ";   no procedures reconstructed"
            yield ""
            return
        for procedure in segment.procedures:
            yield from self._render_procedure(procedure)
        yield ""

    def _render_enum(self, enum: ASTEnumDecl) -> Iterable[str]:
        yield f"enum {enum.name} {{"
        for member in enum.members:
            yield f"  {member.name} = 0x{member.value:04X}"
        yield "}"

    def _render_procedure(self, procedure: ASTProcedure) -> Iterable[str]:
        entry_repr = procedure.entry.render()
        exit_entries = ", ".join(exit.render() for exit in procedure.exits) or "-"
        succ_map = ", ".join(
            f"{label}->[{', '.join(targets)}]"
            for label, targets in sorted(procedure.successor_map.items())
        ) or "-"
        pred_map = ", ".join(
            f"{label}->[{', '.join(targets)}]"
            for label, targets in sorted(procedure.predecessor_map.items())
        ) or "-"
        yield (
            f"procedure {procedure.name} entry{{{entry_repr}}} "
            f"exits=[{exit_entries}] cfg{{succ_map={{ {succ_map} }} pred_map={{ {pred_map} }}}}"
        )
        for block in procedure.blocks:
            yield from self._render_block(block)
        yield ""

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        successor_blocks = block.successors or ()
        successors = ", ".join(target.label for target in successor_blocks)
        yield f"  block {block.label} offset=0x{block.start_offset:04X} succ=[{successors}]"
        for statement in block.statements:
            yield f"    {statement.render()}"


__all__ = ["ASTTextRenderer"]
