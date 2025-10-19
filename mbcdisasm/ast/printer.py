"""Serialise reconstructed AST structures into a textual report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import (
    ASTBlock,
    ASTEnumDecl,
    ASTProcedure,
    ASTProgram,
    ASTSegment,
    ASTStatement,
)


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
        reasons = ",".join(procedure.entry_reasons) or "unspecified"
        exits = ",".join(f"0x{offset:04X}" for offset in procedure.exit_offsets) or "?"
        yield (
            f"procedure {procedure.name} entry=0x{procedure.entry_offset:04X} "
            f"reasons={reasons} exits=[{exits}]"
        )
        if procedure.body:
            yield "  body {"
            yield from self._render_structured_statements(procedure.body, indent=4)
            yield "  }"
        else:
            yield "  body {}"
        if procedure.blocks:
            yield "  ; cfg"
            for block in procedure.blocks:
                yield from self._render_block(block)
        else:
            yield "  ; cfg unavailable"
        yield ""

    def _render_structured_statements(
        self, statements: Iterable[ASTStatement], *, indent: int
    ) -> Iterable[str]:
        prefix = " " * indent
        for statement in statements:
            yield f"{prefix}{statement.render()}"

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        successors = ", ".join(target.label for target in block.successors)
        yield f"  block {block.label} offset=0x{block.start_offset:04X} succ=[{successors}]"
        for statement in block.statements:
            yield f"    {statement.render()}"


__all__ = ["ASTTextRenderer"]
