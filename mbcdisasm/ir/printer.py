"""Utilities for serialising the normalised IR into a text format."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from .model import IRBlock, IRProgram, IRResourceSection, IRSegment


class IRTextRenderer:
    """Render :class:`IRProgram` instances into a stable textual form."""

    def render(self, program: IRProgram) -> str:
        lines: List[str] = []
        lines.append("; normalizer metrics: " + program.metrics.describe())
        if program.resources:
            lines.extend(self._render_resources(program.resources))
        for segment in program.segments:
            lines.extend(self._render_segment(segment))
        return "\n".join(lines) + "\n"

    def write(self, program: IRProgram, output_path: Path) -> None:
        output_path.write_text(self.render(program), "utf-8")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _render_resources(self, sections: Tuple["IRResourceSection", ...]) -> Iterable[str]:
        yield "; resources:"
        for section in sections:
            yield f";   section {section.name}"
            for entry in section.entries:
                summary = entry.describe()
                yield f";     {summary}"

    def _render_segment(self, segment: IRSegment) -> Iterable[str]:
        header = (
            f"; segment {segment.index} offset=0x{segment.start:06X} "
            f"length={segment.length}"
        )
        yield header
        yield "; metrics: " + segment.metrics.describe()
        for block in segment.blocks:
            yield from self._render_block(block)
        yield ""

    def _render_block(self, block: IRBlock) -> Iterable[str]:
        yield f"block {block.label} offset=0x{block.start_offset:06X}"
        if block.annotations:
            for note in block.annotations:
                yield f"  ; {note}"
        for node in block.nodes:
            describe = getattr(node, "describe", None)
            if callable(describe):
                yield f"  {describe()}"
            else:
                yield f"  {node!r}"


__all__ = ["IRTextRenderer"]
