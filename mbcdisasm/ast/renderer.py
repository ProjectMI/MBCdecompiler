"""Text renderer for :mod:`mbcdisasm.ast` structures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import ASTBlock, ASTDominatorInfo, ASTFunction, ASTLoop, ASTProgram


class ASTRenderer:
    """Render :class:`ASTProgram` instances into a stable textual form."""

    def render(self, program: ASTProgram) -> str:
        lines: List[str] = []
        for function in program.functions:
            lines.extend(self._render_function(function))
        return "\n".join(lines) + ("\n" if lines else "")

    def write(self, program: ASTProgram, output_path: Path) -> None:
        output_path.write_text(self.render(program), "utf-8")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _render_function(self, function: ASTFunction) -> Iterable[str]:
        header = (
            f"function segment={function.segment_index} name={function.name} "
            f"entry={function.entry}"
        )
        yield header
        for block in function.blocks:
            yield from self._render_block(block)
        yield from self._render_dom_info("dominators", function.dominators)
        yield from self._render_dom_info("post_dominators", function.post_dominators)
        yield from self._render_loops(function.loops)
        yield ""

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        preds = ", ".join(block.predecessors) or "-"
        succs = ", ".join(block.successors) or "-"
        yield (
            f"  block {block.label} offset=0x{block.start_offset:06X} "
            f"preds=[{preds}] succs=[{succs}]"
        )
        for node in block.body:
            describe = getattr(node, "describe", None)
            if callable(describe):
                yield f"    {describe()}"
            else:
                yield f"    {node!r}"
        yield f"    terminator {block.terminator.describe()}"

    def _render_dom_info(
        self, label: str, infos: Iterable[ASTDominatorInfo]
    ) -> Iterable[str]:
        infos = list(infos)
        if not infos:
            yield f"  {label}: (empty)"
            return
        yield f"  {label}:"
        for info in infos:
            dominators = ", ".join(info.dominators)
            immediate = info.immediate or "-"
            yield f"    {info.block}: dom=[{dominators}] idom={immediate}"

    def _render_loops(self, loops: Iterable[ASTLoop]) -> Iterable[str]:
        loops = list(loops)
        if not loops:
            yield "  loops: none"
            return
        yield "  loops:"
        for loop in loops:
            nodes = ", ".join(loop.nodes)
            latches = ", ".join(loop.latches)
            yield f"    header={loop.header} nodes=[{nodes}] latches=[{latches}]"


__all__ = ["ASTRenderer"]
