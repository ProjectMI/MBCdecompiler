"""Pretty-printer for the reconstructed AST."""

from __future__ import annotations

from typing import Iterable, List

from .model import ASTBlock, ASTFunction, ASTProgram


class ASTRenderer:
    """Render :class:`ASTProgram` instances to a textual format."""

    def render(self, program: ASTProgram) -> str:
        lines: List[str] = []
        for function in program.functions:
            lines.extend(self._render_function(function))
        return "\n".join(lines) + ("\n" if lines else "")

    def write(self, program: ASTProgram, output_path) -> None:
        output_path.write_text(self.render(program), "utf-8")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _render_function(self, function: ASTFunction) -> Iterable[str]:
        header = (
            f"; function segment={function.segment_index} name={function.name} "
            f"entry={function.entry_block} offset=0x{function.entry_offset:06X}"
        )
        yield header
        for block in function.blocks:
            yield from self._render_block(block)
        yield "  dominators:"
        for info in function.dominators:
            members = ", ".join(info.members)
            yield f"    {info.label}: [{members}]"
        yield "  post_dominators:"
        for info in function.post_dominators:
            members = ", ".join(info.members)
            yield f"    {info.label}: [{members}]"
        if function.loops:
            yield "  loops:"
            for loop in function.loops:
                nodes = ", ".join(loop.nodes)
                latches = ", ".join(loop.latches)
                yield f"    header={loop.header} nodes=[{nodes}] latches=[{latches}]"
        else:
            yield "  loops: (none)"
        yield ""

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        offset = "?" if block.offset is None else f"0x{block.offset:06X}"
        suffix = " synthetic" if block.synthetic else ""
        yield f"  block {block.label}{suffix} offset={offset}"
        if block.annotations:
            for note in block.annotations:
                yield f"    ; {note}"
        preds = ", ".join(block.predecessors) if block.predecessors else "(entry)"
        succs = ", ".join(block.successors) if block.successors else "(exit)"
        yield f"    preds: {preds}"
        yield f"    succs: {succs}"
        if block.statements:
            yield "    statements:"
            for node in block.statements:
                describe = getattr(node, "describe", None)
                rendered = describe() if callable(describe) else repr(node)
                yield f"      {rendered}"
        else:
            yield "    statements: (none)"
        yield f"    terminator: {block.terminator.text}"


__all__ = ["ASTRenderer"]
