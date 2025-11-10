"""Rendering helpers for the AST representation."""

from __future__ import annotations

from typing import List

from .model import ASTBlock, ASTFunction, ASTProgram


def _describe_node(node) -> str:
    describe = getattr(node, "describe", None)
    if callable(describe):
        return describe()
    return repr(node)


class ASTRenderer:
    """Render the AST into a stable textual format."""

    def render(self, program: ASTProgram) -> str:
        lines: List[str] = []
        for function in program.functions:
            self._render_function(function, lines)
        return "\n".join(lines)

    def _render_function(self, function: ASTFunction, lines: List[str]) -> None:
        header = (
            f"function {function.name} segment={function.segment_index} "
            f"entry={function.entry_block} offset=0x{function.entry_offset:04X}"
        )
        if function.aliases:
            aliases = ", ".join(
                f"{alias.name}@0x{alias.entry_offset:04X}" for alias in function.aliases
            )
            header += f" aliases=[{aliases}]"
        lines.append(header)
        lines.append("  blocks:")
        for block in function.blocks:
            self._render_block(block, lines)
        self._render_dominators("dominators", function.dominators, lines)
        self._render_dominators("post_dominators", function.post_dominators, lines)
        if function.loops:
            lines.append("  loops:")
            for loop in function.loops:
                loop_line = (
                    f"    header={loop.header} latches=[{', '.join(loop.latches)}] "
                    f"blocks=[{', '.join(loop.blocks)}]"
                )
                lines.append(loop_line)
        lines.append("")

    def _render_block(self, block: ASTBlock, lines: List[str]) -> None:
        preds = ", ".join(block.predecessors)
        succs = ", ".join(block.successors)
        synthetic = " synthetic" if block.synthetic else ""
        lines.append(
            f"    block {block.label} offset=0x{block.start_offset:04X} "
            f"preds=[{preds}] succs=[{succs}]{synthetic}"
        )
        for statement in block.statements:
            lines.append(f"      {_describe_node(statement)}")
        lines.append(f"      terminator {block.terminator.describe()}")

    def _render_dominators(self, title, info, lines: List[str]) -> None:
        lines.append(f"  {title} (root={info.root}):")
        for block in sorted(info.dominators.keys()):
            idom = info.immediate.get(block)
            idom_text = idom if idom is not None else "-"
            doms = ", ".join(info.dominators[block])
            lines.append(f"    {block}: idom={idom_text} dom=[{doms}]")

