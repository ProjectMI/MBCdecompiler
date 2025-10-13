"""Conversion helpers that lift the IR into a structured AST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ..ir import IRBlock, IRFunctionPrologue, IRNode, IRProgram, IRSegment
from .model import ASTBlock, ASTFunction, ASTProgram, ASTSegment, ASTStatement


@dataclass
class _FunctionBuilder:
    """Internal accumulator used while grouping blocks into functions."""

    name: Optional[str] = None
    entry_offset: Optional[int] = None
    blocks: List[ASTBlock] | None = None

    def ensure(self) -> None:
        if self.blocks is None:
            self.blocks = []

    def add_block(self, block: ASTBlock) -> None:
        self.ensure()
        self.blocks.append(block)
        if self.entry_offset is None:
            self.entry_offset = block.start_offset
        if self.name is None:
            self.name = block.label

    def finalise(self) -> Optional[ASTFunction]:
        if not self.blocks:
            return None
        name = self.name or f"fn_{self.blocks[0].label}"
        entry = self.entry_offset or self.blocks[0].start_offset
        return ASTFunction(name=name, entry_offset=entry, blocks=tuple(self.blocks))


class ASTBuilder:
    """Lift :class:`~mbcdisasm.ir.model.IRProgram` into an AST representation."""

    def build(self, program: IRProgram) -> ASTProgram:
        segments = tuple(self._build_segment(segment) for segment in program.segments)
        return ASTProgram(segments=segments, metrics=program.metrics, string_pool=program.string_pool)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_segment(self, segment: IRSegment) -> ASTSegment:
        functions: List[ASTFunction] = []
        current = _FunctionBuilder()

        for block in segment.blocks:
            has_prologue = self._contains_prologue(block.nodes)
            if has_prologue and current.blocks:
                function = current.finalise()
                if function is not None:
                    functions.append(function)
                current = _FunctionBuilder(name=self._derive_function_name(block), entry_offset=block.start_offset)
            elif has_prologue:
                current = _FunctionBuilder(name=self._derive_function_name(block), entry_offset=block.start_offset)

            ast_block = self._build_block(block)
            current.add_block(ast_block)

        final_function = current.finalise()
        if final_function is not None:
            functions.append(final_function)

        return ASTSegment(
            index=segment.index,
            start=segment.start,
            length=segment.length,
            functions=tuple(functions),
            metrics=segment.metrics,
        )

    def _build_block(self, block: IRBlock) -> ASTBlock:
        statements = tuple(self._lift_statements(block.nodes))
        return ASTBlock(
            label=block.label,
            start_offset=block.start_offset,
            statements=statements,
            annotations=block.annotations,
        )

    def _lift_statements(self, nodes: Sequence[IRNode]) -> Iterable[ASTStatement]:
        for node in nodes:
            describe = getattr(node, "describe", None)
            if callable(describe):
                text = describe()
            else:
                text = repr(node)
            yield ASTStatement(text=text)

    @staticmethod
    def _contains_prologue(nodes: Sequence[IRNode]) -> bool:
        return any(isinstance(node, IRFunctionPrologue) for node in nodes)

    @staticmethod
    def _derive_function_name(block: IRBlock) -> str:
        # Prefer explicit labels as the function name to keep the output stable.
        return block.label


__all__ = ["ASTBuilder"]
