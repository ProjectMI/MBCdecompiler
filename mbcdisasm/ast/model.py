"""Dataclasses describing the lifted abstract syntax tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from mbcdisasm.ir.model import IRBlock


@dataclass(frozen=True)
class ASTStatement:
    """Single statement appearing in an AST block."""

    text: str


@dataclass(frozen=True)
class ASTBlock:
    """Structured representation of a basic block within a function."""

    label: str
    start_offset: int
    statements: Tuple[ASTStatement, ...]
    successors: Tuple[str, ...]

    @classmethod
    def from_ir_block(
        cls,
        block: IRBlock,
        *,
        statements: Tuple[ASTStatement, ...],
        successors: Tuple[str, ...],
    ) -> "ASTBlock":
        return cls(
            label=block.label,
            start_offset=block.start_offset,
            statements=statements,
            successors=successors,
        )


@dataclass(frozen=True)
class ASTFunction:
    """Logical function constructed from the IR control-flow graph."""

    name: str
    segment_index: int
    entry_label: str
    start_offset: int
    blocks: Tuple[ASTBlock, ...]


@dataclass(frozen=True)
class ASTProgram:
    """Root container for the lifted AST."""

    functions: Tuple[ASTFunction, ...]

