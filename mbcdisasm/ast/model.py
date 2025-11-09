"""Data structures that describe the reconstructed AST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..ir.model import IRNode


@dataclass(frozen=True)
class ASTEdge:
    """Single outgoing edge from a block terminator."""

    kind: str
    target: str


@dataclass(frozen=True)
class ASTTerminatorView:
    """Serialisable summary of a block terminator."""

    kind: str
    text: str
    edges: Tuple[ASTEdge, ...]


@dataclass(frozen=True)
class ASTBlock:
    """Basic block recovered during the AST phase."""

    label: str
    offset: int | None
    statements: Tuple[IRNode, ...]
    terminator: ASTTerminatorView
    successors: Tuple[str, ...]
    predecessors: Tuple[str, ...]
    annotations: Tuple[str, ...]
    synthetic: bool = False


@dataclass(frozen=True)
class ASTDominatorInfo:
    """Dominator or post-dominator membership for a block."""

    label: str
    members: Tuple[str, ...]


@dataclass(frozen=True)
class ASTLoop:
    """Natural loop detected in the control-flow graph."""

    header: str
    nodes: Tuple[str, ...]
    latches: Tuple[str, ...]


@dataclass(frozen=True)
class ASTFunction:
    """High level representation of a function in the AST programme."""

    name: str
    segment_index: int
    entry_block: str
    entry_offset: int
    blocks: Tuple[ASTBlock, ...]
    dominators: Tuple[ASTDominatorInfo, ...]
    post_dominators: Tuple[ASTDominatorInfo, ...]
    loops: Tuple[ASTLoop, ...]


@dataclass(frozen=True)
class ASTProgram:
    """Container for the reconstructed AST."""

    functions: Tuple[ASTFunction, ...]


__all__ = [
    "ASTProgram",
    "ASTFunction",
    "ASTBlock",
    "ASTLoop",
    "ASTDominatorInfo",
    "ASTTerminatorView",
    "ASTEdge",
]
