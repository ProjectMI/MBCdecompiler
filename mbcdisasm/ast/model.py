"""Dataclasses describing the high-level abstract syntax tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ..ir import IRStringConstant, NormalizerMetrics


@dataclass(frozen=True)
class ASTStatement:
    """Single lifted statement in a basic block."""

    text: str


@dataclass(frozen=True)
class ASTBlock:
    """Structured representation of a basic block."""

    label: str
    start_offset: int
    statements: Tuple[ASTStatement, ...]
    annotations: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ASTFunction:
    """Group of blocks forming a single routine."""

    name: str
    entry_offset: int
    blocks: Tuple[ASTBlock, ...]


@dataclass(frozen=True)
class ASTSegment:
    """Collection of functions reconstructed for a segment."""

    index: int
    start: int
    length: int
    functions: Tuple[ASTFunction, ...]
    metrics: NormalizerMetrics


@dataclass(frozen=True)
class ASTProgram:
    """Top-level AST representation covering all segments."""

    segments: Tuple[ASTSegment, ...]
    metrics: NormalizerMetrics
    string_pool: Tuple[IRStringConstant, ...] = tuple()


__all__ = [
    "ASTStatement",
    "ASTBlock",
    "ASTFunction",
    "ASTSegment",
    "ASTProgram",
]
