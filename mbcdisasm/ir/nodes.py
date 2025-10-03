"""Intermediate representation nodes for the normalisation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence, Tuple


class MemSpace(Enum):
    """Memory space classification for indirect accesses."""

    FRAME = auto()
    GLOBAL = auto()
    CONST = auto()


@dataclass(frozen=True)
class IRNode:
    """Base class for all intermediate representation nodes."""

    pass


@dataclass(frozen=True)
class IRLiteral(IRNode):
    """A literal value pushed on to the stack."""

    value: int
    size: int = 1


@dataclass(frozen=True)
class IRBuildArray(IRNode):
    """Build an array from literal elements."""

    elements: Tuple[IRNode, ...]

    @classmethod
    def from_sequence(cls, elements: Sequence[IRNode]) -> "IRBuildArray":
        return cls(tuple(elements))


@dataclass(frozen=True)
class IRBuildMap(IRNode):
    """Build a map/dictionary from literal key/value pairs."""

    pairs: Tuple[Tuple[IRNode, IRNode], ...]

    @classmethod
    def from_pairs(
        cls, pairs: Sequence[Tuple[IRNode, IRNode]]
    ) -> "IRBuildMap":
        return cls(tuple(tuple(pair) for pair in pairs))


@dataclass(frozen=True)
class IRBuildTuple(IRNode):
    """Build a tuple with positional semantics."""

    elements: Tuple[IRNode, ...]

    @classmethod
    def from_sequence(cls, elements: Sequence[IRNode]) -> "IRBuildTuple":
        return cls(tuple(elements))


@dataclass(frozen=True)
class IRCall(IRNode):
    """Function or helper invocation."""

    target: int
    args: Tuple[IRNode, ...]
    tail: bool = False

    @classmethod
    def from_args(
        cls, target: int, args: Sequence[IRNode], *, tail: bool = False
    ) -> "IRCall":
        return cls(target=target, args=tuple(args), tail=tail)


@dataclass(frozen=True)
class IRReturn(IRNode):
    """Return from the current routine with a fixed arity."""

    arity: int


@dataclass(frozen=True)
class IRSlot:
    """A typed slot within a memory space."""

    space: MemSpace
    index: int


@dataclass(frozen=True)
class IRLoad(IRNode):
    """Load a value from an indirect slot."""

    slot: IRSlot


@dataclass(frozen=True)
class IRStore(IRNode):
    """Store a value into an indirect slot."""

    slot: IRSlot
    value: IRNode


@dataclass(frozen=True)
class IRIf(IRNode):
    """Conditional branch based on an expression."""

    predicate: IRNode
    then_target: int
    else_target: int


@dataclass(frozen=True)
class IRTestSetBranch(IRNode):
    """A branch that assigns the predicate to a dedicated variable."""

    var: str
    expr: IRNode
    then_target: int
    else_target: int


@dataclass(frozen=True)
class IRBlock:
    """A normalised basic block."""

    nodes: Tuple[IRNode, ...] = field(default_factory=tuple)

    @classmethod
    def from_nodes(cls, nodes: Sequence[IRNode]) -> "IRBlock":
        return cls(nodes=tuple(nodes))
