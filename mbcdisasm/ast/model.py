"""Dataclasses describing the reconstructed AST representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from ..ir.model import IRSlot, MemRef, MemSpace, SSAValueKind


class ASTExpression:
    """Base class for AST expressions."""

    __slots__ = ()


@dataclass(frozen=True)
class ASTLiteral(ASTExpression):
    """Literal numeric value used by the AST."""

    value: int
    kind: Optional[SSAValueKind] = None


@dataclass(frozen=True)
class ASTStringLiteral(ASTExpression):
    """Inline string literal produced by the IR normaliser."""

    value: str


@dataclass(frozen=True)
class ASTIdentifier(ASTExpression):
    """Reference to a named SSA value."""

    name: str
    kind: Optional[SSAValueKind] = None
    definition: Optional[ASTExpression] = None


@dataclass(frozen=True)
class ASTUnknownExpr(ASTExpression):
    """Fallback wrapper around expressions the builder could not classify."""

    text: str


@dataclass(frozen=True)
class ASTLoadExpr(ASTExpression):
    """Direct slot load expression."""

    slot: IRSlot
    kind: Optional[SSAValueKind] = None
    ref: Optional[MemRef] = None


@dataclass(frozen=True)
class ASTIndirectLoadExpr(ASTExpression):
    """Indirect memory load expression."""

    base: Optional[ASTExpression]
    offset: Optional[ASTExpression]
    pointer: Optional[ASTExpression] = None
    ref: Optional[MemRef] = None
    kind: Optional[SSAValueKind] = None


@dataclass(frozen=True)
class ASTCallResultExpr(ASTExpression):
    """Expression that represents the result of a call."""

    target_offset: int
    index: int
    symbol: Optional[str] = None
    call_type: str = "call"
    kind: Optional[SSAValueKind] = None


class ASTStatement:
    """Base class for AST statements."""

    __slots__ = ()


@dataclass(frozen=True)
class ASTAssignment(ASTStatement):
    """Assignment statement produced from a load-like IR node."""

    target: ASTIdentifier
    expr: ASTExpression


@dataclass(frozen=True)
class ASTStoreStatement(ASTStatement):
    """Store statement that writes a value to a direct slot."""

    slot: IRSlot
    value: ASTExpression


@dataclass(frozen=True)
class ASTIndirectStoreStatement(ASTStatement):
    """Store statement that targets an indirect reference."""

    base: Optional[ASTExpression]
    value: ASTExpression
    offset: Optional[ASTExpression]
    pointer: Optional[ASTExpression] = None
    ref: Optional[MemRef] = None


@dataclass(frozen=True)
class ASTCallStatement(ASTStatement):
    """Function call statement."""

    target_offset: int
    args: Tuple[ASTExpression, ...]
    returns: Tuple[str, ...] = field(default_factory=tuple)
    call_type: str = "call"
    tail: bool = False
    varargs: bool = False
    symbol: Optional[str] = None
    target_label: Optional[str] = None
    predicate: Optional[str] = None
    convention: Optional[str] = None
    cleanup: Tuple[str, ...] = field(default_factory=tuple)
    cleanup_mask: Optional[int] = None
    arity: Optional[int] = None


@dataclass(frozen=True)
class ASTReturnStatement(ASTStatement):
    """Return statement."""

    values: Tuple[ASTExpression, ...]
    varargs: bool = False
    mask: Optional[int] = None
    cleanup: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ASTBranchStatement(ASTStatement):
    """Conditional branch statement."""

    condition: ASTExpression
    kind: str
    then_label: Optional[str]
    then_offset: Optional[int]
    else_label: Optional[str]
    else_offset: Optional[int]


@dataclass(frozen=True)
class ASTSwitchCase:
    """Single switch dispatch case."""

    key: int
    target_label: Optional[str]
    target_offset: int
    symbol: Optional[str] = None


@dataclass(frozen=True)
class ASTSwitchStatement(ASTStatement):
    """Dispatch statement built from an IR table switch."""

    cases: Tuple[ASTSwitchCase, ...]
    helper_offset: Optional[int] = None
    helper_symbol: Optional[str] = None
    default_label: Optional[str] = None
    default_offset: Optional[int] = None


@dataclass(frozen=True)
class ASTUnstructuredStatement(ASTStatement):
    """Fallback statement that embeds the IR description verbatim."""

    text: str


@dataclass(frozen=True)
class ASTEdge:
    """Control-flow edge emitted by the CFG builder."""

    kind: str
    target_label: Optional[str]
    target_offset: Optional[int]
    detail: Optional[str] = None


@dataclass(frozen=True)
class ASTBlock:
    """Single basic block within the AST programme."""

    label: str
    start_offset: int
    statements: Tuple[ASTStatement, ...]
    successors: Tuple[ASTEdge, ...]
    annotations: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ASTProcedure:
    """Procedure reconstructed from the CFG."""

    entry_label: str
    start_offset: int
    block_labels: Tuple[str, ...]
    return_count: int
    tail_call_count: int
    call_targets: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ASTSegment:
    """AST representation for a container segment."""

    index: int
    start: int
    length: int
    blocks: Tuple[ASTBlock, ...]
    procedures: Tuple[ASTProcedure, ...]
    metrics: "ASTMetrics"


@dataclass(frozen=True)
class ASTProgram:
    """Top-level AST programme."""

    segments: Tuple[ASTSegment, ...]
    metrics: "ASTMetrics"


@dataclass
class ASTMetrics:
    """Aggregate metrics collected during AST reconstruction."""

    blocks: int = 0
    procedures: int = 0
    statements: int = 0
    structured_statements: int = 0
    fallback_statements: int = 0
    loads: int = 0
    indirect_loads: int = 0
    stores: int = 0
    indirect_stores: int = 0
    calls: int = 0
    tail_calls: int = 0
    returns: int = 0
    branches: int = 0
    switches: int = 0
    constants: int = 0
    typed_values: int = 0

    def observe(self, other: "ASTMetrics") -> None:
        """Accumulate values from ``other`` into this instance."""

        self.blocks += other.blocks
        self.procedures += other.procedures
        self.statements += other.statements
        self.structured_statements += other.structured_statements
        self.fallback_statements += other.fallback_statements
        self.loads += other.loads
        self.indirect_loads += other.indirect_loads
        self.stores += other.stores
        self.indirect_stores += other.indirect_stores
        self.calls += other.calls
        self.tail_calls += other.tail_calls
        self.returns += other.returns
        self.branches += other.branches
        self.switches += other.switches
        self.constants += other.constants
        self.typed_values += other.typed_values

    def describe(self) -> str:
        """Return a stable textual summary of the metrics."""

        parts = [
            f"blocks={self.blocks}",
            f"procedures={self.procedures}",
            f"statements={self.statements}",
            f"structured={self.structured_statements}",
            f"fallback={self.fallback_statements}",
            f"loads={self.loads}",
            f"indirect_loads={self.indirect_loads}",
            f"stores={self.stores}",
            f"indirect_stores={self.indirect_stores}",
            f"calls={self.calls}",
            f"tail_calls={self.tail_calls}",
            f"returns={self.returns}",
            f"branches={self.branches}",
            f"switches={self.switches}",
            f"constants={self.constants}",
            f"typed={self.typed_values}",
        ]
        return " ".join(parts)


__all__ = [
    "ASTExpression",
    "ASTLiteral",
    "ASTStringLiteral",
    "ASTIdentifier",
    "ASTUnknownExpr",
    "ASTLoadExpr",
    "ASTIndirectLoadExpr",
    "ASTCallResultExpr",
    "ASTStatement",
    "ASTAssignment",
    "ASTStoreStatement",
    "ASTIndirectStoreStatement",
    "ASTCallStatement",
    "ASTReturnStatement",
    "ASTBranchStatement",
    "ASTSwitchCase",
    "ASTSwitchStatement",
    "ASTUnstructuredStatement",
    "ASTEdge",
    "ASTBlock",
    "ASTProcedure",
    "ASTSegment",
    "ASTProgram",
    "ASTMetrics",
]
