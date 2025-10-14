"""Data structures used by the AST reconstruction stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, Optional, Tuple

from ..ir.model import IRBlock, IRSegment, IRSlot, IRStackEffect, IRFunctionPrologue
from ..ir.model import (
    IRCall,
    IRTailCall,
    IRCallReturn,
    IRReturn,
    IRTerminator,
)
from ..ir.model import CallPredicate, SSAValueKind


class CFGEdgeKind(Enum):
    """Classification of a control-flow edge inside a segment."""

    FALLTHROUGH = auto()
    BRANCH_TRUE = auto()
    BRANCH_FALSE = auto()
    CALL_PREDICATE_TRUE = auto()
    CALL_PREDICATE_FALSE = auto()
    EXCEPTION = auto()


@dataclass(frozen=True)
class CFGEdge:
    """Directed edge connecting two :class:`CFGNode` objects."""

    source: int
    target: int
    kind: CFGEdgeKind
    label: Optional[str] = None


@dataclass
class CallSite:
    """Description of a call-like node encountered inside a block."""

    block_offset: int
    target: int
    args: Tuple[str, ...]
    tail: bool = False
    returns: Tuple[str, ...] = field(default_factory=tuple)
    varargs: bool = False
    arity: Optional[int] = None
    convention: Optional[IRStackEffect] = None
    cleanup_mask: Optional[int] = None
    symbol: Optional[str] = None
    predicate: Optional[CallPredicate] = None


@dataclass
class CFGNode:
    """Wrapper that exposes graph relationships for an :class:`IRBlock`."""

    block: IRBlock
    successors: list[CFGEdge] = field(default_factory=list)
    predecessors: list[CFGEdge] = field(default_factory=list)
    call_sites: list[CallSite] = field(default_factory=list)
    exits: bool = False

    def add_successor(self, edge: CFGEdge) -> None:
        self.successors.append(edge)

    def add_predecessor(self, edge: CFGEdge) -> None:
        self.predecessors.append(edge)

    def register_call(self, call: CallSite) -> None:
        self.call_sites.append(call)


@dataclass
class SegmentCFG:
    """Control-flow graph recovered from a normalised segment."""

    segment: IRSegment
    nodes: dict[int, CFGNode]
    edges: Tuple[CFGEdge, ...]

    def entry_offsets(self) -> Tuple[int, ...]:
        return tuple(sorted(self.nodes))


@dataclass
class Procedure:
    """Group of blocks that form a logical procedure."""

    entry_offset: int
    blocks: Tuple[int, ...]
    exits: Tuple[int, ...]
    call_sites: Tuple[CallSite, ...]
    tail_calls: Tuple[CallSite, ...]
    callees: Tuple[int, ...]
    prologue: Optional[IRFunctionPrologue] = None

    @property
    def name(self) -> str:
        return f"proc_{self.entry_offset:04X}"


# ---------------------------------------------------------------------------
# AST expressions
# ---------------------------------------------------------------------------


class ASTExpression:
    """Base class for reconstructed expressions."""

    def depth(self) -> int:
        return 1


@dataclass(frozen=True)
class UnknownExpr(ASTExpression):
    """Placeholder for expressions that could not be resolved."""

    reason: str = "unknown"


@dataclass(frozen=True)
class RawExpr(ASTExpression):
    """Carry opaque textual fragments produced by earlier passes."""

    text: str


@dataclass(frozen=True)
class LiteralExpr(ASTExpression):
    """Literal integer value."""

    value: int


@dataclass(frozen=True)
class VariableExpr(ASTExpression):
    """Reference to a SSA value rendered as a symbolic variable."""

    name: str
    kind: SSAValueKind = SSAValueKind.UNKNOWN


@dataclass(frozen=True)
class LoadExpr(ASTExpression):
    """Read of a VM slot."""

    slot: IRSlot


@dataclass(frozen=True)
class CallExpr(ASTExpression):
    """Call expression used for both direct invocations and helper wrappers."""

    target: int
    args: Tuple[ASTExpression, ...]
    symbol: Optional[str] = None
    arity: Optional[int] = None
    convention: Optional[IRStackEffect] = None
    cleanup_mask: Optional[int] = None
    predicate: Optional[CallPredicate] = None

    def depth(self) -> int:
        base = 1
        if not self.args:
            return base
        return base + max(arg.depth() for arg in self.args)


# ---------------------------------------------------------------------------
# AST statements
# ---------------------------------------------------------------------------


class ASTStatement:
    """Base class for AST statements."""

    def expressions(self) -> Tuple[ASTExpression, ...]:  # pragma: no cover - overridden
        return tuple()

    def variables(self) -> Tuple[VariableExpr, ...]:  # pragma: no cover - overridden
        return tuple()


@dataclass(frozen=True)
class AssignStatement(ASTStatement):
    """Assignment between SSA aliases."""

    target: VariableExpr
    value: ASTExpression

    def expressions(self) -> Tuple[ASTExpression, ...]:
        return (self.value,)

    def variables(self) -> Tuple[VariableExpr, ...]:
        return (self.target,)


@dataclass(frozen=True)
class StoreStatement(ASTStatement):
    """Store into a VM slot."""

    slot: IRSlot
    value: ASTExpression

    def expressions(self) -> Tuple[ASTExpression, ...]:
        return (self.value,)


@dataclass(frozen=True)
class CallStatement(ASTStatement):
    """Invocation with optional assigned return values."""

    call: CallExpr
    results: Tuple[VariableExpr, ...] = tuple()
    tail: bool = False
    varargs: bool = False

    def expressions(self) -> Tuple[ASTExpression, ...]:
        return (self.call,)

    def variables(self) -> Tuple[VariableExpr, ...]:
        return self.results


@dataclass(frozen=True)
class ReturnStatement(ASTStatement):
    """Return from the current procedure."""

    values: Tuple[ASTExpression, ...]
    varargs: bool = False

    def expressions(self) -> Tuple[ASTExpression, ...]:
        return self.values


@dataclass(frozen=True)
class BranchStatement(ASTStatement):
    """Conditional branch extracted from structured nodes."""

    condition: ASTExpression
    then_target: int
    else_target: int

    def expressions(self) -> Tuple[ASTExpression, ...]:
        return (self.condition,)


@dataclass(frozen=True)
class PrologueStatement(ASTStatement):
    """Specialised representation of :class:`IRFunctionPrologue`."""

    target: VariableExpr
    value: ASTExpression
    then_target: int
    else_target: int

    def expressions(self) -> Tuple[ASTExpression, ...]:
        return (self.value,)

    def variables(self) -> Tuple[VariableExpr, ...]:
        return (self.target,)


@dataclass(frozen=True)
class UnknownStatement(ASTStatement):
    """Fallback wrapper for nodes that lack a dedicated representation."""

    text: str


@dataclass(frozen=True)
class ASTBlock:
    """Block decorated with reconstructed statements."""

    node: CFGNode
    statements: Tuple[ASTStatement, ...]


@dataclass(frozen=True)
class ASTProcedure:
    """Procedure enriched with AST blocks."""

    procedure: Procedure
    blocks: Tuple[ASTBlock, ...]


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Summary numbers describing the quality of the reconstruction."""

    procedures: int = 0
    cfg_nodes: int = 0
    cfg_edges: int = 0
    entry_points: int = 0
    assigned_blocks: int = 0
    typed_variable_ratio: float = 0.0
    constant_expression_ratio: float = 0.0
    call_resolution_ratio: float = 0.0
    max_expression_depth: int = 0
    average_statements_per_block: float = 0.0

    def describe(self) -> str:
        return (
            "procedures={procedures} nodes={cfg_nodes} edges={cfg_edges} "
            "entries={entry_points} coverage={assigned_blocks} "
            "typed={typed:.2f} const={const:.2f} call_args={call:.2f} "
            "depth={depth} avg_stmt={avg:.2f}".format(
                procedures=self.procedures,
                cfg_nodes=self.cfg_nodes,
                cfg_edges=self.cfg_edges,
                entry_points=self.entry_points,
                assigned_blocks=self.assigned_blocks,
                typed=self.typed_variable_ratio,
                const=self.constant_expression_ratio,
                call=self.call_resolution_ratio,
                depth=self.max_expression_depth,
                avg=self.average_statements_per_block,
            )
        )


@dataclass(frozen=True)
class ASTSegment:
    """AST representation associated with a single segment."""

    cfg: SegmentCFG
    procedures: Tuple[ASTProcedure, ...]
    metrics: ReconstructionMetrics


@dataclass(frozen=True)
class ASTProgram:
    """Reconstructed AST for an entire container."""

    segments: Tuple[ASTSegment, ...]
    metrics: ReconstructionMetrics

