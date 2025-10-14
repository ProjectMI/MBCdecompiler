"""Data structures for the reconstructed abstract syntax tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ..ir.model import IRSlot, MemRef, SSAValueKind


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASTExpression:
    """Base class for all expression nodes."""

    def render(self) -> str:  # pragma: no cover - overridden in subclasses
        raise NotImplementedError

    def kind(self) -> SSAValueKind:
        """Return the inferred :class:`SSAValueKind` for the expression."""

        return SSAValueKind.UNKNOWN


@dataclass(frozen=True)
class ASTUnknown(ASTExpression):
    """Fallback expression used when reconstruction fails."""

    token: str

    def render(self) -> str:
        return f"?({self.token})"


@dataclass(frozen=True)
class ASTLiteral(ASTExpression):
    """Literal constant extracted from stack traffic."""

    value: int
    width: int = 16

    def render(self) -> str:
        digits = max(1, self.width // 4)
        return f"0x{self.value:0{digits}X}"

    def kind(self) -> SSAValueKind:
        if self.width <= 8:
            return SSAValueKind.BYTE
        return SSAValueKind.WORD


@dataclass(frozen=True)
class ASTIdentifier(ASTExpression):
    """Named SSA value."""

    name: str
    kind_hint: SSAValueKind = SSAValueKind.UNKNOWN

    def render(self) -> str:
        return self.name

    def kind(self) -> SSAValueKind:
        return self.kind_hint


@dataclass(frozen=True)
class ASTSlotRef(ASTExpression):
    """Reference to a VM slot."""

    slot: IRSlot

    def render(self) -> str:
        space = self.slot.space.name.lower()
        index = f"0x{self.slot.index:04X}"
        return f"{space}[{index}]"

    def kind(self) -> SSAValueKind:
        return SSAValueKind.POINTER


@dataclass(frozen=True)
class ASTMemRefExpr(ASTExpression):
    """Reference to a symbolic memory location."""

    ref: MemRef

    def render(self) -> str:
        return self.ref.describe()

    def kind(self) -> SSAValueKind:
        return SSAValueKind.POINTER


@dataclass(frozen=True)
class ASTIndirectLoadExpr(ASTExpression):
    """Load through an indirect pointer."""

    pointer: ASTExpression
    offset: ASTExpression
    ref: MemRef | None = None

    def render(self) -> str:
        base = self.pointer.render()
        displacement = self.offset.render()
        if self.ref is not None:
            return f"load {self.ref.describe()}[{base} + {displacement}]"
        return f"load {base} + {displacement}"

    def kind(self) -> SSAValueKind:
        pointer_kind = self.pointer.kind()
        if pointer_kind is SSAValueKind.BYTE:
            return SSAValueKind.BYTE
        if pointer_kind is SSAValueKind.BOOLEAN:
            return SSAValueKind.BOOLEAN
        return SSAValueKind.WORD


@dataclass(frozen=True)
class ASTCallExpr(ASTExpression):
    """Call expression with resolved argument expressions."""

    target: int
    args: Tuple[ASTExpression, ...]
    symbol: str | None = None
    tail: bool = False
    varargs: bool = False

    def render(self) -> str:
        rendered_args = ", ".join(arg.render() for arg in self.args)
        target_repr = f"0x{self.target:04X}"
        if self.symbol:
            target_repr = f"{self.symbol}({target_repr})"
        prefix = "tail " if self.tail else ""
        suffix = ", ..." if self.varargs else ""
        return f"{prefix}call {target_repr}({rendered_args}{suffix})"


@dataclass(frozen=True)
class ASTCallResult(ASTExpression):
    """View of a single value returned from a call expression."""

    call: ASTCallExpr
    index: int

    def render(self) -> str:
        return f"{self.call.render()}[{self.index}]"


# ---------------------------------------------------------------------------
# statements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASTStatement:
    """Base class for AST statements."""

    def render(self) -> str:  # pragma: no cover - overridden in subclasses
        raise NotImplementedError


@dataclass(frozen=True)
class ASTAssign(ASTStatement):
    """Assign an expression to a target identifier."""

    target: ASTIdentifier
    value: ASTExpression

    def render(self) -> str:
        return f"{self.target.render()} = {self.value.render()}"


@dataclass(frozen=True)
class ASTStore(ASTStatement):
    """Store ``value`` into the location described by ``target``."""

    target: ASTExpression
    value: ASTExpression

    def render(self) -> str:
        return f"store {self.value.render()} -> {self.target.render()}"


@dataclass(frozen=True)
class ASTCallStatement(ASTStatement):
    """Call that yields one or more return values."""

    call: ASTCallExpr
    returns: Tuple[ASTIdentifier, ...] = field(default_factory=tuple)

    def render(self) -> str:
        if not self.returns:
            return self.call.render()
        outputs = ", ".join(ret.render() for ret in self.returns)
        return f"{outputs} = {self.call.render()}"


@dataclass(frozen=True)
class ASTTailCall(ASTStatement):
    """Tail call used as a return."""

    call: ASTCallExpr
    returns: Tuple[ASTExpression, ...]

    def render(self) -> str:
        rendered = ", ".join(expr.render() for expr in self.returns)
        suffix = f" returns [{rendered}]" if rendered else ""
        return f"tail {self.call.render()}{suffix}"


@dataclass(frozen=True)
class ASTReturn(ASTStatement):
    """Return from the current procedure."""

    values: Tuple[ASTExpression, ...]
    varargs: bool = False
    mask: int | None = None

    def render(self) -> str:
        if self.varargs:
            rendered = ", ".join(expr.render() for expr in self.values)
            payload = f"varargs({rendered})" if rendered else "varargs"
        else:
            rendered = ", ".join(expr.render() for expr in self.values)
            payload = f"[{rendered}]"
        mask = "" if self.mask is None else f" mask=0x{self.mask:04X}"
        return f"return {payload}{mask}"


@dataclass(frozen=True)
class ASTBranch(ASTStatement):
    """Generic conditional branch."""

    condition: ASTExpression
    then_target: int
    else_target: int

    def render(self) -> str:
        condition = self.condition.render()
        return (
            f"if {condition} then 0x{self.then_target:04X} "
            f"else 0x{self.else_target:04X}"
        )


@dataclass(frozen=True)
class ASTTestSet(ASTStatement):
    """Branch that stores a predicate before testing it."""

    var: ASTExpression
    expr: ASTExpression
    then_target: int
    else_target: int

    def render(self) -> str:
        return (
            f"testset {self.var.render()} = {self.expr.render()} "
            f"then 0x{self.then_target:04X} else 0x{self.else_target:04X}"
        )


@dataclass(frozen=True)
class ASTFlagCheck(ASTStatement):
    """Branch that checks a VM flag."""

    flag: int
    then_target: int
    else_target: int

    def render(self) -> str:
        return (
            f"flag 0x{self.flag:04X} ? then 0x{self.then_target:04X} "
            f"else 0x{self.else_target:04X}"
        )


@dataclass(frozen=True)
class ASTFunctionPrologue(ASTStatement):
    """Reconstructed function prologue sequence."""

    var: ASTExpression
    expr: ASTExpression
    then_target: int
    else_target: int

    def render(self) -> str:
        return (
            f"prologue {self.var.render()} = {self.expr.render()} "
            f"then 0x{self.then_target:04X} else 0x{self.else_target:04X}"
        )


@dataclass(frozen=True)
class ASTComment(ASTStatement):
    """Fallback wrapper for nodes that currently lack dedicated support."""

    text: str

    def render(self) -> str:
        return f"; {self.text}"


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASTBlock:
    """Single basic block in the reconstructed AST."""

    label: str
    start_offset: int
    statements: Tuple[ASTStatement, ...]
    successors: Tuple[int, ...]


@dataclass(frozen=True)
class ASTProcedure:
    """Group of blocks that form a reconstructed procedure."""

    name: str
    entry_offset: int
    entry_reasons: Tuple[str, ...]
    blocks: Tuple[ASTBlock, ...]
    exit_offsets: Tuple[int, ...]


@dataclass(frozen=True)
class ASTSegment:
    """AST view of a container segment."""

    index: int
    start: int
    length: int
    procedures: Tuple[ASTProcedure, ...]


@dataclass
class ASTMetrics:
    """Success metrics for the reconstruction pipeline."""

    procedure_count: int = 0
    block_count: int = 0
    edge_count: int = 0
    value_count: int = 0
    resolved_values: int = 0
    call_sites: int = 0
    call_args: int = 0
    resolved_call_args: int = 0
    load_count: int = 0
    resolved_loads: int = 0
    store_count: int = 0
    resolved_stores: int = 0

    def observe_values(self, resolved: int, total: int = 1) -> None:
        self.value_count += total
        self.resolved_values += resolved

    def observe_call_args(self, resolved: int, total: int) -> None:
        self.call_args += total
        self.resolved_call_args += resolved

    def observe_load(self, resolved: bool) -> None:
        self.load_count += 1
        if resolved:
            self.resolved_loads += 1

    def observe_store(self, resolved: bool) -> None:
        self.store_count += 1
        if resolved:
            self.resolved_stores += 1

    def describe(self) -> str:
        def ratio(numerator: int, denominator: int) -> str:
            if denominator == 0:
                return "1.00"
            return f"{numerator / denominator:.2f}"

        return (
            "procedures={proc} blocks={blocks} edges={edges} "
            "values={val_res}/{val_total}({val_ratio}) "
            "calls={calls} call_args={arg_res}/{arg_total}({arg_ratio}) "
            "loads={load_res}/{load_total}({load_ratio}) "
            "stores={store_res}/{store_total}({store_ratio})"
        ).format(
            proc=self.procedure_count,
            blocks=self.block_count,
            edges=self.edge_count,
            val_res=self.resolved_values,
            val_total=self.value_count,
            val_ratio=ratio(self.resolved_values, self.value_count),
            calls=self.call_sites,
            arg_res=self.resolved_call_args,
            arg_total=self.call_args,
            arg_ratio=ratio(self.resolved_call_args, self.call_args),
            load_res=self.resolved_loads,
            load_total=self.load_count,
            load_ratio=ratio(self.resolved_loads, self.load_count),
            store_res=self.resolved_stores,
            store_total=self.store_count,
            store_ratio=ratio(self.resolved_stores, self.store_count),
        )


@dataclass(frozen=True)
class ASTProgram:
    """Root node for the reconstructed AST programme."""

    segments: Tuple[ASTSegment, ...]
    metrics: ASTMetrics


__all__ = [
    "ASTExpression",
    "ASTUnknown",
    "ASTLiteral",
    "ASTIdentifier",
    "ASTSlotRef",
    "ASTMemRefExpr",
    "ASTIndirectLoadExpr",
    "ASTCallExpr",
    "ASTCallResult",
    "ASTStatement",
    "ASTAssign",
    "ASTStore",
    "ASTCallStatement",
    "ASTTailCall",
    "ASTReturn",
    "ASTBranch",
    "ASTTestSet",
    "ASTFlagCheck",
    "ASTFunctionPrologue",
    "ASTComment",
    "ASTBlock",
    "ASTProcedure",
    "ASTSegment",
    "ASTMetrics",
    "ASTProgram",
]
