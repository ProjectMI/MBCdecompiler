"""Data structures for the reconstructed abstract syntax tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

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
class ASTBankedRefExpr(ASTExpression):
    """Reference to a banked memory location."""

    ref: MemRef
    register: int
    register_value: int | None = None
    pointer: ASTExpression | None = None
    offset: ASTExpression | None = None

    def render(self) -> str:
        parts = [self.ref.describe()]
        if self.pointer is not None:
            parts.append(f"ptr={self.pointer.render()}")
        if self.offset is not None:
            parts.append(f"offset={self.offset.render()}")
        if self.register_value is not None:
            parts.append(f"page=0x{self.register_value:04X}")
        else:
            parts.append(f"page_reg=0x{self.register:04X}")
        return "banked_ref " + " ".join(parts)

    def kind(self) -> SSAValueKind:
        return SSAValueKind.POINTER


@dataclass(frozen=True)
class ASTBankedLoadExpr(ASTExpression):
    """Load a value from banked memory."""

    ref: MemRef
    register: int
    register_value: int | None = None
    pointer: ASTExpression | None = None
    offset: ASTExpression | None = None

    def render(self) -> str:
        parts = [self.ref.describe()]
        if self.pointer is not None:
            parts.append(f"ptr={self.pointer.render()}")
        if self.offset is not None:
            parts.append(f"offset={self.offset.render()}")
        if self.register_value is not None:
            parts.append(f"page=0x{self.register_value:04X}")
        else:
            parts.append(f"page_reg=0x{self.register:04X}")
        return "banked_load " + " ".join(parts)

    def kind(self) -> SSAValueKind:
        return SSAValueKind.WORD


@dataclass(frozen=True)
class ASTStackEffect(ASTExpression):
    """Human readable view of a stack effect."""

    mnemonic: str
    operand: int = 0
    pops: int = 0
    operand_role: str | None = None
    operand_alias: str | None = None

    def render(self) -> str:
        details: List[str] = []
        if self.pops:
            details.append(f"pop={self.pops}")
        include_operand = bool(self.operand_role or self.operand_alias)
        if not include_operand:
            include_operand = bool(self.operand)
        if include_operand:
            operand = f"0x{self.operand:04X}"
            if self.operand_alias:
                alias = self.operand_alias
                operand = alias if alias == operand else f"{alias}({operand})"
            if self.operand_role:
                details.append(f"{self.operand_role}={operand}")
            else:
                details.append(f"operand={operand}")
        if not details:
            return self.mnemonic
        rendered = ", ".join(details)
        return f"{self.mnemonic}({rendered})"


@dataclass(frozen=True)
class ASTCallFrame:
    """Representation of the ABI scaffolding around a call."""

    parameters: Tuple[str, ...] = field(default_factory=tuple)
    convention: ASTStackEffect | None = None
    cleanup: Tuple[ASTStackEffect, ...] = field(default_factory=tuple)
    return_mask: int | None = None

    def render(self) -> str:
        parts: List[str] = []
        if self.parameters:
            params = ", ".join(self.parameters)
            parts.append(f"params=[{params}]")
        if self.convention is not None:
            parts.append(f"shuffle={self.convention.render()}")
        if self.cleanup:
            rendered = ", ".join(effect.render() for effect in self.cleanup)
            parts.append(f"cleanup=[{rendered}]")
        if self.return_mask is not None:
            parts.append(f"mask=0x{self.return_mask:04X}")
        if not parts:
            return "frame()"
        return "frame(" + " ".join(parts) + ")"


@dataclass(frozen=True)
class ASTCallExpr(ASTExpression):
    """Call expression with resolved argument expressions."""

    target: int
    args: Tuple[ASTExpression, ...]
    symbol: str | None = None
    tail: bool = False
    varargs: bool = False
    frame: ASTCallFrame | None = None

    def render(self) -> str:
        rendered_args = ", ".join(arg.render() for arg in self.args)
        target_repr = f"0x{self.target:04X}"
        if self.symbol:
            target_repr = f"{self.symbol}({target_repr})"
        prefix = "tail " if self.tail else ""
        suffix = ", ..." if self.varargs else ""
        frame_note = f" {self.frame.render()}" if self.frame is not None else ""
        return f"{prefix}call {target_repr}({rendered_args}{suffix}){frame_note}"


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


@dataclass
class ASTStatement:
    """Base class for AST statements."""

    def render(self) -> str:  # pragma: no cover - overridden in subclasses
        raise NotImplementedError


@dataclass
class ASTAssign(ASTStatement):
    """Assign an expression to a target identifier."""

    target: ASTIdentifier
    value: ASTExpression

    def render(self) -> str:
        return f"{self.target.render()} = {self.value.render()}"


@dataclass
class ASTStore(ASTStatement):
    """Store ``value`` into the location described by ``target``."""

    target: ASTExpression
    value: ASTExpression

    def render(self) -> str:
        return f"store {self.value.render()} -> {self.target.render()}"


@dataclass
class ASTCallStatement(ASTStatement):
    """Call that yields one or more return values."""

    call: ASTCallExpr
    returns: Tuple[ASTIdentifier, ...] = field(default_factory=tuple)

    def render(self) -> str:
        if not self.returns:
            return self.call.render()
        outputs = ", ".join(ret.render() for ret in self.returns)
        return f"{outputs} = {self.call.render()}"


@dataclass
class ASTIORead(ASTStatement):
    """I/O read effect emitted by helper façades."""

    port: str

    def render(self) -> str:
        return f"io.read({self.port})"


@dataclass
class ASTIOWrite(ASTStatement):
    """I/O write effect emitted by helper façades."""

    port: str
    mask: int | None = None

    def render(self) -> str:
        mask = "" if self.mask is None else f", mask=0x{self.mask:04X}"
        return f"io.write({self.port}{mask})"


@dataclass
class ASTTailCall(ASTStatement):
    """Tail call used as a return."""

    call: ASTCallExpr
    returns: Tuple[ASTExpression, ...]

    def render(self) -> str:
        rendered = ", ".join(expr.render() for expr in self.returns)
        suffix = f" returns [{rendered}]" if rendered else ""
        call_repr = self.call.render()
        if self.call.tail and call_repr.startswith("tail "):
            call_repr = call_repr[len("tail ") :]
        return f"tail {call_repr}{suffix}"


@dataclass
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
        mask_note = ""
        if self.mask is not None:
            mask_note = f" mask=0x{self.mask:04X}"
        return f"return {payload}{mask_note}"


@dataclass
class ASTBranch(ASTStatement):
    """Generic conditional branch with CFG links."""

    condition: ASTExpression
    then_branch: "ASTBlock | None" = None
    else_branch: "ASTBlock | None" = None
    then_hint: str | None = None
    else_hint: str | None = None

    def render(self) -> str:
        condition = self.condition.render()
        then_label = self.then_branch.label if self.then_branch else self.then_hint or "?"
        else_label = self.else_branch.label if self.else_branch else self.else_hint or "?"
        return f"if {condition} then {then_label} else {else_label}"


@dataclass
class ASTTestSet(ASTStatement):
    """Branch that stores a predicate before testing it."""

    var: ASTExpression
    expr: ASTExpression
    then_branch: "ASTBlock | None" = None
    else_branch: "ASTBlock | None" = None
    then_hint: str | None = None
    else_hint: str | None = None

    def render(self) -> str:
        then_label = self.then_branch.label if self.then_branch else self.then_hint or "?"
        else_label = self.else_branch.label if self.else_branch else self.else_hint or "?"
        return (
            f"testset {self.var.render()} = {self.expr.render()} "
            f"then {then_label} else {else_label}"
        )


@dataclass
class ASTFlagCheck(ASTStatement):
    """Branch that checks a VM flag."""

    flag: int
    then_branch: "ASTBlock | None" = None
    else_branch: "ASTBlock | None" = None
    then_hint: str | None = None
    else_hint: str | None = None

    def render(self) -> str:
        then_label = self.then_branch.label if self.then_branch else self.then_hint or "?"
        else_label = self.else_branch.label if self.else_branch else self.else_hint or "?"
        return f"flag 0x{self.flag:04X} ? then {then_label} else {else_label}"


@dataclass
class ASTFunctionPrologue(ASTStatement):
    """Reconstructed function prologue sequence."""

    var: ASTExpression
    expr: ASTExpression
    then_branch: "ASTBlock | None" = None
    else_branch: "ASTBlock | None" = None
    then_hint: str | None = None
    else_hint: str | None = None

    def render(self) -> str:
        then_label = self.then_branch.label if self.then_branch else self.then_hint or "?"
        else_label = self.else_branch.label if self.else_branch else self.else_hint or "?"
        return (
            f"prologue {self.var.render()} = {self.expr.render()} "
            f"then {then_label} else {else_label}"
        )


@dataclass
class ASTComment(ASTStatement):
    """Fallback wrapper for nodes that currently lack dedicated support."""

    text: str

    def render(self) -> str:
        return f"; {self.text}"


@dataclass
class ASTSwitchCase:
    """Single case handled by a helper dispatch."""

    key: int
    target: int
    symbol: str | None = None

    def render(self) -> str:
        target_repr = f"0x{self.target:04X}"
        if self.symbol:
            target_repr = f"{self.symbol}({target_repr})"
        return f"0x{self.key:04X}->{target_repr}"


@dataclass
class ASTDispatchTable(ASTStatement):
    """Address table extracted from a helper-driven dispatch."""

    cases: Tuple[ASTSwitchCase, ...]
    helper: int | None = None
    helper_symbol: str | None = None
    default: int | None = None

    def render(self) -> str:
        parts: List[str] = []
        if self.helper is not None:
            helper_repr = f"0x{self.helper:04X}"
            if self.helper_symbol:
                helper_repr = f"{self.helper_symbol}({helper_repr})"
            parts.append(f"helper={helper_repr}")
        if self.default is not None:
            parts.append(f"default=0x{self.default:04X}")
        prefix = "dispatch.data"
        if parts:
            prefix += " " + " ".join(parts)
        rendered_cases = ", ".join(case.render() for case in self.cases)
        return f"{prefix} cases=[{rendered_cases}]"


@dataclass
class ASTSwitch(ASTStatement):
    """Explicit switch extracted from a helper dispatch call."""

    selector: ASTExpression
    cases: Tuple[ASTSwitchCase, ...]
    helper: int | None = None
    helper_symbol: str | None = None
    default: int | None = None

    def render(self) -> str:
        helper_note = ""
        if self.helper is not None:
            helper_repr = f"0x{self.helper:04X}"
            if self.helper_symbol:
                helper_repr = f"{self.helper_symbol}({helper_repr})"
            helper_note = f" helper={helper_repr}"
        default_note = ""
        if self.default is not None:
            default_note = f" default=0x{self.default:04X}"
        rendered_cases = ", ".join(case.render() for case in self.cases)
        return f"switch {self.selector.render()} cases=[{rendered_cases}]{default_note}{helper_note}"


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------


@dataclass
class ASTBlock:
    """Single basic block in the reconstructed AST."""

    label: str
    start_offset: int
    statements: Tuple[ASTStatement, ...]
    successors: Tuple["ASTBlock", ...]


@dataclass(frozen=True)
class ASTFrameSlot:
    """Summary of how a stack frame slot is used."""

    index: int
    reads: int = 0
    writes: int = 0

    def render(self) -> str:
        return f"slot_0x{self.index:04X}(r={self.reads},w={self.writes})"


@dataclass(frozen=True)
class ASTFrameParameter:
    """Formal parameter reconstructed from frame traffic."""

    slot: int
    identifier: ASTIdentifier

    def render(self) -> str:
        return f"{self.identifier.render()}@slot_0x{self.slot:04X}"


@dataclass(frozen=True)
class ASTFrameModel:
    """Lightweight view of the reconstructed stack frame."""

    size: int
    slots: Tuple[ASTFrameSlot, ...] = field(default_factory=tuple)
    parameters: Tuple[ASTFrameParameter, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ASTProcedure:
    """Group of blocks that form a reconstructed procedure."""

    name: str
    entry_offset: int
    entry_reasons: Tuple[str, ...]
    blocks: Tuple[ASTBlock, ...]
    exit_offsets: Tuple[int, ...]
    frame: ASTFrameModel


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
    "ASTBankedRefExpr",
    "ASTBankedLoadExpr",
    "ASTCallExpr",
    "ASTCallResult",
    "ASTStatement",
    "ASTAssign",
    "ASTStore",
    "ASTCallStatement",
    "ASTIORead",
    "ASTIOWrite",
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
