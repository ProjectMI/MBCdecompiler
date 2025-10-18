"""Data structures for the reconstructed abstract syntax tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..constants import OPERAND_ALIASES
from ..ir.model import IRSlot, MemRef, SSAValueKind


def _format_operand(value: int, alias: Optional[str] = None) -> str:
    """Format ``value`` using the shared operand alias table."""

    hex_value = f"0x{value:04X}"
    alias_text = alias or OPERAND_ALIASES.get(value)
    if alias_text:
        upper = alias_text.upper()
        if upper.startswith("0X"):
            alias_text = upper
        if alias_text == hex_value:
            return alias_text
        return f"{alias_text}({hex_value})"
    return hex_value


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


@dataclass(frozen=True)
class ASTTupleExpr(ASTExpression):
    """Tuple literal composed of concrete AST expressions."""

    items: Tuple[ASTExpression, ...]

    def render(self) -> str:
        inner = ", ".join(item.render() for item in self.items)
        return f"({inner})"


# ---------------------------------------------------------------------------
# statements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASTFrameEffect:
    """Side effect applied while constructing a call frame."""

    kind: str
    operand: Optional[int] = None
    alias: Optional[str] = None
    pops: int = 0

    def render(self) -> str:
        details: List[str] = []
        if self.pops:
            details.append(f"pop={self.pops}")
        if self.operand is not None:
            details.append(f"value={_format_operand(self.operand, self.alias)}")
        if not details:
            return self.kind
        inner = ", ".join(details)
        return f"{self.kind}({inner})"


@dataclass(frozen=True)
class ASTCallFrameSlot:
    """Assignment of a value to a frame slot prior to a helper call."""

    index: int
    value: ASTExpression

    def render(self) -> str:
        return f"slot[{self.index}]={self.value.render()}"


@dataclass(frozen=True)
class ASTFinallyStep:
    """Single action performed by a structured epilogue."""

    kind: str
    operand: Optional[int] = None
    alias: Optional[str] = None
    pops: int = 0

    def render(self) -> str:
        details: List[str] = []
        if self.pops:
            details.append(f"pop={self.pops}")
        if self.operand is not None:
            details.append(f"value={_format_operand(self.operand, self.alias)}")
        if not details:
            return self.kind
        inner = ", ".join(details)
        return f"{self.kind}({inner})"


@dataclass(frozen=True)
class ASTFinally:
    """Structured epilogue attached to a return statement."""

    steps: Tuple[ASTFinallyStep, ...]

    def render(self) -> str:
        if not self.steps:
            return "[]"
        rendered = ", ".join(step.render() for step in self.steps)
        return f"[{rendered}]"


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
class ASTCallFrame(ASTStatement):
    """Explicit description of the call frame layout for helper invocations."""

    slots: Tuple[ASTCallFrameSlot, ...] = field(default_factory=tuple)
    effects: Tuple[ASTFrameEffect, ...] = field(default_factory=tuple)
    live_mask: Optional[int] = None

    def render(self) -> str:
        parts: List[str] = []
        if self.slots:
            rendered_slots = ", ".join(slot.render() for slot in self.slots)
            parts.append(f"slots=[{rendered_slots}]")
        if self.effects:
            rendered_effects = ", ".join(effect.render() for effect in self.effects)
            parts.append(f"effects=[{rendered_effects}]")
        if self.live_mask is not None:
            parts.append(f"live={_format_operand(self.live_mask)}")
        suffix = " " + " ".join(parts) if parts else ""
        return f"call_frame{suffix}"


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

    value: Optional[ASTExpression]
    varargs: bool = False
    finally_branch: Optional[ASTFinally] = None

    def render(self) -> str:
        if self.varargs:
            if self.value is None:
                payload = "varargs"
            elif isinstance(self.value, ASTTupleExpr):
                payload = f"varargs{self.value.render()}"
            else:
                payload = f"varargs({self.value.render()})"
        else:
            if self.value is None:
                payload = "()"
            else:
                payload = self.value.render()
        suffix = ""
        if self.finally_branch is not None:
            suffix = f" finally {self.finally_branch.render()}"
        return f"return {payload}{suffix}"


@dataclass
class ASTBranch(ASTStatement):
    """Generic conditional branch with CFG links."""

    condition: ASTExpression
    then_branch: "ASTBlock | None" = None
    else_branch: "ASTBlock | None" = None
    then_hint: str | None = None
    else_hint: str | None = None
    then_offset: int | None = None
    else_offset: int | None = None

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
    then_offset: int | None = None
    else_offset: int | None = None

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
    then_offset: int | None = None
    else_offset: int | None = None

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
    then_offset: int | None = None
    else_offset: int | None = None

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


@dataclass(frozen=True)
class ASTEnumMember:
    """Single member of an enum declaration."""

    name: str
    value: int


@dataclass
class ASTEnumDecl:
    """Named enumeration reconstructed from dispatch helpers."""

    name: str
    members: Tuple[ASTEnumMember, ...]


@dataclass
class ASTSwitchCase:
    """Single case handled by a helper dispatch."""

    key: int
    target: int
    symbol: str | None = None
    key_alias: str | None = None

    def render(self) -> str:
        target_repr = f"0x{self.target:04X}"
        if self.symbol:
            target_repr = f"{self.symbol}({target_repr})"
        key_repr = f"0x{self.key:04X}"
        if self.key_alias:
            key_repr = f"{self.key_alias}(0x{self.key:04X})"
        return f"{key_repr}->{target_repr}"


@dataclass
class ASTSwitch(ASTStatement):
    """Normalised jump table reconstructed from helper dispatch patterns."""

    call: ASTCallExpr | None
    cases: Tuple[ASTSwitchCase, ...]
    helper: int | None = None
    helper_symbol: str | None = None
    default: int | None = None
    index_expr: ASTExpression | None = None
    index_mask: int | None = None
    index_base: int | None = None
    kind: str | None = None
    enum_name: str | None = None

    def render(self) -> str:
        header = f"switch({self._render_index_expr()})"
        body_lines: List[str] = []
        for case in self.cases:
            label = self._format_case_label(case)
            target = self._format_case_target(case.target, case.symbol)
            body_lines.append(f"    case {label}: goto {target};")
        if self.default is not None:
            default_target = self._format_case_target(self.default, None)
            body_lines.append(f"    default: goto {default_target};")
        if not body_lines:
            return f"{header} {{}}"
        body = "\n".join(body_lines)
        return f"{header} {{\n{body}\n}}"

    def _render_index_expr(self) -> str:
        if self.index_expr is not None:
            expr = self.index_expr.render()
        elif self.call is not None:
            expr = self.call.render()
        else:
            expr = "?"
        if self.index_mask is not None:
            expr = f"({expr} & 0x{self.index_mask:04X})"
        return expr

    @staticmethod
    def _format_case_target(target: int, symbol: str | None) -> str:
        target_repr = f"0x{target:04X}"
        if symbol:
            return f"{symbol}({target_repr})"
        return target_repr

    @staticmethod
    def _format_case_label(case: ASTSwitchCase) -> str:
        if case.key_alias:
            return case.key_alias
        return f"0x{case.key:04X}"


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
    enums: Tuple[ASTEnumDecl, ...] = ()


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
    enums: Tuple[ASTEnumDecl, ...] = ()


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
    "ASTTupleExpr",
    "ASTStatement",
    "ASTAssign",
    "ASTStore",
    "ASTFrameEffect",
    "ASTCallFrameSlot",
    "ASTCallStatement",
    "ASTCallFrame",
    "ASTIORead",
    "ASTIOWrite",
    "ASTTailCall",
    "ASTReturn",
    "ASTFinallyStep",
    "ASTFinally",
    "ASTBranch",
    "ASTTestSet",
    "ASTFlagCheck",
    "ASTFunctionPrologue",
    "ASTComment",
    "ASTEnumDecl",
    "ASTEnumMember",
    "ASTBlock",
    "ASTProcedure",
    "ASTSegment",
    "ASTMetrics",
    "ASTProgram",
]
