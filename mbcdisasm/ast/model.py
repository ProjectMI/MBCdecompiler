"""Data structures for the reconstructed abstract syntax tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Tuple

from ..constants import OPERAND_ALIASES
from ..ir.model import IRSlot, MemRef, MemSpace, SSAValueKind


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


def _escape_string(value: str) -> str:
    """Escape control characters for canonical string rendering."""

    escaped: List[str] = []
    for char in value:
        code = ord(char)
        if char == "\\":
            escaped.append("\\\\")
        elif char == "\"":
            escaped.append("\\\"")
        elif 0x20 <= code <= 0x7E:
            escaped.append(char)
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif char == "\t":
            escaped.append("\\t")
        else:
            escaped.append(f"\\x{code:02X}")
    return "".join(escaped)


def _escape_bytes(value: bytes) -> str:
    """Render byte literals using a stable escaped representation."""

    escaped: List[str] = []
    for byte in value:
        if 0x20 <= byte <= 0x7E and byte not in {0x22, 0x5C}:
            escaped.append(chr(byte))
        elif byte == 0x0A:
            escaped.append("\\n")
        elif byte == 0x0D:
            escaped.append("\\r")
        elif byte == 0x09:
            escaped.append("\\t")
        else:
            escaped.append(f"\\x{byte:02X}")
    return "".join(escaped)


class ASTEffectKind(Enum):
    """Classification of the observable side effect for a statement."""

    PURE = "pure"
    READ_ONLY = "read_only"
    MUTABLE = "mutable"
    IO = "io"
    UNKNOWN = "unknown"


class ASTCastKind(Enum):
    """Supported cast conversions in the canonical AST."""

    ZERO_EXTEND = "zero_extend"
    SIGN_EXTEND = "sign_extend"
    TRUNCATE = "truncate"
    TO_BOOLEAN = "to_boolean"
    TO_FLOAT = "to_float"
    FROM_FLOAT = "from_float"


class ASTMemoryAccessKind(Enum):
    """Nature of the memory access expressed by :class:`ASTMemoryAccess`."""

    FIELD = "field"
    ELEMENT = "element"
    SLICE = "slice"
    VIEW = "view"


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
class ASTNumericLiteral(ASTExpression):
    """Literal integer constant with explicit width and sign."""

    value: int
    bits: int = 16
    signed: bool = False

    def render(self) -> str:
        digits = max(1, (self.bits + 3) // 4)
        if self.signed:
            magnitude = abs(self.value)
            sign = "+" if self.value >= 0 else "-"
        else:
            mask = (1 << self.bits) - 1
            magnitude = self.value & mask
            sign = "+"
        return f"{sign}0x{magnitude:0{digits}X}"

    def kind(self) -> SSAValueKind:
        if self.bits <= 8:
            return SSAValueKind.BYTE
        if self.bits <= 16:
            return SSAValueKind.WORD
        return SSAValueKind.UNKNOWN


@dataclass(frozen=True)
class ASTFloatLiteral(ASTExpression):
    """Literal floating point constant with explicit precision."""

    value: float
    bits: int = 32

    def render(self) -> str:
        precision = max(1, self.bits // 8)
        magnitude = abs(self.value)
        sign = "+" if self.value >= 0 else "-"
        formatted = format(magnitude, f".{precision}g")
        return f"{sign}{formatted}f{self.bits}"

    def kind(self) -> SSAValueKind:
        return SSAValueKind.WORD


@dataclass(frozen=True)
class ASTBooleanLiteral(ASTExpression):
    """Canonical boolean literal."""

    value: bool

    def render(self) -> str:
        return "true" if self.value else "false"

    def kind(self) -> SSAValueKind:
        return SSAValueKind.BOOLEAN


@dataclass(frozen=True)
class ASTStringLiteral(ASTExpression):
    """Canonical string literal with explicit encoding."""

    value: str
    encoding: str = "utf-8"

    def render(self) -> str:
        escaped = _escape_string(self.value)
        return f'"{escaped}"@{self.encoding}'


@dataclass(frozen=True)
class ASTBytesLiteral(ASTExpression):
    """Canonical byte literal with explicit encoding."""

    data: bytes
    encoding: str = "binary"

    def render(self) -> str:
        escaped = _escape_bytes(self.data)
        return f'b"{escaped}"@{self.encoding}'


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
class ASTCastExpr(ASTExpression):
    """Explicit safe cast between compatible scalar types."""

    value: ASTExpression
    kind_name: ASTCastKind
    precondition: Optional[str] = None
    postcondition: Optional[str] = None

    def render(self) -> str:
        inner = self.value.render()
        extras: List[str] = []
        if self.precondition:
            extras.append(f"pre={self.precondition}")
        if self.postcondition:
            extras.append(f"post={self.postcondition}")
        suffix = "" if not extras else " [" + ", ".join(extras) + "]"
        return f"cast.{self.kind_name.value}({inner}){suffix}"

    def kind(self) -> SSAValueKind:
        return self.value.kind()


@dataclass(frozen=True)
class ASTSlice:
    """Slice specification used by :class:`ASTMemoryAccess`."""

    start: Optional[ASTExpression] = None
    stop: Optional[ASTExpression] = None
    stride: Optional[ASTExpression] = None

    def render(self) -> str:
        parts: List[str] = []
        parts.append(self.start.render() if self.start else "")
        parts.append(self.stop.render() if self.stop else "")
        if self.stride is not None:
            parts.append(self.stride.render())
        text = ":".join(parts)
        if self.stride is None:
            return text
        return text


@dataclass(frozen=True)
class ASTMemoryAccess(ASTExpression):
    """Canonical representation of structured memory accesses."""

    ref: MemRef
    access_kind: ASTMemoryAccessKind = ASTMemoryAccessKind.FIELD
    base: Optional[ASTExpression] = None
    index: Optional[ASTExpression] = None
    slice: Optional[ASTSlice] = None
    alias_class: Optional[str] = None

    def _root_name(self) -> str:
        if self.ref.symbol:
            return self.ref.symbol.replace(":", ".")
        region = self.ref.region or "mem"
        if region.startswith("mem") or region in {"frame", "global", "const", "io"}:
            return region
        return f"mem.{region}"

    def _field_name(self) -> str:
        if self.ref.symbol:
            return self.ref.symbol.split(".")[-1]
        if self.ref.offset is not None:
            width = 4 if self.ref.offset > 0xFF else 2
            return f"field_0x{self.ref.offset:0{width}X}"
        if self.ref.base is not None:
            return f"field_base_0x{self.ref.base:04X}"
        return "field"

    def _alias_suffix(self) -> str:
        return "" if not self.alias_class else f" @{self.alias_class}"

    def render(self) -> str:
        root = self._root_name()
        if self.access_kind is ASTMemoryAccessKind.FIELD:
            field = self._field_name()
            if self.base is not None:
                base = self.base.render()
            else:
                base = root
            return f"{base}.{field}{self._alias_suffix()}"
        if self.access_kind is ASTMemoryAccessKind.ELEMENT:
            base = self.base.render() if self.base is not None else root
            index = self.index.render() if self.index is not None else "0"
            return f"{base}[{index}]{self._alias_suffix()}"
        if self.access_kind is ASTMemoryAccessKind.SLICE:
            base = self.base.render() if self.base is not None else root
            slice_text = self.slice.render() if self.slice is not None else ":"
            return f"{base}[{slice_text}]{self._alias_suffix()}"
        base = self.base.render() if self.base is not None else root
        details: List[str] = [self.ref.describe()]
        if self.index is not None:
            details.append(f"index={self.index.render()}")
        if self.slice is not None:
            details.append(f"slice={self.slice.render()}")
        description = " ".join(details)
        return f"view({base} {description}){self._alias_suffix()}"

    def kind(self) -> SSAValueKind:
        return SSAValueKind.UNKNOWN


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

    effect: ASTEffectKind = field(default=ASTEffectKind.PURE, init=False)

    def __post_init__(self) -> None:
        if not hasattr(self, "effect") or self.effect is None:
            self.effect = ASTEffectKind.PURE

    def render(self) -> str:  # pragma: no cover - overridden in subclasses
        raise NotImplementedError

    def _render_effect_suffix(self) -> str:
        if self.effect is ASTEffectKind.PURE:
            return ""
        return f" {{effect={self.effect.value}}}"


@dataclass
class ASTAssign(ASTStatement):
    """Assign an expression to a target identifier."""

    target: ASTIdentifier
    value: ASTExpression

    def render(self) -> str:
        return f"{self.target.render()} = {self.value.render()}{self._render_effect_suffix()}"


@dataclass
class ASTStore(ASTStatement):
    """Store ``value`` into the location described by ``target``."""

    target: ASTMemoryAccess
    value: ASTExpression

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.MUTABLE

    def render(self) -> str:
        return f"store {self.value.render()} -> {self.target.render()}{self._render_effect_suffix()}"


@dataclass
class ASTCallStatement(ASTStatement):
    """Call that yields one or more return values."""

    call: ASTCallExpr
    returns: Tuple[ASTIdentifier, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.UNKNOWN

    def render(self) -> str:
        if not self.returns:
            return f"{self.call.render()}{self._render_effect_suffix()}"
        outputs = ", ".join(ret.render() for ret in self.returns)
        return f"{outputs} = {self.call.render()}{self._render_effect_suffix()}"


@dataclass
class ASTCallFrame(ASTStatement):
    """Explicit description of the call frame layout for helper invocations."""

    slots: Tuple[ASTCallFrameSlot, ...] = field(default_factory=tuple)
    effects: Tuple[ASTFrameEffect, ...] = field(default_factory=tuple)
    live_mask: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.READ_ONLY if (self.slots or self.effects) else ASTEffectKind.PURE

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
        return f"call_frame{suffix}{self._render_effect_suffix()}"


@dataclass
class ASTFrameProtocol(ASTStatement):
    """Aggregated description of the frame policy prior to returning."""

    masks: Tuple[Tuple[int, Optional[str]], ...] = field(default_factory=tuple)
    teardown: int = 0
    drops: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.masks or self.teardown or self.drops:
            self.effect = ASTEffectKind.READ_ONLY

    def render(self) -> str:
        parts: List[str] = []
        if self.masks:
            rendered_masks = ", ".join(
                _format_operand(value, alias)
                for value, alias in self.masks
            )
            parts.append(f"masks=[{rendered_masks}]")
        if self.teardown:
            parts.append(f"teardown={self.teardown}")
        if self.drops:
            parts.append(f"drops={self.drops}")
        suffix = " " + " ".join(parts) if parts else ""
        return f"frame_protocol{suffix}{self._render_effect_suffix()}"


@dataclass
class ASTIORead(ASTStatement):
    """I/O read effect emitted by helper façades."""

    port: str

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.IO

    def render(self) -> str:
        return f"io.read({self.port}){self._render_effect_suffix()}"


@dataclass
class ASTIOWrite(ASTStatement):
    """I/O write effect emitted by helper façades."""

    port: str
    mask: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.IO

    def render(self) -> str:
        mask = "" if self.mask is None else f", mask=0x{self.mask:04X}"
        return f"io.write({self.port}{mask}){self._render_effect_suffix()}"


@dataclass
class ASTTailCall(ASTStatement):
    """Tail call used as a return."""

    call: ASTCallExpr
    returns: Tuple[ASTExpression, ...]
    finally_branch: Optional[ASTFinally] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.UNKNOWN

    def render(self) -> str:
        rendered = ", ".join(expr.render() for expr in self.returns)
        suffix = f" returns [{rendered}]" if rendered else ""
        call_repr = self.call.render()
        if self.call.tail and call_repr.startswith("tail "):
            call_repr = call_repr[len("tail ") :]
        finally_suffix = ""
        if self.finally_branch is not None:
            finally_suffix = f" finally {self.finally_branch.render()}"
        return f"tail {call_repr}{suffix}{finally_suffix}{self._render_effect_suffix()}"


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
        return f"return {payload}{suffix}{self._render_effect_suffix()}"


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
        return f"if {condition} then {then_label} else {else_label}{self._render_effect_suffix()}"


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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.MUTABLE

    def render(self) -> str:
        then_label = self.then_branch.label if self.then_branch else self.then_hint or "?"
        else_label = self.else_branch.label if self.else_branch else self.else_hint or "?"
        return (
            f"testset {self.var.render()} = {self.expr.render()} "
            f"then {then_label} else {else_label}{self._render_effect_suffix()}"
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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.READ_ONLY

    def render(self) -> str:
        then_label = self.then_branch.label if self.then_branch else self.then_hint or "?"
        else_label = self.else_branch.label if self.else_branch else self.else_hint or "?"
        return f"flag 0x{self.flag:04X} ? then {then_label} else {else_label}{self._render_effect_suffix()}"


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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.READ_ONLY

    def render(self) -> str:
        then_label = self.then_branch.label if self.then_branch else self.then_hint or "?"
        else_label = self.else_branch.label if self.else_branch else self.else_hint or "?"
        return (
            f"prologue {self.var.render()} = {self.expr.render()} "
            f"then {then_label} else {else_label}{self._render_effect_suffix()}"
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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.effect = ASTEffectKind.READ_ONLY

    def _render_helper(self) -> str | None:
        if self.helper is None:
            return None
        helper_repr = f"0x{self.helper:04X}"
        if self.helper_symbol:
            helper_repr = f"{self.helper_symbol}({helper_repr})"
        return f"helper={helper_repr}"

    def _render_index(self) -> str:
        expr = None
        if self.index_expr is not None:
            expr = self.index_expr.render()
        elif self.call is not None:
            expr = self.call.render()
        if self.index_mask is not None:
            mask_text = f"0x{self.index_mask:04X}"
            if expr:
                expr = f"{expr} & {mask_text}"
            else:
                expr = f"& {mask_text}"
        if not expr:
            expr = "?"
        return f"index={expr}"

    def render(self) -> str:
        parts: List[str] = [self._render_index()]
        if self.enum_name:
            parts.append(f"enum={self.enum_name}")
        rendered_cases = ", ".join(case.render() for case in self.cases)
        parts.append(f"table=[{rendered_cases}]")
        if self.default is not None:
            parts.append(f"default=0x{self.default:04X}")
        if self.index_base is not None:
            parts.append(f"base=0x{self.index_base:04X}")
        helper_note = self._render_helper()
        if helper_note:
            parts.append(helper_note)
        if self.kind:
            parts.append(f"kind={self.kind}")
        return f"Switch{{{', '.join(parts)}}}{self._render_effect_suffix()}"


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


@dataclass(frozen=True)
class ASTAliasModel:
    """Description of the aliasing guarantees for the AST."""

    parameter_policy: str = "noalias"
    global_regions: Tuple[str, ...] = ("frame", "global", "const")

    def describe(self) -> str:
        regions = ", ".join(self.global_regions)
        return f"parameters={self.parameter_policy} regions=[{regions}]"


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
    aliasing: ASTAliasModel
    enums: Tuple[ASTEnumDecl, ...] = ()


__all__ = [
    "ASTEffectKind",
    "ASTCastKind",
    "ASTMemoryAccessKind",
    "ASTExpression",
    "ASTUnknown",
    "ASTNumericLiteral",
    "ASTFloatLiteral",
    "ASTBooleanLiteral",
    "ASTStringLiteral",
    "ASTBytesLiteral",
    "ASTIdentifier",
    "ASTSlotRef",
    "ASTCastExpr",
    "ASTSlice",
    "ASTMemoryAccess",
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
    "ASTFrameProtocol",
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
    "ASTSwitchCase",
    "ASTSwitch",
    "ASTBlock",
    "ASTProcedure",
    "ASTSegment",
    "ASTAliasModel",
    "ASTMetrics",
    "ASTProgram",
]
