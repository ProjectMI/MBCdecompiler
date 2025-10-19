"""Data structures for the reconstructed abstract syntax tree."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, List, Optional, Tuple

from ..constants import OPERAND_ALIASES
from ..ir.model import SSAValueKind


class ASTNumericSign(Enum):
    """Signedness annotation attached to numeric literals."""

    UNSIGNED = "unsigned"
    SIGNED = "signed"


class ASTEffectCategory(Enum):
    """Side-effect categories recognised by the canonical AST."""

    PURE = "pure"
    READ = "readonly"
    MUTABLE = "mutable"
    IO = "io"
    UNKNOWN = "unknown"


class ASTAliasKind(Enum):
    """Alias annotations attached to structured memory locations."""

    NOALIAS = "noalias"
    REGION = "region"
    UNKNOWN = "unknown"


class ASTConversionKind(Enum):
    """Small, fixed set of safe conversion primitives."""

    ZERO_EXTEND = "zero_extend"
    SIGN_EXTEND = "sign_extend"
    TRUNCATE = "truncate"
    NORMALISE = "normalise"
    TO_BOOLEAN = "to_boolean"
    TO_POINTER = "to_pointer"


class ASTArgumentMode(Enum):
    """Calling convention for passing an argument to a call site."""

    VALUE = "value"
    BORROW = "borrow"
    MOVE = "move"


@dataclass(frozen=True)
class ASTAliasInfo:
    """Alias metadata describing how a location can alias with others."""

    kind: ASTAliasKind
    label: Optional[str] = None

    def render(self) -> str:
        if self.label:
            return f"{self.kind.value}:{self.label}"
        return self.kind.value


@dataclass(frozen=True)
class ASTAccessPathElement:
    """Base class for structured memory access path elements."""

    def render(self) -> str:  # pragma: no cover - overridden in subclasses
        raise NotImplementedError


@dataclass(frozen=True)
class ASTFieldElement(ASTAccessPathElement):
    """Access a named field of an aggregate."""

    name: str

    def render(self) -> str:
        return f".{self.name}"


@dataclass(frozen=True)
class ASTIndexElement(ASTAccessPathElement):
    """Index into a sequence using an AST expression."""

    index: "ASTExpression"

    def render(self) -> str:
        return f"[{self.index.render()}]"


@dataclass(frozen=True)
class ASTSliceElement(ASTAccessPathElement):
    """Access a slice of a sequence."""

    start: int
    length: int

    def render(self) -> str:
        end = self.start + self.length
        return f"[{self._render_bound(self.start)}:{self._render_bound(end)}]"

    @staticmethod
    def _render_bound(value: int) -> str:
        return f"0x{value:04X}"


def _default_alias() -> ASTAliasInfo:
    return ASTAliasInfo(ASTAliasKind.UNKNOWN)


def _escape_string(text: str) -> str:
    escaped: List[str] = []
    for char in text:
        code = ord(char)
        if char in {"\\", '"'}:
            escaped.append(f"\\{char}")
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif char == "\t":
            escaped.append("\\t")
        elif 0x20 <= code <= 0x7E:
            escaped.append(char)
        else:
            escaped.append(f"\\x{code:02X}")
    return "".join(escaped)


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

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.PURE

    def render(self) -> str:  # pragma: no cover - overridden in subclasses
        raise NotImplementedError

    def kind(self) -> SSAValueKind:
        """Return the inferred :class:`SSAValueKind` for the expression."""

        return SSAValueKind.UNKNOWN

    def effect(self) -> ASTEffectCategory:
        """Return the side-effect classification for the expression."""

        return self.effect_category


@dataclass(frozen=True)
class ASTUnknown(ASTExpression):
    """Fallback expression used when reconstruction fails."""

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.UNKNOWN
    token: str

    def render(self) -> str:
        return f"?({self.token})"


@dataclass(frozen=True)
class ASTIntegerLiteral(ASTExpression):
    """Integer literal with explicit signedness and width."""

    value: int
    bits: int = 16
    sign: ASTNumericSign = ASTNumericSign.UNSIGNED
    radix: int = 16

    def render(self) -> str:
        sign_char = "+" if self.value >= 0 else "-"
        magnitude = abs(self.value)
        if self.radix == 10:
            magnitude_text = f"{magnitude:d}"
            prefix = ""
        elif self.radix == 2:
            width = max(1, self.bits)
            magnitude_text = f"{magnitude:0{width}b}"
            prefix = "0b"
        else:
            width = max(1, math.ceil(self.bits / 4))
            magnitude_text = f"{magnitude:0{width}X}"
            prefix = "0x"
        return f"int<{self.sign.value},{self.bits}>({sign_char}{prefix}{magnitude_text})"

    def kind(self) -> SSAValueKind:
        if self.bits <= 8:
            return SSAValueKind.BYTE
        return SSAValueKind.WORD


@dataclass(frozen=True)
class ASTUnaryExpr(ASTExpression):
    """Unary operator applied to a single operand."""

    op: str
    operand: "ASTExpression"

    def render(self) -> str:
        if self.op == "not":
            return f"not {self.operand.render()}"
        return f"{self.op}({self.operand.render()})"


@dataclass(frozen=True)
class ASTFlagExpr(ASTExpression):
    """Boolean view over a VM flag used in structured conditions."""

    flag: int

    def render(self) -> str:
        return f"flag(0x{self.flag:04X})"

    def kind(self) -> SSAValueKind:
        return SSAValueKind.BOOLEAN


@dataclass(frozen=True)
class ASTRealLiteral(ASTExpression):
    """Floating-point literal with explicit encoding metadata."""

    value: float
    precision: int = 32
    encoding: str = "ieee754"

    def render(self) -> str:
        return f"real<{self.encoding},{self.precision}>({self.value})"

    def kind(self) -> SSAValueKind:
        return SSAValueKind.WORD


@dataclass(frozen=True)
class ASTBooleanLiteral(ASTExpression):
    """Boolean literal in its canonical representation."""

    value: bool

    def render(self) -> str:
        return f"bool({'true' if self.value else 'false'})"

    def kind(self) -> SSAValueKind:
        return SSAValueKind.BOOLEAN


@dataclass(frozen=True)
class ASTStringLiteral(ASTExpression):
    """String literal with explicit encoding."""

    text: str
    encoding: str = "utf-8"

    def render(self) -> str:
        return f'str[{self.encoding}]("{_escape_string(self.text)}")'


@dataclass(frozen=True)
class ASTBytesLiteral(ASTExpression):
    """Opaque byte literal with explicit encoding."""

    data: bytes
    encoding: str = "hex"

    def render(self) -> str:
        if self.encoding == "hex":
            payload = self.data.hex().upper()
        else:
            payload = self.data.decode(self.encoding)
        return f"bytes[{self.encoding}]({payload})"

    def kind(self) -> SSAValueKind:
        if len(self.data) == 1:
            return SSAValueKind.BYTE
        return SSAValueKind.UNKNOWN


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
class ASTSafeCastExpr(ASTExpression):
    """Safe conversion constrained to the approved conversion set."""

    operand: ASTExpression
    conversion: ASTConversionKind
    target_kind: SSAValueKind = SSAValueKind.UNKNOWN
    precondition: Optional[str] = None
    postcondition: Optional[str] = None

    def render(self) -> str:
        metadata = [self.conversion.value]
        if self.precondition:
            metadata.append(f"requires={self.precondition}")
        if self.postcondition:
            metadata.append(f"ensures={self.postcondition}")
        return f"cast[{', '.join(metadata)}]({self.operand.render()})"

    def kind(self) -> SSAValueKind:
        return self.target_kind


@dataclass(frozen=True)
class ASTMemoryLocation:
    """Structured representation of an addressable location."""

    base: str | ASTExpression
    path: Tuple[ASTAccessPathElement, ...] = ()
    alias: ASTAliasInfo = field(default_factory=_default_alias)

    def render(self) -> str:
        base_repr = self.base.render() if isinstance(self.base, ASTExpression) else self.base
        alias_repr = self.alias.render()
        alias_suffix = ""
        if self.alias.kind is not ASTAliasKind.UNKNOWN or self.alias.label:
            alias_suffix = f"@{alias_repr}"
        trail = "".join(element.render() for element in self.path)
        return f"{base_repr}{alias_suffix}{trail}"


@dataclass(frozen=True)
class ASTMemoryRead(ASTExpression):
    """Read a value from a structured location."""

    location: ASTMemoryLocation
    value_kind: SSAValueKind | None = None

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.READ

    def render(self) -> str:
        return self.location.render()

    def kind(self) -> SSAValueKind:
        if self.value_kind is not None:
            return self.value_kind
        return SSAValueKind.UNKNOWN


@dataclass(frozen=True)
class ASTCallArg:
    """Single argument passed to a call expression with explicit mode."""

    value: "ASTExpression"
    mode: ASTArgumentMode = ASTArgumentMode.VALUE

    def render(self) -> str:
        prefix = ""
        if self.mode is ASTArgumentMode.BORROW:
            prefix = "borrow "
        elif self.mode is ASTArgumentMode.MOVE:
            prefix = "move "
        return f"{prefix}{self.value.render()}"


@dataclass(frozen=True)
class ASTCallExpr(ASTExpression):
    """Call expression with resolved argument expressions."""

    target: int
    args: Tuple[ASTCallArg, ...]
    symbol: str | None = None
    tail: bool = False
    varargs: bool = False

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.UNKNOWN

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

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.UNKNOWN

    def render(self) -> str:  # pragma: no cover - overridden in subclasses
        raise NotImplementedError

    def effect(self) -> ASTEffectCategory:
        """Return the side-effect classification for the statement."""

        return self.effect_category


@dataclass
class ASTAssign(ASTStatement):
    """Assign an expression to a target identifier."""

    target: ASTIdentifier
    value: ASTExpression

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.PURE

    def render(self) -> str:
        return f"{self.target.render()} = {self.value.render()}"


@dataclass
class ASTMemoryWrite(ASTStatement):
    """Store ``value`` into ``location``."""

    location: ASTMemoryLocation
    value: ASTExpression

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.MUTABLE

    def render(self) -> str:
        return f"{self.location.render()} = {self.value.render()}"


@dataclass
class ASTCallStatement(ASTStatement):
    """Call that yields one or more return values."""

    call: ASTCallExpr
    returns: Tuple[ASTIdentifier, ...] = field(default_factory=tuple)

    def render(self) -> str:
        if not self.returns:
            return self.call.render()
        rendered = ", ".join(identifier.render() for identifier in self.returns)
        return f"{self.call.render()} -> ({rendered})"


@dataclass
class ASTCallFrame(ASTStatement):
    """Structured representation of a helper call frame."""

    slots: Tuple[ASTCallFrameSlot, ...]
    effects: Tuple[ASTFrameEffect, ...]
    live_mask: Optional[int] = None

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.MUTABLE

    def render(self) -> str:
        rendered_slots = ", ".join(slot.render() for slot in self.slots)
        rendered_effects = ", ".join(effect.render() for effect in self.effects)
        details = []
        if rendered_slots:
            details.append(f"slots=[{rendered_slots}]")
        if rendered_effects:
            details.append(f"effects=[{rendered_effects}]")
        if self.live_mask is not None:
            details.append(f"live_mask=0x{self.live_mask:04X}")
        joined = " ".join(details)
        return f"frame{{{joined}}}"


@dataclass
class ASTFrameProtocol(ASTStatement):
    """Summary of structured epilogue behaviour."""

    masks: Tuple[Tuple[int, Optional[str]], ...]
    teardown: int
    drops: int

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.MUTABLE

    def render(self) -> str:
        rendered_masks = []
        for value, alias in self.masks:
            mask_text = f"0x{value:04X}"
            if alias:
                mask_text = f"{alias}({mask_text})"
            rendered_masks.append(mask_text)
        masks = ", ".join(rendered_masks)
        return f"frame.protocol[masks=[{masks}], teardown={self.teardown}, drops={self.drops}]"


@dataclass
class ASTIORead(ASTStatement):
    """I/O read effect emitted by helper façades."""

    port: str

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.READ

    def render(self) -> str:
        return f"io.read({self.port})"


@dataclass
class ASTIOWrite(ASTStatement):
    """I/O write effect emitted by helper façades."""

    port: str
    mask: int | None = None

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.IO

    def render(self) -> str:
        mask = "" if self.mask is None else f", mask=0x{self.mask:04X}"
        return f"io.write({self.port}{mask})"


@dataclass
class ASTTailCall(ASTStatement):
    """Tail call used as a return."""

    call: ASTCallExpr
    returns: Tuple[ASTExpression, ...]
    finally_branch: Optional[ASTFinally] = None

    def render(self) -> str:
        rendered = ", ".join(expr.render() for expr in self.returns)
        suffix = f" returns [{rendered}]" if rendered else ""
        call_repr = self.call.render()
        if self.call.tail and call_repr.startswith("tail "):
            call_repr = call_repr[len("tail ") :]
        finally_suffix = ""
        if self.finally_branch is not None:
            finally_suffix = f" finally {self.finally_branch.render()}"
        return f"tail {call_repr}{suffix}{finally_suffix}"


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

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.PURE

    def render(self) -> str:
        return f"; {self.text}"


@dataclass
class ASTBreak(ASTStatement):
    """Exit from the innermost loop."""

    level: int = 1

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.PURE

    def render(self) -> str:
        if self.level == 1:
            return "break"
        return f"break {self.level}"


@dataclass
class ASTContinue(ASTStatement):
    """Continue with the next iteration of the innermost loop."""

    level: int = 1

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.PURE

    def render(self) -> str:
        if self.level == 1:
            return "continue"
        return f"continue {self.level}"


@dataclass
class ASTThrow(ASTStatement):
    """Raise an exception from the current scope."""

    value: Optional[ASTExpression] = None

    def render(self) -> str:
        if self.value is None:
            return "throw"
        return f"throw {self.value.render()}"


@dataclass(frozen=True)
class ASTCatch:
    """Catch handler for :class:`ASTTry`."""

    exception: Optional[str]
    body: Tuple["ASTStatement", ...]


@dataclass
class ASTTry(ASTStatement):
    """Structured exception handling block."""

    body: Tuple["ASTStatement", ...]
    handlers: Tuple[ASTCatch, ...] = ()
    finaliser: Tuple["ASTStatement", ...] = ()

    def render(self) -> str:
        parts = ["try {"]
        body = "; ".join(stmt.render() for stmt in self.body)
        parts.append(body or "pass")
        parts.append("}")
        for handler in self.handlers:
            label = handler.exception or "default"
            handler_body = "; ".join(stmt.render() for stmt in handler.body) or "pass"
            parts.append(f" catch {label} {{ {handler_body} }}")
        if self.finaliser:
            final_body = "; ".join(stmt.render() for stmt in self.finaliser)
            parts.append(f" finally {{ {final_body or 'pass'} }}")
        return "".join(parts)


@dataclass
class ASTIf(ASTStatement):
    """Structured conditional."""

    condition: ASTExpression | ASTStatement
    then_branch: Tuple["ASTStatement", ...]
    else_branch: Tuple["ASTStatement", ...] = ()

    def render(self) -> str:
        then_body = "; ".join(stmt.render() for stmt in self.then_branch) or "pass"
        cond = self.condition.render() if isinstance(self.condition, ASTExpression) else self.condition.render()
        if not self.else_branch:
            return f"if {cond} then {{ {then_body} }}"
        else_body = "; ".join(stmt.render() for stmt in self.else_branch) or "pass"
        return (
            f"if {cond} then {{ {then_body} }} "
            f"else {{ {else_body} }}"
        )


@dataclass
class ASTWhile(ASTStatement):
    """Canonical pre-test loop."""

    condition: ASTExpression | ASTStatement
    body: Tuple["ASTStatement", ...]

    def render(self) -> str:
        inner = "; ".join(stmt.render() for stmt in self.body) or "pass"
        cond = self.condition.render() if isinstance(self.condition, ASTExpression) else self.condition.render()
        return f"while {cond} do {{ {inner} }}"


@dataclass
class ASTIntrinsic(ASTStatement):
    """Encapsulated intrinsic/assembly primitive."""

    mnemonic: str
    operand: Optional[int] = None
    alias: Optional[str] = None
    annotations: Tuple[str, ...] = ()

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.UNKNOWN

    def render(self) -> str:
        details: List[str] = []
        if self.operand is not None:
            details.append(f"operand={_format_operand(self.operand, self.alias)}")
        if self.annotations:
            rendered = ", ".join(self.annotations)
            details.append(f"notes=[{rendered}]")
        inner = ", ".join(details)
        suffix = f"({inner})" if inner else ""
        return f"intrinsic {self.mnemonic}{suffix}"


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
    body: Tuple["ASTStatement", ...] = ()

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
        return f"Switch{{{', '.join(parts)}}}"


@dataclass
class ASTBlock:
    """Block of sequential statements with outgoing CFG edges."""

    label: str
    start_offset: int
    statements: Tuple[ASTStatement, ...]
    successors: Tuple["ASTBlock", ...] | None = None


@dataclass
class ASTProcedure:
    """Single reconstructed procedure."""

    name: str
    entry_offset: int
    blocks: Tuple[ASTBlock, ...]
    entry_reasons: Tuple[str, ...] = ()
    exit_offsets: Tuple[int, ...] = ()
    body: Tuple[ASTStatement, ...] = ()


@dataclass
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
    "ASTIntegerLiteral",
    "ASTUnaryExpr",
    "ASTFlagExpr",
    "ASTRealLiteral",
    "ASTBooleanLiteral",
    "ASTStringLiteral",
    "ASTBytesLiteral",
    "ASTIdentifier",
    "ASTSafeCastExpr",
    "ASTMemoryLocation",
    "ASTFieldElement",
    "ASTIndexElement",
    "ASTSliceElement",
    "ASTAliasInfo",
    "ASTAliasKind",
    "ASTArgumentMode",
    "ASTNumericSign",
    "ASTConversionKind",
    "ASTEffectCategory",
    "ASTMemoryRead",
    "ASTCallArg",
    "ASTCallExpr",
    "ASTCallResult",
    "ASTTupleExpr",
    "ASTStatement",
    "ASTAssign",
    "ASTMemoryWrite",
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
    "ASTBreak",
    "ASTContinue",
    "ASTThrow",
    "ASTCatch",
    "ASTTry",
    "ASTIf",
    "ASTWhile",
    "ASTIntrinsic",
    "ASTSwitch",
    "ASTSwitchCase",
    "ASTEnumDecl",
    "ASTEnumMember",
    "ASTBlock",
    "ASTProcedure",
    "ASTSegment",
    "ASTMetrics",
    "ASTProgram",
]
