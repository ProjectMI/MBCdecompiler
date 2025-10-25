"""Data structures for the reconstructed abstract syntax tree."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class ASTAliasInfo:
    """Alias metadata describing how a location can alias with others."""

    kind: ASTAliasKind
    label: Optional[str] = None

    def render(self) -> str:
        if self.label:
            return f"{self.kind.value}({self.label})"
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


@dataclass(frozen=True)
class ASTMemoryAddress:
    """Canonical description of an addressable region."""

    kind: "ASTAddressSpace"
    region: Optional[str] = None
    bank: Optional[int] = None
    page: Optional[int] = None
    page_alias: Optional[str] = None
    page_register: Optional[int] = None
    base_register: Optional[int] = None
    base_offset: Optional[int] = None
    offset: Optional[int] = None
    symbol: Optional[str] = None

    def describe(self) -> List[str]:
        parts: List[str] = [f"type={self.kind.value}"]
        if self.region and self.region != self.kind.value:
            parts.append(f"region={self.region}")
        if self.symbol:
            parts.append(f"symbol={self.symbol}")
        if self.bank is not None:
            parts.append(f"bank=0x{self.bank:04X}")
        if self.page_alias:
            parts.append(f"page={self.page_alias}")
        elif self.page is not None:
            parts.append(f"page=0x{self.page:02X}")
        if self.page_register is not None:
            formatted = _format_operand(self.page_register)
            parts.append(f"page_reg={formatted}")
        if self.base_register is not None:
            parts.append(f"base_reg=0x{self.base_register:04X}")
        if self.base_offset is not None:
            parts.append(f"base=0x{self.base_offset:04X}")
        if self.offset is not None:
            parts.append(f"offset=0x{self.offset:04X}")
        return parts

    def render(self) -> str:
        parts = self.describe()
        inner = ", ".join(parts)
        return f"addr{{{inner}}}" if inner else "addr{}"


def _default_alias() -> ASTAliasInfo:
    return ASTAliasInfo(ASTAliasKind.UNKNOWN)


class ASTAddressSpace(Enum):
    """Canonical address space identifiers used by memory locations."""

    FRAME = "frame"
    GLOBAL = "global"
    CONST = "const"
    BANKED = "banked"
    MEMORY = "mem"
    IO = "io"
    UNKNOWN = "unknown"


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
        canonical = OPERAND_ALIASES.get(value)
        if alias_text == hex_value or alias_text == canonical:
            return alias_text
        return f"{alias_text}({hex_value})"
    return hex_value


# ---------------------------------------------------------------------------
# effect modelling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASTBitField:
    """Typed bitfield used by effect descriptors."""

    width: int
    value: int
    alias: Optional[str] = None

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("bitfield width must be positive")
        if self.value < 0:
            raise ValueError("bitfield value must be non-negative")
        max_value = (1 << self.width) - 1
        if self.value > max_value:
            raise ValueError(
                f"bitfield value 0x{self.value:X} exceeds width {self.width}"
            )

    def render(self) -> str:
        width = self.width
        digits = max(1, (width + 3) // 4)
        value_repr = f"0x{self.value:0{digits}X}"
        alias_repr = _format_operand(self.value, self.alias)
        if alias_repr != value_repr:
            value_repr = alias_repr
        return f"mask[{width}]={value_repr}"


class ASTEffect:
    """Base class for canonicalised side effects."""

    domain_order: ClassVar[int] = 0

    def render(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def order_key(self) -> Tuple[int, ...]:
        """Ordering key enforcing the canonical sequencing of effects."""

        return (self.domain_order,)


class ASTIOOperation(Enum):
    WRITE = "write"
    READ = "read"
    STEP = "step"
    BRIDGE = "bridge"


class ASTHelperOperation(Enum):
    INVOKE = "invoke"
    FANOUT = "fanout"
    MASK_LOW = "mask_low"
    MASK_HIGH = "mask_high"
    FORMAT = "format"
    DISPATCH = "dispatch"
    REDUCE = "reduce"
    WRAPPER = "wrapper"


@dataclass(frozen=True)
class ASTIOEffect(ASTEffect):
    """I/O channel interaction."""

    domain_order: ClassVar[int] = 3
    _order_map: ClassVar[Dict[ASTIOOperation, int]] = {
        ASTIOOperation.WRITE: 0,
        ASTIOOperation.READ: 1,
        ASTIOOperation.STEP: 2,
        ASTIOOperation.BRIDGE: 3,
    }

    operation: ASTIOOperation
    port: str
    mask: Optional[ASTBitField] = None

    def order_key(self) -> Tuple[int, ...]:
        return (
            self.domain_order,
            self._order_map.get(self.operation, len(self._order_map)),
            hash(self.port) & 0xFFFF,
        )

    def render(self) -> str:
        mask_text = ""
        if self.mask is not None:
            mask_text = f", {self.mask.render()}"
        return f"io.{self.operation.value}({self.port}{mask_text})"


class ASTFrameEffect(ASTEffect):
    """Base class for frame level effects."""

    domain_order: ClassVar[int] = 1
    action_order: ClassVar[int] = 0

    def order_key(self) -> Tuple[int, ...]:
        return (self.domain_order, self.action_order)


@dataclass(frozen=True)
class ASTFrameResetEffect(ASTFrameEffect):
    """Reset one or more frame channels using a mask."""

    action_order: ClassVar[int] = 0

    mask: Optional[ASTBitField] = None
    channel: Optional[str] = None

    def order_key(self) -> Tuple[int, ...]:
        return (
            self.domain_order,
            self.action_order,
            self.channel or "",
            self.mask.value if self.mask is not None else -1,
            self.mask.width if self.mask is not None else 0,
        )

    def render(self) -> str:
        parts: List[str] = []
        if self.channel:
            parts.append(f"channel={self.channel}")
        if self.mask is not None:
            parts.append(self.mask.render())
        inner = ", ".join(parts)
        return f"frame.reset({inner})" if inner else "frame.reset()"


@dataclass(frozen=True)
class ASTFrameMaskEffect(ASTFrameEffect):
    """Install a live return mask for the active frame."""

    action_order: ClassVar[int] = 1

    mask: ASTBitField
    channel: Optional[str] = None

    def order_key(self) -> Tuple[int, ...]:
        return (
            self.domain_order,
            self.action_order,
            self.channel or "",
            self.mask.value,
            self.mask.width,
        )

    def render(self) -> str:
        parts: List[str] = [self.mask.render()]
        if self.channel:
            parts.append(f"channel={self.channel}")
        inner = ", ".join(parts)
        return f"frame.mask({inner})"


@dataclass(frozen=True)
class ASTFrameWriteEffect(ASTFrameEffect):
    """Write a structured value into the call frame metadata."""

    action_order: ClassVar[int] = 2

    channel: str
    value: Optional[ASTBitField] = None

    def order_key(self) -> Tuple[int, ...]:
        value_key = -1
        width_key = 0
        if self.value is not None:
            value_key = self.value.value
            width_key = self.value.width
        return (
            self.domain_order,
            self.action_order,
            self.channel,
            value_key,
            width_key,
        )

    def render(self) -> str:
        if self.value is None:
            return f"frame.write(channel={self.channel})"
        return f"frame.write(channel={self.channel}, {self.value.render()})"


@dataclass(frozen=True)
class ASTFrameTeardownEffect(ASTFrameEffect):
    """Tear down the active call frame by popping stack slots."""

    action_order: ClassVar[int] = 3

    pops: int

    def order_key(self) -> Tuple[int, ...]:
        return (self.domain_order, self.action_order, self.pops)

    def render(self) -> str:
        return f"frame.teardown(pops={self.pops})"


@dataclass(frozen=True)
class ASTFrameDropEffect(ASTFrameEffect):
    """Drop transient stack slots after a helper interaction."""

    action_order: ClassVar[int] = 4

    pops: int

    def order_key(self) -> Tuple[int, ...]:
        return (self.domain_order, self.action_order, self.pops)

    def render(self) -> str:
        return f"frame.drop(pops={self.pops})"


@dataclass(frozen=True)
class ASTFrameChannelEffect(ASTFrameEffect):
    """Catch-all channel write used for scheduler/page helpers."""

    action_order: ClassVar[int] = 5

    channel: str
    value: Optional[ASTBitField] = None

    def order_key(self) -> Tuple[int, ...]:
        value_key = -1
        width_key = 0
        if self.value is not None:
            value_key = self.value.value
            width_key = self.value.width
        return (
            self.domain_order,
            self.action_order,
            self.channel,
            value_key,
            width_key,
        )

    def render(self) -> str:
        if self.value is None:
            return f"frame.channel({self.channel})"
        return f"frame.channel({self.channel}, {self.value.render()})"


@dataclass(frozen=True)
class ASTFrameProtocolChannel:
    """Named protocol channel captured during frame teardown."""

    name: str
    mask: Optional[ASTBitField] = None

    def render(self) -> str:
        if self.mask is None:
            return self.name
        return f"{self.name}({self.mask.render()})"


@dataclass(frozen=True)
class ASTFrameProtocolEffect(ASTEffect):
    """Summary of post-call frame protocol actions."""

    domain_order: ClassVar[int] = 0
    masks: Tuple[ASTBitField, ...]
    channels: Tuple["ASTFrameProtocolChannel", ...] = ()
    teardown: int = 0
    drops: int = 0

    def order_key(self) -> Tuple[int, ...]:
        return (
            self.domain_order,
            len(self.masks),
            len(self.channels),
            self.teardown,
            self.drops,
        )

    def render(self) -> str:
        mask_text = ", ".join(mask.render() for mask in self.masks)
        parts = [f"masks=[{mask_text}]" if mask_text else "masks=[]"]
        if self.channels:
            channel_text = ", ".join(channel.render() for channel in self.channels)
            parts.append(f"channels=[{channel_text}]")
        else:
            parts.append("channels=[]")
        if self.teardown:
            parts.append(f"teardown={self.teardown}")
        if self.drops:
            parts.append(f"drops={self.drops}")
        inner = ", ".join(parts)
        return f"frame.protocol({inner})"


@dataclass(frozen=True)
class ASTHelperEffect(ASTEffect):
    """Helper façade interactions used during epilogues."""

    domain_order: ClassVar[int] = 2
    _order_map: ClassVar[Dict[ASTHelperOperation, int]] = {
        ASTHelperOperation.INVOKE: 0,
        ASTHelperOperation.FANOUT: 1,
        ASTHelperOperation.MASK_LOW: 2,
        ASTHelperOperation.MASK_HIGH: 3,
        ASTHelperOperation.FORMAT: 4,
        ASTHelperOperation.DISPATCH: 5,
        ASTHelperOperation.REDUCE: 6,
        ASTHelperOperation.WRAPPER: 7,
    }

    operation: ASTHelperOperation
    target: Optional[int] = None
    symbol: Optional[str] = None
    mask: Optional[ASTBitField] = None

    def order_key(self) -> Tuple[int, ...]:
        return (
            self.domain_order,
            self._order_map.get(self.operation, len(self._order_map)),
            self.target or 0,
        )

    def render(self) -> str:
        parts: List[str] = []
        if self.target is not None:
            target_repr = f"0x{self.target:04X}"
            if self.symbol:
                target_repr = f"{self.symbol}({target_repr})"
            parts.append(f"target={target_repr}")
        elif self.symbol:
            parts.append(f"symbol={self.symbol}")
        if self.mask is not None:
            parts.append(self.mask.render())
        inner = ", ".join(parts)
        return (
            f"helpers.{self.operation.value}({inner})"
            if inner
            else f"helpers.{self.operation.value}()"
        )


def _render_effects(effects: Sequence[ASTEffect]) -> str:
    if not effects:
        return "[]"
    rendered = ", ".join(effect.render() for effect in effects)
    return f"[{rendered}]"


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
class ASTAddressOrigin:
    """Normalised description of the originating address space."""

    space: ASTAddressSpace
    label: Optional[str] = None

    def render(self) -> str:
        mapping = {
            ASTAddressSpace.FRAME: "stack",
            ASTAddressSpace.GLOBAL: "global",
            ASTAddressSpace.CONST: "const",
            ASTAddressSpace.BANKED: "mem",
            ASTAddressSpace.MEMORY: "mem",
            ASTAddressSpace.IO: "io",
        }
        base = mapping.get(self.space, self.space.value)
        if self.label:
            return f"{base}[label={self.label}]"
        return base


@dataclass(frozen=True)
class ASTMemoryLocation:
    """Structured representation of an addressable location."""

    address: Optional[ASTMemoryAddress] = None
    origin: Optional[ASTAddressOrigin] = None
    displacement: Optional["ASTExpression"] = None
    path: Tuple[ASTAccessPathElement, ...] = ()
    alias: ASTAliasInfo = field(default_factory=_default_alias)

    def render(self) -> str:
        parts: List[str] = []
        if self.address is not None:
            parts.extend(self.address.describe())
        if self.origin is not None:
            parts.append(f"origin={self.origin.render()}")
        if self.displacement is not None:
            parts.append(f"delta={self.displacement.render()}")
        if self.alias.kind is not ASTAliasKind.UNKNOWN or self.alias.label:
            parts.append(f"alias={self.alias.render()}")
        inner = ", ".join(parts)
        base_repr = f"addr{{{inner}}}" if inner else "addr{}"
        trail = "".join(element.render() for element in self.path)
        return f"{base_repr}{trail}"


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
class ASTCallExpr(ASTExpression):
    """Call expression with resolved argument operands."""

    target: int
    operands: Tuple[ASTCallOperand, ...]
    symbol: Optional[str] = None
    tail: bool = False
    varargs: bool = False

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.UNKNOWN

    def render(self) -> str:
        rendered_args = ", ".join(operand.render() for operand in self.operands)
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


class ASTCallOperandKind(Enum):
    """Canonical operand kinds used at call sites."""

    VALUE = "value"
    IMMEDIATE = "immediate"
    TRACE = "trace"
    STACK = "stackref"


class ASTCallOperand:
    """Base class for typed call operands."""

    kind: ClassVar[ASTCallOperandKind] = ASTCallOperandKind.VALUE

    def render(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_expression(self) -> ASTExpression:
        raise NotImplementedError

    def is_resolved(self) -> bool:
        return not isinstance(self.to_expression(), ASTUnknown)


@dataclass(frozen=True)
class ASTImmediateOperand(ASTCallOperand):
    """Immediate argument resolved to a literal value."""

    token: str
    literal: ASTIntegerLiteral

    kind: ClassVar[ASTCallOperandKind] = ASTCallOperandKind.IMMEDIATE

    def render(self) -> str:
        return f"imm({self.literal.render()})"

    def to_expression(self) -> ASTExpression:
        return self.literal


@dataclass(frozen=True)
class ASTTraceOperand(ASTCallOperand):
    """Operand sourced from a trace reference."""

    token: str
    label: str
    offset: int

    kind: ClassVar[ASTCallOperandKind] = ASTCallOperandKind.TRACE

    def render(self) -> str:
        return f"trace(label={self.label}, offset=0x{self.offset:06X})"

    def to_expression(self) -> ASTExpression:
        return ASTIdentifier(self.token, SSAValueKind.IDENTIFIER)


@dataclass(frozen=True)
class ASTStackOperand(ASTCallOperand):
    """Operand referencing a slot in a typed stack space."""

    token: str
    location: Optional[ASTMemoryLocation] = None
    label: Optional[str] = None
    value_kind: SSAValueKind = SSAValueKind.UNKNOWN

    kind: ClassVar[ASTCallOperandKind] = ASTCallOperandKind.STACK

    def render(self) -> str:
        if self.location is not None:
            return f"stackref({self.location.render()})"
        label = self.label or self.token
        return f"stackref({label})"

    def to_expression(self) -> ASTExpression:
        if self.location is not None:
            return ASTMemoryRead(location=self.location, value_kind=self.value_kind)
        label = self.label or self.token
        return ASTIdentifier(label, self.value_kind)


@dataclass(frozen=True)
class ASTValueOperand(ASTCallOperand):
    """Operand backed by an arbitrary AST expression."""

    token: str
    value: ASTExpression

    kind: ClassVar[ASTCallOperandKind] = ASTCallOperandKind.VALUE

    def render(self) -> str:
        return f"value({self.value.render()})"

    def to_expression(self) -> ASTExpression:
        return self.value


@dataclass(frozen=True)
class ASTCallArgumentSlot:
    """Assignment of a value to a call frame argument slot."""

    index: int
    operand: ASTCallOperand

    def render(self) -> str:
        return f"arg[{self.index}]={self.operand.render()}"


@dataclass(frozen=True)
class ASTCallReturnSlot:
    """Description of a returned value slot for a call."""

    index: int
    kind: SSAValueKind = SSAValueKind.UNKNOWN
    name: Optional[str] = None

    def render(self) -> str:
        prefix = self.kind.name.lower()
        label = self.name or f"{prefix}{self.index}"
        return f"ret[{self.index}]={label}:{prefix}"


@dataclass(frozen=True)
class ASTCallABI:
    """Canonical calling convention metadata."""

    slots: Tuple[ASTCallArgumentSlot, ...] = ()
    returns: Tuple[ASTCallReturnSlot, ...] = ()
    effects: Tuple[ASTEffect, ...] = ()
    live_mask: Optional[ASTBitField] = None
    tail: bool = False

    def __post_init__(self) -> None:
        if self.slots:
            ordered_slots = tuple(sorted(self.slots, key=lambda slot: slot.index))
            object.__setattr__(self, "slots", ordered_slots)
        if self.returns:
            ordered_returns = tuple(sorted(self.returns, key=lambda slot: slot.index))
            object.__setattr__(self, "returns", ordered_returns)
        if self.effects:
            ordered_effects = tuple(
                sorted(self.effects, key=lambda effect: effect.order_key())
            )
            object.__setattr__(self, "effects", ordered_effects)

    def render(self) -> str:
        parts: List[str] = []
        if self.slots:
            rendered_slots = ", ".join(slot.render() for slot in self.slots)
            parts.append(f"slots=[{rendered_slots}]")
        if self.returns:
            rendered_returns = ", ".join(slot.render() for slot in self.returns)
            parts.append(f"returns=[{rendered_returns}]")
        if self.effects:
            parts.append(f"effects={_render_effects(self.effects)}")
        if self.live_mask is not None:
            parts.append(f"return_mask={self.live_mask.render()}")
        if self.tail:
            parts.append("tail=true")
        inner = ", ".join(parts)
        return f"abi{{{inner}}}" if inner else "abi{}"


class ASTSymbolTypeFamily(Enum):
    """High-level taxonomy used by :class:`ASTSymbolSignature` entries."""

    VALUE = "value"
    ADDRESS = "address"
    TOKEN = "token"
    FLAG = "flag"
    OPAQUE = "opaque"
    EFFECT = "effect"


@dataclass(frozen=True)
class ASTSymbolType:
    """Structured type descriptor recorded in symbol table entries."""

    family: ASTSymbolTypeFamily
    width: Optional[int] = None
    sign: Optional[ASTNumericSign] = None
    space: Optional[str] = None

    def render(self) -> str:
        if self.family is ASTSymbolTypeFamily.VALUE:
            sign = self.sign.value if self.sign is not None else "any"
            width = self.width or 16
            return f"value<{sign},{width}>"
        if self.family is ASTSymbolTypeFamily.ADDRESS:
            width = self.width or 16
            space = self.space or "mem"
            return f"address<{space},{width}>"
        if self.family is ASTSymbolTypeFamily.TOKEN:
            return "token"
        if self.family is ASTSymbolTypeFamily.FLAG:
            return "flag"
        if self.family is ASTSymbolTypeFamily.EFFECT:
            return "effect"
        return "opaque"


@dataclass(frozen=True)
class ASTSignatureValue:
    """Single argument or return slot recorded in a symbol signature."""

    index: int
    type: ASTSymbolType
    name: Optional[str] = None

    def render(self) -> str:
        label = self.name or f"{self.type.family.value}{self.index}"
        return f"{label}:{self.type.render()}"


@dataclass(frozen=True)
class ASTSymbolSignature:
    """Synthesised call signature describing a known symbol."""

    address: int
    name: str
    arguments: Tuple[ASTSignatureValue, ...] = ()
    returns: Tuple[ASTSignatureValue, ...] = ()
    calling_conventions: Tuple[str, ...] = ()
    attributes: Tuple[str, ...] = ()
    effects: Tuple[str, ...] = ()

    def render(self) -> str:
        args = ", ".join(value.render() for value in self.arguments) or "-"
        if not self.returns:
            rets = "-"
        elif len(self.returns) == 1:
            rets = self.returns[0].render()
        else:
            rendered = ", ".join(value.render() for value in self.returns)
            rets = f"tuple({rendered})"
        conventions = ", ".join(self.calling_conventions) or "call"
        attributes = ", ".join(self.attributes)
        effects = ", ".join(self.effects)
        parts = [
            f"args=[{args}]",
            f"returns=[{rets}]",
            f"cc=[{conventions}]",
        ]
        if attributes:
            parts.append(f"attrs=[{attributes}]")
        if effects:
            parts.append(f"effects=[{effects}]")
        inner = " ".join(parts)
        return f"symbol {self.name}(0x{self.address:04X}) {inner}"


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
    abi: Optional[ASTCallABI] = None

    def render(self) -> str:
        call_repr = self.call.render()
        if self.returns:
            rendered = ", ".join(identifier.render() for identifier in self.returns)
            call_repr = f"{call_repr} -> ({rendered})"
        if self.abi is not None:
            return f"{call_repr} {self.abi.render()}"
        return call_repr


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
    mask: Optional[ASTBitField] = None

    effect_category: ClassVar[ASTEffectCategory] = ASTEffectCategory.IO

    def render(self) -> str:
        mask = "" if self.mask is None else f", {self.mask.render()}"
        return f"io.write({self.port}{mask})"


@dataclass
class ASTTerminator(ASTStatement):
    """Base class for normalised block terminators."""


@dataclass(frozen=True)
class ASTReturnPayload:
    """Structured representation of return values."""

    values: Tuple[ASTExpression, ...] = ()
    varargs: bool = False

    def render(self) -> str:
        if self.varargs:
            if not self.values:
                return "varargs"
            inner = ", ".join(value.render() for value in self.values)
            return f"varargs({inner})"
        if not self.values:
            return "()"
        if len(self.values) == 1:
            return self.values[0].render()
        inner = ", ".join(value.render() for value in self.values)
        return f"({inner})"


@dataclass
class ASTReturn(ASTTerminator):
    """Return from the current procedure."""

    payload: ASTReturnPayload
    effects: Tuple[ASTEffect, ...] = ()

    def __post_init__(self) -> None:
        if self.effects:
            self.effects = tuple(sorted(self.effects, key=lambda eff: eff.order_key()))

    def render(self) -> str:
        return f"return {self.payload.render()} effects={_render_effects(self.effects)}"


@dataclass
class ASTTailCall(ASTTerminator):
    """Tail call used as a return."""

    call: ASTCallExpr
    payload: ASTReturnPayload = field(default_factory=ASTReturnPayload)
    abi: Optional[ASTCallABI] = None
    effects: Tuple[ASTEffect, ...] = ()

    def __post_init__(self) -> None:
        if self.effects:
            self.effects = tuple(sorted(self.effects, key=lambda eff: eff.order_key()))

    def render(self) -> str:
        call_repr = self.call.render()
        if call_repr.startswith("tail "):
            call_repr = call_repr[len("tail ") :]
        result = f"tail {call_repr}"
        payload_repr = self.payload.render()
        if payload_repr != "()":
            result = f"{result} -> {payload_repr}"
        if self.abi is not None:
            result = f"{result} {self.abi.render()}"
        return f"{result} effects={_render_effects(self.effects)}"


@dataclass
class ASTJump(ASTTerminator):
    """Unconditional branch to another basic block."""

    target: "ASTBlock | None" = None
    target_offset: int | None = None
    hint: str | None = None

    def render(self) -> str:
        if self.target is not None:
            destination = self.target.label
        elif self.hint is not None:
            destination = self.hint
        elif self.target_offset is not None:
            destination = f"0x{self.target_offset:04X}"
        else:
            destination = "?"
        return f"jump {destination}"


@dataclass
class ASTBranch(ASTTerminator):
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
class ASTTestSet(ASTTerminator):
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
class ASTFlagCheck(ASTTerminator):
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
class ASTFunctionPrologue(ASTTerminator):
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


@dataclass(frozen=True)
class ASTDispatchHelper:
    """Metadata about the helper responsible for dispatch operations."""

    address: int
    symbol: Optional[str] = None

    def render(self) -> str:
        helper_repr = f"0x{self.address:04X}"
        if self.symbol:
            helper_repr = f"{self.symbol}({helper_repr})"
        return helper_repr


@dataclass(frozen=True)
class ASTDispatchIndex:
    """Description of the index expression used by a dispatch."""

    expression: ASTExpression | None = None
    mask: Optional[int] = None
    base: Optional[int] = None

    def render_expression(self) -> str:
        return self.expression.render() if self.expression is not None else "?"


@dataclass
class ASTSwitch(ASTStatement):
    """Normalised jump table reconstructed from helper dispatch patterns."""

    call: ASTCallExpr | None
    cases: Tuple[ASTSwitchCase, ...]
    index: ASTDispatchIndex = field(default_factory=ASTDispatchIndex)
    helper: Optional[ASTDispatchHelper] = None
    default: int | None = None
    kind: str | None = None
    enum_name: str | None = None
    abi: Optional["ASTCallABI"] = None

    def render(self) -> str:
        parts: List[str] = [f"index={self.index.render_expression()}"]
        if self.index.mask is not None:
            parts.append(f"mask=0x{self.index.mask:04X}")
        if self.index.base is not None:
            parts.append(f"base=0x{self.index.base:04X}")
        rendered_cases = ", ".join(case.render() for case in self.cases)
        parts.append(f"table=[{rendered_cases}]")
        if self.default is not None:
            parts.append(f"default=0x{self.default:04X}")
        if self.enum_name:
            parts.append(f"enum={self.enum_name}")
        if self.helper is not None:
            parts.append(f"helper={self.helper.render()}")
        if self.kind:
            parts.append(f"kind={self.kind}")
        if self.abi is not None:
            parts.append(f"abi={self.abi.render()}")
        return f"Switch{{{', '.join(parts)}}}"


@dataclass
class ASTBlock:
    """Block of sequential statements with outgoing CFG edges."""

    label: str
    start_offset: int
    body: Tuple[ASTStatement, ...]
    terminator: ASTTerminator
    successors: Tuple["ASTBlock", ...] = field(default_factory=tuple)
    predecessors: Tuple["ASTBlock", ...] = field(default_factory=tuple)

    @property
    def statements(self) -> Tuple[ASTStatement, ...]:
        return self.body + (self.terminator,)

    @statements.setter
    def statements(self, value: Tuple[ASTStatement, ...]) -> None:
        if not value:
            raise ValueError("block statements must include a terminator")
        terminator = value[-1]
        if not isinstance(terminator, ASTTerminator):
            raise ValueError("block terminator must be an ASTTerminator")
        self.body = tuple(value[:-1])
        self.terminator = terminator


@dataclass(frozen=True)
class ASTEntryReason:
    """Structured reason used to explain procedure entry points."""

    kind: str
    detail: Optional[str] = None

    def render(self) -> str:
        return f"{self.kind}({self.detail})" if self.detail else self.kind


@dataclass(frozen=True)
class ASTEntryPoint:
    """Canonical description of a procedure entry block."""

    label: str
    offset: int
    reasons: Tuple[ASTEntryReason, ...]

    def render(self) -> str:
        reason_text = ", ".join(reason.render() for reason in self.reasons)
        return f"label={self.label}, offset=0x{self.offset:04X}, reasons=[{reason_text}]"


@dataclass(frozen=True)
class ASTExitReason:
    """Explanation for why a particular block is considered an exit."""

    kind: str

    def render(self) -> str:
        return self.kind


@dataclass(frozen=True)
class ASTExitPoint:
    """Summary of a reconstructed exit block."""

    label: str
    offset: int
    reasons: Tuple[ASTExitReason, ...] = ()

    def render(self) -> str:
        reason_text = ", ".join(reason.render() for reason in self.reasons)
        suffix = f" {{{reason_text}}}" if reason_text else ""
        return f"{self.label}@0x{self.offset:04X}{suffix}"


class ASTProcedureResultKind(Enum):
    """Categorisation of the reconstructed procedure result layout."""

    VOID = "void"
    FIXED = "fixed"
    SPARSE = "sparse"
    VARIADIC = "variadic"


@dataclass(frozen=True)
class ASTProcedureResultSlot:
    """Typed description of a single returned value slot."""

    index: int
    type: ASTSymbolType
    required: bool = True

    def render(self) -> str:
        flavour = "required" if self.required else "optional"
        return f"{flavour}[{self.index}]={self.type.render()}"


@dataclass(frozen=True)
class ASTProcedureResult:
    """Summary of the result slots produced by a procedure."""

    kind: ASTProcedureResultKind
    required_slots: Tuple[int, ...] = ()
    optional_slots: Tuple[int, ...] = ()
    slots: Tuple[ASTProcedureResultSlot, ...] = ()
    varargs: bool = False
    vararg_type: Optional[ASTSymbolType] = None

    def render(self) -> str:
        parts: List[str] = [f"kind={self.kind.value}"]
        include_slot_lists = not self.slots
        if include_slot_lists and self.required_slots:
            required = ", ".join(str(index) for index in self.required_slots)
            parts.append(f"required=[{required}]")
        if include_slot_lists and self.optional_slots:
            optional = ", ".join(str(index) for index in self.optional_slots)
            parts.append(f"optional=[{optional}]")
        if self.slots:
            slot_desc = ", ".join(slot.render() for slot in self.slots)
            parts.append(f"slots=[{slot_desc}]")
        if self.varargs:
            parts.append("varargs=true")
        if self.vararg_type is not None:
            parts.append(f"vararg_type={self.vararg_type.render()}")
        inner = ", ".join(parts)
        return f"result{{{inner}}}"


@dataclass(frozen=True)
class ASTProcedureAlias:
    """Alias pointing at an equivalent procedure located elsewhere."""

    segment: int
    offset: int

    def render(self) -> str:
        return f"seg={self.segment} offset=0x{self.offset:04X}"


@dataclass
class ASTProcedure:
    """Single reconstructed procedure."""

    name: str
    blocks: Tuple[ASTBlock, ...]
    entry: ASTEntryPoint
    exits: Tuple[ASTExitPoint, ...]
    result: "ASTProcedureResult"
    successor_map: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    predecessor_map: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    aliases: Tuple[ASTProcedureAlias, ...] = field(default_factory=tuple)

    @property
    def entry_offset(self) -> int:
        return self.entry.offset


@dataclass
class ASTSegment:
    """AST view of a container segment."""

    index: int
    start: int
    length: int
    procedures: Tuple[ASTProcedure, ...]
    enums: Tuple[ASTEnumDecl, ...] = ()
    kind: str = "code"


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
    symbols: Tuple[ASTSymbolSignature, ...] = ()


__all__ = [
    "ASTExpression",
    "ASTUnknown",
    "ASTIntegerLiteral",
    "ASTRealLiteral",
    "ASTBooleanLiteral",
    "ASTStringLiteral",
    "ASTBytesLiteral",
    "ASTIdentifier",
    "ASTSafeCastExpr",
    "ASTMemoryLocation",
    "ASTAddressOrigin",
    "ASTMemoryAddress",
    "ASTAccessPathElement",
    "ASTFieldElement",
    "ASTIndexElement",
    "ASTSliceElement",
    "ASTAliasInfo",
    "ASTAliasKind",
    "ASTNumericSign",
    "ASTConversionKind",
    "ASTEffectCategory",
    "ASTAddressSpace",
    "ASTBitField",
    "ASTEffect",
    "ASTMemoryRead",
    "ASTCallOperandKind",
    "ASTCallOperand",
    "ASTImmediateOperand",
    "ASTTraceOperand",
    "ASTStackOperand",
    "ASTValueOperand",
    "ASTCallExpr",
    "ASTCallResult",
    "ASTTupleExpr",
    "ASTStatement",
    "ASTAssign",
    "ASTMemoryWrite",
    "ASTIOEffect",
    "ASTFrameEffect",
    "ASTFrameMaskEffect",
    "ASTFrameResetEffect",
    "ASTFrameWriteEffect",
    "ASTFrameTeardownEffect",
    "ASTFrameDropEffect",
    "ASTFrameChannelEffect",
    "ASTFrameProtocolChannel",
    "ASTFrameProtocolEffect",
    "ASTHelperEffect",
    "ASTCallArgumentSlot",
    "ASTCallReturnSlot",
    "ASTCallABI",
    "ASTCallStatement",
    "ASTSymbolTypeFamily",
    "ASTSymbolType",
    "ASTSignatureValue",
    "ASTSymbolSignature",
    "ASTIORead",
    "ASTIOWrite",
    "ASTTerminator",
    "ASTTailCall",
    "ASTJump",
    "ASTReturn",
    "ASTReturnPayload",
    "ASTBranch",
    "ASTTestSet",
    "ASTFlagCheck",
    "ASTFunctionPrologue",
    "ASTComment",
    "ASTEnumDecl",
    "ASTEnumMember",
    "ASTBlock",
    "ASTEntryReason",
    "ASTEntryPoint",
    "ASTExitReason",
    "ASTExitPoint",
    "ASTProcedureResultKind",
    "ASTProcedureResult",
    "ASTProcedureResultSlot",
    "ASTProcedureAlias",
    "ASTProcedure",
    "ASTSwitchCase",
    "ASTDispatchHelper",
    "ASTDispatchIndex",
    "ASTSwitch",
    "ASTSegment",
    "ASTMetrics",
    "ASTProgram",
]
