"""Dataclasses describing the normalised intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from ..constants import IO_PORT_NAME, OPERAND_ALIASES


class MemSpace(Enum):
    """High level classification of indirect memory accesses."""

    FRAME = auto()
    GLOBAL = auto()
    CONST = auto()


class SSAValueKind(Enum):
    """Lightweight annotation attached to SSA values."""

    UNKNOWN = auto()
    BYTE = auto()
    WORD = auto()
    POINTER = auto()
    IO = auto()
    PAGE_REGISTER = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()


@dataclass(frozen=True)
class IRSlot:
    """Address of a VM slot used by :class:`IRLoad` and :class:`IRStore`."""

    space: MemSpace
    index: int


@dataclass(frozen=True)
class MemRef:
    """Symbolic description of an indirect memory reference."""

    region: str
    bank: Optional[int] = None
    base: Optional[int] = None
    page: Optional[int] = None
    offset: Optional[int] = None
    symbol: Optional[str] = None
    page_alias: Optional[str] = None

    def describe(self) -> str:
        region = self.page_alias or self.region or "mem"
        prefix = region
        if not prefix.startswith(("mem", "io")) and prefix not in {"frame", "global", "const"}:
            prefix = f"mem.{prefix}"

        details = []
        location = self._format_location()
        if location:
            details.append(location)
        if self.base is not None:
            details.append(f"base=0x{self.base:04X}")
        if self.bank is not None and self.page_alias is None and not self._has_region_alias():
            details.append(f"bank=0x{self.bank:04X}")

        rendered = prefix if not details else f"{prefix}[{', '.join(details)}]"
        return rendered

    def _format_location(self) -> Optional[str]:
        if self.page is not None and self.offset is not None:
            if self.page_alias:
                combined = (self.page << 8) | (self.offset & 0xFF)
                return f"0x{combined:04X}"
            return f"0x{self.page:02X}:0x{self.offset:02X}"
        if self.offset is not None:
            width = 4 if self.offset > 0xFF else 2
            return f"0x{self.offset:0{width}X}"
        if self.page is not None:
            return f"page=0x{self.page:02X}"
        return None

    def _has_region_alias(self) -> bool:
        return bool(self.region and not self.region.startswith("mem"))


def _render_ascii(data: bytes) -> str:
    printable: List[str] = []
    for byte in data:
        if 0x20 <= byte <= 0x7E:
            printable.append(chr(byte))
        elif byte in {0x09, 0x0A, 0x0D}:
            printable.append({0x09: "\\t", 0x0A: "\\n", 0x0D: "\\r"}[byte])
        else:
            printable.append(f"\\x{byte:02x}")
    return "".join(printable)


@dataclass(frozen=True)
class IRNode:
    """Base class for IR nodes.

    The class only exists to make type signatures more explicit.  Subclasses do
    not rely on runtime inheritance checks so the dataclasses can remain frozen
    which keeps the structures hashable and easy to compare in tests.
    """


@dataclass(frozen=True)
class CallPredicate:
    """Description of the control-flow predicate derived from a call result."""

    kind: str
    var: Optional[str] = None
    expr: Optional[str] = None
    then_target: Optional[int] = None
    else_target: Optional[int] = None
    flag: Optional[int] = None

    def describe(self) -> str:
        if self.kind == "testset":
            var = self.var or "var"
            expr = self.expr or "expr"
            then_target = f"0x{self.then_target:04X}" if self.then_target is not None else "?"
            else_target = f"0x{self.else_target:04X}" if self.else_target is not None else "?"
            return f"testset {var}={expr} then={then_target} else={else_target}"
        if self.kind == "flag":
            flag = f"0x{self.flag:04X}" if self.flag is not None else "?"
            then_target = f"0x{self.then_target:04X}" if self.then_target is not None else "?"
            else_target = f"0x{self.else_target:04X}" if self.else_target is not None else "?"
            return f"flag {flag} then={then_target} else={else_target}"
        then_target = f"0x{self.then_target:04X}" if self.then_target is not None else "?"
        else_target = f"0x{self.else_target:04X}" if self.else_target is not None else "?"
        return f"{self.kind} then={then_target} else={else_target}"


@dataclass(frozen=True)
class IRCall(IRNode):
    """Invocation of another routine."""

    target: int
    args: Tuple[str, ...]
    tail: bool = False
    arity: Optional[int] = None
    convention: Optional[IRStackEffect] = None
    cleanup_mask: Optional[int] = None
    cleanup: Tuple[IRStackEffect, ...] = field(default_factory=tuple)
    symbol: Optional[str] = None
    predicate: Optional[CallPredicate] = None

    def describe(self) -> str:
        suffix = " tail" if self.tail else ""
        args = ", ".join(self.args)
        target_repr = f"0x{self.target:04X}"
        if self.symbol:
            target_repr = f"{self.symbol}({target_repr})"
        details = []
        if self.arity is not None:
            details.append(f"arity={self.arity}")
        if self.convention is not None:
            details.append(f"convention={self.convention.describe()}")
        if self.cleanup_mask is not None:
            details.append(f"mask={_format_operand(self.cleanup_mask)}")
        if self.cleanup:
            rendered = ", ".join(step.describe() for step in self.cleanup)
            details.append(f"cleanup=[{rendered}]")
        if self.predicate is not None:
            details.append(f"predicate={self.predicate.describe()}")
        extra = f" {' '.join(details)}" if details else ""
        return f"call{suffix} target={target_repr} args=[{args}]{extra}"


@dataclass(frozen=True)
class IRTailCall(IRNode):
    """Tail call that forwards control to ``call`` and returns its values."""

    call: IRCall
    returns: Tuple[str, ...]
    varargs: bool = False
    cleanup: Tuple[IRStackEffect, ...] = field(default_factory=tuple)
    cleanup_mask: Optional[int] = None

    @property
    def target(self) -> int:
        return self.call.target

    @property
    def args(self) -> Tuple[str, ...]:
        return self.call.args

    @property
    def tail(self) -> bool:
        return True

    @property
    def arity(self) -> Optional[int]:
        return self.call.arity

    @property
    def convention(self) -> Optional[IRStackEffect]:
        return self.call.convention

    @property
    def symbol(self) -> Optional[str]:
        return self.call.symbol

    @property
    def predicate(self) -> Optional[CallPredicate]:
        return self.call.predicate

    def describe(self) -> str:
        call_repr = self.call.describe()
        details: List[str] = []
        if self.returns:
            rendered = ", ".join(self.returns)
            details.append(f"returns=[{rendered}]")
        if self.varargs:
            details.append("varargs")
        if self.cleanup:
            rendered = ", ".join(step.describe() for step in self.cleanup)
            details.append(f"cleanup=[{rendered}]")
        if self.cleanup_mask is not None:
            details.append(f"mask={_format_operand(self.cleanup_mask)}")
        suffix = "" if not details else " " + " ".join(details)
        return f"tailcall {call_repr}{suffix}"


@dataclass(frozen=True)
class IRStackEffect:
    """Canonical representation of stack shuffles and teardowns."""

    mnemonic: str
    operand: int = 0
    pops: int = 0
    operand_role: Optional[str] = None
    operand_alias: Optional[str] = None

    def describe(self) -> str:
        details = []
        if self.pops:
            details.append(f"pop={self.pops}")
        include_operand = bool(self.operand_role or self.operand_alias)
        if not include_operand:
            include_operand = bool(self.operand) or self.mnemonic not in {"stack_teardown"}
        if include_operand:
            hex_value = f"0x{self.operand:04X}"
            alias = self.operand_alias
            if alias:
                alias_text = str(alias)
                if alias_text.upper().startswith("0X"):
                    alias_text = alias_text.upper()
                value = alias_text if alias_text == hex_value else f"{alias_text}({hex_value})"
            else:
                value = hex_value
            if self.operand_role:
                details.append(f"{self.operand_role}={value}")
            else:
                details.append(f"operand={value}")
        if not details:
            return self.mnemonic
        inner = ", ".join(details)
        return f"{self.mnemonic}({inner})"


@dataclass(frozen=True)
class IRReturn(IRNode):
    """Return from the current routine."""

    values: Tuple[str, ...]
    varargs: bool = False
    cleanup: Tuple[IRStackEffect, ...] = field(default_factory=tuple)
    mask: Optional[int] = None

    def describe(self) -> str:
        cleanup = ""
        if self.cleanup:
            rendered = ", ".join(step.describe() for step in self.cleanup)
            cleanup = f" cleanup=[{rendered}]"
        if self.mask is not None:
            mask_repr = _format_operand(self.mask)
            cleanup = f" mask={mask_repr}{cleanup}"
        if self.varargs:
            if self.values:
                return f"return varargs({', '.join(self.values)}){cleanup}"
            return f"return varargs{cleanup}"
        values = ", ".join(self.values)
        return f"return [{values}]{cleanup}"


@dataclass(frozen=True)
class IRLiteral(IRNode):
    """Push a literal value onto the VM stack."""

    value: int
    mode: int
    source: str
    annotations: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        return f"lit(0x{self.value:04X})"


@dataclass(frozen=True)
class IRLiteralChunk(IRNode):
    """Inline ASCII chunk embedded directly in the code stream."""

    data: bytes
    source: str
    annotations: Tuple[str, ...] = field(default_factory=tuple)
    symbol: Optional[str] = None

    def describe(self) -> str:
        if self.symbol:
            rendered = f"str({self.symbol})"
            if self.annotations:
                rendered += " " + ", ".join(self.annotations)
            return rendered

        note = f"ascii({_render_ascii(self.data)})"
        if self.annotations:
            note += " " + ", ".join(self.annotations)
        return note


@dataclass(frozen=True)
class IRStringConstant(IRNode):
    """Entry in the global ASCII constant pool."""

    name: str
    data: bytes
    segments: Tuple[bytes, ...]
    source: str

    def describe(self) -> str:
        if len(self.segments) == 1:
            body = _render_ascii(self.segments[0])
            payload = f"ascii({body})"
        else:
            parts = ", ".join(f"ascii({_render_ascii(segment)})" for segment in self.segments)
            payload = f"[{parts}]"
        return f"const {self.name} = {payload}"


@dataclass(frozen=True)
class IRBuildArray(IRNode):
    """Aggregate literal values into a positional container."""

    elements: Tuple[str, ...]

    def describe(self) -> str:
        values = ", ".join(self.elements)
        return f"array([{values}])"


@dataclass(frozen=True)
class IRBuildMap(IRNode):
    """Aggregate literal values into a key/value container."""

    entries: Tuple[Tuple[str, str], ...]

    def describe(self) -> str:
        pairs = ", ".join(f"{key}:{value}" for key, value in self.entries)
        return f"map([{pairs}])"


@dataclass(frozen=True)
class IRBuildTuple(IRNode):
    """Aggregate literal values into an immutable tuple."""

    elements: Tuple[str, ...]

    def describe(self) -> str:
        values = ", ".join(self.elements)
        return f"tuple([{values}])"


@dataclass(frozen=True)
class IRLiteralBlock(IRNode):
    """Canonical representation of literal marker chains."""

    triplets: Tuple[Tuple[int, int, int], ...]
    reducer: Optional[str] = None
    reducer_operand: Optional[int] = None
    tail: Tuple[int, ...] = tuple()
    annotations: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        chunks = []
        for a, b, c in self.triplets:
            chunks.append(f"(0x{a:04X}, 0x{b:04X}, 0x{c:04X})")
        base = "literal_block[" + ", ".join(chunks) + "]"
        if self.tail:
            tail_repr = ", ".join(f"0x{value:04X}" for value in self.tail)
            base += f" tail=[{tail_repr}]"
        if self.reducer:
            operand = (
                f" 0x{self.reducer_operand:04X}" if self.reducer_operand is not None else ""
            )
            base += f" via {self.reducer}{operand}"
        if self.annotations:
            base += " " + ", ".join(self.annotations)
        return base


@dataclass(frozen=True)
class IRIf(IRNode):
    """Standard conditional branch."""

    condition: str
    then_target: int
    else_target: int

    def describe(self) -> str:
        return (
            f"if cond={self.condition} then=0x{self.then_target:04X} "
            f"else=0x{self.else_target:04X}"
        )


@dataclass(frozen=True)
class IRTestSetBranch(IRNode):
    """Branch that stores the predicate before testing it."""

    var: str
    expr: str
    then_target: int
    else_target: int

    def describe(self) -> str:
        return (
            f"testset {self.var}={self.expr} then=0x{self.then_target:04X} "
            f"else=0x{self.else_target:04X}"
        )


@dataclass(frozen=True)
class IRFlagCheck(IRNode):
    """Specialised branch that tests one of the global VM flags."""

    flag: int
    then_target: int
    else_target: int

    def describe(self) -> str:
        return (
            f"check_flag flag=0x{self.flag:04X} then=0x{self.then_target:04X} "
            f"else=0x{self.else_target:04X}"
        )


@dataclass(frozen=True)
class IRFunctionPrologue(IRNode):
    """Grouped representation for the standard Lua-like function prologue."""

    var: str
    expr: str
    then_target: int
    else_target: int

    def describe(self) -> str:
        return (
            f"function_prologue {self.var}={self.expr} "
            f"then=0x{self.then_target:04X} else=0x{self.else_target:04X}"
        )


@dataclass(frozen=True)
class IRLoad(IRNode):
    """Load a value from a VM slot."""

    slot: IRSlot
    target: str = "stack"

    def describe(self) -> str:
        return f"load {self.slot.space.name.lower()}[{self.slot.index}] -> {self.target}"


@dataclass(frozen=True)
class IRStore(IRNode):
    """Store a value into a VM slot."""

    slot: IRSlot
    value: str = "stack"

    def describe(self) -> str:
        return f"store {self.value} -> {self.slot.space.name.lower()}[{self.slot.index}]"


@dataclass(frozen=True)
class IRIndirectLoad(IRNode):
    """Read a value through an indirect slot pointer."""

    base: str
    offset: int
    target: str
    base_slot: Optional[IRSlot] = None
    ref: Optional[MemRef] = None
    pointer: Optional[str] = None
    offset_source: Optional[str] = None

    def describe(self) -> str:
        if self.ref is not None:
            ptr = self.pointer or self.base
            offset = self.offset_source or f"0x{self.offset:04X}"
            return f"load {self.ref.describe()} ptr={ptr} offset={offset} -> {self.target}"

        prefix = self.base
        if self.base_slot is not None:
            slot = f"{self.base_slot.space.name.lower()}[0x{self.base_slot.index:04X}]"
            if self.base:
                prefix = f"{slot} ({self.base})"
            else:
                prefix = slot
        ptr = self.pointer or prefix
        offset = self.offset_source or f"0x{self.offset:04X}"
        return f"indirect_load ptr={ptr} offset={offset} -> {self.target}"


@dataclass(frozen=True)
class IRIndirectStore(IRNode):
    """Write a value through an indirect slot pointer."""

    base: str
    value: str
    offset: int
    base_slot: Optional[IRSlot] = None
    ref: Optional[MemRef] = None
    pointer: Optional[str] = None
    offset_source: Optional[str] = None

    def describe(self) -> str:
        if self.ref is not None:
            ptr = self.pointer or self.base
            offset = self.offset_source or f"0x{self.offset:04X}"
            return f"store {self.value} -> {self.ref.describe()} ptr={ptr} offset={offset}"

        prefix = self.base
        if self.base_slot is not None:
            slot = f"{self.base_slot.space.name.lower()}[0x{self.base_slot.index:04X}]"
            if self.base:
                prefix = f"{slot} ({self.base})"
            else:
                prefix = slot
        ptr = self.pointer or prefix
        offset = self.offset_source or f"0x{self.offset:04X}"
        return f"indirect_store {self.value} -> ptr={ptr} offset={offset}"


@dataclass(frozen=True)
class IRIORead(IRNode):
    """Read a value from the shared IO port."""

    port: str = IO_PORT_NAME

    def describe(self) -> str:
        if self.port != IO_PORT_NAME:
            return f"io.read(port={self.port})"
        return "io.read()"


@dataclass(frozen=True)
class IRIOWrite(IRNode):
    """Write configuration bits to the shared IO port."""

    mask: Optional[int] = None
    port: str = IO_PORT_NAME

    def describe(self) -> str:
        details = [f"port={self.port}"]
        if self.mask is not None:
            details.append(f"mask=0x{self.mask:04X}")
        if details:
            return f"io.write({', '.join(details)})"
        return "io.write()"


@dataclass(frozen=True)
class IRStackDuplicate(IRNode):
    """Duplicate the value currently at the top of the VM stack."""

    value: str
    copies: int = 2

    def describe(self) -> str:
        if self.copies <= 1:
            return f"dup {self.value}"
        return f"dup {self.value} -> copies={self.copies}"


@dataclass(frozen=True)
class IRAsciiPreamble(IRNode):
    """Helper node that replaces the common ASCII initialisation prologue."""

    loader_operand: int
    mode_operand: int
    shuffle_operand: int

    def describe(self) -> str:
        return (
            f"ascii_preamble load=0x{self.loader_operand:04X} "
            f"mode=0x{self.mode_operand:04X} shuffle=0x{self.shuffle_operand:04X}"
        )


@dataclass(frozen=True)
class IRCallPreparation(IRNode):
    """Grouped stack permutations that prepare arguments for helper calls."""

    steps: Tuple[Tuple[str, int], ...]

    def describe(self) -> str:
        rendered = ", ".join(f"{mnemonic}(0x{operand:04X})" for mnemonic, operand in self.steps)
        return f"prep_call_args[{rendered}]"


@dataclass(frozen=True)
class IRCallCleanup(IRNode):
    """Helper sequence that discharges temporary call frames or return values."""

    steps: Tuple[IRStackEffect, ...]
    pops: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pops", sum(step.pops for step in self.steps))

    def describe(self) -> str:
        rendered = ", ".join(step.describe() for step in self.steps)
        suffix = f" pop={self.pops}" if self.pops else ""
        return f"cleanup_call[{rendered}]{suffix}"


@dataclass(frozen=True)
class IRTailcallFrame(IRNode):
    """Canonical frame setup that precedes the VM tailcall helpers."""

    steps: Tuple[Tuple[str, int], ...]

    def describe(self) -> str:
        rendered = ", ".join(f"{mnemonic}(0x{operand:04X})" for mnemonic, operand in self.steps)
        return f"prep_tailcall[{rendered}]"


@dataclass(frozen=True)
class IRTablePatch(IRNode):
    """Collapses the recurring 0x66xx table patch sequences."""

    operations: Tuple[Tuple[str, int], ...]
    annotations: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        rendered = ", ".join(f"{mnemonic}(0x{operand:04X})" for mnemonic, operand in self.operations)
        note = f"table_patch[{rendered}]"
        if self.annotations:
            note += " " + ", ".join(self.annotations)
        return note


@dataclass(frozen=True)
class IRDispatchCase:
    """Single dispatch case extracted from a table patch."""

    key: int
    target: int
    symbol: Optional[str] = None

    def describe(self) -> str:
        target_repr = f"0x{self.target:04X}"
        if self.symbol:
            target_repr = f"{self.symbol}({target_repr})"
        return f"{self.key}->{target_repr}"


@dataclass(frozen=True)
class IRSwitchDispatch(IRNode):
    """High level representation of dispatch helper tables."""

    cases: Tuple[IRDispatchCase, ...]
    helper: Optional[int] = None
    helper_symbol: Optional[str] = None
    default: Optional[int] = None

    def describe(self) -> str:
        helper_details = "helper=?"
        if self.helper is not None:
            helper_repr = f"0x{self.helper:04X}"
            if self.helper_symbol:
                helper_repr = f"{self.helper_symbol}({helper_repr})"
            helper_details = f"helper={helper_repr}"
        case_text = ", ".join(case.describe() for case in self.cases)
        description = f"dispatch {helper_details} cases=[{case_text}]"
        if self.default is not None:
            default_repr = f"0x{self.default:04X}"
            description += f" default={default_repr}"
        return description


@dataclass(frozen=True)
class IRTableBuilderBegin(IRNode):
    """Marks the start of a table construction pipeline."""

    mode: int
    prologue: Tuple[Tuple[str, int], ...]
    annotations: Tuple[str, ...] = field(default_factory=tuple)
    headers: Tuple[str, ...] = field(default_factory=tuple)
    descriptors: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        ops = ", ".join(f"{mnemonic}(0x{operand:04X})" for mnemonic, operand in self.prologue)
        details = [f"mode=0x{self.mode:02X}"]
        if ops:
            details.append(f"prologue=[{ops}]")
        if self.headers:
            rendered = ", ".join(self.headers)
            details.append(f"headers=[{rendered}]")
        if self.descriptors:
            rendered = ", ".join(self.descriptors)
            details.append(f"descriptors=[{rendered}]")
        if self.annotations:
            details.extend(self.annotations)
        return "table_begin " + " ".join(details)


@dataclass(frozen=True)
class IRTableBuilderEmit(IRNode):
    """Table body emitted by the table construction pipeline."""

    mode: int
    kind: str
    operations: Tuple[Tuple[str, int], ...]
    annotations: Tuple[str, ...] = field(default_factory=tuple)
    parameters: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        ops = ", ".join(f"{mnemonic}(0x{operand:04X})" for mnemonic, operand in self.operations)
        details = [f"mode=0x{self.mode:02X}", f"kind={self.kind}", f"ops=[{ops}]"]
        if self.parameters:
            params = ", ".join(self.parameters)
            details.append(f"params=[{params}]")
        extra = [note for note in self.annotations if note not in {self.kind, f"mode=0x{self.mode:02X}"}]
        if extra:
            details.extend(extra)
        return "table_emit " + " ".join(details)


@dataclass(frozen=True)
class IRTableBuilderCommit(IRNode):
    """Represents the control-flow guard terminating a table builder."""

    guard: str
    commit_target: int
    fallback_target: int
    fallback_literal: Optional[int] = None
    parameters: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        base = (
            f"table_commit guard={self.guard} "
            f"commit=0x{self.commit_target:04X} "
            f"fallback=0x{self.fallback_target:04X}"
        )
        if self.fallback_literal is not None:
            base += f" literal=0x{self.fallback_literal:04X}"
        if self.parameters:
            rendered = ", ".join(self.parameters)
            base += f" params=[{rendered}]"
        return base


@dataclass(frozen=True)
class IRAsciiFinalize(IRNode):
    """Normalises helper invocations that terminate ASCII aggregation blocks."""

    helper: int
    summary: str

    def describe(self) -> str:
        return f"ascii_finalize helper=0x{self.helper:04X} source={self.summary}"


@dataclass(frozen=True)
class IRStackDrop(IRNode):
    """Discard the value currently residing at the top of the VM stack."""

    value: str

    def describe(self) -> str:
        return f"drop {self.value}"


@dataclass(frozen=True)
class IRPageRegister(IRNode):
    """Interaction with the VM page register latch."""

    register: int
    value: Optional[str] = None
    literal: Optional[int] = None

    def describe(self) -> str:
        operand = _format_operand(self.register)
        if self.value:
            return f"page_register {operand}={self.value}"
        return f"page_register[{operand}]"


@dataclass(frozen=True)
class IRAsciiHeader(IRNode):
    """Captures dense ASCII banners embedded at block boundaries."""

    chunks: Tuple[str, ...]

    def describe(self) -> str:
        rendered = ", ".join(self.chunks)
        return f"ascii_header[{rendered}]"


@dataclass(frozen=True)
class IRCallReturn(IRNode):
    """Compact representation for immediate call/return templates."""

    target: int
    args: Tuple[str, ...]
    tail: bool
    returns: Tuple[str, ...]
    varargs: bool = False
    cleanup: Tuple[IRStackEffect, ...] = field(default_factory=tuple)
    arity: Optional[int] = None
    convention: Optional[IRStackEffect] = None
    cleanup_mask: Optional[int] = None
    symbol: Optional[str] = None
    predicate: Optional[CallPredicate] = None

    def describe(self) -> str:
        prefix = "call_return tail" if self.tail else "call_return"
        args = ", ".join(self.args)
        ret = "varargs" if self.varargs else ", ".join(self.returns)
        if self.varargs and self.returns:
            ret = f"varargs({ret})"
        cleanup = ""
        if self.cleanup:
            rendered = ", ".join(step.describe() for step in self.cleanup)
            cleanup = f" cleanup=[{rendered}]"
        target_repr = f"0x{self.target:04X}"
        if self.symbol:
            target_repr = f"{self.symbol}({target_repr})"
        details = []
        if self.arity is not None:
            details.append(f"arity={self.arity}")
        if self.convention is not None:
            details.append(f"convention={self.convention.describe()}")
        if self.cleanup_mask is not None:
            details.append(f"mask={_format_operand(self.cleanup_mask)}")
        if self.predicate is not None:
            details.append(f"predicate={self.predicate.describe()}")
        extra = f" {' '.join(details)}" if details else ""
        return (
            f"{prefix} target={target_repr} args=[{args}] returns=[{ret}]"
            f"{cleanup}{extra}"
        )


@dataclass(frozen=True)
class IRRaw(IRNode):
    """Fallback wrapper for instructions that have not been normalised."""

    mnemonic: str
    operand: int
    operand_role: Optional[str] = None
    operand_alias: Optional[str] = None
    annotations: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        note = ""
        if self.annotations:
            note = " " + ", ".join(self.annotations)
        hex_value = f"0x{self.operand:04X}"
        alias = self.operand_alias
        if alias:
            alias_text = str(alias)
            if alias_text.upper().startswith("0X"):
                alias_text = alias_text.upper()
            value = alias_text if alias_text == hex_value else f"{alias_text}({hex_value})"
        else:
            value = hex_value

        if self.operand_role:
            detail = f"{self.operand_role}={value}"
        else:
            detail = f"operand={value}"

        if self.mnemonic.startswith("op_"):
            rendered = f"raw {self.mnemonic} {detail}"
        else:
            rendered = f"{self.mnemonic}({detail})"
        return f"{rendered}{note}"


@dataclass(frozen=True)
class IRTailcallReturn(IRNode):
    """Bundle that represents a tail call immediately followed by a return."""

    target: int
    args: Tuple[str, ...]
    returns: int
    varargs: bool = False
    cleanup: Tuple[IRStackEffect, ...] = field(default_factory=tuple)
    tail: bool = True
    arity: Optional[int] = None
    convention: Optional[IRStackEffect] = None
    cleanup_mask: Optional[int] = None
    symbol: Optional[str] = None
    predicate: Optional[CallPredicate] = None

    def describe(self) -> str:
        target_repr = f"0x{self.target:04X}"
        if self.symbol:
            target_repr = f"{self.symbol}({target_repr})"
        args = ", ".join(self.args)
        details = [f"returns={self.returns}"]
        if self.varargs:
            details.append("varargs")
        if self.convention is not None:
            details.append(f"convention={self.convention.describe()}")
        if self.cleanup_mask is not None:
            details.append(f"mask={_format_operand(self.cleanup_mask)}")
        if self.cleanup:
            rendered = ", ".join(step.describe() for step in self.cleanup)
            details.append(f"cleanup=[{rendered}]")
        if self.arity is not None:
            details.append(f"arity={self.arity}")
        if self.predicate is not None:
            details.append(f"predicate={self.predicate.describe()}")
        suffix = " ".join(details)
        return f"tailcall_return target={target_repr} args=[{args}] {suffix}".rstrip()


@dataclass(frozen=True)
class IRConditionMask(IRNode):
    """High level view of flag fan-out and RET_MASK terminators."""

    source: str
    mask: int

    def describe(self) -> str:
        mask_repr = _format_operand(self.mask)
        return f"condition_mask source={self.source} mask={mask_repr}"


@dataclass(frozen=True)
class IRBlock:
    """Single basic block in the IR programme."""

    label: str
    start_offset: int
    nodes: Tuple[IRNode, ...]
    annotations: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IRSegment:
    """Collection of IR blocks corresponding to a container segment."""

    index: int
    start: int
    length: int
    blocks: Tuple[IRBlock, ...]
    metrics: "NormalizerMetrics"


@dataclass(frozen=True)
class IRProgram:
    """Normalised representation for the entire container."""

    segments: Tuple[IRSegment, ...]
    metrics: "NormalizerMetrics"
    string_pool: Tuple[IRStringConstant, ...] = tuple()


@dataclass
class NormalizerMetrics:
    """Aggregate counts recorded during normalisation."""

    calls: int = 0
    tail_calls: int = 0
    returns: int = 0
    literals: int = 0
    literal_chunks: int = 0
    aggregates: int = 0
    testset_branches: int = 0
    if_branches: int = 0
    loads: int = 0
    stores: int = 0
    reduce_replaced: int = 0
    raw_remaining: int = 0

    def observe(self, other: "NormalizerMetrics") -> None:
        """Accumulate values from ``other`` into this instance."""

        self.calls += other.calls
        self.tail_calls += other.tail_calls
        self.returns += other.returns
        self.literals += other.literals
        self.literal_chunks += other.literal_chunks
        self.aggregates += other.aggregates
        self.testset_branches += other.testset_branches
        self.if_branches += other.if_branches
        self.loads += other.loads
        self.stores += other.stores
        self.reduce_replaced += other.reduce_replaced
        self.raw_remaining += other.raw_remaining

    def describe(self) -> str:
        """Return a stable textual summary of the metrics."""

        parts = [
            f"calls={self.calls}",
            f"tail_calls={self.tail_calls}",
            f"returns={self.returns}",
            f"literals={self.literals}",
            f"literal_chunks={self.literal_chunks}",
            f"aggregates={self.aggregates}",
            f"testset_branches={self.testset_branches}",
            f"if_branches={self.if_branches}",
            f"loads={self.loads}",
            f"stores={self.stores}",
            f"reduce_replaced={self.reduce_replaced}",
            f"raw_remaining={self.raw_remaining}",
        ]
        return " ".join(parts)


__all__ = [
    "IRProgram",
    "IRSegment",
    "IRBlock",
    "IRCall",
    "IRTailCall",
    "IRReturn",
    "IRBuildArray",
    "IRBuildMap",
    "IRBuildTuple",
    "IRLiteralBlock",
    "IRIf",
    "IRTestSetBranch",
    "IRFlagCheck",
    "IRFunctionPrologue",
    "IRLoad",
    "IRStore",
    "IRIndirectLoad",
    "IRIndirectStore",
    "IRIORead",
    "IRIOWrite",
    "IRStackDuplicate",
    "IRStackDrop",
    "IRPageRegister",
    "IRAsciiHeader",
    "IRCallReturn",
    "IRTailcallReturn",
    "IRConditionMask",
    "IRLiteral",
    "IRLiteralChunk",
    "IRStringConstant",
    "IRAsciiPreamble",
    "IRCallPreparation",
    "IRTailcallFrame",
    "IRTablePatch",
    "IRDispatchCase",
    "IRSwitchDispatch",
    "IRAsciiFinalize",
    "IRSlot",
    "MemRef",
    "IRRaw",
    "MemSpace",
    "SSAValueKind",
    "NormalizerMetrics",
]


def _format_operand(value: int) -> str:
    hex_value = f"0x{value:04X}"
    alias = OPERAND_ALIASES.get(value)
    if alias:
        alias_text = alias if isinstance(alias, str) else str(alias)
        upper = alias_text.upper()
        if upper.startswith("0X"):
            alias_text = upper
        return alias_text if alias_text == hex_value else f"{alias_text}({hex_value})"
    return hex_value

