"""Dataclasses describing the normalised intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple


class MemSpace(Enum):
    """High level classification of indirect memory accesses."""

    FRAME = auto()
    GLOBAL = auto()
    CONST = auto()


class SSAValueKind(Enum):
    """Lightweight annotation attached to SSA values."""

    UNKNOWN = auto()
    INTEGER = auto()
    ADDRESS = auto()
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
    page: Optional[int] = None
    offset: Optional[int] = None
    symbol: Optional[str] = None

    def describe(self) -> str:
        tag = self.symbol or self.region
        parts = [tag]
        if self.bank is not None:
            parts.append(f"bank=0x{self.bank:04X}")
        if self.page is not None:
            parts.append(f"page=0x{self.page:04X}")
        if self.offset is not None:
            parts.append(f"off=0x{self.offset:04X}")
        return "mem[" + " ".join(parts) + "]"


@dataclass(frozen=True)
class IRNode:
    """Base class for IR nodes.

    The class only exists to make type signatures more explicit.  Subclasses do
    not rely on runtime inheritance checks so the dataclasses can remain frozen
    which keeps the structures hashable and easy to compare in tests.
    """


@dataclass(frozen=True)
class IRCall(IRNode):
    """Invocation of another routine."""

    target: int
    args: Tuple[str, ...]
    tail: bool = False

    def describe(self) -> str:
        suffix = " tail" if self.tail else ""
        args = ", ".join(self.args)
        return f"call{suffix} target=0x{self.target:04X} args=[{args}]"


@dataclass(frozen=True)
class IRStackEffect:
    """Canonical representation of stack shuffles and teardowns."""

    mnemonic: str
    operand: int = 0
    pops: int = 0

    def describe(self) -> str:
        details = []
        if self.pops:
            details.append(f"pop={self.pops}")
        if self.operand or self.mnemonic not in {"stack_teardown"}:
            details.append(f"operand=0x{self.operand:04X}")
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

    def describe(self) -> str:
        cleanup = ""
        if self.cleanup:
            rendered = ", ".join(step.describe() for step in self.cleanup)
            cleanup = f" cleanup=[{rendered}]"
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

    def describe(self) -> str:
        printable = []
        for byte in self.data:
            if 0x20 <= byte <= 0x7E:
                printable.append(chr(byte))
            elif byte in {0x09, 0x0A, 0x0D}:
                printable.append({0x09: "\\t", 0x0A: "\\n", 0x0D: "\\r"}[byte])
            else:
                printable.append(f"\\x{byte:02x}")
        text = "".join(printable)
        note = f"ascii({text})"
        if self.annotations:
            note += " " + ", ".join(self.annotations)
        return note


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
        return base


@dataclass(frozen=True)
class IRAsciiWrapperCall(IRNode):
    """Tailored representation of helper calls guarded by inline ASCII chunks."""

    target: int
    args: Tuple[str, ...]
    ascii_chunks: Tuple[str, ...]
    tail: bool = False

    def describe(self) -> str:
        ascii_repr = ", ".join(self.ascii_chunks)
        prefix = "ascii_wrapper_call tail" if self.tail else "ascii_wrapper_call"
        return f"{prefix} target=0x{self.target:04X} ascii=[{ascii_repr}] args=[{', '.join(self.args)}]"


@dataclass(frozen=True)
class IRTailcallAscii(IRNode):
    """Tail calls immediately followed by ASCII driven conditionals."""

    target: int
    args: Tuple[str, ...]
    ascii_chunks: Tuple[str, ...]
    condition: str
    then_target: int
    else_target: int

    def describe(self) -> str:
        ascii_repr = ", ".join(self.ascii_chunks)
        return (
            f"tailcall_ascii target=0x{self.target:04X} cond={self.condition} "
            f"then=0x{self.then_target:04X} else=0x{self.else_target:04X} ascii=[{ascii_repr}]"
        )


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
    memref: Optional[MemRef] = None

    def describe(self) -> str:
        if self.memref is not None:
            via = ""
            if self.base and self.memref.symbol != self.base:
                via = f" via {self.base}"
            return f"load {self.memref.describe()}{via} -> {self.target}"
        prefix = self.base
        if self.base_slot is not None:
            slot = f"{self.base_slot.space.name.lower()}[0x{self.base_slot.index:04X}]"
            if self.base:
                prefix = f"{slot} ({self.base})"
            else:
                prefix = slot
        return f"indirect_load base={prefix} offset=0x{self.offset:04X} -> {self.target}"


@dataclass(frozen=True)
class IRIndirectStore(IRNode):
    """Write a value through an indirect slot pointer."""

    base: str
    value: str
    offset: int
    base_slot: Optional[IRSlot] = None
    memref: Optional[MemRef] = None

    def describe(self) -> str:
        if self.memref is not None:
            via = ""
            if self.base and self.memref.symbol != self.base:
                via = f" via {self.base}"
            return f"store {self.value} -> {self.memref.describe()}{via}"
        prefix = self.base
        if self.base_slot is not None:
            slot = f"{self.base_slot.space.name.lower()}[0x{self.base_slot.index:04X}]"
            if self.base:
                prefix = f"{slot} ({self.base})"
            else:
                prefix = slot
        return f"indirect_store {self.value} -> {prefix} offset=0x{self.offset:04X}"


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

    def describe(self) -> str:
        rendered = ", ".join(f"{mnemonic}(0x{operand:04X})" for mnemonic, operand in self.operations)
        return f"table_patch[{rendered}]"


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
        return f"{prefix} target=0x{self.target:04X} args=[{args}] returns=[{ret}]{cleanup}"


@dataclass(frozen=True)
class IRRaw(IRNode):
    """Fallback wrapper for instructions that have not been normalised."""

    mnemonic: str
    operand: int
    annotations: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        note = ""
        if self.annotations:
            note = " " + ", ".join(self.annotations)
        return f"raw {self.mnemonic} operand=0x{self.operand:04X}{note}"


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
    "IRReturn",
    "IRBuildArray",
    "IRBuildMap",
    "IRBuildTuple",
    "IRLiteralBlock",
    "IRAsciiWrapperCall",
    "IRTailcallAscii",
    "IRIf",
    "IRTestSetBranch",
    "IRFlagCheck",
    "IRFunctionPrologue",
    "IRLoad",
    "IRStore",
    "IRIndirectLoad",
    "IRIndirectStore",
    "IRStackDuplicate",
    "IRStackDrop",
    "IRAsciiHeader",
    "IRCallReturn",
    "IRLiteral",
    "IRLiteralChunk",
    "IRAsciiPreamble",
    "IRCallPreparation",
    "IRTailcallFrame",
    "IRTablePatch",
    "IRAsciiFinalize",
    "IRSlot",
    "IRRaw",
    "MemSpace",
    "SSAValueKind",
    "NormalizerMetrics",
]
