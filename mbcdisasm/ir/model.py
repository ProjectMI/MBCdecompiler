"""Dataclasses describing the normalised intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple


class MemSpace(Enum):
    """High level classification of indirect memory accesses."""

    FRAME = auto()
    GLOBAL = auto()
    CONST = auto()


@dataclass(frozen=True)
class IRSlot:
    """Address of a VM slot used by :class:`IRLoad` and :class:`IRStore`."""

    space: MemSpace
    index: int


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
class IRReturn(IRNode):
    """Return from the current routine."""

    values: Tuple[str, ...]
    varargs: bool = False

    def describe(self) -> str:
        if self.varargs:
            if self.values:
                return f"return varargs({', '.join(self.values)})"
            return "return varargs"
        values = ", ".join(self.values)
        return f"return [{values}]"


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
class IRLiteralBlock(IRNode):
    """Compact representation of mirrored literal bootstrap blocks."""

    pair: Tuple[int, int]
    terminator: int
    count: int
    reducers: int = 0
    mirrored: bool = False

    def describe(self) -> str:
        first, second = self.pair
        pair_text = f"0x{first:04X}/0x{second:04X}"
        suffix = " mirrored" if self.mirrored else ""
        reducer_note = f" reducers={self.reducers}" if self.reducers else ""
        return (
            f"literal_block pair={pair_text} term=0x{self.terminator:04X} "
            f"count={self.count}{suffix}{reducer_note}"
        )


@dataclass(frozen=True)
class IRAsciiPrologue(IRNode):
    """Marker that denotes the start of an inline ASCII payload."""

    marker_operand: int
    layout_operand: int
    shuffle_operand: int

    def describe(self) -> str:
        return (
            "ascii_prologue marker=0x"
            f"{self.marker_operand:04X} layout=0x{self.layout_operand:04X} "
            f"shuffle=0x{self.shuffle_operand:04X}"
        )


@dataclass(frozen=True)
class IRCallArgPrep(IRNode):
    """Normalised wrapper for call argument stack preparation helpers."""

    steps: Tuple[Tuple[str, int], ...]

    def describe(self) -> str:
        parts = ", ".join(f"{mnemonic}(0x{operand:04X})" for mnemonic, operand in self.steps)
        return f"prep_call_args[{parts}]"


@dataclass(frozen=True)
class IRTailcallPrep(IRNode):
    """Structured representation of tailcall frame preparation."""

    steps: Tuple[Tuple[str, int], ...]

    def describe(self) -> str:
        parts = ", ".join(f"{mnemonic}(0x{operand:04X})" for mnemonic, operand in self.steps)
        return f"prep_tailcall[{parts}]"


@dataclass(frozen=True)
class IRTableChunk(IRNode):
    """Capture the recurring table patch helpers used across scripts."""

    base_operand: int
    key_operand: int
    value_operand: int
    extras: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)

    def describe(self) -> str:
        extras = ""
        if self.extras:
            extras = ", extras=[" + ", ".join(
                f"{mnemonic}(0x{operand:04X})" for mnemonic, operand in self.extras
            ) + "]"
        return (
            "table_chunk base=0x"
            f"{self.base_operand:04X} key=0x{self.key_operand:04X} "
            f"value=0x{self.value_operand:04X}{extras}"
        )


@dataclass(frozen=True)
class IRAsciiBlock(IRNode):
    """Aggregate consecutive ASCII chunks finalised by helper calls."""

    data: bytes
    helper_operand: int
    annotations: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        printable = []
        for byte in self.data:
            if 0x20 <= byte <= 0x7E:
                printable.append(chr(byte))
            elif byte in {0x09, 0x0A, 0x0D}:
                printable.append({0x09: "\t", 0x0A: "\n", 0x0D: "\r"}[byte])
            else:
                printable.append(f"\\x{byte:02x}")
        text = "".join(printable)
        note = f"ascii_block({text}) helper=0x{self.helper_operand:04X}"
        if self.annotations:
            note += " " + ", ".join(self.annotations)
        return note


@dataclass(frozen=True)
class IRCheckFlag(IRNode):
    """Branch that explicitly checks a VM flag constant."""

    flag: int
    then_target: int
    else_target: int

    def describe(self) -> str:
        name = f"FLAG_{self.flag:04X}"
        return (
            f"check_flag {name} then=0x{self.then_target:04X} "
            f"else=0x{self.else_target:04X}"
        )


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
class IRStackDuplicate(IRNode):
    """Duplicate the value currently at the top of the VM stack."""

    value: str
    copies: int = 2

    def describe(self) -> str:
        if self.copies <= 1:
            return f"dup {self.value}"
        return f"dup {self.value} -> copies={self.copies}"


@dataclass(frozen=True)
class IRStackDrop(IRNode):
    """Discard the value currently residing at the top of the VM stack."""

    value: str

    def describe(self) -> str:
        return f"drop {self.value}"


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
    "IRLiteralBlock",
    "IRAsciiPrologue",
    "IRCallArgPrep",
    "IRTailcallPrep",
    "IRTableChunk",
    "IRAsciiBlock",
    "IRCheckFlag",
    "IRBuildArray",
    "IRBuildMap",
    "IRBuildTuple",
    "IRIf",
    "IRTestSetBranch",
    "IRLoad",
    "IRStore",
    "IRStackDuplicate",
    "IRStackDrop",
    "IRLiteral",
    "IRLiteralChunk",
    "IRSlot",
    "IRRaw",
    "MemSpace",
    "NormalizerMetrics",
]
