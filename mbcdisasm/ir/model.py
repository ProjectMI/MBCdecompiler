"""Data structures for the normalised intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, List, Sequence, Tuple


class MemSpace(Enum):
    """Addressing domains recognised by the normaliser."""

    FRAME = auto()
    GLOBAL = auto()
    CONST = auto()


@dataclass(frozen=True)
class IRTemp:
    """Named temporary produced by IR operations."""

    name: str


@dataclass(frozen=True)
class IRLiteral:
    """Literal value created by the normaliser."""

    value: int

    def render(self) -> str:
        return f"#{self.value}"


@dataclass(frozen=True)
class IRSlot:
    """Reference to an addressable VM slot."""

    space: MemSpace
    index: int

    def render(self) -> str:
        space = self.space.name.lower()
        return f"{space}[0x{self.index:04X}]"


class IRNode:
    """Base class for IR operations to ease type checking."""


@dataclass
class IRBuildArray(IRNode):
    result: IRTemp
    elements: Sequence[object]

    def render(self) -> str:
        values = ", ".join(_render_value(value) for value in self.elements)
        return f"{self.result.name} = build_array([{values}])"


@dataclass
class IRBuildTuple(IRNode):
    result: IRTemp
    elements: Sequence[object]

    def render(self) -> str:
        values = ", ".join(_render_value(value) for value in self.elements)
        return f"{self.result.name} = build_tuple({values})"


@dataclass
class IRBuildMap(IRNode):
    result: IRTemp
    items: Sequence[Tuple[object, object]]

    def render(self) -> str:
        parts = []
        for key, value in self.items:
            parts.append(f"{_render_value(key)}: {_render_value(value)}")
        inside = ", ".join(parts)
        return f"{self.result.name} = build_map({{{inside}}})"


@dataclass
class IRCall(IRNode):
    target: int
    args: Sequence[object]
    result: IRTemp | None
    tail: bool = False

    def render(self) -> str:
        args = ", ".join(_render_value(arg) for arg in self.args)
        prefix = "tailcall" if self.tail else "call"
        if self.result is None:
            return f"{prefix} fn@0x{self.target:04X}({args})"
        return f"{self.result.name} = {prefix} fn@0x{self.target:04X}({args})"


@dataclass
class IRReturn(IRNode):
    values: Sequence[object]

    def render(self) -> str:
        payload = ", ".join(_render_value(value) for value in self.values)
        return f"return {payload}"


@dataclass
class IRIf(IRNode):
    predicate: object
    then_target: str
    else_target: str

    def render(self) -> str:
        return (
            f"if {_render_value(self.predicate)} then goto {self.then_target} "
            f"else goto {self.else_target}"
        )


@dataclass
class IRTestSetBranch(IRNode):
    target: IRTemp
    expression: object
    then_target: str
    else_target: str

    def render(self) -> str:
        expr = _render_value(self.expression)
        return (
            f"{self.target.name} = testset {expr} ? goto {self.then_target} "
            f": goto {self.else_target}"
        )


@dataclass
class IRLoad(IRNode):
    result: IRTemp
    slot: IRSlot

    def render(self) -> str:
        return f"{self.result.name} = load {self.slot.render()}"


@dataclass
class IRStore(IRNode):
    slot: IRSlot
    value: object

    def render(self) -> str:
        return f"store {self.slot.render()} <- {_render_value(self.value)}"


@dataclass
class IRBasicBlock:
    label: str
    operations: List[IRNode] = field(default_factory=list)

    def render(self) -> List[str]:
        lines = [f"block {self.label}:"]
        for op in self.operations:
            lines.append(f"  {op.render()}")
        return lines


@dataclass
class NormalizerMetrics:
    calls: int = 0
    tail_calls: int = 0
    returns: int = 0
    aggregates: int = 0
    testset_branches: int = 0
    if_branches: int = 0
    loads: int = 0
    stores: int = 0
    reduce_replaced: int = 0
    raw_remaining: int = 0

    def merge(self, other: "NormalizerMetrics") -> None:
        for field in ("calls", "tail_calls", "returns", "aggregates", "testset_branches", "if_branches", "loads", "stores", "reduce_replaced", "raw_remaining"):
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def render(self) -> List[str]:
        return [
            "metrics:",
            f"  calls={self.calls}",
            f"  tail_calls={self.tail_calls}",
            f"  returns={self.returns}",
            f"  aggregates={self.aggregates}",
            f"  testset_branches={self.testset_branches}",
            f"  if_branches={self.if_branches}",
            f"  loads={self.loads}",
            f"  stores={self.stores}",
            f"  reduce_replaced={self.reduce_replaced}",
            f"  raw_remaining={self.raw_remaining}",
        ]


@dataclass
class IRProgram:
    blocks: Sequence[IRBasicBlock]
    metrics: NormalizerMetrics

    def render(self) -> List[str]:
        lines: List[str] = []
        lines.extend(self.metrics.render())
        for block in self.blocks:
            lines.extend(block.render())
        return lines


def _render_value(value: object) -> str:
    if isinstance(value, IRTemp):
        return value.name
    if isinstance(value, IRLiteral):
        return value.render()
    if isinstance(value, IRSlot):
        return value.render()
    return str(value)
