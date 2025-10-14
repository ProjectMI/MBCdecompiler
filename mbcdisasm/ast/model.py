"""Dataclasses describing the reconstructed AST model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from mbcdisasm.ir.model import IRSlot, MemRef


@dataclass(frozen=True)
class ASTExpression:
    """Lightweight description of an expression tree node."""

    kind: str
    value: Any = None
    operands: Tuple["ASTExpression", ...] = field(default_factory=tuple)
    type_hint: Optional[str] = None

    def describe(self) -> str:
        """Render the expression into a stable textual form."""

        if self.kind == "literal":
            if isinstance(self.value, int):
                return f"0x{self.value:04X}" if self.value > 9 else str(self.value)
            return str(self.value)
        if self.kind == "identifier":
            return str(self.value)
        if self.kind == "slot":
            slot = self.value
            assert isinstance(slot, IRSlot)
            return f"{slot.space.name.lower()}[0x{slot.index:04X}]"
        if self.kind == "indirect_load":
            base = self.operands[0].describe() if self.operands else "ptr"
            offset = self.value.get("offset_repr", "?")
            ref = self.value.get("ref")
            if ref is not None:
                assert isinstance(ref, MemRef)
                return f"load {ref.describe()} via {base}"
            return f"load *({base} + {offset})"
        if self.kind == "indirect_store":
            base = self.operands[0].describe() if self.operands else "ptr"
            offset = self.value.get("offset_repr", "?")
            ref = self.value.get("ref")
            if ref is not None:
                assert isinstance(ref, MemRef)
                return f"store -> {ref.describe()} via {base}"
            return f"store -> *({base} + {offset})"
        if self.kind == "call_target":
            return str(self.value)
        if self.kind == "raw":
            return str(self.value)
        if self.kind == "tuple":
            parts = ", ".join(expr.describe() for expr in self.operands)
            return f"({parts})"
        return str(self.value)


@dataclass(frozen=True)
class ASTStatement:
    """Single AST level statement extracted from the IR."""

    kind: str
    text: str
    expressions: Tuple[ASTExpression, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ASTBlock:
    """AST block paired with CFG successor information."""

    label: str
    start_offset: int
    successors: Tuple[str, ...]
    statements: Tuple[ASTStatement, ...]

    def describe_successors(self) -> str:
        targets = ", ".join(self.successors)
        return targets or "<exit>"


@dataclass(frozen=True)
class ASTProcedure:
    """Procedure reconstructed from the CFG and IR blocks."""

    name: str
    entry_label: str
    entry_offset: int
    blocks: Tuple[ASTBlock, ...]
    exits: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ASTSegment:
    """Segment level container for reconstructed procedures."""

    index: int
    start: int
    length: int
    procedures: Tuple[ASTProcedure, ...]
    entry_offsets: Tuple[int, ...]
    dangling_targets: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ASTProgram:
    """Full AST reconstruction covering the entire container."""

    segments: Tuple[ASTSegment, ...]
    metrics: "ASTMetrics"


@dataclass
class ASTMetrics:
    """Aggregate counters describing reconstruction quality."""

    procedures: int = 0
    blocks: int = 0
    cfg_edges: int = 0
    calls: int = 0
    tail_calls: int = 0
    returns: int = 0
    loads: int = 0
    stores: int = 0
    indirect_loads: int = 0
    indirect_stores: int = 0
    literals: int = 0
    resolved_identifiers: int = 0
    unknown_identifiers: int = 0
    dangling_targets: int = 0

    def describe(self) -> str:
        parts = [
            f"procedures={self.procedures}",
            f"blocks={self.blocks}",
            f"edges={self.cfg_edges}",
            f"calls={self.calls}",
            f"tail_calls={self.tail_calls}",
            f"returns={self.returns}",
            f"loads={self.loads}",
            f"stores={self.stores}",
            f"indirect_loads={self.indirect_loads}",
            f"indirect_stores={self.indirect_stores}",
            f"literals={self.literals}",
            f"resolved={self.resolved_identifiers}",
            f"unknown={self.unknown_identifiers}",
            f"dangling_targets={self.dangling_targets}",
        ]
        return " ".join(parts)

    def observe_calls(self, tail: bool = False) -> None:
        if tail:
            self.tail_calls += 1
        else:
            self.calls += 1

    def observe_returns(self) -> None:
        self.returns += 1

    def observe_load(self, indirect: bool = False) -> None:
        if indirect:
            self.indirect_loads += 1
        else:
            self.loads += 1

    def observe_store(self, indirect: bool = False) -> None:
        if indirect:
            self.indirect_stores += 1
        else:
            self.stores += 1

    def observe_literals(self, count: int = 1) -> None:
        self.literals += count

    def observe_identifiers(self, resolved: int, unknown: int) -> None:
        self.resolved_identifiers += resolved
        self.unknown_identifiers += unknown

    def observe_dangling(self, count: int) -> None:
        self.dangling_targets += count


__all__ = [
    "ASTProgram",
    "ASTSegment",
    "ASTProcedure",
    "ASTBlock",
    "ASTStatement",
    "ASTExpression",
    "ASTMetrics",
]
