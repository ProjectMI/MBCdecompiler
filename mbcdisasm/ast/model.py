"""Data model for reconstructed abstract syntax trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

from ..ir.model import IRBlock, IRProgram, IRSegment, IRSlot, MemSpace, SSAValueKind


@dataclass(frozen=True)
class CFGNode:
    """Single node in the reconstructed control-flow graph."""

    label: str
    block: IRBlock
    segment_index: int
    successors: Tuple[str, ...] = field(default_factory=tuple)
    predecessors: Tuple[str, ...] = field(default_factory=tuple)

    def with_successors(self, successors: Iterable[str]) -> "CFGNode":
        return CFGNode(
            label=self.label,
            block=self.block,
            segment_index=self.segment_index,
            successors=tuple(successors),
            predecessors=self.predecessors,
        )

    def with_predecessors(self, predecessors: Iterable[str]) -> "CFGNode":
        return CFGNode(
            label=self.label,
            block=self.block,
            segment_index=self.segment_index,
            successors=self.successors,
            predecessors=tuple(predecessors),
        )


@dataclass(frozen=True)
class ControlFlowGraph:
    """Dense representation of the block-level control flow graph."""

    nodes: Mapping[str, CFGNode]
    offsets: Mapping[int, str]

    def successors(self, label: str) -> Tuple[str, ...]:
        node = self.nodes[label]
        return node.successors

    def predecessors(self, label: str) -> Tuple[str, ...]:
        node = self.nodes[label]
        return node.predecessors

    def node(self, label: str) -> CFGNode:
        return self.nodes[label]

    def segment_index(self, label: str) -> int:
        return self.nodes[label].segment_index


@dataclass(frozen=True)
class ASTExpression:
    """Expression node used inside :class:`ASTStatement`."""

    kind: str
    value: Optional[object] = None
    operands: Tuple["ASTExpression", ...] = tuple()
    alias: Optional[str] = None
    type: SSAValueKind = SSAValueKind.UNKNOWN

    def describe(self) -> str:
        if self.kind == "literal":
            value_repr = self.value
            if isinstance(value_repr, int):
                value_repr = f"0x{value_repr:04X}"
            return f"literal({value_repr})"
        if self.kind == "load" and isinstance(self.value, IRSlot):
            slot = self.value
            location = f"{slot.space.name.lower()}[{slot.index}]"
            if self.alias:
                return f"load {location} -> {self.alias}"
            return f"load {location}"
        if self.kind == "ref" and self.alias:
            type_suffix = f":{self.type.name.lower()}" if self.type is not SSAValueKind.UNKNOWN else ""
            return f"ref({self.alias}{type_suffix})"
        if not self.operands:
            payload = self.value if self.value is not None else "?"
            return f"{self.kind}({payload})"
        rendered = ", ".join(operand.describe() for operand in self.operands)
        suffix = f" as {self.alias}" if self.alias else ""
        return f"{self.kind}[{rendered}]{suffix}".strip()


@dataclass(frozen=True)
class ASTStatement:
    """Structured statement reconstructed from IR nodes."""

    kind: str
    target: Optional[str]
    expr: Optional[ASTExpression]
    args: Tuple[ASTExpression, ...] = tuple()
    info: Tuple[str, ...] = tuple()

    def describe(self) -> str:
        if self.kind == "assign":
            expr = self.expr.describe() if self.expr else "?"
            return f"{self.target} = {expr}" if self.target else f"assign {expr}"
        if self.kind == "store":
            expr = self.expr.describe() if self.expr else "?"
            target = self.target or "slot"
            return f"store {target} <= {expr}"
        if self.kind in {"call", "tail_call", "call_return"}:
            args = ", ".join(arg.describe() for arg in self.args)
            meta = " ".join(self.info)
            details = f" ({meta})" if meta else ""
            call_repr = self.target or self.kind
            return f"{call_repr}({args}){details}".rstrip()
        if self.kind == "return":
            args = ", ".join(arg.describe() for arg in self.args)
            suffix = " " + " ".join(self.info) if self.info else ""
            return f"return [{args}]{suffix}".rstrip()
        if self.kind == "branch":
            meta = " ".join(self.info)
            expr = self.expr.describe() if self.expr else "?"
            return f"branch {expr} {meta}".rstrip()
        if self.kind == "prologue":
            meta = " ".join(self.info)
            expr = self.expr.describe() if self.expr else "?"
            return f"prologue {expr} {meta}".rstrip()
        return f"{self.kind}"


@dataclass(frozen=True)
class ASTBlock:
    """Reconstructed block containing structured statements."""

    label: str
    start_offset: int
    statements: Tuple[ASTStatement, ...]


@dataclass(frozen=True)
class ASTProcedure:
    """Grouping of blocks that form a logical procedure."""

    name: str
    entry: str
    blocks: Tuple[str, ...]
    exits: Tuple[str, ...]
    sources: Tuple[str, ...] = tuple()


@dataclass(frozen=True)
class ASTSegment:
    """Segment level container for procedures and blocks."""

    index: int
    start: int
    length: int
    blocks: Tuple[ASTBlock, ...]
    procedures: Tuple[ASTProcedure, ...]


@dataclass
class ASTMetrics:
    """Heuristic counters that gauge reconstruction quality."""

    procedures: int = 0
    blocks: int = 0
    statements: int = 0
    expressions: int = 0
    constants: int = 0
    resolved_references: int = 0
    unresolved_references: int = 0
    typed_expressions: int = 0
    call_arguments: int = 0
    store_values: int = 0
    return_values: int = 0

    def observe_expression(self, expr: ASTExpression) -> None:
        self.expressions += 1
        if expr.kind == "literal":
            self.constants += 1
        if expr.type is not SSAValueKind.UNKNOWN:
            self.typed_expressions += 1

    def record_reference(self, *, resolved: bool) -> None:
        if resolved:
            self.resolved_references += 1
        else:
            self.unresolved_references += 1

    def describe(self) -> str:
        parts = [
            f"procedures={self.procedures}",
            f"blocks={self.blocks}",
            f"statements={self.statements}",
            f"expressions={self.expressions}",
            f"constants={self.constants}",
            f"resolved_refs={self.resolved_references}",
            f"unresolved_refs={self.unresolved_references}",
            f"typed_exprs={self.typed_expressions}",
            f"call_args={self.call_arguments}",
            f"stores={self.store_values}",
            f"returns={self.return_values}",
        ]
        return " ".join(parts)


@dataclass(frozen=True)
class ASTProgram:
    """Top level bundle for AST reconstruction results."""

    segments: Tuple[ASTSegment, ...]
    cfg: ControlFlowGraph
    metrics: ASTMetrics


__all__ = [
    "CFGNode",
    "ControlFlowGraph",
    "ASTExpression",
    "ASTStatement",
    "ASTBlock",
    "ASTProcedure",
    "ASTSegment",
    "ASTMetrics",
    "ASTProgram",
]
