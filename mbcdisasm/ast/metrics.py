"""Metrics helpers for AST reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .model import (
    ASTExpression,
    ReconstructionMetrics,
    VariableExpr,
    LiteralExpr,
    CallExpr,
    UnknownExpr,
)


@dataclass
class MetricCounter:
    """Accumulate raw counts while building the AST."""

    procedures: int = 0
    cfg_nodes: int = 0
    cfg_edges: int = 0
    entry_points: int = 0
    assigned_blocks: int = 0
    typed_variables: int = 0
    total_variables: int = 0
    literal_expressions: int = 0
    total_expressions: int = 0
    call_arguments: int = 0
    resolved_call_arguments: int = 0
    max_expression_depth: int = 0
    statements: int = 0
    blocks: int = 0

    def observe_variables(self, variables: Iterable[VariableExpr]) -> None:
        for var in variables:
            self.total_variables += 1
            if var.kind.name != "UNKNOWN":
                self.typed_variables += 1

    def observe_expressions(self, expressions: Iterable[ASTExpression]) -> None:
        for expr in expressions:
            self.total_expressions += 1
            depth = expr.depth()
            if depth > self.max_expression_depth:
                self.max_expression_depth = depth
            if isinstance(expr, LiteralExpr):
                self.literal_expressions += 1
            if isinstance(expr, CallExpr):
                self.observe_call_arguments(expr.args)

    def observe_call_arguments(self, args: Iterable[ASTExpression]) -> None:
        resolved = 0
        total = 0
        for arg in args:
            total += 1
            if not isinstance(arg, UnknownExpr):
                resolved += 1
        self.call_arguments += total
        self.resolved_call_arguments += resolved

    def observe_block(self, statement_count: int) -> None:
        self.blocks += 1
        self.statements += statement_count

    def merge(self, other: "MetricCounter") -> None:
        self.procedures += other.procedures
        self.cfg_nodes += other.cfg_nodes
        self.cfg_edges += other.cfg_edges
        self.entry_points += other.entry_points
        self.assigned_blocks += other.assigned_blocks
        self.typed_variables += other.typed_variables
        self.total_variables += other.total_variables
        self.literal_expressions += other.literal_expressions
        self.total_expressions += other.total_expressions
        self.call_arguments += other.call_arguments
        self.resolved_call_arguments += other.resolved_call_arguments
        self.max_expression_depth = max(self.max_expression_depth, other.max_expression_depth)
        self.statements += other.statements
        self.blocks += other.blocks

    def to_metrics(self) -> ReconstructionMetrics:
        typed_ratio = (
            self.typed_variables / self.total_variables if self.total_variables else 0.0
        )
        literal_ratio = (
            self.literal_expressions / self.total_expressions if self.total_expressions else 0.0
        )
        call_ratio = (
            self.resolved_call_arguments / self.call_arguments if self.call_arguments else 0.0
        )
        average_statements = (self.statements / self.blocks) if self.blocks else 0.0
        return ReconstructionMetrics(
            procedures=self.procedures,
            cfg_nodes=self.cfg_nodes,
            cfg_edges=self.cfg_edges,
            entry_points=self.entry_points,
            assigned_blocks=self.assigned_blocks,
            typed_variable_ratio=typed_ratio,
            constant_expression_ratio=literal_ratio,
            call_resolution_ratio=call_ratio,
            max_expression_depth=self.max_expression_depth,
            average_statements_per_block=average_statements,
        )

