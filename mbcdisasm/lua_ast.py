"""Lightweight Lua abstract syntax tree helpers.

The high level reconstructor historically emitted raw strings which made it
extremely difficult to reason about control-flow once more advanced
restructuring heuristics entered the scene.  This module provides a compact set
of dataclasses that model a Lua-like language.  They are intentionally
minimalistic – the goal is not to be a perfect AST representation but rather a
tool that captures the constructs we emit (assignments, control flow
statements, helper invocations, table constructors, …) in a structured manner.

Statements expose an :meth:`emit` method which receives a :class:`LuaWriter`
instance.  Expressions simply render to strings which keeps the integration
points small while still allowing higher level passes (such as the control-flow
structurer) to analyse the emitted code before it becomes immutable text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from .lua_formatter import LuaWriter


# ---------------------------------------------------------------------------
# expression nodes
# ---------------------------------------------------------------------------


class LuaExpression:
    """Base class for all Lua expression nodes."""

    def render(self) -> str:
        raise NotImplementedError


@dataclass
class LiteralExpr(LuaExpression):
    value: str
    py_value: object | None = None

    def render(self) -> str:
        return self.value


@dataclass
class NameExpr(LuaExpression):
    name: str

    def render(self) -> str:
        return self.name


@dataclass
class UnaryExpr(LuaExpression):
    operator: str
    operand: LuaExpression

    def render(self) -> str:
        inner = self.operand.render()
        if isinstance(self.operand, BinaryExpr):
            inner = f"({inner})"
        return f"{self.operator}{inner}"


@dataclass
class BinaryExpr(LuaExpression):
    left: LuaExpression
    operator: str
    right: LuaExpression

    def render(self) -> str:
        lhs = self.left.render()
        rhs = self.right.render()
        return f"{lhs} {self.operator} {rhs}"


@dataclass
class CallExpr(LuaExpression):
    callee: LuaExpression
    arguments: Sequence[LuaExpression] = field(default_factory=list)

    def render(self) -> str:
        args = ", ".join(arg.render() for arg in self.arguments)
        return f"{self.callee.render()}({args})"


@dataclass
class MethodCallExpr(LuaExpression):
    target: LuaExpression
    method: str
    arguments: Sequence[LuaExpression] = field(default_factory=list)

    def render(self) -> str:
        args = ", ".join(arg.render() for arg in self.arguments)
        return f"{self.target.render()}:{self.method}({args})"


@dataclass
class TableField:
    key: Optional[LuaExpression]
    value: LuaExpression

    def render(self) -> str:
        if self.key is None:
            return self.value.render()
        key = self.key.render()
        return f"[{key}] = {self.value.render()}"


@dataclass
class TableExpr(LuaExpression):
    fields: Sequence[TableField] = field(default_factory=list)

    def render(self) -> str:
        if not self.fields:
            return "{}"
        rendered = [field.render() for field in self.fields]
        return f"{{{', '.join(rendered)}}}"


# ---------------------------------------------------------------------------
# statement nodes
# ---------------------------------------------------------------------------


class LuaStatement:
    """Base class for all emitted Lua statements."""

    def emit(self, writer: LuaWriter) -> None:
        raise NotImplementedError


@dataclass
class CommentStatement(LuaStatement):
    text: str

    def emit(self, writer: LuaWriter) -> None:
        writer.write_comment(self.text)


@dataclass
class BlankLine(LuaStatement):
    def emit(self, writer: LuaWriter) -> None:  # pragma: no cover - trivial
        writer.write_line("")


@dataclass
class Assignment(LuaStatement):
    targets: Sequence[LuaExpression]
    value: LuaExpression
    is_local: bool = True

    def emit(self, writer: LuaWriter) -> None:
        lhs = ", ".join(target.render() for target in self.targets)
        prefix = "local " if self.is_local else ""
        writer.write_line(f"{prefix}{lhs} = {self.value.render()}")


@dataclass
class MultiAssignment(LuaStatement):
    targets: Sequence[LuaExpression]
    values: Sequence[LuaExpression]
    is_local: bool = True

    def emit(self, writer: LuaWriter) -> None:
        lhs = ", ".join(target.render() for target in self.targets)
        rhs = ", ".join(value.render() for value in self.values)
        prefix = "local " if self.is_local else ""
        writer.write_line(f"{prefix}{lhs} = {rhs}")


@dataclass
class CallStatement(LuaStatement):
    expression: LuaExpression

    def emit(self, writer: LuaWriter) -> None:
        writer.write_line(self.expression.render())


@dataclass
class ReturnStatement(LuaStatement):
    values: Sequence[LuaExpression] = field(default_factory=list)

    def emit(self, writer: LuaWriter) -> None:
        if not self.values:
            writer.write_line("return")
        else:
            rendered = ", ".join(value.render() for value in self.values)
            writer.write_line(f"return {rendered}")


@dataclass
class BreakStatement(LuaStatement):
    def emit(self, writer: LuaWriter) -> None:
        writer.write_line("break")


@dataclass
class BlockStatement(LuaStatement):
    statements: List[LuaStatement] = field(default_factory=list)

    def extend(self, other: Iterable[LuaStatement]) -> None:
        for statement in other:
            self.statements.append(statement)

    def emit(self, writer: LuaWriter) -> None:
        for statement in self.statements:
            statement.emit(writer)


@dataclass
class IfClause:
    condition: Optional[LuaExpression]
    body: BlockStatement

    def emit(self, writer: LuaWriter, *, is_first: bool, is_else: bool) -> None:
        if is_else:
            writer.write_line("else")
        elif is_first:
            writer.write_line(f"if {self.condition.render()} then")
        else:
            writer.write_line(f"elseif {self.condition.render()} then")
        with writer.indented():
            self.body.emit(writer)


@dataclass
class IfStatement(LuaStatement):
    clauses: Sequence[IfClause]

    def emit(self, writer: LuaWriter) -> None:
        for index, clause in enumerate(self.clauses):
            is_else = clause.condition is None
            clause.emit(
                writer,
                is_first=index == 0,
                is_else=is_else,
            )
        writer.write_line("end")


@dataclass
class WhileStatement(LuaStatement):
    condition: LuaExpression
    body: BlockStatement

    def emit(self, writer: LuaWriter) -> None:
        writer.write_line(f"while {self.condition.render()} do")
        with writer.indented():
            self.body.emit(writer)
        writer.write_line("end")


@dataclass
class RepeatStatement(LuaStatement):
    body: BlockStatement
    condition: LuaExpression

    def emit(self, writer: LuaWriter) -> None:
        writer.write_line("repeat")
        with writer.indented():
            self.body.emit(writer)
        writer.write_line(f"until {self.condition.render()}")


@dataclass
class NumericForStatement(LuaStatement):
    variable: LuaExpression
    start: LuaExpression
    stop: LuaExpression
    step: Optional[LuaExpression]
    body: BlockStatement

    def emit(self, writer: LuaWriter) -> None:
        header = (
            f"for {self.variable.render()} = {self.start.render()}, {self.stop.render()}"
        )
        if self.step is not None:
            header += f", {self.step.render()}"
        header += " do"
        writer.write_line(header)
        with writer.indented():
            self.body.emit(writer)
        writer.write_line("end")


@dataclass
class GenericForStatement(LuaStatement):
    variables: Sequence[LuaExpression]
    iterator: Sequence[LuaExpression]
    body: BlockStatement

    def emit(self, writer: LuaWriter) -> None:
        vars_rendered = ", ".join(var.render() for var in self.variables)
        iter_rendered = ", ".join(expr.render() for expr in self.iterator)
        writer.write_line(f"for {vars_rendered} in {iter_rendered} do")
        with writer.indented():
            self.body.emit(writer)
        writer.write_line("end")


@dataclass
class SwitchCase:
    values: Sequence[LuaExpression]
    body: BlockStatement


@dataclass
class SwitchStatement(LuaStatement):
    expression: LuaExpression
    cases: Sequence[SwitchCase]
    default: Optional[BlockStatement] = None

    def emit(self, writer: LuaWriter) -> None:
        writer.write_comment("switch emulation")
        for index, case in enumerate(self.cases):
            condition = " or ".join(
                f"{self.expression.render()} == {value.render()}" for value in case.values
            )
            keyword = "if" if index == 0 else "elseif"
            writer.write_line(f"{keyword} {condition} then")
            with writer.indented():
                case.body.emit(writer)
        if self.default is not None:
            writer.write_line("else")
            with writer.indented():
                self.default.emit(writer)
        writer.write_line("end")


def wrap_block(statements: Iterable[LuaStatement]) -> BlockStatement:
    block = BlockStatement()
    block.extend(statements)
    return block

