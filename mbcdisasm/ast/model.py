"""Dataclasses representing the lifted abstract syntax tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASTExpression:
    """Base node for expressions."""

    def describe(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class ASTIdentifier(ASTExpression):
    """Named SSA value or synthetic temporary."""

    name: str

    def describe(self) -> str:  # pragma: no cover - trivial
        return self.name


@dataclass(frozen=True)
class ASTLiteral(ASTExpression):
    """Immediate literal value encountered during lifting."""

    value: int

    def describe(self) -> str:
        return f"0x{self.value:04X}"


@dataclass(frozen=True)
class ASTSlotReference(ASTExpression):
    """Reference to a VM slot in one of the known spaces."""

    space: str
    index: int

    def describe(self) -> str:
        return f"{self.space}[0x{self.index:04X}]"


@dataclass(frozen=True)
class ASTIndirectReference(ASTExpression):
    """Pointer dereference produced from IRIndirectLoad/Store."""

    base: str
    offset: int
    pointer: Optional[str] = None
    ref: Optional[str] = None
    offset_source: Optional[str] = None

    def describe(self) -> str:
        pointer = self.pointer or self.base
        if self.offset_source:
            offset = self.offset_source
        else:
            offset = f"0x{self.offset:04X}"
        if self.ref:
            return f"*{self.ref}(ptr={pointer}, offset={offset})"
        return f"*{pointer}+{offset}"


@dataclass(frozen=True)
class ASTCallExpression(ASTExpression):
    """Call expression."""

    target: int
    args: Tuple[ASTExpression, ...]
    symbol: Optional[str] = None

    def describe(self) -> str:
        rendered_args = ", ".join(arg.describe() for arg in self.args)
        target = f"0x{self.target:04X}"
        if self.symbol:
            target = f"{self.symbol}({target})"
        return f"call {target}({rendered_args})"


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASTStatement:
    """Base node for lifted statements."""

    def describe(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class ASTExpressionStatement(ASTStatement):
    """Wrapper around standalone expressions."""

    expr: ASTExpression

    def describe(self) -> str:
        return self.expr.describe()


@dataclass(frozen=True)
class ASTAssignment(ASTStatement):
    """Assignment to a named temporary."""

    target: ASTIdentifier
    value: ASTExpression

    def describe(self) -> str:
        return f"{self.target.describe()} = {self.value.describe()}"


@dataclass(frozen=True)
class ASTStore(ASTStatement):
    """Store the supplied expression to a VM slot."""

    destination: ASTSlotReference
    value: ASTExpression

    def describe(self) -> str:
        return f"{self.destination.describe()} := {self.value.describe()}"


@dataclass(frozen=True)
class ASTIndirectStore(ASTStatement):
    """Store through an indirect pointer."""

    destination: ASTIndirectReference
    value: ASTExpression

    def describe(self) -> str:
        return f"{self.destination.describe()} := {self.value.describe()}"


@dataclass(frozen=True)
class ASTCallStatement(ASTStatement):
    """Invoke another routine."""

    call: ASTCallExpression
    tail: bool = False
    returns: Tuple[str, ...] = field(default_factory=tuple)
    cleanup: Tuple[str, ...] = field(default_factory=tuple)
    predicate: Optional[str] = None

    def describe(self) -> str:
        base = self.call.describe()
        if self.tail:
            base = "tail " + base
        if self.returns:
            base += " -> [" + ", ".join(self.returns) + "]"
        if self.cleanup:
            base += " cleanup=[" + ", ".join(self.cleanup) + "]"
        if self.predicate:
            base += f" predicate={self.predicate}"
        return base


@dataclass(frozen=True)
class ASTReturn(ASTStatement):
    """Return from the current procedure."""

    values: Tuple[ASTExpression, ...]
    varargs: bool = False
    cleanup: Tuple[str, ...] = field(default_factory=tuple)
    mask: Optional[int] = None

    def describe(self) -> str:
        rendered = ", ".join(value.describe() for value in self.values)
        prefix = "return"
        if self.varargs:
            prefix = "return varargs"
        details = []
        if rendered:
            details.append(f"[{rendered}]")
        if self.mask is not None:
            details.append(f"mask=0x{self.mask:04X}")
        if self.cleanup:
            details.append("cleanup=[" + ", ".join(self.cleanup) + "]")
        if not details:
            return prefix
        return f"{prefix} {' '.join(details)}"


@dataclass(frozen=True)
class ASTConditional(ASTStatement):
    """Structured conditional guard."""

    condition: str
    then_target: Optional[int]
    else_target: Optional[int]
    kind: str = "if"

    def describe(self) -> str:
        then_part = f"then=0x{self.then_target:04X}" if self.then_target is not None else "then=fallthrough"
        else_part = f"else=0x{self.else_target:04X}" if self.else_target is not None else "else=fallthrough"
        return f"{self.kind} {self.condition} {then_part} {else_part}"


@dataclass(frozen=True)
class ASTSwitch(ASTStatement):
    """Structured dispatch statement."""

    subject: str
    cases: Tuple[Tuple[int, Optional[str]], ...]
    default: Optional[int]

    def describe(self) -> str:
        rendered_cases = ", ".join(
            f"0x{key:04X}->" + (symbol if symbol is not None else "?")
            for key, symbol in self.cases
        )
        default = f" default=0x{self.default:04X}" if self.default is not None else ""
        return f"switch {self.subject} cases=[{rendered_cases}]" + default


@dataclass(frozen=True)
class ASTRawStatement(ASTStatement):
    """Fallback wrapper around IR nodes that don't have a dedicated form."""

    text: str

    def describe(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Blocks, functions, programme container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASTBlock:
    """Structured block produced after lifting."""

    label: str
    start_offset: int
    statements: Tuple[ASTStatement, ...]


@dataclass(frozen=True)
class ASTFunction:
    """Top-level function entry recovered from the control-flow graph."""

    name: str
    entry_offset: int
    blocks: Tuple[ASTBlock, ...]
    attributes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ASTProgram:
    """Root object for the lifted AST."""

    functions: Tuple[ASTFunction, ...]


__all__ = [
    "ASTProgram",
    "ASTFunction",
    "ASTBlock",
    "ASTStatement",
    "ASTExpression",
    "ASTAssignment",
    "ASTExpressionStatement",
    "ASTStore",
    "ASTIndirectStore",
    "ASTCallStatement",
    "ASTReturn",
    "ASTConditional",
    "ASTSwitch",
    "ASTRawStatement",
    "ASTIdentifier",
    "ASTLiteral",
    "ASTSlotReference",
    "ASTIndirectReference",
    "ASTCallExpression",
]
