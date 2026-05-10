from __future__ import annotations

"""Small data objects used by the experimental pseudo-AST layer."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AstStatement:
    offset: int
    file_offset: int
    kind: str
    text: str
    opcode: int
    mnemonic: str
    operands: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ast_payload(
    *,
    statements: list[AstStatement],
    residual_stack: list[str],
    underflows: int,
    warning: str | None = None,
) -> dict[str, Any]:
    return {
        "format": "experimental_stack_ast_v0",
        "warning": warning or (
            "This is a symbolic stack AST seed, not a final decompilation. "
            "It is meant to preserve expression intent and control-flow anchors for later structuring."
        ),
        "statement_count": len(statements),
        "underflow_placeholders": underflows,
        "residual_stack": list(residual_stack[-16:]),
        "statements": [stmt.to_dict() for stmt in statements],
        "source": "\n".join(stmt.text for stmt in statements),
    }
