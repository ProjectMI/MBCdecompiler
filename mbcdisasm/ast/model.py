"""High-level AST representation derived from the normalised IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..ir.model import IRNode


@dataclass(frozen=True)
class ASTSwitchCase:
    """Single case in a structured switch terminator."""

    key: str
    target: str


@dataclass(frozen=True)
class ASTTerminator:
    """Canonical description of structured control-flow exits."""

    kind: str
    origin: Optional[IRNode] = None
    description: Optional[str] = None
    target: Optional[str] = None
    then_target: Optional[str] = None
    else_target: Optional[str] = None
    branch_kind: Optional[str] = None
    condition: Optional[str] = None
    cases: Tuple[ASTSwitchCase, ...] = tuple()
    default_target: Optional[str] = None

    def describe(self) -> str:
        """Render the terminator into a stable textual description."""

        if self.kind == "jump":
            target = self.target or "?"
            return f"jump -> {target}"
        if self.kind == "branch":
            cond = self.condition or "?"
            then_target = self.then_target or "?"
            else_target = self.else_target or "?"
            prefix = "branch"
            if self.branch_kind:
                prefix += f"[{self.branch_kind}]"
            return f"{prefix} {cond} then={then_target} else={else_target}"
        if self.kind == "switch":
            rendered_cases = ", ".join(f"{case.key}->{case.target}" for case in self.cases)
            default = self.default_target or "?"
            return f"switch cases=[{rendered_cases}] default={default}"
        if self.kind in {"return", "tailcall", "tailcall_return", "call_return", "terminator"}:
            if self.description:
                return self.description
            if self.origin is not None:
                describe = getattr(self.origin, "describe", None)
                if callable(describe):
                    return describe()
            return self.kind
        if self.description:
            return self.description
        if self.origin is not None:
            describe = getattr(self.origin, "describe", None)
            if callable(describe):
                return describe()
        return self.kind or "?"


@dataclass(frozen=True)
class ASTBlock:
    """Structured basic block with canonical control-flow metadata."""

    label: str
    start_offset: int
    body: Tuple[IRNode, ...]
    terminator: ASTTerminator
    predecessors: Tuple[str, ...]
    successors: Tuple[str, ...]


@dataclass(frozen=True)
class ASTDominatorInfo:
    """Summary of dominator relationships for a single block."""

    block: str
    dominators: Tuple[str, ...]
    immediate: Optional[str]


@dataclass(frozen=True)
class ASTLoop:
    """Natural loop recovered from the CFG."""

    header: str
    nodes: Tuple[str, ...]
    latches: Tuple[str, ...]


@dataclass(frozen=True)
class ASTFunction:
    """Structured view of a single function body."""

    segment_index: int
    name: str
    entry: str
    blocks: Tuple[ASTBlock, ...]
    dominators: Tuple[ASTDominatorInfo, ...]
    post_dominators: Tuple[ASTDominatorInfo, ...]
    loops: Tuple[ASTLoop, ...]


@dataclass(frozen=True)
class ASTProgram:
    """Top-level structured representation for the entire container."""

    functions: Tuple[ASTFunction, ...]


__all__ = [
    "ASTBlock",
    "ASTDominatorInfo",
    "ASTFunction",
    "ASTLoop",
    "ASTProgram",
    "ASTSwitchCase",
    "ASTTerminator",
]
