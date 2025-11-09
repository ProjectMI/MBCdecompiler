"""Data structures for the high level abstract syntax tree stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from ..ir.model import IRNode


@dataclass(frozen=True)
class ASTTerminator:
    """Canonical control-flow terminator description."""

    kind: str
    targets: Tuple[str, ...] = field(default_factory=tuple)
    detail: Optional[object] = None

    def describe(self) -> str:
        """Return a stable textual description of the terminator."""

        if self.kind == "goto" and self.targets:
            return f"goto {self.targets[0]}"
        if self.kind in {"return", "tailcall", "tailcall_return", "call_return"}:
            node = self.detail
            if isinstance(node, IRNode):
                return node.describe()
        if self.kind == "switch":
            node = self.detail
            if isinstance(node, IRNode):
                return node.describe()
            cases = ", ".join(self.targets)
            return f"switch [{cases}]"
        if self.kind == "predicate" and self.detail is not None:
            return str(self.detail)
        node = self.detail
        if isinstance(node, IRNode):
            return node.describe()
        return self.kind


@dataclass(frozen=True)
class ASTBlock:
    """Single block in the AST control-flow graph."""

    label: str
    start_offset: int
    statements: Tuple[IRNode, ...]
    terminator: ASTTerminator
    predecessors: Tuple[str, ...]
    successors: Tuple[str, ...]
    synthetic: bool = False


@dataclass(frozen=True)
class DominatorInfo:
    """Summary of a dominator/post-dominator tree."""

    root: str
    immediate: Mapping[str, Optional[str]]
    dominators: Mapping[str, Tuple[str, ...]]

    @staticmethod
    def freeze(
        root: str,
        immediate: Mapping[str, Optional[str]],
        dominators: Mapping[str, Sequence[str]],
    ) -> "DominatorInfo":
        """Create an immutable instance from mutable mappings."""

        immut_idom = MappingProxyType(dict(immediate))
        immut_dom = MappingProxyType({
            block: tuple(sorted(doms)) for block, doms in dominators.items()
        })
        return DominatorInfo(root=root, immediate=immut_idom, dominators=immut_dom)


@dataclass(frozen=True)
class ASTLoop:
    """Description of a natural loop detected in the CFG."""

    header: str
    latches: Tuple[str, ...]
    blocks: Tuple[str, ...]


@dataclass(frozen=True)
class ASTFunction:
    """High level representation of a single function."""

    segment_index: int
    name: str
    entry_block: str
    entry_offset: int
    blocks: Tuple[ASTBlock, ...]
    dominators: DominatorInfo
    post_dominators: DominatorInfo
    loops: Tuple[ASTLoop, ...]


@dataclass(frozen=True)
class ASTProgram:
    """High level AST representation for the entire container."""

    functions: Tuple[ASTFunction, ...]

