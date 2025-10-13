"""Lift IR segments into a lightweight AST representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from mbcdisasm.ir.model import IRBlock, IRFunctionPrologue, IRProgram, IRSegment

from .cfg import CFGBuilder, ControlFlowGraph
from .model import ASTBlock, ASTFunction, ASTProgram, ASTStatement


@dataclass
class _FunctionPartition:
    entry: str
    labels: Set[str]


class ASTBuilder:
    """Construct an :class:`ASTProgram` from an :class:`IRProgram`."""

    def __init__(self, *, cfg_builder: Optional[CFGBuilder] = None) -> None:
        self._cfg_builder = cfg_builder or CFGBuilder()

    def build(self, program: IRProgram) -> ASTProgram:
        functions: List[ASTFunction] = []
        for segment in program.segments:
            cfg = self._cfg_builder.build(segment)
            functions.extend(self._build_segment(segment, cfg))
        return ASTProgram(functions=tuple(functions))

    # ------------------------------------------------------------------
    # segment handling
    # ------------------------------------------------------------------
    def _build_segment(
        self, segment: IRSegment, cfg: ControlFlowGraph
    ) -> Iterable[ASTFunction]:
        partitions = self._partition_segment(segment, cfg)
        label_to_block: Dict[str, IRBlock] = {block.label: block for block in segment.blocks}

        for partition in partitions:
            blocks = self._order_blocks(segment.blocks, partition.labels)
            ast_blocks = [
                self._build_block(label_to_block[block.label], cfg, partition.labels)
                for block in blocks
            ]
            if not blocks:
                continue
            entry_block = blocks[0]
            name = f"seg{segment.index}_fn_{entry_block.start_offset:04X}"
            function = ASTFunction(
                name=name,
                segment_index=segment.index,
                entry_label=partition.entry,
                start_offset=entry_block.start_offset,
                blocks=tuple(ast_blocks),
            )
            yield function

    def _partition_segment(
        self, segment: IRSegment, cfg: ControlFlowGraph
    ) -> Sequence[_FunctionPartition]:
        blocks = list(segment.blocks)
        if not blocks:
            return []

        entries = self._discover_entries(blocks)
        assigned: Set[str] = set()
        partitions: List[_FunctionPartition] = []

        for entry in entries:
            if entry in assigned:
                continue
            reachable = self._collect_reachable(cfg, entry, assigned)
            partitions.append(_FunctionPartition(entry=entry, labels=reachable))
            assigned.update(reachable)

        for block in blocks:
            if block.label in assigned:
                continue
            reachable = self._collect_reachable(cfg, block.label, assigned)
            partitions.append(_FunctionPartition(entry=block.label, labels=reachable))
            assigned.update(reachable)

        return partitions

    def _discover_entries(self, blocks: Sequence[IRBlock]) -> List[str]:
        entries: List[str] = []
        for index, block in enumerate(blocks):
            if self._is_prologue(block) or index == 0:
                entries.append(block.label)
        return entries

    def _collect_reachable(
        self,
        cfg: ControlFlowGraph,
        start_label: str,
        assigned: Set[str],
    ) -> Set[str]:
        visited: Set[str] = set()
        pending: List[str] = [start_label]
        while pending:
            label = pending.pop()
            if label in visited or label in assigned:
                continue
            visited.add(label)
            node = cfg.nodes.get(label)
            if node is None:
                continue
            for successor in node.successors:
                if successor not in assigned:
                    pending.append(successor)
        return visited

    def _order_blocks(
        self, blocks: Sequence[IRBlock], labels: Set[str]
    ) -> List[IRBlock]:
        return [block for block in blocks if block.label in labels]

    def _build_block(
        self, block: IRBlock, cfg: ControlFlowGraph, allowed: Set[str]
    ) -> ASTBlock:
        statements = tuple(
            ASTStatement(text=self._render_statement(node)) for node in block.nodes
        )
        cfg_node = cfg.nodes.get(block.label)
        if cfg_node is None:
            successors: Tuple[str, ...] = tuple()
        else:
            successors = tuple(successor for successor in cfg_node.successors if successor in allowed)
        return ASTBlock.from_ir_block(
            block,
            statements=statements,
            successors=successors,
        )

    def _render_statement(self, node) -> str:
        describe = getattr(node, "describe", None)
        if callable(describe):
            return describe()
        return repr(node)

    @staticmethod
    def _is_prologue(block: IRBlock) -> bool:
        return any(isinstance(node, IRFunctionPrologue) for node in block.nodes)

