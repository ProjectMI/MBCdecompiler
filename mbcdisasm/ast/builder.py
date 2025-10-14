"""Reconstruct a coarse AST on top of the IR normaliser output."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from mbcdisasm.ir.model import (
    IRBlock,
    IRCall,
    IRCallReturn,
    IRFunctionPrologue,
    IRIf,
    IRIndirectLoad,
    IRIndirectStore,
    IRLiteral,
    IRLoad,
    IRRaw,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
    IRProgram,
    IRStackDrop,
    IRStackDuplicate,
    IRStore,
)

from .model import (
    ASTBlock,
    ASTExpression,
    ASTMetrics,
    ASTProcedure,
    ASTProgram,
    ASTSegment,
    ASTStatement,
)

_IDENTIFIER_PREFIXES = {
    "word": "word",
    "byte": "byte",
    "ptr": "pointer",
    "bool": "bool",
    "io": "io",
    "page": "page",
    "ret": "return",
    "id": "identifier",
    "ssa": "ssa",
}


@dataclass(frozen=True)
class _CFGNode:
    """Internal helper structure describing basic block adjacency."""

    label: str
    start_offset: int
    successors: Tuple[str, ...]
    exits: Tuple[str, ...]


class ASTBuilder:
    """High level driver that maps IR segments into AST segments."""

    def build(self, program: IRProgram) -> ASTProgram:
        metrics = ASTMetrics()
        ast_segments: List[ASTSegment] = []
        for segment in program.segments:
            ast_segment = self._build_segment(segment, metrics)
            ast_segments.append(ast_segment)
        return ASTProgram(segments=tuple(ast_segments), metrics=metrics)

    # ------------------------------------------------------------------
    # segment level reconstruction
    # ------------------------------------------------------------------
    def _build_segment(self, segment: IRSegment, metrics: ASTMetrics) -> ASTSegment:
        blocks_by_label: Dict[str, IRBlock] = {block.label: block for block in segment.blocks}
        offsets: Dict[str, int] = {block.label: block.start_offset for block in segment.blocks}
        label_by_offset: Dict[int, str] = {block.start_offset: block.label for block in segment.blocks}

        cfg_nodes, dangling = self._build_cfg(segment.blocks, label_by_offset)
        entry_offsets = self._discover_entries(segment.blocks, label_by_offset, cfg_nodes)
        procedures = self._group_into_procedures(cfg_nodes, entry_offsets)

        ast_procedures: List[ASTProcedure] = []
        for entry_label, block_labels in procedures:
            ir_blocks = [blocks_by_label[label] for label in block_labels]
            ast_blocks = [self._build_block(block, cfg_nodes[label], metrics) for label, block in zip(block_labels, ir_blocks)]
            entry_offset = offsets[entry_label]
            exits = tuple(sorted(cfg_nodes[entry_label].exits))
            procedure = ASTProcedure(
                name=f"proc_{segment.index:02d}_{entry_offset:04X}",
                entry_label=entry_label,
                entry_offset=entry_offset,
                blocks=tuple(ast_blocks),
                exits=exits,
            )
            ast_procedures.append(procedure)
            metrics.procedures += 1
            metrics.blocks += len(ast_blocks)
            metrics.cfg_edges += sum(len(block.successors) for block in ast_blocks)

        metrics.observe_dangling(len(dangling))

        return ASTSegment(
            index=segment.index,
            start=segment.start,
            length=segment.length,
            procedures=tuple(ast_procedures),
            entry_offsets=tuple(sorted(entry_offsets)),
            dangling_targets=tuple(sorted(dangling)),
        )

    # ------------------------------------------------------------------
    # CFG construction and procedure grouping
    # ------------------------------------------------------------------
    def _build_cfg(
        self,
        blocks: Sequence[IRBlock],
        label_by_offset: Dict[int, str],
    ) -> Tuple[Dict[str, _CFGNode], Set[int]]:
        cfg: Dict[str, _CFGNode] = {}
        dangling_targets: Set[int] = set()
        ordered_blocks = sorted(blocks, key=lambda block: block.start_offset)
        offset_to_next: Dict[int, Optional[int]] = {}
        for index, block in enumerate(ordered_blocks):
            next_offset = ordered_blocks[index + 1].start_offset if index + 1 < len(ordered_blocks) else None
            offset_to_next[block.start_offset] = next_offset

        for block in ordered_blocks:
            successors: Set[int] = set()
            exits: Set[str] = set()
            terminated = False
            for node in block.nodes:
                if isinstance(node, (IRIf, IRTestSetBranch, IRFunctionPrologue)):
                    successors.add(node.then_target)
                    successors.add(node.else_target)
                elif isinstance(node, IRSwitchDispatch):
                    for case in node.cases:
                        successors.add(case.target)
                    if node.default is not None:
                        successors.add(node.default)
                elif isinstance(node, (IRReturn, IRTailCall, IRTailcallReturn, IRTerminator)):
                    terminated = True
                    exits.add(node.__class__.__name__)
                elif isinstance(node, IRCallReturn):
                    terminated = True
                    exits.add("IRCallReturn")
            if not terminated:
                fallthrough = offset_to_next.get(block.start_offset)
                if fallthrough is not None:
                    successors.add(fallthrough)
            resolved_successors: List[str] = []
            for target in successors:
                label = label_by_offset.get(target)
                if label is None:
                    dangling_targets.add(target)
                    continue
                resolved_successors.append(label)
            cfg[block.label] = _CFGNode(
                label=block.label,
                start_offset=block.start_offset,
                successors=tuple(sorted(resolved_successors)),
                exits=tuple(sorted(exits)),
            )
        return cfg, dangling_targets

    def _discover_entries(
        self,
        blocks: Sequence[IRBlock],
        label_by_offset: Dict[int, str],
        cfg_nodes: Dict[str, _CFGNode],
    ) -> Set[int]:
        entries: Set[int] = set()
        if blocks:
            entries.add(blocks[0].start_offset)
        for block in blocks:
            for node in block.nodes:
                if isinstance(node, IRFunctionPrologue):
                    entries.add(block.start_offset)
                if isinstance(node, (IRCall, IRCallReturn, IRTailCall, IRTailcallReturn)):
                    label = label_by_offset.get(node.target)
                    if label is not None:
                        entries.add(cfg_nodes[label].start_offset)
        return entries

    def _group_into_procedures(
        self,
        cfg_nodes: Dict[str, _CFGNode],
        entry_offsets: Set[int],
    ) -> List[Tuple[str, List[str]]]:
        offset_to_label = {node.start_offset: node.label for node in cfg_nodes.values()}
        entry_labels = [offset_to_label[offset] for offset in sorted(entry_offsets) if offset in offset_to_label]
        visited: Set[str] = set()
        procedures: List[Tuple[str, List[str]]] = []

        for entry_label in entry_labels:
            if entry_label in visited:
                continue
            owned: Set[str] = set()
            queue: deque[str] = deque([entry_label])
            while queue:
                current = queue.popleft()
                if current in owned:
                    continue
                owned.add(current)
                node = cfg_nodes[current]
                for successor in node.successors:
                    successor_offset = cfg_nodes[successor].start_offset
                    if successor in owned or successor in visited:
                        continue
                    if successor_offset in entry_offsets and successor != entry_label:
                        continue
                    queue.append(successor)
            for label in owned:
                visited.add(label)
            ordered = sorted(owned, key=lambda label: cfg_nodes[label].start_offset)
            procedures.append((entry_label, ordered))

        remaining = [label for label in cfg_nodes if label not in visited]
        for label in remaining:
            procedures.append((label, [label]))
        procedures.sort(key=lambda item: cfg_nodes[item[0]].start_offset)
        return procedures

    # ------------------------------------------------------------------
    # block -> statement lowering
    # ------------------------------------------------------------------
    def _build_block(
        self,
        block: IRBlock,
        cfg_node: _CFGNode,
        metrics: ASTMetrics,
    ) -> ASTBlock:
        value_map: Dict[str, ASTExpression] = {}
        statements: List[ASTStatement] = []
        resolved = 0
        unknown = 0

        for node in block.nodes:
            builder = getattr(self, f"_handle_{node.__class__.__name__.lower()}", None)
            if callable(builder):
                statement, produced = builder(node, value_map, metrics)
                if statement is not None:
                    statements.append(statement)
                resolved += produced[0]
                unknown += produced[1]
            else:
                statements.append(ASTStatement(kind="raw", text=self._describe(node)))

        metrics.observe_identifiers(resolved, unknown)
        return ASTBlock(
            label=block.label,
            start_offset=block.start_offset,
            successors=cfg_node.successors,
            statements=tuple(statements),
        )

    # ------------------------------------------------------------------
    # node handlers
    # ------------------------------------------------------------------
    def _handle_irliteral(
        self,
        node: IRLiteral,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        expr = ASTExpression(kind="literal", value=node.value)
        metrics.observe_literals()
        text = f"literal 0x{node.value:04X} ({node.source})"
        return ASTStatement(kind="literal", text=text, expressions=(expr,)), (0, 0)

    def _handle_irload(
        self,
        node: IRLoad,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        target = self._ensure_identifier(node.target, value_map)
        source = ASTExpression(kind="slot", value=node.slot)
        value_map[node.target] = source
        metrics.observe_load()
        text = f"{node.target} = load {source.describe()}"
        return ASTStatement(kind="load", text=text, expressions=(target, source)), (1, 0)

    def _handle_irstore(
        self,
        node: IRStore,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        value = self._resolve_value(node.value, value_map)
        target = ASTExpression(kind="slot", value=node.slot)
        metrics.observe_store()
        text = f"store {value.describe()} -> {target.describe()}"
        return ASTStatement(kind="store", text=text, expressions=(value, target)), self._resolved_flags(value)

    def _handle_irindirectload(
        self,
        node: IRIndirectLoad,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        base = self._resolve_value(node.base, value_map)
        offset_repr = f"0x{node.offset:04X}"
        value = ASTExpression(
            kind="indirect_load",
            value={"offset_repr": offset_repr, "ref": node.ref},
            operands=(base,),
        )
        value_map[node.target] = value
        metrics.observe_load(indirect=True)
        target = self._ensure_identifier(node.target, value_map)
        text = f"{node.target} = {value.describe()}"
        return ASTStatement(kind="indirect_load", text=text, expressions=(target, value)), self._merge_flags(base, value)

    def _handle_irindirectstore(
        self,
        node: IRIndirectStore,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        base = self._resolve_value(node.base, value_map)
        value = self._resolve_value(node.value, value_map)
        offset_repr = f"0x{node.offset:04X}"
        target = ASTExpression(
            kind="indirect_store",
            value={"offset_repr": offset_repr, "ref": node.ref},
            operands=(base,),
        )
        metrics.observe_store(indirect=True)
        text = f"store {value.describe()} -> {target.describe()}"
        return ASTStatement(kind="indirect_store", text=text, expressions=(value, target)), self._merge_flags(base, value)

    def _handle_ircall(
        self,
        node: IRCall,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        args = [self._resolve_value(arg, value_map) for arg in node.args]
        metrics.observe_calls(tail=node.tail)
        target = self._format_call_target(node)
        suffix = ' tail' if node.tail else ''
        text = f"call{suffix} {target}({', '.join(expr.describe() for expr in args)})"
        if node.predicate is not None:
            text += f" predicate={node.predicate.describe()}"
        return ASTStatement(kind="call", text=text, expressions=tuple(args)), self._sum_flags(args)

    def _handle_ircallreturn(
        self,
        node: IRCallReturn,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        args = [self._resolve_value(arg, value_map) for arg in node.args]
        returns = [self._resolve_value(ret, value_map) for ret in node.returns]
        metrics.observe_calls(tail=node.tail)
        metrics.observe_returns()
        target = self._format_call_target(node)
        suffix = ' tail' if node.tail else ''
        text = (
            f"call_return{suffix} {target} args=[{', '.join(arg.describe() for arg in args)}] "
            f"returns=[{', '.join(ret.describe() for ret in returns)}]"
        ).strip()
        return ASTStatement(
            kind="call_return",
            text=text,
            expressions=tuple(args + returns),
        ), self._merge_lists(args, returns)

    def _handle_irreturn(
        self,
        node: IRReturn,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        values = [self._resolve_value(value, value_map) for value in node.values]
        metrics.observe_returns()
        if node.varargs:
            text = "return varargs"
            if values:
                text += "(" + ", ".join(expr.describe() for expr in values) + ")"
        else:
            text = "return " + ", ".join(expr.describe() for expr in values)
        return ASTStatement(kind="return", text=text, expressions=tuple(values)), self._sum_flags(values)

    def _handle_irtestsetbranch(
        self,
        node: IRTestSetBranch,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        var_expr = self._resolve_value(node.var, value_map)
        expr_expr = self._resolve_value(node.expr, value_map)
        text = (
            f"testset {var_expr.describe()} = {expr_expr.describe()} then=0x{node.then_target:04X} "
            f"else=0x{node.else_target:04X}"
        )
        return ASTStatement(
            kind="testset",
            text=text,
            expressions=(var_expr, expr_expr),
        ), self._merge_flags(var_expr, expr_expr)

    def _handle_irif(
        self,
        node: IRIf,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        condition = self._resolve_value(node.condition, value_map)
        text = f"if {condition.describe()} then=0x{node.then_target:04X} else=0x{node.else_target:04X}"
        return ASTStatement(kind="if", text=text, expressions=(condition,)), self._resolved_flags(condition)

    def _handle_irraw(
        self,
        node: IRRaw,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        return ASTStatement(kind="raw", text=self._describe(node)), (0, 0)

    def _handle_irstackduplicate(
        self,
        node: IRStackDuplicate,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        value = self._resolve_value(node.value, value_map)
        text = f"dup {value.describe()} copies={node.copies}"
        return ASTStatement(kind="stack", text=text, expressions=(value,)), self._resolved_flags(value)

    def _handle_irstackdrop(
        self,
        node: IRStackDrop,
        value_map: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[Optional[ASTStatement], Tuple[int, int]]:
        value = self._resolve_value(node.value, value_map)
        text = f"drop {value.describe()}"
        return ASTStatement(kind="stack", text=text, expressions=(value,)), self._resolved_flags(value)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _describe(self, node: object) -> str:
        describe = getattr(node, "describe", None)
        if callable(describe):
            return describe()
        return repr(node)

    def _format_call_target(self, node: IRCall) -> str:
        target = f"0x{node.target:04X}"
        if getattr(node, "symbol", None):
            return f"{node.symbol}({target})"
        return target

    def _resolve_value(
        self,
        reference: str,
        value_map: Dict[str, ASTExpression],
    ) -> ASTExpression:
        if reference in value_map:
            return value_map[reference]
        expr = self._ensure_identifier(reference, value_map)
        value_map[reference] = expr
        return expr

    def _ensure_identifier(
        self,
        name: str,
        value_map: Dict[str, ASTExpression],
    ) -> ASTExpression:
        if name in value_map:
            return value_map[name]
        kind = "unknown"
        for prefix, hint in _IDENTIFIER_PREFIXES.items():
            if name.startswith(prefix):
                kind = hint
                break
        expr = ASTExpression(kind="identifier", value=name, type_hint=kind if kind != "unknown" else None)
        value_map[name] = expr
        return expr

    def _resolved_flags(self, expr: ASTExpression) -> Tuple[int, int]:
        if expr.kind == "identifier" and expr.type_hint is None:
            return (0, 1)
        if expr.kind == "raw":
            return (0, 1)
        return (1, 0)

    def _merge_flags(self, *exprs: ASTExpression) -> Tuple[int, int]:
        resolved = 0
        unknown = 0
        for expr in exprs:
            res, unk = self._resolved_flags(expr)
            resolved += res
            unknown += unk
        return resolved, unknown

    def _merge_lists(self, lhs: List[ASTExpression], rhs: List[ASTExpression]) -> Tuple[int, int]:
        resolved = 0
        unknown = 0
        for expr in lhs + rhs:
            res, unk = self._resolved_flags(expr)
            resolved += res
            unknown += unk
        return resolved, unknown

    def _sum_flags(self, exprs: Iterable[ASTExpression]) -> Tuple[int, int]:
        resolved = 0
        unknown = 0
        for expr in exprs:
            res, unk = self._resolved_flags(expr)
            resolved += res
            unknown += unk
        return resolved, unknown
