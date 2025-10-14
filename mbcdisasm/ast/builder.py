"""AST reconstruction pipeline built on top of the normalised IR."""

from __future__ import annotations

from collections import deque
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..ir.model import (
    IRAsciiHeader,
    IRBlock,
    IRCall,
    IRCallReturn,
    IRConditionMask,
    IRFlagCheck,
    IRFunctionPrologue,
    IRIf,
    IRIndirectLoad,
    IRIndirectStore,
    IRIORead,
    IRIOWrite,
    IRLoad,
    IRLiteral,
    IRLiteralBlock,
    IRLiteralChunk,
    IRNode,
    IRProgram,
    IRReturn,
    IRSegment,
    IRStackDuplicate,
    IRStackEffect,
    IRStackDrop,
    IRStore,
    IRSwitchDispatch,
    IRTailCall,
    IRTailcallReturn,
    IRTestSetBranch,
    IRTerminator,
    MemRef,
    SSAValueKind,
)
from .model import (
    ASTAssignment,
    ASTBlock,
    ASTBranchStatement,
    ASTCallResultExpr,
    ASTCallStatement,
    ASTEdge,
    ASTIdentifier,
    ASTIndirectLoadExpr,
    ASTIndirectStoreStatement,
    ASTLiteral,
    ASTMetrics,
    ASTProcedure,
    ASTProgram,
    ASTReturnStatement,
    ASTSegment,
    ASTStatement,
    ASTStoreStatement,
    ASTSwitchCase,
    ASTSwitchStatement,
    ASTUnstructuredStatement,
    ASTUnknownExpr,
    ASTLoadExpr,
    ASTStringLiteral,
    ASTExpression,
)


class ASTBuilder:
    """Convert :class:`IRProgram` instances into high-level AST structures."""

    _NAME_KIND_RE = re.compile(r"^(?P<prefix>[a-z]+)[0-9]+$")

    def build_program(self, program: IRProgram) -> ASTProgram:
        """Build an :class:`ASTProgram` from ``program``."""

        segments: List[ASTSegment] = []
        aggregate_metrics = ASTMetrics()

        for segment in program.segments:
            built = self._build_segment(segment)
            segments.append(built)
            aggregate_metrics.observe(built.metrics)

        return ASTProgram(segments=tuple(segments), metrics=aggregate_metrics)

    # ------------------------------------------------------------------
    # segment processing
    # ------------------------------------------------------------------
    def _build_segment(self, segment: IRSegment) -> ASTSegment:
        blocks = segment.blocks
        offset_to_label = {block.start_offset: block.label for block in blocks}
        label_to_offset = {block.label: block.start_offset for block in blocks}
        order = {block.label: index for index, block in enumerate(blocks)}

        successors = self._build_successor_map(blocks, offset_to_label, label_to_offset)

        block_objects: List[ASTBlock] = []
        block_metrics = ASTMetrics()
        block_map: Dict[str, ASTBlock] = {}

        for block in blocks:
            edges = successors.get(block.label, ())
            ast_block, metrics = self._lower_block(block, edges, offset_to_label, label_to_offset)
            block_metrics.observe(metrics)
            block_objects.append(ast_block)
            block_map[block.label] = ast_block

        entries = self._discover_entries(blocks, offset_to_label)
        assignment, entry_set = self._partition_procedures(blocks, successors, entries)
        procedures = self._build_procedures(block_map, assignment, entry_set, order)

        segment_metrics = ASTMetrics()
        segment_metrics.observe(block_metrics)
        segment_metrics.procedures = len(procedures)

        return ASTSegment(
            index=segment.index,
            start=segment.start,
            length=segment.length,
            blocks=tuple(block_objects),
            procedures=tuple(procedures),
            metrics=segment_metrics,
        )

    def _build_successor_map(
        self,
        blocks: Sequence[IRBlock],
        offset_to_label: Dict[int, str],
        label_to_offset: Dict[str, int],
    ) -> Dict[str, Tuple[ASTEdge, ...]]:
        mapping: Dict[str, Tuple[ASTEdge, ...]] = {}
        for index, block in enumerate(blocks):
            fallthrough = blocks[index + 1].label if index + 1 < len(blocks) else None
            edges = tuple(self._block_successors(block, fallthrough, offset_to_label, label_to_offset))
            mapping[block.label] = edges
        return mapping

    def _block_successors(
        self,
        block: IRBlock,
        fallthrough: Optional[str],
        offset_to_label: Dict[int, str],
        label_to_offset: Dict[str, int],
    ) -> Iterable[ASTEdge]:
        edges: List[ASTEdge] = []
        for node in reversed(block.nodes):
            terminator = self._is_explicit_terminator(node)
            if terminator:
                return edges
            if isinstance(node, IRIf):
                edges.append(self._edge("then", node.then_target, offset_to_label))
                edges.append(self._edge("else", node.else_target, offset_to_label))
                return edges
            if isinstance(node, IRTestSetBranch):
                edges.append(self._edge("then", node.then_target, offset_to_label))
                edges.append(self._edge("else", node.else_target, offset_to_label))
                return edges
            if isinstance(node, IRFlagCheck):
                edges.append(self._edge("then", node.then_target, offset_to_label))
                edges.append(self._edge("else", node.else_target, offset_to_label))
                return edges
            if isinstance(node, IRFunctionPrologue):
                edges.append(self._edge("then", node.then_target, offset_to_label))
                edges.append(self._edge("else", node.else_target, offset_to_label))
                return edges
            if isinstance(node, IRSwitchDispatch):
                for case in node.cases:
                    detail = f"case=0x{case.key:04X}"
                    edges.append(self._edge("case", case.target, offset_to_label, detail))
                if node.default is not None:
                    edges.append(self._edge("default", node.default, offset_to_label))
                return edges
        if fallthrough is not None:
            edges.append(
                ASTEdge(
                    kind="fallthrough",
                    target_label=fallthrough,
                    target_offset=label_to_offset.get(fallthrough),
                    detail=None,
                )
            )
        return edges

    def _edge(
        self,
        kind: str,
        target_offset: Optional[int],
        offset_to_label: Dict[int, str],
        detail: Optional[str] = None,
    ) -> ASTEdge:
        label = offset_to_label.get(target_offset) if target_offset is not None else None
        return ASTEdge(kind=kind, target_label=label, target_offset=target_offset, detail=detail)

    def _is_explicit_terminator(self, node: IRNode) -> bool:
        if isinstance(node, (IRReturn, IRTerminator, IRCallReturn, IRTailcallReturn)):
            return True
        if isinstance(node, (IRTailCall,)):
            return True
        if isinstance(node, IRCall) and node.tail:
            return True
        return False

    # ------------------------------------------------------------------
    # procedure discovery
    # ------------------------------------------------------------------
    def _discover_entries(
        self, blocks: Sequence[IRBlock], offset_to_label: Dict[int, str]
    ) -> Set[str]:
        entries: Set[str] = set()
        if blocks:
            entries.add(blocks[0].label)
        for block in blocks:
            for node in block.nodes:
                if isinstance(node, IRFunctionPrologue):
                    entries.add(block.label)
                elif isinstance(node, (IRCall, IRCallReturn, IRTailCall, IRTailcallReturn)):
                    label = offset_to_label.get(node.target)
                    if label:
                        entries.add(label)
                elif isinstance(node, IRSwitchDispatch):
                    for case in node.cases:
                        label = offset_to_label.get(case.target)
                        if label:
                            entries.add(label)
                    if node.default is not None:
                        label = offset_to_label.get(node.default)
                        if label:
                            entries.add(label)
        return entries

    def _partition_procedures(
        self,
        blocks: Sequence[IRBlock],
        successors: Dict[str, Tuple[ASTEdge, ...]],
        entries: Set[str],
    ) -> Tuple[Dict[str, str], Set[str]]:
        order = {block.label: index for index, block in enumerate(blocks)}
        assignment: Dict[str, str] = {}
        entry_set = set(entries)

        for entry in sorted(entry_set, key=lambda label: order.get(label, 0)):
            if entry in assignment:
                continue
            if entry not in order:
                continue
            queue = deque([entry])
            while queue:
                label = queue.popleft()
                if label in assignment:
                    continue
                assignment[label] = entry
                for edge in successors.get(label, ()):  # type: ignore[arg-type]
                    target = edge.target_label
                    if not target:
                        continue
                    if target in entry_set and target != entry:
                        continue
                    queue.append(target)

        for block in blocks:
            if block.label not in assignment:
                assignment[block.label] = block.label
                entry_set.add(block.label)

        return assignment, entry_set

    def _build_procedures(
        self,
        block_map: Dict[str, ASTBlock],
        assignment: Dict[str, str],
        entries: Set[str],
        order: Dict[str, int],
    ) -> List[ASTProcedure]:
        procedures: List[ASTProcedure] = []
        for entry in sorted(entries, key=lambda label: order.get(label, 0)):
            members = [label for label, root in assignment.items() if root == entry]
            if not members:
                continue
            members.sort(key=lambda label: order.get(label, 0))
            start_offset = block_map.get(entry).start_offset if entry in block_map else 0
            return_count = 0
            tail_call_count = 0
            call_targets: Set[str] = set()
            for label in members:
                block = block_map.get(label)
                if not block:
                    continue
                for statement in block.statements:
                    if isinstance(statement, ASTReturnStatement):
                        return_count += 1
                    elif isinstance(statement, ASTCallStatement):
                        if statement.tail or "tail" in statement.call_type:
                            tail_call_count += 1
                        target_repr = None
                        if statement.symbol:
                            target_repr = statement.symbol
                        elif statement.target_label:
                            target_repr = statement.target_label
                        else:
                            target_repr = f"0x{statement.target_offset:04X}"
                        call_targets.add(target_repr)
            procedures.append(
                ASTProcedure(
                    entry_label=entry,
                    start_offset=start_offset,
                    block_labels=tuple(members),
                    return_count=return_count,
                    tail_call_count=tail_call_count,
                    call_targets=tuple(sorted(call_targets)),
                )
            )
        return procedures

    # ------------------------------------------------------------------
    # block lowering
    # ------------------------------------------------------------------
    def _lower_block(
        self,
        block: IRBlock,
        successors: Sequence[ASTEdge],
        offset_to_label: Dict[int, str],
        label_to_offset: Dict[str, int],
    ) -> Tuple[ASTBlock, ASTMetrics]:
        env: Dict[str, ASTExpression] = {}
        statements: List[ASTStatement] = []
        metrics = ASTMetrics()
        metrics.blocks = 1

        for node in block.nodes:
            statement = self._convert_node(node, env, metrics, offset_to_label)
            if statement is not None:
                statements.append(statement)

        return (
            ASTBlock(
                label=block.label,
                start_offset=block.start_offset,
                statements=tuple(statements),
                successors=tuple(successors),
                annotations=block.annotations,
            ),
            metrics,
        )

    def _convert_node(
        self,
        node: IRNode,
        env: Dict[str, ASTExpression],
        metrics: ASTMetrics,
        offset_to_label: Dict[int, str],
    ) -> Optional[ASTStatement]:
        if isinstance(node, IRLoad):
            target_kind = self._infer_kind(node.target)
            expr = ASTLoadExpr(slot=node.slot, kind=target_kind)
            env[node.target] = expr
            target = ASTIdentifier(name=node.target, kind=target_kind, definition=expr)
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.loads += 1
            if target_kind is not None:
                metrics.typed_values += 1
            return ASTAssignment(target=target, expr=expr)

        if isinstance(node, IRIndirectLoad):
            target_kind = self._infer_kind(node.target)
            base_expr = self._resolve_expr(node.base, env, metrics)
            offset_expr = (
                self._resolve_expr(node.offset_source, env, metrics)
                if node.offset_source
                else ASTLiteral(value=node.offset & 0xFFFF, kind=self._literal_kind(node.offset))
            )
            pointer_expr = (
                self._resolve_expr(node.pointer, env, metrics) if node.pointer else None
            )
            expr = ASTIndirectLoadExpr(
                base=base_expr,
                offset=offset_expr,
                pointer=pointer_expr,
                ref=node.ref,
                kind=target_kind,
            )
            env[node.target] = expr
            target = ASTIdentifier(name=node.target, kind=target_kind, definition=expr)
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.indirect_loads += 1
            if target_kind is not None:
                metrics.typed_values += 1
            return ASTAssignment(target=target, expr=expr)

        if isinstance(node, IRStore):
            value = self._resolve_expr(node.value, env, metrics)
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.stores += 1
            return ASTStoreStatement(slot=node.slot, value=value)

        if isinstance(node, IRIndirectStore):
            base_expr = self._resolve_expr(node.base, env, metrics)
            value_expr = self._resolve_expr(node.value, env, metrics)
            offset_expr = (
                self._resolve_expr(node.offset_source, env, metrics)
                if node.offset_source
                else ASTLiteral(value=node.offset & 0xFFFF, kind=self._literal_kind(node.offset))
            )
            pointer_expr = (
                self._resolve_expr(node.pointer, env, metrics) if node.pointer else None
            )
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.indirect_stores += 1
            return ASTIndirectStoreStatement(
                base=base_expr,
                value=value_expr,
                offset=offset_expr,
                pointer=pointer_expr,
                ref=node.ref,
            )

        if isinstance(node, IRCall):
            args = tuple(self._resolve_expr(arg, env, metrics) for arg in node.args)
            call = ASTCallStatement(
                target_offset=node.target,
                args=args,
                returns=tuple(),
                call_type="call",
                tail=node.tail,
                varargs=False,
                symbol=node.symbol,
                target_label=offset_to_label.get(node.target),
                predicate=node.predicate.describe() if node.predicate else None,
                convention=node.convention.describe() if node.convention else None,
                cleanup=tuple(effect.describe() for effect in node.cleanup),
                cleanup_mask=node.cleanup_mask,
                arity=node.arity,
            )
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.calls += 1
            if node.tail:
                metrics.tail_calls += 1
            return call

        if isinstance(node, IRCallReturn):
            args = tuple(self._resolve_expr(arg, env, metrics) for arg in node.args)
            call = ASTCallStatement(
                target_offset=node.target,
                args=args,
                returns=node.returns,
                call_type="call_return" if not node.tail else "tail_return",
                tail=node.tail,
                varargs=node.varargs,
                symbol=node.symbol,
                target_label=offset_to_label.get(node.target),
                predicate=node.predicate.describe() if node.predicate else None,
                convention=node.convention.describe() if node.convention else None,
                cleanup=tuple(effect.describe() for effect in node.cleanup),
                cleanup_mask=node.cleanup_mask,
                arity=node.arity,
            )
            for index, name in enumerate(node.returns):
                kind = self._infer_kind(name)
                env[name] = ASTCallResultExpr(
                    target_offset=node.target,
                    index=index,
                    symbol=node.symbol,
                    call_type=call.call_type,
                    kind=kind,
                )
                if kind is not None:
                    metrics.typed_values += 1
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.calls += 1
            if node.tail:
                metrics.tail_calls += 1
            return call

        if isinstance(node, IRTailCall):
            args = tuple(self._resolve_expr(arg, env, metrics) for arg in node.call.args)
            call = ASTCallStatement(
                target_offset=node.call.target,
                args=args,
                returns=node.returns,
                call_type="tail_call",
                tail=True,
                varargs=node.varargs,
                symbol=node.call.symbol,
                target_label=offset_to_label.get(node.call.target),
                predicate=node.call.predicate.describe() if node.call.predicate else None,
                convention=node.call.convention.describe() if node.call.convention else None,
                cleanup=tuple(effect.describe() for effect in node.call.cleanup),
                cleanup_mask=node.cleanup_mask or node.call.cleanup_mask,
                arity=node.call.arity,
            )
            for index, name in enumerate(node.returns):
                kind = self._infer_kind(name)
                env[name] = ASTCallResultExpr(
                    target_offset=node.call.target,
                    index=index,
                    symbol=node.call.symbol,
                    call_type="tail_call",
                    kind=kind,
                )
                if kind is not None:
                    metrics.typed_values += 1
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.calls += 1
            metrics.tail_calls += 1
            return call

        if isinstance(node, IRTailcallReturn):
            args = tuple(self._resolve_expr(arg, env, metrics) for arg in node.args)
            call = ASTCallStatement(
                target_offset=node.target,
                args=args,
                returns=node.returns,
                call_type="tail_return",
                tail=True,
                varargs=node.varargs,
                symbol=node.symbol,
                target_label=offset_to_label.get(node.target),
                predicate=node.predicate.describe() if node.predicate else None,
                convention=node.convention.describe() if node.convention else None,
                cleanup=tuple(effect.describe() for effect in node.cleanup),
                cleanup_mask=node.cleanup_mask,
                arity=node.arity,
            )
            for index, name in enumerate(node.returns):
                kind = self._infer_kind(name)
                env[name] = ASTCallResultExpr(
                    target_offset=node.target,
                    index=index,
                    symbol=node.symbol,
                    call_type="tail_return",
                    kind=kind,
                )
                if kind is not None:
                    metrics.typed_values += 1
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.calls += 1
            metrics.tail_calls += 1
            return call

        if isinstance(node, IRReturn):
            values = tuple(self._resolve_expr(value, env, metrics) for value in node.values)
            cleanup = tuple(effect.describe() for effect in node.cleanup)
            statement = ASTReturnStatement(
                values=values,
                varargs=node.varargs,
                mask=node.mask,
                cleanup=cleanup,
            )
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.returns += 1
            return statement

        if isinstance(node, IRIf):
            condition = self._resolve_expr(node.condition, env, metrics)
            statement = ASTBranchStatement(
                condition=condition,
                kind="if",
                then_label=offset_to_label.get(node.then_target),
                then_offset=node.then_target,
                else_label=offset_to_label.get(node.else_target),
                else_offset=node.else_target,
            )
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.branches += 1
            return statement

        if isinstance(node, IRTestSetBranch):
            text = f"{node.var}={node.expr}"
            condition = ASTUnknownExpr(text=text)
            statement = ASTBranchStatement(
                condition=condition,
                kind="testset",
                then_label=offset_to_label.get(node.then_target),
                then_offset=node.then_target,
                else_label=offset_to_label.get(node.else_target),
                else_offset=node.else_target,
            )
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.branches += 1
            return statement

        if isinstance(node, IRFlagCheck):
            condition = ASTUnknownExpr(text=f"flag(0x{node.flag:04X})")
            statement = ASTBranchStatement(
                condition=condition,
                kind="flag",
                then_label=offset_to_label.get(node.then_target),
                then_offset=node.then_target,
                else_label=offset_to_label.get(node.else_target),
                else_offset=node.else_target,
            )
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.branches += 1
            return statement

        if isinstance(node, IRFunctionPrologue):
            condition = ASTUnknownExpr(text=f"{node.var}={node.expr}")
            statement = ASTBranchStatement(
                condition=condition,
                kind="prologue",
                then_label=offset_to_label.get(node.then_target),
                then_offset=node.then_target,
                else_label=offset_to_label.get(node.else_target),
                else_offset=node.else_target,
            )
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.branches += 1
            return statement

        if isinstance(node, IRSwitchDispatch):
            cases = []
            for case in node.cases:
                cases.append(
                    ASTSwitchCase(
                        key=case.key,
                        target_label=offset_to_label.get(case.target),
                        target_offset=case.target,
                        symbol=case.symbol,
                    )
                )
            statement = ASTSwitchStatement(
                cases=tuple(cases),
                helper_offset=node.helper,
                helper_symbol=node.helper_symbol,
                default_label=offset_to_label.get(node.default) if node.default is not None else None,
                default_offset=node.default,
            )
            metrics.statements += 1
            metrics.structured_statements += 1
            metrics.switches += 1
            return statement

        if isinstance(
            node,
            (
                IRLiteral,
                IRLiteralChunk,
                IRLiteralBlock,
                IRAsciiHeader,
                IRStackDuplicate,
                IRStackDrop,
                IRConditionMask,
                IRIORead,
                IRIOWrite,
            ),
        ):
            # Preserve the IR textual description for nodes we do not handle explicitly.
            description = self._describe(node)
            metrics.statements += 1
            metrics.fallback_statements += 1
            return ASTUnstructuredStatement(text=description)

        description = self._describe(node)
        metrics.statements += 1
        metrics.fallback_statements += 1
        return ASTUnstructuredStatement(text=description)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _describe(self, node: IRNode) -> str:
        describe = getattr(node, "describe", None)
        if callable(describe):
            return describe()
        return repr(node)

    def _resolve_expr(
        self,
        value: Optional[str],
        env: Dict[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> ASTExpression:
        if value is None:
            return ASTUnknownExpr(text="?")
        token = value.strip()
        if not token:
            return ASTUnknownExpr(text="?")
        definition = env.get(token)
        if definition is not None:
            if isinstance(definition, (ASTLiteral, ASTStringLiteral)):
                metrics.constants += 1
                if isinstance(definition, ASTLiteral) and definition.kind is not None:
                    metrics.typed_values += 1
                return definition
            kind = self._infer_kind(token)
            if kind is not None:
                metrics.typed_values += 1
            return ASTIdentifier(name=token, kind=kind, definition=definition)

        if token.startswith("lit(") and token.endswith(")"):
            inner = token[4:-1]
            try:
                literal_value = int(inner, 16)
            except ValueError:
                return ASTUnknownExpr(text=token)
            kind = self._literal_kind(literal_value)
            metrics.constants += 1
            if kind is not None:
                metrics.typed_values += 1
            return ASTLiteral(value=literal_value, kind=kind)

        if token.startswith("str(") or token.startswith("ascii("):
            metrics.constants += 1
            return ASTStringLiteral(value=token)

        if token.startswith("0x"):
            try:
                literal_value = int(token, 16)
            except ValueError:
                return ASTUnknownExpr(text=token)
            kind = self._literal_kind(literal_value)
            metrics.constants += 1
            if kind is not None:
                metrics.typed_values += 1
            return ASTLiteral(value=literal_value, kind=kind)

        match = self._NAME_KIND_RE.match(token)
        if match:
            kind = self._infer_kind(token)
            if kind is not None:
                metrics.typed_values += 1
            return ASTIdentifier(name=token, kind=kind)

        return ASTUnknownExpr(text=token)

    def _infer_kind(self, name: Optional[str]) -> Optional[SSAValueKind]:
        if not name:
            return None
        match = self._NAME_KIND_RE.match(name)
        if not match:
            return None
        prefix = match.group("prefix")
        mapping = {
            "word": SSAValueKind.WORD,
            "ptr": SSAValueKind.POINTER,
            "byte": SSAValueKind.BYTE,
            "bool": SSAValueKind.BOOLEAN,
            "page": SSAValueKind.PAGE_REGISTER,
            "id": SSAValueKind.IDENTIFIER,
            "io": SSAValueKind.IO,
        }
        return mapping.get(prefix, None)

    def _literal_kind(self, value: int) -> Optional[SSAValueKind]:
        if value < 0:
            value &= 0xFFFFFFFF
        if value <= 0xFF:
            return SSAValueKind.BYTE
        return SSAValueKind.WORD


__all__ = ["ASTBuilder"]
