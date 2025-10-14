"""Utilities for serialising AST programmes into a textual format."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

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
    ASTLoadExpr,
    ASTProgram,
    ASTReturnStatement,
    ASTSegment,
    ASTStatement,
    ASTStoreStatement,
    ASTSwitchCase,
    ASTSwitchStatement,
    ASTUnstructuredStatement,
    ASTUnknownExpr,
    ASTStringLiteral,
    ASTExpression,
)
from ..ir.model import IRSlot, SSAValueKind


class ASTTextRenderer:
    """Render :class:`ASTProgram` instances into a stable textual form."""

    def render(self, program: ASTProgram) -> str:
        lines: List[str] = []
        lines.append("; ast metrics: " + program.metrics.describe())
        for segment in program.segments:
            lines.extend(self._render_segment(segment))
        return "\n".join(lines) + "\n"

    def write(self, program: ASTProgram, output_path: Path) -> None:
        output_path.write_text(self.render(program), "utf-8")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _render_segment(self, segment: ASTSegment) -> Iterable[str]:
        header = (
            f"; segment {segment.index} offset=0x{segment.start:06X} "
            f"length={segment.length}"
        )
        yield header
        yield "; metrics: " + segment.metrics.describe()
        if segment.procedures:
            yield "; procedures:"
            for procedure in segment.procedures:
                blocks = ", ".join(procedure.block_labels)
                calls = ", ".join(procedure.call_targets)
                details = (
                    f"blocks=[{blocks}] returns={procedure.return_count} "
                    f"tail_calls={procedure.tail_call_count}"
                )
                if calls:
                    details += f" calls=[{calls}]"
                yield f";   {procedure.entry_label} {details}".rstrip()
        for block in segment.blocks:
            yield from self._render_block(block)
        yield ""

    def _render_block(self, block: ASTBlock) -> Iterable[str]:
        yield f"block {block.label} offset=0x{block.start_offset:06X}"
        for note in block.annotations:
            yield f"  ; {note}"
        for statement in block.statements:
            yield "  " + self._render_statement(statement)
        for edge in block.successors:
            yield "  ; edge " + self._render_edge(edge)

    def _render_statement(self, statement: ASTStatement) -> str:
        if isinstance(statement, ASTAssignment):
            target = self._render_identifier(statement.target)
            expr = self._render_expr(statement.expr)
            return f"{target} = {expr}"
        if isinstance(statement, ASTStoreStatement):
            slot = self._render_slot(statement.slot)
            value = self._render_expr(statement.value)
            return f"store {slot} <= {value}"
        if isinstance(statement, ASTIndirectStoreStatement):
            base = self._render_expr(statement.base) if statement.base else "?"
            value = self._render_expr(statement.value)
            offset = self._render_expr(statement.offset) if statement.offset else "?"
            details = [f"base={base}", f"offset={offset}", f"value={value}"]
            if statement.pointer is not None:
                details.append(f"ptr={self._render_expr(statement.pointer)}")
            if statement.ref is not None:
                details.append(f"ref={statement.ref.describe()}")
            return "store_indirect " + " ".join(details)
        if isinstance(statement, ASTCallStatement):
            prefix = statement.call_type
            args = ", ".join(self._render_expr(arg) for arg in statement.args)
            target = f"0x{statement.target_offset:04X}"
            if statement.symbol:
                target = f"{statement.symbol}({target})"
            if statement.target_label:
                target += f" -> {statement.target_label}"
            details: List[str] = [f"target={target}", f"args=[{args}]"]
            if statement.returns:
                details.append(
                    "returns=[" + ", ".join(statement.returns) + "]"
                )
            if statement.varargs:
                details.append("varargs")
            if statement.arity is not None:
                details.append(f"arity={statement.arity}")
            if statement.convention:
                details.append(f"convention={statement.convention}")
            if statement.cleanup_mask is not None:
                details.append(f"mask=0x{statement.cleanup_mask:04X}")
            if statement.cleanup:
                rendered = ", ".join(statement.cleanup)
                details.append(f"cleanup=[{rendered}]")
            if statement.predicate:
                details.append(f"predicate={statement.predicate}")
            return f"{prefix} " + " ".join(details)
        if isinstance(statement, ASTReturnStatement):
            values = ", ".join(self._render_expr(value) for value in statement.values)
            prefix = "return"
            if statement.varargs:
                if statement.values:
                    values = f"varargs({values})"
                else:
                    values = "varargs"
            result = f"{prefix} [{values}]" if statement.values or statement.varargs else "return"
            if statement.mask is not None:
                result += f" mask=0x{statement.mask:04X}"
            if statement.cleanup:
                rendered = ", ".join(statement.cleanup)
                result += f" cleanup=[{rendered}]"
            return result
        if isinstance(statement, ASTBranchStatement):
            condition = self._render_expr(statement.condition)
            then_label = statement.then_label or f"0x{statement.then_offset:04X}" if statement.then_offset is not None else "?"
            else_label = statement.else_label or f"0x{statement.else_offset:04X}" if statement.else_offset is not None else "?"
            return (
                f"branch kind={statement.kind} cond={condition} "
                f"then={then_label} else={else_label}"
            )
        if isinstance(statement, ASTSwitchStatement):
            cases = ", ".join(self._render_case(case) for case in statement.cases)
            helper = "helper=?"
            if statement.helper_offset is not None:
                helper = f"helper=0x{statement.helper_offset:04X}"
                if statement.helper_symbol:
                    helper = f"helper={statement.helper_symbol}(0x{statement.helper_offset:04X})"
            result = f"switch {helper} cases=[{cases}]"
            if statement.default_offset is not None:
                target = statement.default_label or f"0x{statement.default_offset:04X}"
                result += f" default={target}"
            return result
        if isinstance(statement, ASTUnstructuredStatement):
            return f"raw {statement.text}"
        return repr(statement)

    def _render_case(self, case: ASTSwitchCase) -> str:
        target = case.target_label or f"0x{case.target_offset:04X}"
        if case.symbol:
            target = f"{case.symbol}({target})"
        return f"0x{case.key:04X}->{target}"

    def _render_edge(self, edge: ASTEdge) -> str:
        target = edge.target_label
        if target is None and edge.target_offset is not None:
            target = f"0x{edge.target_offset:04X}"
        if target is None:
            target = "?"
        parts = [f"kind={edge.kind}", f"target={target}"]
        if edge.detail:
            parts.append(edge.detail)
        return " ".join(parts)

    def _render_expr(self, expr: ASTExpression) -> str:
        if isinstance(expr, ASTLiteral):
            width = 2 if expr.kind is SSAValueKind.BYTE else 4
            return f"0x{expr.value:0{width}X}"
        if isinstance(expr, ASTStringLiteral):
            return expr.value
        if isinstance(expr, ASTIdentifier):
            base = expr.name
            kind = self._render_kind(expr.kind)
            if kind:
                base = f"{base}:{kind}"
            if expr.definition is not None and not isinstance(expr.definition, ASTIdentifier):
                base += f"({self._render_expr(expr.definition)})"
            return base
        if isinstance(expr, ASTLoadExpr):
            slot = self._render_slot(expr.slot)
            if expr.ref is not None:
                target = expr.ref.describe()
            else:
                target = slot
            return f"load {target}"
        if isinstance(expr, ASTIndirectLoadExpr):
            parts = []
            if expr.base is not None:
                parts.append(f"base={self._render_expr(expr.base)}")
            if expr.offset is not None:
                parts.append(f"offset={self._render_expr(expr.offset)}")
            if expr.pointer is not None:
                parts.append(f"ptr={self._render_expr(expr.pointer)}")
            if expr.ref is not None:
                parts.append(f"ref={expr.ref.describe()}")
            return "indirect_load[" + ", ".join(parts) + "]"
        if isinstance(expr, ASTCallResultExpr):
            target = f"0x{expr.target_offset:04X}"
            if expr.symbol:
                target = f"{expr.symbol}({target})"
            label = self._render_kind(expr.kind)
            prefix = f"call_result[{expr.index}]"
            if label:
                prefix = f"{prefix}:{label}"
            return f"{prefix}@{target}"
        if isinstance(expr, ASTUnknownExpr):
            return expr.text
        return repr(expr)

    def _render_identifier(self, identifier: ASTIdentifier) -> str:
        base = identifier.name
        kind = self._render_kind(identifier.kind)
        if kind:
            base = f"{base}:{kind}"
        return base

    def _render_slot(self, slot: IRSlot) -> str:
        index = f"0x{slot.index:04X}"
        return f"{slot.space.name.lower()}[{index}]"

    def _render_kind(self, kind: Optional[SSAValueKind]) -> str:
        if kind is None:
            return ""
        return kind.name.lower()


__all__ = ["ASTTextRenderer"]
