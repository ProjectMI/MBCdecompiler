"""AST reconstruction pipeline built on top of the CFG."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..ir.model import (
    IRAsciiHeader,
    IRAsciiPreamble,
    IRAsciiFinalize,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    IRCallCleanup,
    IRCallPreparation,
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
    IRPageRegister,
    IRReturn,
    IRStackDuplicate,
    IRStackDrop,
    IRStore,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
    IRRaw,
)
from ..ir.model import IRProgram, IRSegment, SSAValueKind
from .cfg import CFGBuilder, ProcedureResolver
from .metrics import MetricCounter
from .model import (
    ASTBlock,
    ASTExpression,
    ASTProgram,
    ASTProcedure,
    ASTSegment,
    AssignStatement,
    BranchStatement,
    CallExpr,
    CallStatement,
    CFGNode,
    LiteralExpr,
    LoadExpr,
    Procedure,
    PrologueStatement,
    RawExpr,
    ReturnStatement,
    StoreStatement,
    UnknownExpr,
    UnknownStatement,
    VariableExpr,
    ReconstructionMetrics,
)


_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LITERAL_RE = re.compile(r"^(?:0x[0-9A-Fa-f]+|-?\d+)$")
_CONST_LITERAL_RE = re.compile(r"^const_0x([0-9A-Fa-f]+)$")
_KIND_PREFIX = {
    "byte": SSAValueKind.BYTE,
    "word": SSAValueKind.WORD,
    "ptr": SSAValueKind.POINTER,
    "io": SSAValueKind.IO,
    "page": SSAValueKind.PAGE_REGISTER,
    "bool": SSAValueKind.BOOLEAN,
    "id": SSAValueKind.IDENTIFIER,
}


def _infer_kind(name: str) -> SSAValueKind:
    for prefix, kind in _KIND_PREFIX.items():
        if name.startswith(prefix):
            return kind
    return SSAValueKind.UNKNOWN


def _parse_literal(text: str) -> Optional[int]:
    if not text:
        return None
    match = _CONST_LITERAL_RE.match(text)
    if match:
        return int(match.group(1), 16)
    if _LITERAL_RE.match(text):
        base = 16 if text.startswith("0x") or text.startswith("0X") else 10
        return int(text, base)
    return None


@dataclass
class ValueTable:
    """Associate SSA aliases with reconstructed expressions."""

    values: Dict[str, ASTExpression]

    def __init__(self) -> None:
        self.values = {}

    def assign(self, name: str, expr: ASTExpression) -> None:
        if not name or name == "stack":
            return
        if not _ALIAS_RE.match(name):
            return
        self.values[name] = expr

    def resolve(self, name: Optional[str]) -> ASTExpression:
        if not name:
            return UnknownExpr("missing")
        if name in self.values:
            return self.values[name]
        literal = _parse_literal(name)
        if literal is not None:
            return LiteralExpr(literal)
        if _ALIAS_RE.match(name):
            return VariableExpr(name=name, kind=_infer_kind(name))
        return RawExpr(name)


class StatementBuilder:
    """Convert IR nodes into AST statements."""

    def __init__(self, table: ValueTable) -> None:
        self.table = table

    def build_statements(self, node: CFGNode) -> Tuple:
        statements: List = []
        for ir in node.block.nodes:
            stmt = self._dispatch(ir)
            if stmt is not None:
                statements.append(stmt)
        return tuple(statements)

    # pylint: disable=too-many-return-statements
    def _dispatch(self, node) -> Optional:
        if isinstance(node, IRFunctionPrologue):
            target = self._variable(node.var)
            value = self._expression(node.expr)
            self.table.assign(node.var, value)
            return PrologueStatement(target=target, value=value, then_target=node.then_target, else_target=node.else_target)
        if isinstance(node, IRLoad):
            value = LoadExpr(slot=node.slot)
            target = self._variable(node.target)
            self.table.assign(node.target, value)
            return AssignStatement(target=target, value=value)
        if isinstance(node, IRStore):
            value = self._expression(node.value)
            return StoreStatement(slot=node.slot, value=value)
        if isinstance(node, IRCall):
            call_expr = self._call_expr(node)
            return CallStatement(call=call_expr, tail=node.tail, varargs=False)
        if isinstance(node, IRTailCall):
            call_expr = self._call_expr(node)
            results = tuple(self._variable(name) for name in node.returns)
            if len(results) == 1:
                self.table.assign(results[0].name, call_expr)
            return CallStatement(call=call_expr, results=results, tail=True, varargs=node.varargs)
        if isinstance(node, IRTailcallReturn):
            call_expr = self._call_expr(node)
            results = tuple(self._variable(name) for name in node.returns)
            if len(results) == 1:
                self.table.assign(results[0].name, call_expr)
            return CallStatement(call=call_expr, results=results, tail=True, varargs=node.varargs)
        if isinstance(node, IRCallReturn):
            call_expr = self._call_expr(node)
            results = tuple(self._variable(name) for name in node.returns)
            if len(results) == 1:
                self.table.assign(results[0].name, call_expr)
            return CallStatement(call=call_expr, results=results, tail=node.tail, varargs=node.varargs)
        if isinstance(node, IRReturn):
            values = tuple(self._expression(value) for value in node.values)
            return ReturnStatement(values=values, varargs=node.varargs)
        if isinstance(node, IRIf):
            condition = self._expression(node.condition)
            return BranchStatement(condition=condition, then_target=node.then_target, else_target=node.else_target)
        if isinstance(node, IRTestSetBranch):
            condition = self._expression(node.expr)
            target = self._variable(node.var)
            self.table.assign(node.var, condition)
            return BranchStatement(condition=condition, then_target=node.then_target, else_target=node.else_target)
        if isinstance(node, IRFlagCheck):
            condition = RawExpr(f"flag_{node.flag:04X}")
            return BranchStatement(condition=condition, then_target=node.then_target, else_target=node.else_target)
        if isinstance(node, (IRLiteral, IRLiteralBlock, IRLiteralChunk)):
            literal = RawExpr(node.describe())
            return UnknownStatement(text=literal.text)
        if isinstance(node, (IRAsciiHeader, IRAsciiPreamble, IRAsciiFinalize)):
            return UnknownStatement(text=node.describe())
        if isinstance(node, IRStackDuplicate):
            source = self._expression(node.value)
            return UnknownStatement(text=f"duplicate {source}")
        if isinstance(node, IRStackDrop):
            return UnknownStatement(text=node.describe())
        if isinstance(node, (IRCallPreparation, IRCallCleanup)):
            return UnknownStatement(text=node.describe())
        if isinstance(node, (IRIndirectLoad, IRIndirectStore, IRIORead, IRIOWrite)):
            return UnknownStatement(text=node.describe())
        if isinstance(node, (IRBuildArray, IRBuildMap, IRBuildTuple)):
            return UnknownStatement(text=node.describe())
        if isinstance(node, IRConditionMask):
            return UnknownStatement(text=node.describe())
        if isinstance(node, IRPageRegister):
            return UnknownStatement(text=node.describe())
        if isinstance(node, IRRaw):
            return UnknownStatement(text=node.describe())
        if isinstance(node, IRTerminator):
            return UnknownStatement(text=node.describe())
        return None

    def _variable(self, name: str) -> VariableExpr:
        if not name or name == "stack":
            return VariableExpr(name="ssa", kind=SSAValueKind.UNKNOWN)
        kind = _infer_kind(name)
        return VariableExpr(name=name, kind=kind)

    def _expression(self, value: Optional[str]) -> ASTExpression:
        if value is None:
            return UnknownExpr("missing")
        return self.table.resolve(value)

    def _call_expr(self, node) -> CallExpr:
        args = tuple(self.table.resolve(arg) for arg in node.args)
        return CallExpr(
            target=node.target,
            args=args,
            symbol=getattr(node, "symbol", None),
            arity=getattr(node, "arity", None),
            convention=getattr(node, "convention", None),
            cleanup_mask=getattr(node, "cleanup_mask", None),
            predicate=getattr(node, "predicate", None),
        )


class ProcedureASTBuilder:
    """Build :class:`ASTProcedure` objects from discovered procedures."""

    def __init__(self, procedure: Procedure, node_map: Dict[int, CFGNode], metrics: MetricCounter) -> None:
        self.procedure = procedure
        self.nodes = node_map
        self.metrics = metrics
        self.table = ValueTable()
        self.statement_builder = StatementBuilder(self.table)

    def build(self) -> ASTProcedure:
        blocks: List[ASTBlock] = []
        for offset in self.procedure.blocks:
            cfg_node = self.nodes[offset]
            statements = self.statement_builder.build_statements(cfg_node)
            blocks.append(ASTBlock(node=cfg_node, statements=statements))
            self.metrics.observe_block(len(statements))
            for stmt in statements:
                self.metrics.observe_variables(stmt.variables())
                self.metrics.observe_expressions(stmt.expressions())
        return ASTProcedure(procedure=self.procedure, blocks=tuple(blocks))


class ASTBuilder:
    """High level orchestration for AST reconstruction."""

    def __init__(self) -> None:
        self.cfg_builder = CFGBuilder()
        self.resolver = ProcedureResolver()

    def build_segment(self, segment: IRSegment) -> Tuple[ASTSegment, MetricCounter]:
        cfg = self.cfg_builder.build_segment(segment)
        procedures = self.resolver.detect(cfg)
        metrics = MetricCounter()
        metrics.cfg_nodes = len(cfg.nodes)
        metrics.cfg_edges = len(cfg.edges)
        metrics.entry_points = len(procedures)
        metrics.procedures = len(procedures)
        metrics.assigned_blocks = sum(len(proc.blocks) for proc in procedures)

        ast_procedures: List[ASTProcedure] = []
        for procedure in procedures:
            builder = ProcedureASTBuilder(procedure, cfg.nodes, metrics)
            ast_proc = builder.build()
            ast_procedures.append(ast_proc)

        segment_metrics = metrics.to_metrics()
        ast_segment = ASTSegment(cfg=cfg, procedures=tuple(ast_procedures), metrics=segment_metrics)
        return ast_segment, metrics

    def build_program(self, program: IRProgram) -> ASTProgram:
        segments: List[ASTSegment] = []
        aggregate = MetricCounter()
        for segment in program.segments:
            ast_segment, metrics = self.build_segment(segment)
            segments.append(ast_segment)
            aggregate.merge(metrics)
        program_metrics = aggregate.to_metrics()
        return ASTProgram(segments=tuple(segments), metrics=program_metrics)

