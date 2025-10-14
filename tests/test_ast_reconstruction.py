"""Tests for the AST reconstruction pipeline."""

from mbcdisasm.ast import ASTBuilder, CFGBuilder, ProcedureResolver
from mbcdisasm.ast.model import (
    AssignStatement,
    CallStatement,
    LiteralExpr,
    LoadExpr,
    ReconstructionMetrics,
    VariableExpr,
)
from mbcdisasm.ir.model import (
    IRBlock,
    IRCall,
    IRFunctionPrologue,
    IRLoad,
    IRReturn,
    IRSegment,
    IRSlot,
    NormalizerMetrics,
    MemSpace,
)


def _make_segment() -> IRSegment:
    metrics = NormalizerMetrics()
    block0 = IRBlock(
        label="block_0",
        start_offset=0x1000,
        nodes=(
            IRFunctionPrologue(
                var="bool0",
                expr="bool1",
                then_target=0x1010,
                else_target=0x1020,
            ),
        ),
    )
    block1 = IRBlock(
        label="block_1",
        start_offset=0x1010,
        nodes=(
            IRLoad(IRSlot(MemSpace.GLOBAL, 1), target="word0"),
            IRCall(target=0x2000, args=("word0", "const_0x10")),
        ),
    )
    block2 = IRBlock(
        label="block_2",
        start_offset=0x1020,
        nodes=(IRReturn(values=("word0",), varargs=False),),
    )
    block3 = IRBlock(
        label="callee_entry",
        start_offset=0x2000,
        nodes=(
            IRFunctionPrologue(
                var="bool2",
                expr="bool3",
                then_target=0x2008,
                else_target=0x200C,
            ),
        ),
    )
    block4 = IRBlock(
        label="callee_then",
        start_offset=0x2008,
        nodes=(IRReturn(values=("bool2",), varargs=False),),
    )
    block5 = IRBlock(
        label="callee_else",
        start_offset=0x200C,
        nodes=(IRReturn(values=("bool3",), varargs=False),),
    )
    return IRSegment(
        index=0,
        start=0,
        length=0x40,
        blocks=(block0, block1, block2, block3, block4, block5),
        metrics=metrics,
    )


def test_cfg_builder_creates_edges() -> None:
    segment = _make_segment()
    builder = CFGBuilder()
    cfg = builder.build_segment(segment)

    assert len(cfg.nodes) == 6
    entry = cfg.nodes[0x1000]
    successors = {edge.target for edge in entry.successors}
    assert successors == {0x1010, 0x1020}

    call_node = cfg.nodes[0x1010]
    assert call_node.call_sites and call_node.call_sites[0].target == 0x2000


def test_procedure_resolver_detects_boundaries() -> None:
    segment = _make_segment()
    cfg = CFGBuilder().build_segment(segment)
    resolver = ProcedureResolver()
    procedures = resolver.detect(cfg)

    entries = {proc.entry_offset for proc in procedures}
    assert entries == {0x1000, 0x2000}

    primary = next(proc for proc in procedures if proc.entry_offset == 0x1000)
    assert set(primary.blocks) == {0x1000, 0x1010, 0x1020}
    assert primary.callees == (0x2000,)

    callee = next(proc for proc in procedures if proc.entry_offset == 0x2000)
    assert set(callee.blocks) == {0x2000, 0x2008, 0x200C}


def test_ast_builder_produces_statements_and_metrics() -> None:
    segment = _make_segment()
    builder = ASTBuilder()
    ast_segment, metrics = builder.build_segment(segment)

    assert isinstance(ast_segment.metrics, ReconstructionMetrics)
    assert ast_segment.metrics.procedures == 2
    assert metrics.cfg_nodes == 6

    proc = next(
        procedure
        for procedure in ast_segment.procedures
        if procedure.procedure.entry_offset == 0x1010 or procedure.procedure.entry_offset == 0x1000
    )

    block = next(block for block in proc.blocks if block.node.block.start_offset == 0x1010)
    assert isinstance(block.statements[0], AssignStatement)
    assign = block.statements[0]
    assert isinstance(assign.target, VariableExpr)
    assert assign.target.name == "word0"
    assert isinstance(assign.value, LoadExpr)
    assert assign.value.slot.space is MemSpace.GLOBAL

    call_stmt = next(stmt for stmt in block.statements if isinstance(stmt, CallStatement))
    assert isinstance(call_stmt.call.args[0], LoadExpr)
    assert isinstance(call_stmt.call.args[1], LiteralExpr)

    assert ast_segment.metrics.typed_variable_ratio > 0.0
    assert ast_segment.metrics.call_resolution_ratio > 0.0

