from pathlib import Path

from mbcdisasm import IRNormalizer
from mbcdisasm.constants import RET_MASK
from mbcdisasm.ast import (
    ASTBranch,
    ASTBuilder,
    ASTCallFrame,
    ASTCallStatement,
    ASTReturn,
    ASTSwitch,
    ASTTailCall,
)
from mbcdisasm.ir.model import (
    IRBlock,
    IRCall,
    IRCallReturn,
    IRDispatchCase,
    IRDispatchIndex,
    IRIf,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
    IRTailCall,
    IRSlot,
    MemSpace,
    NormalizerMetrics,
)

from tests.test_ir_normalizer import build_container


def test_ast_builder_reconstructs_cfg(tmp_path: Path) -> None:
    container, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    builder = ASTBuilder()
    ast_program = builder.build(program)

    assert ast_program.metrics.procedure_count >= 1
    assert ast_program.metrics.block_count >= 1
    assert ast_program.metrics.call_sites >= 1
    assert ast_program.metrics.call_args >= ast_program.metrics.resolved_call_args

    segment = ast_program.segments[0]
    assert segment.procedures
    procedure = segment.procedures[0]
    assert procedure.entry_reasons
    assert procedure.blocks

    block = procedure.blocks[0]
    assert block.successors is not None
    rendered = [statement.render() for statement in block.statements]
    assert any("call" in line for line in rendered)
    assert any(line.startswith("return") for line in rendered)
    assert procedure.name == f"proc_{procedure.entry_offset:04X}"

    summary = ast_program.metrics.describe()
    assert "procedures=" in summary
    assert "calls=" in summary


def test_ast_builder_splits_after_return_sequences() -> None:
    segment = IRSegment(
        index=0,
        start=0x0100,
        length=0x20,
        blocks=(
            IRBlock(
                label="block_entry",
                start_offset=0x0100,
                nodes=(IRReturn(values=("ret0",), varargs=False),),
            ),
            IRBlock(
                label="block_follow",
                start_offset=0x0110,
                nodes=(
                    IRLoad(slot=IRSlot(MemSpace.FRAME, 0), target="word0"),
                    IRReturn(values=("ret0",), varargs=False),
                ),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    ast_segment = ast_program.segments[0]
    entries = [procedure.entry_offset for procedure in ast_segment.procedures]
    assert entries == [0x0100, 0x0110]
    assert all(
        procedure.name == f"proc_{procedure.entry_offset:04X}"
        for procedure in ast_segment.procedures
    )
    first = ast_segment.procedures[0]
    assert first.exit_offsets == (0x0100,)
    assert len(first.blocks) == 1


def test_ast_builder_uses_postdominators_for_exits() -> None:
    segment = IRSegment(
        index=0,
        start=0x0200,
        length=0x40,
        blocks=(
            IRBlock(
                label="block_cond",
                start_offset=0x0200,
                nodes=(IRIf(condition="bool0", then_target=0x0210, else_target=0x0220),),
            ),
            IRBlock(
                label="block_then",
                start_offset=0x0210,
                nodes=(IRReturn(values=("ret0",), varargs=False),),
            ),
            IRBlock(
                label="block_else",
                start_offset=0x0220,
                nodes=(IRReturn(values=("ret1",), varargs=False),),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    ast_segment = ast_program.segments[0]
    assert len(ast_segment.procedures) == 1
    procedure = ast_segment.procedures[0]
    assert tuple(sorted(procedure.exit_offsets)) == (0x0210, 0x0220)
    assert {block.start_offset for block in procedure.blocks} == {0x0200, 0x0210, 0x0220}


def test_ast_builder_converts_dispatch_with_trailing_table() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x01, target=0x2222, symbol="proc_2222"),),
        helper=0x1111,
        helper_symbol="helper_1111",
        default=0x3333,
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0100,
        nodes=(dispatch, IRCall(target=0x1111, args=(), symbol="helper_1111"), IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0100,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    assert all("dispatch.data" not in stmt.render() for stmt in statements)
    switch_stmt = statements[0]
    assert isinstance(switch_stmt, ASTSwitch)
    assert switch_stmt.helper == 0x1111
    assert switch_stmt.cases[0].key == 0x01
    assert switch_stmt.cases[0].target == 0x2222
    assert isinstance(statements[1], ASTReturn)


def test_ast_builder_converts_dispatch_with_leading_call() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x02, target=0x4444, symbol=None),),
        helper=0x5555,
        helper_symbol=None,
        default=None,
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0200,
        nodes=(IRCall(target=0x5555, args=()), dispatch, IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0200,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    assert all("dispatch.data" not in stmt.render() for stmt in statements)
    call_stmt = statements[0]
    assert isinstance(call_stmt, ASTCallStatement)
    assert call_stmt.call.target == 0x5555
    assert call_stmt.call.symbol is None


def test_ast_builder_pairs_dispatch_with_call_return_metadata() -> None:
    dispatch = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x00, target=0x3333, symbol=None),
            IRDispatchCase(key=0x01, target=0x4444, symbol=None),
        ),
        helper=0x7777,
        helper_symbol="helper_7777",
        default=None,
    )
    call = IRCallReturn(
        target=0x7777,
        args=("word0", "lit(0x0007)", "lit(0x0001)", "lit(0x9999)"),
        tail=False,
        returns=("ret0",),
        symbol="helper_7777",
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0240,
        nodes=(dispatch, call),
    )
    segment = IRSegment(
        index=0,
        start=0x0240,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    switch_stmt = next(statement for statement in statements if isinstance(statement, ASTSwitch))
    assert switch_stmt.index_source == "word0"
    assert switch_stmt.index_mask == 0x0007
    assert switch_stmt.index_base == 0x0001
    rendered = switch_stmt.render()
    assert "index=word0 & 0x0007" in rendered
    assert "base=0x0001" in rendered

    call_stmt = next(statement for statement in statements if isinstance(statement, ASTCallStatement))
    assert call_stmt.returns


def test_ast_builder_inserts_switch_before_call_return_when_table_trails() -> None:
    dispatch = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x10, target=0x5555, symbol=None),
            IRDispatchCase(key=0x11, target=0x6666, symbol=None),
        ),
        helper=0x8888,
        helper_symbol=None,
        default=None,
    )
    call = IRCallReturn(
        target=0x8888,
        args=("ptr_index", "lit(0x000F)"),
        tail=False,
        returns=("out0", "out1"),
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0250,
        nodes=(call, dispatch),
    )
    segment = IRSegment(
        index=0,
        start=0x0250,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    switch_stmt = next(statement for statement in statements if isinstance(statement, ASTSwitch))
    assert switch_stmt.index_source == "ptr_index"
    assert switch_stmt.index_mask == 0x000F
    call_stmt = next(statement for statement in statements if isinstance(statement, ASTCallStatement))
    assert len(call_stmt.returns) == 2

def test_ast_builder_simplifies_single_case_dispatch_to_call() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x07, target=0xAAAA, symbol=None),),
        helper=0x6060,
        helper_symbol="helper_6060",
        default=None,
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0220,
        nodes=(dispatch, IRCall(target=0x6060, args=(), symbol="helper_6060")),
    )
    segment = IRSegment(
        index=0,
        start=0x0220,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    assert all("dispatch.data" not in stmt.render() for stmt in statements)
    assert isinstance(statements[0], ASTCallStatement)
    assert statements[0].call.target == 0x6060
    assert statements[0].call.symbol == "helper_6060"


def test_ast_switch_marks_io_dispatch() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x10, target=0x2000, symbol=None),),
        helper=0x00F0,
        helper_symbol="io.flush_tail",
        default=0x1234,
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0250,
        nodes=(dispatch, IRCall(target=0x00F0, args=(), symbol="io.flush_tail"), IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0250,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    switch_stmt = statements[0]
    assert isinstance(switch_stmt, ASTSwitch)
    assert switch_stmt.kind == "io"


def test_ast_builder_drops_redundant_tailcall_after_switch() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x05, target=0x3000, symbol=None),),
        helper=0x7777,
        helper_symbol=None,
    )
    helper_call = IRCall(target=0x7777, args=())
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0300,
        nodes=(
            dispatch,
            helper_call,
            IRTailCall(call=IRCall(target=0x7777, args=()), returns=()),
        ),
    )
    segment = IRSegment(
        index=0,
        start=0x0300,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    assert any(isinstance(stmt, ASTCallStatement) for stmt in statements)
    assert not any(isinstance(stmt, ASTTailCall) for stmt in statements)


def test_ast_builder_prunes_redundant_branch_blocks() -> None:
    segment = IRSegment(
        index=0,
        start=0x0400,
        length=0x20,
        blocks=(
            IRBlock(
                label="block_entry",
                start_offset=0x0400,
                nodes=(IRIf(condition="stack_top", then_target=0x0410, else_target=0x0410),),
            ),
            IRBlock(
                label="block_return",
                start_offset=0x0410,
                nodes=(IRReturn(values=("ret0",), varargs=False),),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    segment_ast = ast_program.segments[0]
    assert len(segment_ast.procedures) == 1
    procedure = segment_ast.procedures[0]
    assert {block.start_offset for block in procedure.blocks} == {0x0410}
    assert all(
        not isinstance(statement, ASTBranch)
        for block in procedure.blocks
        for statement in block.statements
    )


def test_ast_builder_drops_empty_procedures() -> None:
    segment = IRSegment(
        index=0,
        start=0x0500,
        length=0x10,
        blocks=(
            IRBlock(
                label="block_empty",
                start_offset=0x0500,
                nodes=tuple(),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    assert not ast_program.segments[0].procedures
    assert ast_program.metrics.procedure_count == 0


def test_ast_switch_carries_index_metadata() -> None:
    dispatch = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x03, target=0x8888, symbol=None),
            IRDispatchCase(key=0x04, target=0x9999, symbol=None),
        ),
        helper=0x7777,
        helper_symbol=None,
        default=None,
        index=IRDispatchIndex(source="word0", mask=0x0007, base=0x0001),
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0300,
        nodes=(dispatch, IRCall(target=0x7777, args=()), IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0300,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    switch_stmt = next(statement for statement in statements if isinstance(statement, ASTSwitch))
    assert switch_stmt.index_source == "word0"
    assert switch_stmt.index_mask == 0x0007
    assert switch_stmt.index_base == 0x0001
    rendered = switch_stmt.render()
    assert "index=word0 & 0x0007" in rendered
    assert "base=0x0001" in rendered


def test_ast_builder_emits_call_frame_and_finally(tmp_path: Path) -> None:
    container, knowledge = build_container(tmp_path)
    program = IRNormalizer(knowledge).normalise_container(container)
    ast_program = ASTBuilder().build(program)

    segment = ast_program.segments[0]
    procedure = segment.procedures[0]
    block = procedure.blocks[0]

    frame = next(statement for statement in block.statements if isinstance(statement, ASTCallFrame))
    assert frame.live_mask == RET_MASK
    assert len(frame.slots) == 2

    return_stmt = next(statement for statement in block.statements if isinstance(statement, ASTReturn))
    assert return_stmt.finally_branch is not None
    kinds = [step.kind for step in return_stmt.finally_branch.steps]
    assert "stack_teardown" in kinds
