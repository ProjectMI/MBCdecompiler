from pathlib import Path

from mbcdisasm import IRNormalizer
from mbcdisasm.ast import ASTBuilder, ASTDispatchTable, ASTReturn, ASTSwitch
from mbcdisasm.ir.model import (
    IRAbiEffect,
    IRBlock,
    IRCall,
    IRCallReturn,
    IRDispatchCase,
    IRIf,
    IRLoad,
    IRProgram,
    IRReturn,
    IRStackEffect,
    IRSegment,
    IRSwitchDispatch,
    IRSlot,
    IRStore,
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
    assert procedure.frame.size >= 0
    assert isinstance(procedure.frame.parameters, tuple)

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

    assert isinstance(statements[0], ASTDispatchTable)
    assert isinstance(statements[1], ASTSwitch)
    assert statements[1].helper == 0x1111
    assert statements[1].cases[0].key == 0x01
    assert statements[1].cases[0].target == 0x2222
    assert isinstance(statements[2], ASTReturn)


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

    assert isinstance(statements[0], ASTDispatchTable)
    assert isinstance(statements[1], ASTSwitch)
    assert statements[1].helper == 0x5555
    assert statements[1].cases[0].key == 0x02
    assert statements[1].cases[0].target == 0x4444


def test_ast_builder_records_frame_and_return_mask() -> None:
    block = IRBlock(
        label="entry",
        start_offset=0x1000,
        nodes=(
            IRLoad(slot=IRSlot(MemSpace.FRAME, 0x0000), target="arg0"),
            IRStore(slot=IRSlot(MemSpace.FRAME, 0x0001), value="word1"),
            IRReturn(
                values=("arg0",),
                varargs=False,
                abi_effects=(IRAbiEffect(kind="return_mask", operand=0x0003),),
            ),
        ),
    )
    segment = IRSegment(
        index=0,
        start=0x1000,
        length=0x10,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    procedure = ast_program.segments[0].procedures[0]

    assert procedure.frame.slots[0].index == 0x0000
    assert procedure.frame.slots[0].reads == 1
    assert procedure.frame.parameters[0].identifier.render() == "arg0"

    return_stmt = next(
        stmt for stmt in procedure.blocks[0].statements if isinstance(stmt, ASTReturn)
    )
    assert return_stmt.mask == 0x0001


def test_ast_builder_attaches_call_frame_details() -> None:
    call = IRCallReturn(
        target=0x1234,
        args=("arg_a", "arg_b"),
        tail=False,
        returns=("ret0",),
        cleanup=(IRStackEffect(mnemonic="stack_teardown", pops=1),),
        arity=2,
        convention=IRStackEffect(
            mnemonic="stack_shuffle",
            operand=0x4B08,
            operand_alias="CALL_SHUFFLE_STD",
        ),
        symbol="helper",  # pragma: no mutate
        abi_effects=(IRAbiEffect(kind="return_mask", operand=0x0003),),
    )
    block = IRBlock(
        label="entry",
        start_offset=0x2000,
        nodes=(
            call,
            IRReturn(values=("ret0",), varargs=False),
        ),
    )
    segment = IRSegment(
        index=0,
        start=0x2000,
        length=0x10,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    call_stmt = next(stmt for stmt in statements if hasattr(stmt, "call"))
    frame = call_stmt.call.frame
    assert frame is not None
    assert frame.parameters == ("arg0", "arg1")
    assert frame.convention is not None
    assert frame.convention.render() == "stack_shuffle(operand=CALL_SHUFFLE_STD(0x4B08))"
    assert frame.cleanup and frame.cleanup[0].render() == "stack_teardown(pop=1)"
    assert frame.return_mask == 0x0001
