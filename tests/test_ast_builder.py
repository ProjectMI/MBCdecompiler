from pathlib import Path

from mbcdisasm import IRNormalizer
from mbcdisasm.ast import ASTBuilder, ASTDataArray, ASTDispatchSwitch
from mbcdisasm.ir.model import (
    IRBuildArray,
    IRBlock,
    IRDispatchCase,
    IRIf,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
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


def test_ast_builder_renders_dispatch_switch() -> None:
    dispatch = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x01, target=0x6623, symbol="dispatch_helper_6623"),
            IRDispatchCase(key=0x02, target=0x6624, symbol=None),
        ),
        helper=0x6623,
        helper_symbol="dispatch_helper_6623",
        default=0x6615,
    )
    segment = IRSegment(
        index=0,
        start=0x0000,
        length=0x10,
        blocks=(
            IRBlock(
                label="entry",
                start_offset=0x0000,
                nodes=(dispatch, IRReturn(values=(), varargs=False)),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    statement = ast_program.segments[0].procedures[0].blocks[0].statements[0]
    assert isinstance(statement, ASTDispatchSwitch)
    assert statement.helper == 0x6623
    assert statement.default == 0x6615
    assert {case.key for case in statement.cases} == {0x01, 0x02}
    rendered = statement.render()
    assert rendered.startswith("switch")
    assert "0x6623" in rendered


def test_ast_builder_converts_address_arrays_to_data() -> None:
    array = IRBuildArray(elements=("lit(0x0001)", "lit(0x0002)", "lit(0x0003)"))
    segment = IRSegment(
        index=0,
        start=0x1000,
        length=0x10,
        blocks=(
            IRBlock(
                label="entry",
                start_offset=0x1000,
                nodes=(array, IRReturn(values=(), varargs=False)),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    statement = ast_program.segments[0].procedures[0].blocks[0].statements[0]
    assert isinstance(statement, ASTDataArray)
    assert [element.render() for element in statement.elements] == [
        "0x0001",
        "0x0002",
        "0x0003",
    ]
    assert statement.render() == "data [0x0001, 0x0002, 0x0003]"
