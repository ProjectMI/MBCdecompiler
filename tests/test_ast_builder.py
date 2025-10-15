from pathlib import Path

from mbcdisasm import IRNormalizer
from mbcdisasm.ast import ASTBuilder, ASTDispatchSwitch, ASTDispatchTable
from mbcdisasm.mbc import MbcContainer
from mbcdisasm.ir.model import (
    IRBlock,
    IRIf,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRSlot,
    MemSpace,
    NormalizerMetrics,
)

from tests.test_ir_normalizer import (
    Segment,
    SegmentDescriptor,
    build_container,
    build_word,
    encode_instructions,
    write_manual,
)


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


def test_ast_builder_expands_switch_dispatch(tmp_path: Path) -> None:
    base = tmp_path / "dispatch"
    base.mkdir()
    knowledge = write_manual(base)

    words = [
        build_word(0, 0x2C, 0x01, 0x6623),
        build_word(4, 0x2C, 0x02, 0x6624),
        build_word(8, 0x28, 0x00, 0x6623),
        build_word(12, 0x30, 0x00, 0x0000),
    ]
    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    builder = ASTBuilder()
    ast_program = builder.build(program)
    procedure = ast_program.segments[0].procedures[0]
    block = procedure.blocks[0]
    statements = block.statements

    assert isinstance(statements[0], ASTDispatchTable)
    assert isinstance(statements[1], ASTDispatchSwitch)
    table = statements[0]
    switch = statements[1]

    assert switch.table is table
    assert table.helper == 0x6623
    assert [case.key for case in table.cases] == [0x01, 0x02]
    assert {case.target.address for case in table.cases} == {0x6623, 0x6624}
    assert tuple(ret.name for ret in switch.returns) == ("ret0",)

    rendered = [stmt.render() for stmt in statements[:2]]
    assert any(render.startswith("dispatch_table") for render in rendered)
    assert any(render.startswith("ret0") or "switch" in render for render in rendered)


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
