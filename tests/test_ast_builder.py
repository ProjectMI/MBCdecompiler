from mbcdisasm.ast import ASTBuilder, ASTPrinter, CFGBuilder
from mbcdisasm.ir.model import (
    IRBlock,
    IRFunctionPrologue,
    IRIf,
    IRLiteral,
    IRProgram,
    IRReturn,
    IRSegment,
    NormalizerMetrics,
)


def build_program(blocks):
    segment = IRSegment(index=0, start=0, length=0, blocks=tuple(blocks), metrics=NormalizerMetrics())
    return IRProgram(segments=(segment,), metrics=NormalizerMetrics())


def test_cfg_builder_extracts_branch_edges() -> None:
    blocks = (
        IRBlock(
            label="block_0",
            start_offset=0,
            nodes=(
                IRIf(condition="test", then_target=4, else_target=8),
            ),
        ),
        IRBlock(
            label="block_1",
            start_offset=4,
            nodes=(
                IRLiteral(value=1, mode=0, source="test"),
            ),
        ),
        IRBlock(
            label="block_2",
            start_offset=8,
            nodes=(
                IRReturn(values=("r0",), varargs=False),
            ),
        ),
    )
    segment = IRSegment(index=0, start=0, length=0, blocks=blocks, metrics=NormalizerMetrics())

    cfg = CFGBuilder().build(segment)
    assert cfg.entry_label == "block_0"
    assert cfg.nodes["block_0"].successors == ("block_1", "block_2")
    assert cfg.nodes["block_1"].successors == ("block_2",)


def test_ast_builder_groups_functions() -> None:
    blocks = (
        IRBlock(
            label="block_0",
            start_offset=0,
            nodes=(
                IRFunctionPrologue(var="arg", expr="stack0", then_target=4, else_target=4),
            ),
        ),
        IRBlock(
            label="block_1",
            start_offset=4,
            nodes=(
                IRReturn(values=("ret0",), varargs=False),
            ),
        ),
        IRBlock(
            label="block_2",
            start_offset=8,
            nodes=(
                IRFunctionPrologue(var="arg", expr="stack1", then_target=12, else_target=12),
            ),
        ),
        IRBlock(
            label="block_3",
            start_offset=12,
            nodes=(
                IRReturn(values=("ret1",), varargs=False),
            ),
        ),
    )
    program = build_program(blocks)

    ast_program = ASTBuilder().build(program)
    assert len(ast_program.functions) == 2

    first, second = ast_program.functions
    assert first.entry_label == "block_0"
    assert [block.label for block in first.blocks] == ["block_0", "block_1"]
    assert second.entry_label == "block_2"
    assert [block.label for block in second.blocks] == ["block_2", "block_3"]

    rendered = ASTPrinter().render(ast_program)
    assert "function seg0_fn_0000" in rendered
    assert "function seg0_fn_0008" in rendered
    assert "succ=[block_1]" in rendered
