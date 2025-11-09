from __future__ import annotations

from mbcdisasm.ast import ASTBuilder
from mbcdisasm.ir.model import (
    IRBlock,
    IRCfgBlock,
    IRCfgEdge,
    IRControlFlowGraph,
    IRFunctionCfg,
    IRIf,
    IRLiteral,
    IRProgram,
    IRReturn,
    IRSegment,
    IRTerminator,
    NormalizerMetrics,
)


def _make_program(blocks: list[IRBlock], cfg_blocks: list[IRCfgBlock]) -> IRProgram:
    segment = IRSegment(
        index=0,
        start=0,
        length=0,
        blocks=tuple(blocks),
        metrics=NormalizerMetrics(),
    )
    function = IRFunctionCfg(
        segment_index=0,
        name="test",
        entry_block=cfg_blocks[0].label,
        entry_offset=cfg_blocks[0].start_offset,
        blocks=tuple(cfg_blocks),
    )
    cfg = IRControlFlowGraph(functions=(function,))
    return IRProgram(
        segments=(segment,),
        metrics=NormalizerMetrics(),
        cfg=cfg,
    )


def test_ast_builder_computes_dominators_and_loops() -> None:
    entry = IRBlock(
        label="entry",
        start_offset=0x0000,
        nodes=(IRTerminator(operand=0),),
    )
    header = IRBlock(
        label="loop_header",
        start_offset=0x0010,
        nodes=(IRIf(condition="cond", then_target=0x20, else_target=0x30),),
    )
    body = IRBlock(
        label="loop_body",
        start_offset=0x0020,
        nodes=(
            IRLiteral(value=1, mode=0, source="test"),
            IRTerminator(operand=0),
        ),
    )
    exit_block = IRBlock(
        label="exit",
        start_offset=0x0030,
        nodes=(IRReturn(values=("x",),),),
    )

    cfg_blocks = [
        IRCfgBlock(
            label="entry",
            start_offset=0x0000,
            terminator="goto",
            edges=(IRCfgEdge("goto", "loop_header"),),
        ),
        IRCfgBlock(
            label="loop_header",
            start_offset=0x0010,
            terminator="if",
            edges=(
                IRCfgEdge("then", "loop_body"),
                IRCfgEdge("else", "exit"),
            ),
        ),
        IRCfgBlock(
            label="loop_body",
            start_offset=0x0020,
            terminator="goto",
            edges=(IRCfgEdge("goto", "loop_header"),),
        ),
        IRCfgBlock(
            label="exit",
            start_offset=0x0030,
            terminator="return",
            edges=tuple(),
        ),
    ]

    program = _make_program([entry, header, body, exit_block], cfg_blocks)
    builder = ASTBuilder()
    ast_program = builder.build(program)

    function = ast_program.functions[0]
    assert function.dominators.immediate["entry"] is None
    assert function.dominators.immediate["loop_header"] == "entry"
    assert function.dominators.immediate["loop_body"] == "loop_header"
    assert function.dominators.immediate["exit"] == "loop_header"

    assert len(function.loops) == 1
    loop = function.loops[0]
    assert loop.header == "loop_header"
    assert loop.latches == ("loop_body",)
    assert set(loop.blocks) == {"loop_header", "loop_body"}


def test_ast_builder_splits_critical_edges() -> None:
    entry = IRBlock(
        label="entry",
        start_offset=0x0000,
        nodes=(IRIf(condition="c0", then_target=0x10, else_target=0x20),),
    )
    block_a = IRBlock(
        label="block_a",
        start_offset=0x0010,
        nodes=(IRIf(condition="c1", then_target=0x30, else_target=0x40),),
    )
    block_b = IRBlock(
        label="block_b",
        start_offset=0x0020,
        nodes=(IRLiteral(value=2, mode=0, source="b"), IRTerminator(operand=0)),
    )
    block_c = IRBlock(
        label="block_c",
        start_offset=0x0030,
        nodes=(IRLiteral(value=3, mode=0, source="c"), IRTerminator(operand=0)),
    )
    join = IRBlock(
        label="join",
        start_offset=0x0040,
        nodes=(IRReturn(values=("y",),),),
    )

    cfg_blocks = [
        IRCfgBlock(
            label="entry",
            start_offset=0x0000,
            terminator="if",
            edges=(IRCfgEdge("then", "block_a"), IRCfgEdge("else", "block_b")),
        ),
        IRCfgBlock(
            label="block_a",
            start_offset=0x0010,
            terminator="if",
            edges=(
                IRCfgEdge("then", "join"),
                IRCfgEdge("else", "block_c"),
            ),
        ),
        IRCfgBlock(
            label="block_b",
            start_offset=0x0020,
            terminator="goto",
            edges=(IRCfgEdge("goto", "join"),),
        ),
        IRCfgBlock(
            label="block_c",
            start_offset=0x0030,
            terminator="goto",
            edges=(IRCfgEdge("goto", "join"),),
        ),
        IRCfgBlock(
            label="join",
            start_offset=0x0040,
            terminator="return",
            edges=tuple(),
        ),
    ]

    program = _make_program([entry, block_a, block_b, block_c, join], cfg_blocks)
    builder = ASTBuilder()
    ast_program = builder.build(program)
    function = ast_program.functions[0]

    synthetic_blocks = [block for block in function.blocks if block.synthetic]
    assert synthetic_blocks, "expected critical edges to be split"
    join_split = [block for block in synthetic_blocks if "join" in block.label]
    assert join_split
    for block in join_split:
        assert block.terminator.kind == "goto"
        assert block.successors == ("join",)


def test_branch_folding_to_goto() -> None:
    entry = IRBlock(
        label="entry",
        start_offset=0x0000,
        nodes=(
            IRIf(condition="always", then_target=0x10, else_target=0x10),
        ),
    )
    target = IRBlock(
        label="target",
        start_offset=0x0010,
        nodes=(IRReturn(values=("z",),),),
    )

    cfg_blocks = [
        IRCfgBlock(
            label="entry",
            start_offset=0x0000,
            terminator="if",
            edges=(
                IRCfgEdge("then", "target"),
                IRCfgEdge("else", "target"),
            ),
        ),
        IRCfgBlock(
            label="target",
            start_offset=0x0010,
            terminator="return",
            edges=tuple(),
        ),
    ]

    program = _make_program([entry, target], cfg_blocks)
    builder = ASTBuilder()
    ast_program = builder.build(program)

    function = ast_program.functions[0]
    entry_block = next(block for block in function.blocks if block.label == "entry")
    assert entry_block.terminator.kind == "goto"
    assert entry_block.terminator.targets == ("target",)

