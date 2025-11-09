from __future__ import annotations

from mbcdisasm.ast import ASTBuilder
from mbcdisasm.ir.model import (
    IRBlock,
    IRIf,
    IRReturn,
    IRProgram,
    IRSegment,
    IRLiteral,
    IRTerminator,
    NormalizerMetrics,
    IRControlFlowGraph,
    IRFunctionCfg,
    IRCfgBlock,
    IRCfgEdge,
)


def _program_from_blocks(function_blocks, cfg_blocks):
    segment = IRSegment(
        index=0,
        start=0,
        length=0,
        blocks=tuple(function_blocks),
        metrics=NormalizerMetrics(),
    )
    function_cfg = IRFunctionCfg(
        segment_index=0,
        name="test_func",
        entry_block=cfg_blocks[0].label,
        entry_offset=function_blocks[0].start_offset,
        blocks=tuple(cfg_blocks),
    )
    cfg = IRControlFlowGraph(functions=(function_cfg,))
    return IRProgram(segments=(segment,), metrics=NormalizerMetrics(), cfg=cfg)


def test_builder_constructs_basic_function():
    entry_if = IRIf(condition="x == 0", then_target=0x10, else_target=0x20)
    entry_block = IRBlock(
        label="block_0",
        start_offset=0x0000,
        nodes=(IRLiteral(0, 0, "entry"), entry_if),
    )
    then_block = IRBlock(
        label="block_1",
        start_offset=0x0010,
        nodes=(IRReturn(values=("value0:unknown",)),),
    )
    else_block = IRBlock(
        label="block_2",
        start_offset=0x0020,
        nodes=(IRReturn(values=("value0:unknown",)),),
    )

    cfg_blocks = (
        IRCfgBlock(
            label="block_0",
            start_offset=0x0000,
            terminator=entry_if.describe(),
            edges=(IRCfgEdge("then", "block_1"), IRCfgEdge("else", "block_2")),
        ),
        IRCfgBlock(
            label="block_1",
            start_offset=0x0010,
            terminator="return",
            edges=tuple(),
        ),
        IRCfgBlock(
            label="block_2",
            start_offset=0x0020,
            terminator="return",
            edges=tuple(),
        ),
    )

    program = _program_from_blocks((entry_block, then_block, else_block), cfg_blocks)
    builder = ASTBuilder()
    ast_program = builder.build(program)

    function = ast_program.functions[0]
    assert {block.label for block in function.blocks} == {"block_0", "block_1", "block_2"}

    branch = next(block for block in function.blocks if block.label == "block_0")
    assert branch.terminator.kind == "branch"
    assert branch.successors == ("block_1", "block_2")

    dom = {info.label: set(info.members) for info in function.dominators}
    assert dom["block_0"] == {"block_0"}
    assert dom["block_1"] == {"block_0", "block_1"}
    assert dom["block_2"] == {"block_0", "block_2"}

    assert function.loops == tuple()


def test_builder_splits_critical_edges_and_folds_branches():
    entry_if = IRIf(condition="flag", then_target=0x10, else_target=0x30)
    fold_branch = IRIf(condition="always", then_target=0x30, else_target=0x30)

    entry_block = IRBlock(
        label="block_entry",
        start_offset=0x0000,
        nodes=(entry_if,),
    )
    fold_block = IRBlock(
        label="block_mid",
        start_offset=0x0010,
        nodes=(fold_branch,),
    )
    join_block = IRBlock(
        label="block_join",
        start_offset=0x0030,
        nodes=(IRReturn(values=("value0:unknown",)),),
    )

    cfg_blocks = (
        IRCfgBlock(
            label="block_entry",
            start_offset=0x0000,
            terminator=entry_if.describe(),
            edges=(IRCfgEdge("then", "block_mid"), IRCfgEdge("else", "block_join")),
        ),
        IRCfgBlock(
            label="block_mid",
            start_offset=0x0010,
            terminator=fold_branch.describe(),
            edges=(IRCfgEdge("then", "block_join"), IRCfgEdge("else", "block_join")),
        ),
        IRCfgBlock(
            label="block_join",
            start_offset=0x0030,
            terminator="return",
            edges=tuple(),
        ),
    )

    program = _program_from_blocks((entry_block, fold_block, join_block), cfg_blocks)
    builder = ASTBuilder()
    ast_program = builder.build(program)
    function = ast_program.functions[0]

    labels = {block.label for block in function.blocks}
    synthetic = [block for block in function.blocks if block.synthetic]
    assert len(synthetic) == 2
    for bridge in synthetic:
        assert bridge.terminator.kind == "goto"
        assert bridge.successors == ("block_join",)
        assert bridge.predecessors == ("block_entry",)

    entry = next(block for block in function.blocks if block.label == "block_entry")
    assert "block_join" not in entry.successors
    assert all(label in entry.successors for label in (synthetic[0].label, synthetic[1].label))

    assert "block_mid" not in labels


def test_builder_discovers_loops():
    loop_branch = IRIf(condition="loop", then_target=0x10, else_target=0x30)
    entry_block = IRBlock(
        label="block_header",
        start_offset=0x0000,
        nodes=(loop_branch,),
    )
    body_block = IRBlock(
        label="block_body",
        start_offset=0x0010,
        nodes=(IRTerminator(operand=0),),
    )
    exit_block = IRBlock(
        label="block_exit",
        start_offset=0x0030,
        nodes=(IRReturn(values=("value0:unknown",)),),
    )

    cfg_blocks = (
        IRCfgBlock(
            label="block_header",
            start_offset=0x0000,
            terminator=loop_branch.describe(),
            edges=(IRCfgEdge("then", "block_body"), IRCfgEdge("else", "block_exit")),
        ),
        IRCfgBlock(
            label="block_body",
            start_offset=0x0010,
            terminator="goto",
            edges=(IRCfgEdge("goto", "block_header"),),
        ),
        IRCfgBlock(
            label="block_exit",
            start_offset=0x0030,
            terminator="return",
            edges=tuple(),
        ),
    )

    program = _program_from_blocks((entry_block, body_block, exit_block), cfg_blocks)
    builder = ASTBuilder()
    ast_program = builder.build(program)
    function = ast_program.functions[0]

    loops = {loop.header: loop for loop in function.loops}
    assert set(loops) == {"block_header"}
    header_loop = loops["block_header"]
    assert set(header_loop.nodes) == {"block_body", "block_header"}
    assert set(header_loop.latches) == {"block_body"}
