from __future__ import annotations

from mbcdisasm import ASTBuilder
from mbcdisasm.ir.model import IRBlock, IRIf, IRProgram, IRReturn, IRSegment, NormalizerMetrics


def build_program(blocks: tuple[IRBlock, ...]) -> IRProgram:
    segment = IRSegment(index=0, start=0, length=0, blocks=blocks, metrics=NormalizerMetrics())
    return IRProgram(segments=(segment,), metrics=NormalizerMetrics())


def test_ast_builder_structures_cfg() -> None:
    blocks = (
        IRBlock("entry", 0x0000, (IRIf("cond", 0x0010, 0x0020),)),
        IRBlock("then", 0x0010, (IRIf("b", 0x0030, 0x0040),)),
        IRBlock("fold", 0x0020, (IRIf("fold", 0x0030, 0x0030),)),
        IRBlock("join", 0x0030, (IRIf("join", 0x0040, 0x0050),)),
        IRBlock("ret1", 0x0040, (IRReturn(("v1",),),)),
        IRBlock("ret2", 0x0050, (IRReturn(("v2",),),)),
        IRBlock("dead", 0x0060, (IRReturn(("dead",),),)),
    )
    program = build_program(blocks)

    builder = ASTBuilder()
    ast_program = builder.build(program)

    assert len(ast_program.functions) == 1
    function = ast_program.functions[0]
    labels = {block.label for block in function.blocks}

    assert "dead" not in labels, "unreachable block should be eliminated"

    fold_block = next(block for block in function.blocks if block.label == "fold")
    assert fold_block.terminator.kind == "jump"

    assert any(block.label.startswith("then_split") for block in function.blocks)
    assert any(block.label.startswith("join_split") for block in function.blocks)

    ret1_block = next(block for block in function.blocks if block.label == "ret1")
    assert all(pred.startswith("then_split") or pred.startswith("join_split") for pred in ret1_block.predecessors)

    dom_map = {info.block: set(info.dominators) for info in function.dominators}
    assert dom_map["entry"] == {"entry"}
    assert "entry" in dom_map["then"]


def test_ast_builder_detects_loops() -> None:
    blocks = (
        IRBlock("entry", 0x0000, (IRIf("loop", 0x0010, 0x0030),)),
        IRBlock("loop_header", 0x0010, (IRIf("cond", 0x0020, 0x0030),)),
        IRBlock("loop_body", 0x0020, (IRIf("body", 0x0010, 0x0030),)),
        IRBlock("exit", 0x0030, (IRReturn(("done",),),)),
    )
    program = build_program(blocks)

    builder = ASTBuilder()
    ast_program = builder.build(program)
    function = ast_program.functions[0]

    loops = {loop.header: loop for loop in function.loops}
    assert "loop_header" in loops
    loop = loops["loop_header"]
    loop_nodes = set(loop.nodes)
    assert {"loop_body", "loop_header"}.issubset(loop_nodes)
    assert any(latch.startswith("loop_body") for latch in loop.latches)

    dom_map = {info.block: set(info.dominators) for info in function.dominators}
    assert dom_map["loop_body"].issuperset({"entry", "loop_header", "loop_body"})

    post_dom_map = {info.block: set(info.dominators) for info in function.post_dominators}
    assert "exit" in post_dom_map["entry"]
    assert "exit" in post_dom_map["loop_body"]
