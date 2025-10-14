from mbcdisasm.ast import ASTBuilder
from mbcdisasm.ast.model import (
    ASTAssignment,
    ASTBranchStatement,
    ASTIdentifier,
    ASTLoadExpr,
    ASTReturnStatement,
    ASTStoreStatement,
)
from mbcdisasm.ir.model import (
    IRBlock,
    IRIf,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRSlot,
    IRStore,
    MemSpace,
    NormalizerMetrics,
)


def build_program(*blocks: IRBlock) -> IRProgram:
    segment = IRSegment(
        index=0,
        start=0x100,
        length=0x40,
        blocks=tuple(blocks),
        metrics=NormalizerMetrics(),
    )
    return IRProgram(segments=(segment,), metrics=NormalizerMetrics())


def test_basic_assignment_store_and_return() -> None:
    slot = IRSlot(space=MemSpace.FRAME, index=0x10)
    load = IRLoad(slot=slot, target="word0")
    store = IRStore(slot=slot, value="word0")
    ret = IRReturn(values=("word0",))
    block = IRBlock(label="block_0", start_offset=0x100, nodes=(load, store, ret))

    program = build_program(block)
    ast_program = ASTBuilder().build_program(program)

    assert len(ast_program.segments) == 1
    segment = ast_program.segments[0]
    assert segment.metrics.blocks == 1
    assert len(segment.blocks) == 1

    ast_block = segment.blocks[0]
    assert len(ast_block.statements) == 3

    assign, store_stmt, ret_stmt = ast_block.statements
    assert isinstance(assign, ASTAssignment)
    assert assign.target.name == "word0"
    assert isinstance(assign.expr, ASTLoadExpr)

    assert isinstance(store_stmt, ASTStoreStatement)
    assert isinstance(store_stmt.value, ASTIdentifier)

    assert isinstance(ret_stmt, ASTReturnStatement)
    assert segment.procedures[0].block_labels == ("block_0",)


def test_cfg_edges_and_branch_detection() -> None:
    slot = IRSlot(space=MemSpace.FRAME, index=0x00)
    cond_load = IRLoad(slot=slot, target="bool0")
    branch = IRIf(condition="bool0", then_target=0x120, else_target=0x140)
    entry = IRBlock(label="block_0", start_offset=0x100, nodes=(cond_load, branch))

    ret_then = IRReturn(values=tuple())
    block_then = IRBlock(label="block_1", start_offset=0x120, nodes=(ret_then,))

    ret_else = IRReturn(values=tuple())
    block_else = IRBlock(label="block_2", start_offset=0x140, nodes=(ret_else,))

    program = build_program(entry, block_then, block_else)
    ast_program = ASTBuilder().build_program(program)

    segment = ast_program.segments[0]
    ast_entry = segment.blocks[0]
    assert any(isinstance(stmt, ASTBranchStatement) for stmt in ast_entry.statements)
    edges = list(ast_entry.successors)
    assert {edge.target_label for edge in edges} == {"block_1", "block_2"}
    assert segment.procedures[0].block_labels == ("block_0", "block_1", "block_2")
