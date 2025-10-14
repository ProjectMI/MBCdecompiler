from mbcdisasm.ast import ASTBuilder, ASTTextRenderer
from mbcdisasm.ir.model import (
    IRBlock,
    IRCall,
    IRFunctionPrologue,
    IRIf,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRSlot,
    IRStore,
    IRRaw,
    MemSpace,
    NormalizerMetrics,
)


def make_segment(blocks):
    metrics = NormalizerMetrics()
    return IRSegment(index=0, start=0, length=0x40, blocks=tuple(blocks), metrics=metrics)


def make_program(blocks):
    segment = make_segment(blocks)
    program_metrics = NormalizerMetrics()
    return IRProgram(segments=(segment,), metrics=program_metrics)


def test_cfg_builder_detects_branch_edges():
    block0 = IRBlock(
        label="block_0",
        start_offset=0x0000,
        nodes=(IRIf(condition="bool0", then_target=0x0010, else_target=0x0020),),
    )
    block1 = IRBlock(
        label="block_1",
        start_offset=0x0010,
        nodes=(IRReturn(values=tuple()),),
    )
    block2 = IRBlock(
        label="block_2",
        start_offset=0x0020,
        nodes=(IRReturn(values=tuple()),),
    )
    program = make_program([block0, block1, block2])

    builder = ASTBuilder()
    ast_program = builder.build(program)

    cfg = ast_program.cfg
    node0 = cfg.node("block_0")
    assert set(node0.successors) == {"block_1", "block_2"}
    assert cfg.successors("block_1") == tuple()
    assert cfg.successors("block_2") == tuple()


def test_procedure_detection_uses_prologue_and_calls():
    prologue_block = IRBlock(
        label="block_0",
        start_offset=0x0000,
        nodes=(
            IRFunctionPrologue(var="r0", expr="cond0", then_target=0x0010, else_target=0x0020),
        ),
    )
    body_block = IRBlock(
        label="block_1",
        start_offset=0x0010,
        nodes=(
            IRCall(target=0x0030, args=("word0",), symbol="callee"),
            IRReturn(values=("ret0",)),
        ),
    )
    else_block = IRBlock(
        label="block_2",
        start_offset=0x0020,
        nodes=(IRReturn(values=tuple()),),
    )
    callee_block = IRBlock(
        label="block_3",
        start_offset=0x0030,
        nodes=(IRReturn(values=tuple()),),
    )

    program = make_program([prologue_block, body_block, else_block, callee_block])
    ast_program = ASTBuilder().build(program)

    segment = ast_program.segments[0]
    assert len(segment.procedures) == 2
    names = {proc.name for proc in segment.procedures}
    assert "proc_0x0000" in names
    assert "callee" in names
    proc = next(proc for proc in segment.procedures if proc.name == "proc_0x0000")
    assert proc.entry == "block_0"
    assert set(proc.blocks) >= {"block_0", "block_1"}


def test_expression_builder_resolves_load_store_and_call():
    load_block = IRBlock(
        label="block_0",
        start_offset=0x0000,
        nodes=(
            IRLoad(slot=IRSlot(space=MemSpace.FRAME, index=1), target="word0"),
            IRLoad(slot=IRSlot(space=MemSpace.GLOBAL, index=2), target="ptr0"),
            IRRaw(
                mnemonic="add",
                operand=0,
                operand_role="target",
                operand_alias="word1",
                annotations=("lhs=word0", "rhs=ptr0"),
            ),
            IRStore(slot=IRSlot(space=MemSpace.FRAME, index=3), value="word1"),
            IRCall(target=0x0040, args=("word1", "word0"), symbol="helper"),
            IRReturn(values=("word1",)),
        ),
    )

    program = make_program([load_block])
    ast_program = ASTBuilder().build(program)
    segment = ast_program.segments[0]
    block = segment.blocks[0]

    statements = block.statements
    assert statements[0].kind == "assign"
    assert statements[0].target == "word0"
    assert statements[1].target == "ptr0"
    assert statements[2].target == "word1"
    assert statements[3].kind == "store"
    assert statements[3].expr.alias == "word1"
    assert statements[3].expr.describe().startswith("add[")
    assert statements[4].kind == "call"
    assert statements[4].target == "helper"
    assert statements[-1].kind == "return"

    metrics = ast_program.metrics
    assert metrics.call_arguments == 2
    assert metrics.return_values == 1
    assert metrics.store_values == 1
    assert metrics.expressions >= 3


def test_ast_metrics_describe_includes_key_fields():
    builder = ASTBuilder()
    program = make_program([])
    ast_program = builder.build(program)
    description = ast_program.metrics.describe()
    assert "procedures=" in description
    assert "expressions=" in description


def test_ast_renderer_outputs_structure(tmp_path):
    load_block = IRBlock(
        label="block_0",
        start_offset=0x0000,
        nodes=(IRReturn(values=tuple()),),
    )
    program = make_program([load_block])
    ast_program = ASTBuilder().build(program)

    renderer = ASTTextRenderer()
    output = renderer.render(ast_program)
    assert "; ast metrics:" in output
    assert "procedure" in output or "block block_0" in output
    path = tmp_path / "ast.txt"
    renderer.write(ast_program, path)
    assert path.exists()

