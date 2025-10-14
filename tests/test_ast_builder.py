from pathlib import Path

from mbcdisasm import IRNormalizer
from mbcdisasm.ast import ASTBuilder, ASTDispatch
from mbcdisasm.ir.model import (
    IRBlock,
    IRDispatchCase,
    IRProgram,
    IRSegment,
    IRSwitchDispatch,
    IRReturn,
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

    summary = ast_program.metrics.describe()
    assert "procedures=" in summary
    assert "calls=" in summary


def test_ast_builder_resolves_dispatch_links() -> None:
    metrics = NormalizerMetrics()
    dispatch_cases = (
        IRDispatchCase(key=0x01, target=0x0010),
        IRDispatchCase(key=0x02, target=0x0020, symbol="handler_two"),
    )
    dispatch_node = IRSwitchDispatch(cases=dispatch_cases, helper=0x6623, helper_symbol="dispatch_helper", default=0x0030)
    blocks = (
        IRBlock(label="entry", start_offset=0x0000, nodes=(dispatch_node,)),
        IRBlock(label="case_a", start_offset=0x0010, nodes=(IRReturn(values=tuple()),)),
        IRBlock(label="case_b", start_offset=0x0020, nodes=(IRReturn(values=tuple()),)),
        IRBlock(label="fallback", start_offset=0x0030, nodes=(IRReturn(values=tuple()),)),
    )
    segment = IRSegment(index=0, start=0, length=0, blocks=blocks, metrics=metrics)
    program = IRProgram(segments=(segment,), metrics=metrics)

    builder = ASTBuilder()
    ast_program = builder.build(program)

    segment_result = ast_program.segments[0]
    procedure = segment_result.procedures[0]
    entry_block = next(block for block in procedure.blocks if block.start_offset == 0x0000)

    assert entry_block.successors
    successor_offsets = {block.start_offset for block in entry_block.successors}
    assert successor_offsets == {0x0010, 0x0020, 0x0030}

    assert entry_block.statements
    dispatch_statement = entry_block.statements[0]
    assert isinstance(dispatch_statement, ASTDispatch)
    case_targets = {case.target.start_offset for case in dispatch_statement.cases}
    assert case_targets == {0x0010, 0x0020}
    assert dispatch_statement.default is not None
    assert dispatch_statement.default.start_offset == 0x0030
    rendered = dispatch_statement.render()
    assert "dispatch helper=dispatch_helper" in rendered
