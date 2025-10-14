from pathlib import Path

from mbcdisasm import IRNormalizer
from mbcdisasm.ast import ASTBuilder
from mbcdisasm.ir.model import (
    IRBlock,
    IRCall,
    IRCallCleanup,
    IRReturn,
    IRSegment,
    IRStackEffect,
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


def test_compact_cfg_collapses_cleanup_targets() -> None:
    builder = ASTBuilder()
    cleanup = IRCallCleanup(steps=(IRStackEffect(mnemonic="stack_teardown", pops=3),))
    blocks = (
        IRBlock(
            label="entry",
            start_offset=0x0000,
            nodes=(IRCall(target=0x0010, args=tuple()),),
        ),
        IRBlock(label="cleanup", start_offset=0x0010, nodes=(cleanup,)),
        IRBlock(
            label="exit",
            start_offset=0x0020,
            nodes=(IRReturn(values=tuple()),),
        ),
    )
    segment = IRSegment(index=0, start=0, length=0, blocks=blocks, metrics=NormalizerMetrics())
    block_map = {block.start_offset: block for block in segment.blocks}

    analyses = builder._build_cfg(segment, block_map)
    entry_reasons = builder._detect_entries(segment, block_map, analyses)
    compacted, updated_entries = builder._compact_cfg(analyses, entry_reasons)

    assert 0x0010 not in compacted
    assert compacted[0x0000].successors == (0x0020,)
    assert compacted[0x0000].fallthrough == 0x0020
    assert updated_entries.get(0x0010) is None
    assert updated_entries.get(0x0020) == ("call_target",)
