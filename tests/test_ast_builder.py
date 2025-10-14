from pathlib import Path

from mbcdisasm import IRNormalizer
from mbcdisasm.ast import ASTBuilder, ASTCallStatement, ASTSwitch
from mbcdisasm.constants import CALL_SHUFFLE_STANDARD
from mbcdisasm.ir.model import (
    IRBlock,
    IRCall,
    IRProgram,
    IRReturn,
    IRSegment,
    IRStackEffect,
    IRSwitchDispatch,
    IRDispatchCase,
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


def test_ast_builder_aliases_tail_helpers_and_switch() -> None:
    call_shuffle = IRCall(
        target=0x0072,
        args=("arg0", "arg1", "arg2"),
        tail=True,
        arity=3,
        convention=IRStackEffect(mnemonic="stack_shuffle", operand=CALL_SHUFFLE_STANDARD),
        cleanup=tuple(),
        symbol="tail_helper_72",
    )
    call_dispatch = IRCall(
        target=0x003E,
        args=tuple(),
        tail=True,
        cleanup=tuple(),
        symbol="tail_helper_3e",
    )
    switch = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=1, target=0x1111, symbol="case_one"),
            IRDispatchCase(key=2, target=0x2222, symbol=None),
        ),
        helper=0x6623,
        helper_symbol="helper_6623",
        default=0x3333,
    )
    program = IRProgram(
        segments=(
            IRSegment(
                index=0,
                start=0,
                length=0,
                blocks=(
                    IRBlock(
                        label="block_0",
                        start_offset=0x0100,
                        nodes=(call_shuffle, call_dispatch, switch, IRReturn(values=("ret0",))),
                    ),
                ),
                metrics=NormalizerMetrics(),
            ),
        ),
        metrics=NormalizerMetrics(),
    )

    builder = ASTBuilder()
    ast_program = builder.build(program)
    block = ast_program.segments[0].procedures[0].blocks[0]

    assert block.statements
    first_stmt = block.statements[0]
    assert isinstance(first_stmt, ASTCallStatement)
    assert first_stmt.call.symbol == "ui.flush"
    rendered_args = [expr.render() for expr in first_stmt.call.args]
    assert rendered_args == ["arg1", "arg0", "arg2"]

    switch_stmt = next(stmt for stmt in block.statements if isinstance(stmt, ASTSwitch))
    assert switch_stmt.helper == 0x6623
    assert switch_stmt.helper_symbol == "helper_6623"
    assert switch_stmt.default == 0x3333
    rendered_cases = [case.render() for case in switch_stmt.cases]
    assert "1->case_one(0x1111)" in rendered_cases[0]
