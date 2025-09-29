from __future__ import annotations

from typing import cast

from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.manual_semantics import InstructionSemantics, StackEffect
from mbcdisasm.stack_analysis import StackSeedAnalyzer
from mbcdisasm.knowledge import KnowledgeBase


def _make_semantics(
    mnemonic: str,
    *,
    inputs: int,
    outputs: int,
) -> InstructionSemantics:
    return InstructionSemantics(
        key=f"{mnemonic}:0",
        mnemonic=mnemonic,
        manual_name=mnemonic,
        summary=None,
        control_flow=None,
        stack_delta=None,
        stack_effect=StackEffect(inputs=inputs, outputs=outputs, delta=None, source="test"),
        tags=(),
        comparison_operator=None,
        enum_values={},
        enum_namespace=None,
        struct_context=None,
        stack_inputs=inputs,
        stack_outputs=outputs,
        uses_operand=False,
        operand_hint=None,
        vm_method="",
        vm_call_style="",
    )


def test_stack_seed_analyzer_records_underflows() -> None:
    consume_two = _make_semantics("consume_two", inputs=2, outputs=0)
    produce_one = _make_semantics("produce_one", inputs=0, outputs=1)

    instructions = [
        IRInstruction(
            offset=0,
            key="consume_two:0",
            mnemonic="consume_two",
            operand=0,
            stack_delta=None,
            control_flow=None,
            semantics=consume_two,
            stack_inputs=2,
            stack_outputs=0,
        ),
        IRInstruction(
            offset=4,
            key="produce_one:0",
            mnemonic="produce_one",
            operand=0,
            stack_delta=None,
            control_flow=None,
            semantics=produce_one,
            stack_inputs=0,
            stack_outputs=1,
        ),
    ]

    block = IRBlock(start=0, instructions=instructions, successors=[])
    program = IRProgram(segment_index=0, blocks={0: block})

    analyzer = StackSeedAnalyzer(program, cast(KnowledgeBase, object()))
    analyzer.max_seed_depth = 0.0
    plan = analyzer.build_plan()
    assert len(plan.seeds[0]) == 0

    underflows = analyzer.underflows()
    assert underflows, "expected at least one underflow event"
    event = underflows[0]
    assert event.block_start == 0
    assert event.deficit >= 1
