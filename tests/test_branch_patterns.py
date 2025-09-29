from __future__ import annotations

from typing import Dict

from mbcdisasm.branch_analysis import (
    BranchDescriptor,
    BranchKind,
    BranchOutcome,
    BranchRegistry,
    BranchStructure,
)
from mbcdisasm.branch_patterns import analyse_branch_patterns
from mbcdisasm.cfg import BasicBlock, ControlFlowGraph
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.manual_semantics import AnnotatedInstruction, InstructionSemantics, StackEffect


def _make_semantics(
    mnemonic: str,
    *,
    control_flow: str | None,
    inputs: int,
    outputs: int,
) -> InstructionSemantics:
    return InstructionSemantics(
        key=f"{mnemonic}:0",
        mnemonic=mnemonic,
        manual_name=mnemonic,
        summary=None,
        control_flow=control_flow,
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


def _basic_block(offset: int, semantics: InstructionSemantics, successors: set[int]) -> BasicBlock:
    word = InstructionWord(offset=offset, raw=0)
    instruction = AnnotatedInstruction(word=word, semantics=semantics)
    return BasicBlock(start=offset, instructions=[instruction], successors=set(successors), predecessors=set())


def _make_graph(blocks: Dict[int, BasicBlock]) -> ControlFlowGraph:
    for block in blocks.values():
        for successor in block.successors:
            if successor in blocks:
                blocks[successor].predecessors.add(block.start)
    return ControlFlowGraph(segment=None, blocks=blocks)


def test_conditional_pattern_detects_if_else() -> None:
    branch_sem = _make_semantics("branch", control_flow="branch", inputs=1, outputs=0)
    fallthrough_sem = _make_semantics("fallthrough", control_flow=None, inputs=0, outputs=0)

    entry = _basic_block(0, branch_sem, {4, 8})
    then_block = _basic_block(4, fallthrough_sem, {8})
    join_block = _basic_block(8, fallthrough_sem, set())

    graph = _make_graph({0: entry, 4: then_block, 8: join_block})

    descriptor = BranchDescriptor(
        offset=0,
        block_start=0,
        kind=BranchKind.CONDITIONAL,
        semantics=branch_sem,
        outcomes=(
            BranchOutcome(target=4, condition="true", description="true"),
            BranchOutcome(target=8, condition="false", description="false"),
        ),
        fallthrough=8,
        operand_hint=None,
        stack_inputs=1,
        stack_outputs=0,
        consumed_symbols=(),
        warnings=(),
    )

    structure = BranchStructure(graph=graph, descriptors={0: descriptor}, predecessors={4: {0}, 8: {0, 4}})
    registry = BranchRegistry(structure)

    patterns = analyse_branch_patterns(registry)
    pattern = patterns.conditional(0)
    assert pattern is not None
    assert pattern.style == "if"
    assert pattern.true_target == 4
    assert pattern.false_target == 8
    assert pattern.join_block == 8


def test_loop_pattern_detects_natural_loop() -> None:
    branch_sem = _make_semantics("loop_branch", control_flow="branch", inputs=1, outputs=0)
    body_sem = _make_semantics("body", control_flow=None, inputs=0, outputs=0)

    header = _basic_block(0, branch_sem, {4, 12})
    body = _basic_block(4, body_sem, {0})
    exit_block = _basic_block(12, body_sem, set())

    graph = _make_graph({0: header, 4: body, 12: exit_block})

    descriptor = BranchDescriptor(
        offset=0,
        block_start=0,
        kind=BranchKind.CONDITIONAL,
        semantics=branch_sem,
        outcomes=(
            BranchOutcome(target=4, condition="true", description="body"),
            BranchOutcome(target=12, condition="false", description="exit"),
        ),
        fallthrough=12,
        operand_hint=None,
        stack_inputs=1,
        stack_outputs=0,
        consumed_symbols=(),
        warnings=(),
    )

    structure = BranchStructure(
        graph=graph,
        descriptors={0: descriptor},
        predecessors={4: {0}, 12: {0}, 0: {4}},
    )
    registry = BranchRegistry(structure)
    patterns = analyse_branch_patterns(registry)

    loop = patterns.loop(0)
    assert loop is not None
    assert 4 in loop.body
    conditional = patterns.conditional(0)
    assert conditional is not None
    assert conditional.style == "loop_guard"
    assert conditional.loop_body_target == 4
    assert conditional.loop_exit_target == 12
