from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.manual_semantics import InstructionSemantics, StackEffect
from mbcdisasm.vm_analysis import (
    VirtualMachineAnalyzer,
    analyze_block_lifetimes,
    analyze_program_lifetimes,
    format_vm_block_trace,
    lifetimes_to_dict,
    lifetimes_to_json,
    render_value_lifetimes,
    render_vm_program,
    vm_block_trace_to_dict,
    vm_block_trace_to_json,
    vm_program_trace_to_dict,
    vm_program_trace_to_json,
    summarise_program,
    count_operations,
)


def _semantics(
    key: str,
    mnemonic: str,
    *,
    delta: float,
    inputs: int,
    outputs: int,
    tags: tuple[str, ...] = (),
    control_flow: str | None = None,
) -> InstructionSemantics:
    return InstructionSemantics(
        key=key,
        mnemonic=mnemonic,
        manual_name=mnemonic,
        summary=f"{mnemonic} summary",
        control_flow=control_flow,
        stack_delta=delta,
        stack_effect=StackEffect(inputs=inputs, outputs=outputs, delta=delta, source="test"),
        tags=tags,
        comparison_operator=None,
        enum_values={},
        enum_namespace=None,
        struct_context=None,
        stack_inputs=inputs,
        stack_outputs=outputs,
        uses_operand=True,
        operand_hint=None,
        vm_method=mnemonic,
        vm_call_style="method",
    )


def _instruction(
    *,
    offset: int,
    key: str,
    operand: int,
    semantics: InstructionSemantics,
) -> IRInstruction:
    return IRInstruction(
        offset=offset,
        key=key,
        mnemonic=semantics.mnemonic,
        operand=operand,
        stack_delta=semantics.stack_delta,
        control_flow=semantics.control_flow,
        semantics=semantics,
        stack_inputs=semantics.stack_inputs or 0,
        stack_outputs=semantics.stack_outputs or 0,
    )


def test_virtual_machine_analyzer_tracks_stack_depth() -> None:
    analyzer = VirtualMachineAnalyzer()
    block = IRBlock(
        start=0x10,
        instructions=[
            _instruction(
                offset=0x10,
                key="01:00",
                operand=4,
                semantics=_semantics("01:00", "push_small", delta=1, inputs=0, outputs=1, tags=("literal",)),
            ),
            _instruction(
                offset=0x14,
                key="02:00",
                operand=0,
                semantics=_semantics("02:00", "consume_pair", delta=-2, inputs=2, outputs=0),
            ),
        ],
        successors=[],
    )

    trace = analyzer.trace_block(block)
    assert len(trace.entry_stack) == 0
    assert trace.instructions[0].state.depth_after == 1
    assert trace.instructions[1].state.depth_before == 1
    warnings = trace.instructions[1].operation.warnings
    assert "underflow" in warnings

    formatted = "\n".join(format_vm_block_trace(trace))
    assert "depth 0->1" in formatted
    assert "depth 1->0" in formatted

    lifetimes = analyze_block_lifetimes(trace)
    literal = lifetimes["literal_0"]
    assert literal.created_offset == 0x10
    assert literal.consumed_offsets == (0x14,)
    assert not literal.survives

    placeholder = lifetimes["missing_0"]
    assert placeholder.created_offset is None
    assert placeholder.consumed_offsets == (0x14,)
    assert not placeholder.survives

    lifetime_lines = render_value_lifetimes(lifetimes)
    assert any("literal_0" in line for line in lifetime_lines)
    lifetime_dict = lifetimes_to_dict(lifetimes)
    assert lifetime_dict["literal_0"]["created_offset"] == 0x10
    lifetime_json = lifetimes_to_json(lifetimes, indent=0)
    assert '"literal_0"' in lifetime_json

    trace_dict = vm_block_trace_to_dict(trace)
    assert trace_dict["start"] == 0x10
    assert trace_dict["instructions"][0]["offset"] == 0x10
    trace_json = vm_block_trace_to_json(trace, indent=0)
    assert '"offset": 16' in trace_json


def test_render_vm_program_includes_summary_header() -> None:
    analyzer = VirtualMachineAnalyzer()
    block_a = IRBlock(
        start=0,
        instructions=[
            _instruction(
                offset=0,
                key="01:00",
                operand=1,
                semantics=_semantics("01:00", "literal_a", delta=1, inputs=0, outputs=1, tags=("literal",)),
            ),
            _instruction(
                offset=4,
                key="02:00",
                operand=2,
                semantics=_semantics("02:00", "literal_b", delta=1, inputs=0, outputs=1, tags=("literal",)),
            ),
        ],
        successors=[0x10],
    )
    block_b = IRBlock(
        start=0x10,
        instructions=[
            _instruction(
                offset=0x10,
                key="03:00",
                operand=0,
                semantics=_semantics("03:00", "merge", delta=-2, inputs=2, outputs=1),
            ),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=7, blocks={block_a.start: block_a, block_b.start: block_b})

    trace = analyzer.trace_program(program)
    rendered = render_vm_program(trace)

    assert "segment 7 vm-trace" in rendered
    assert "instructions=3" in rendered
    assert "block 0x000000" in rendered
    assert "block 0x000010" in rendered

    lifetime_map = analyze_program_lifetimes(trace)
    assert set(lifetime_map) == {0, 0x10}
    block_a_lifetimes = lifetime_map[0]
    assert any(value.survives for value in block_a_lifetimes.values())
    program_dict = vm_program_trace_to_dict(trace)
    assert program_dict["segment_index"] == 7
    assert "0x000000" in program_dict["blocks"]
    program_json = vm_program_trace_to_json(trace, indent=0)
    assert '"segment_index": 7' in program_json
    summary = summarise_program(trace)
    assert "blocks=2" in summary
    assert count_operations(trace) == 3
    assert count_operations(trace.block_order()[0]) == 2
