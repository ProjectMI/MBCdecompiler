from __future__ import annotations

import json

from mbcdisasm.branch_analysis import (
    BranchAnalyzer,
    BranchReport,
    BranchStatistics,
    branch_graph_to_dict,
    branch_graph_to_json,
    branch_reports_to_json,
    branch_statistics_to_json,
    build_branch_report,
    build_branch_reports,
    describe_branches,
    render_branch_graph,
    render_branch_reports,
    render_branch_summary,
    render_branch_timeline,
)
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.manual_semantics import InstructionSemantics, StackEffect


def _semantics(
    key: str,
    name: str,
    *,
    control_flow: str,
    inputs: int,
    outputs: int,
) -> InstructionSemantics:
    return InstructionSemantics(
        key=key,
        mnemonic=name,
        manual_name=name,
        summary=f"{name} summary",
        control_flow=control_flow,
        stack_delta=float(outputs - inputs),
        stack_effect=StackEffect(inputs=inputs, outputs=outputs, delta=float(outputs - inputs), source="test"),
        tags=(),
        comparison_operator=None,
        enum_values={},
        enum_namespace=None,
        struct_context=None,
        stack_inputs=inputs,
        stack_outputs=outputs,
        uses_operand=False,
        operand_hint=None,
        vm_method=name,
        vm_call_style="method",
    )


def _instruction(offset: int, semantics: InstructionSemantics) -> IRInstruction:
    return IRInstruction(
        offset=offset,
        key=semantics.key,
        mnemonic=semantics.mnemonic,
        operand=0,
        stack_delta=semantics.stack_delta,
        control_flow=semantics.control_flow,
        semantics=semantics,
        stack_inputs=semantics.stack_inputs or 0,
        stack_outputs=semantics.stack_outputs or 0,
    )


def test_branch_analyzer_recognises_conditional_branch() -> None:
    branch_sem = _semantics("10:00", "branch_if", control_flow="branch", inputs=1, outputs=0)
    ret_sem = _semantics("11:00", "return", control_flow="return", inputs=1, outputs=0)
    block_a = IRBlock(
        start=0x100,
        instructions=[_instruction(0x100, branch_sem)],
        successors=[0x200, 0x300],
    )
    block_b = IRBlock(start=0x200, instructions=[_instruction(0x200, ret_sem)], successors=[])
    block_c = IRBlock(start=0x300, instructions=[_instruction(0x300, ret_sem)], successors=[])
    program = IRProgram(segment_index=1, blocks={0x100: block_a, 0x200: block_b, 0x300: block_c})

    graph = describe_branches(program)
    assert len(graph.descriptors) == 3
    descriptor = graph.get(0x100)
    assert descriptor is not None
    assert descriptor.block_start == 0x100
    assert descriptor.classification == "branch"
    assert {edge.role for edge in descriptor.edges} == {"true", "false"}
    true_edge = next(edge for edge in descriptor.edges if edge.role == "true")
    assert true_edge.target == 0x300
    false_edge = next(edge for edge in descriptor.edges if edge.role == "false")
    assert false_edge.target == 0x200
    return_edges = [edge for offset, desc in graph.descriptors.items() if offset != 0x100 for edge in desc.edges]
    assert all(edge.role == "return" for edge in return_edges)

    rendered = render_branch_graph(graph)
    assert "segment 1 branches" in rendered
    assert "branch_if [branch]" in rendered
    payload = branch_graph_to_dict(graph)
    assert payload["segment_index"] == 1
    json_blob = branch_graph_to_json(graph, indent=0)
    assert json.loads(json_blob)["segment_index"] == 1


def test_branch_analyzer_captures_jump_and_return() -> None:
    jump_sem = _semantics("20:00", "jump", control_flow="jump", inputs=0, outputs=0)
    ret_sem = _semantics("21:00", "return", control_flow="return", inputs=1, outputs=0)
    block_a = IRBlock(
        start=0x10,
        instructions=[_instruction(0x10, jump_sem)],
        successors=[0x40],
    )
    block_b = IRBlock(
        start=0x40,
        instructions=[_instruction(0x40, ret_sem)],
        successors=[],
    )
    program = IRProgram(segment_index=2, blocks={0x10: block_a, 0x40: block_b})

    analyzer = BranchAnalyzer(program)
    graph = analyzer.analyse()
    descriptor = graph.get(0x10)
    assert descriptor is not None
    assert descriptor.classification == "jump"
    assert any(edge.role == "jump" for edge in descriptor.edges)
    return_descriptor = graph.get(0x40)
    assert return_descriptor is not None
    assert return_descriptor.classification == "return"
    assert any(edge.target is None for edge in return_descriptor.edges)


def test_branch_statistics_and_summary_helpers() -> None:
    branch_sem = _semantics("30:00", "branch_if", control_flow="branch", inputs=1, outputs=0)
    jump_sem = _semantics("31:00", "jump", control_flow="jump", inputs=0, outputs=0)
    ret_sem = _semantics("32:00", "return", control_flow="return", inputs=1, outputs=0)
    block_a = IRBlock(start=0x10, instructions=[_instruction(0x10, branch_sem)], successors=[0x40, 0x20])
    block_b = IRBlock(start=0x20, instructions=[_instruction(0x20, jump_sem)], successors=[0x10])
    block_c = IRBlock(start=0x40, instructions=[_instruction(0x40, ret_sem)], successors=[])
    program = IRProgram(segment_index=5, blocks={0x10: block_a, 0x20: block_b, 0x40: block_c})

    graph = describe_branches(program)
    stats = graph.statistics()
    assert isinstance(stats, BranchStatistics)
    assert stats.total == 3
    assert stats.conditional == 1
    assert stats.jumps == 1
    assert stats.returns == 1
    assert stats.backward_edges >= 1
    json_payload = branch_statistics_to_json(stats, indent=0)
    assert '"total": 3' in json_payload

    summary = render_branch_summary(graph)
    assert "segment 5 branch report" in summary
    assert "total branches: 3" in summary

    single = build_branch_report(program)
    reports = build_branch_reports([program])
    assert len(reports) == 1
    report = reports[0]
    assert report.segment_index == single.segment_index
    assert isinstance(report, BranchReport)
    rendered = render_branch_reports(reports)
    assert "timeline:" in rendered
    assert render_branch_timeline(report.timeline)
    json_reports = branch_reports_to_json(reports, indent=0)
    assert '"segment_index": 5' in json_reports


def test_branch_report_builder_handles_multiple_programs() -> None:
    sem = _semantics("40:00", "return", control_flow="return", inputs=1, outputs=0)
    block = IRBlock(start=0, instructions=[_instruction(0, sem)], successors=[])
    program_a = IRProgram(segment_index=1, blocks={0: block})
    program_b = IRProgram(segment_index=2, blocks={0: block})

    reports = build_branch_reports([program_a, program_b])
    assert [report.segment_index for report in reports] == [1, 2]
    text = render_branch_reports(reports)
    assert text.count("branch report") == 2

