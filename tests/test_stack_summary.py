from pathlib import Path
from typing import Tuple

from mbcdisasm.stack_summary import (
    BlockStackProfile,
    StackProfileAnalyzer,
    StackProfileReport,
    StackProfileSummary,
    analyse_stack_profiles,
    build_stack_profile_report,
    build_stack_profile_reports,
    render_stack_profile_report,
    render_stack_profile_reports,
    render_stack_summary,
    filter_profiles_by_delta,
    filter_profiles_by_entry_requirement,
    stack_profile_reports_to_json,
    stack_profiles_to_csv,
    stack_profiles_to_json,
    write_stack_profile_reports,
)
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.manual_semantics import InstructionSemantics, StackEffect


def _semantics(key: str, mnemonic: str, *, inputs: int, outputs: int) -> InstructionSemantics:
    delta = float(outputs - inputs)
    return InstructionSemantics(
        key=key,
        mnemonic=mnemonic,
        manual_name=mnemonic,
        summary=f"{mnemonic} summary",
        control_flow=None,
        stack_delta=delta,
        stack_effect=StackEffect(inputs=inputs, outputs=outputs, delta=delta, source="test"),
        tags=(),
        comparison_operator=None,
        enum_values={},
        enum_namespace=None,
        struct_context=None,
        stack_inputs=inputs,
        stack_outputs=outputs,
        uses_operand=False,
        operand_hint=None,
        vm_method=mnemonic,
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


def _profile(
    *,
    start: int,
    instructions: int,
    entry: int,
    min_depth: int,
    max_depth: int,
    delta: int,
    warnings: Tuple[str, ...] = (),
) -> BlockStackProfile:
    """Convenience helper for constructing profiles."""

    return BlockStackProfile(
        block_start=start,
        instruction_count=instructions,
        entry_requirement=entry,
        min_depth=min_depth,
        max_depth=max_depth,
        net_delta=delta,
        warnings=warnings,
    )


def test_stack_profile_analyzer_computes_entry_requirements() -> None:
    push = _semantics("10:00", "push", inputs=0, outputs=1)
    consume = _semantics("11:00", "consume", inputs=2, outputs=0)
    block = IRBlock(
        start=0x10,
        instructions=[
            _instruction(0x10, push),
            _instruction(0x14, consume),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=1, blocks={block.start: block})

    analyzer = StackProfileAnalyzer(program)
    profiles = analyzer.analyse()
    assert len(profiles) == 1
    profile = profiles[0]
    assert isinstance(profile, BlockStackProfile)
    assert profile.entry_requirement == 1
    assert profile.max_depth >= 1
    text = render_stack_profile_reports([analyzer.report()])
    assert "segment 1 stack profile" in text


def test_stack_profile_summary_serialisation(tmp_path) -> None:
    pop = _semantics("12:00", "pop", inputs=1, outputs=0)
    dup = _semantics("13:00", "dup", inputs=1, outputs=2)
    block_a = IRBlock(start=0x20, instructions=[_instruction(0x20, dup)], successors=[])
    block_b = IRBlock(start=0x30, instructions=[_instruction(0x30, pop)], successors=[])
    program = IRProgram(segment_index=2, blocks={0x20: block_a, 0x30: block_b})

    profiles = analyse_stack_profiles(program)
    summary = StackProfileAnalyzer(program).summary(profiles)
    assert isinstance(summary, StackProfileSummary)
    summary_text = render_stack_summary(summary)
    assert "blocks analysed" in summary_text
    json_profiles = stack_profiles_to_json(profiles, indent=0)
    assert '"block_start": 32' in json_profiles
    csv_payload = stack_profiles_to_csv(profiles)
    assert "0x000020" in csv_payload
    report = build_stack_profile_report(program)
    assert isinstance(report, StackProfileReport)
    rendered = render_stack_profile_report(report)
    assert "blocks analysed" in rendered
    json_report = stack_profile_reports_to_json([report], indent=0)
    assert '"segment_index": 2' in json_report
    destination = tmp_path / "stack_report.txt"
    write_stack_profile_reports([report], destination)
    assert destination.exists()
def test_stack_profile_reports_multiple_segments(tmp_path) -> None:
    nop = _semantics("14:00", "nop", inputs=0, outputs=0)
    block = IRBlock(start=0, instructions=[_instruction(0, nop)], successors=[])
    program_a = IRProgram(segment_index=3, blocks={0: block})
    program_b = IRProgram(segment_index=4, blocks={0: block})

    reports = build_stack_profile_reports([program_a, program_b])
    assert [report.segment_index for report in reports] == [3, 4]
    rendered = render_stack_profile_reports(reports)
    assert rendered.count("stack profile") == 2
    destination = tmp_path / "profiles.txt"
    write_stack_profile_reports(reports, destination)
    assert destination.exists()


def test_filter_profiles_by_entry_requirement_threshold() -> None:
    profiles = [
        _profile(start=0x10, instructions=1, entry=0, min_depth=0, max_depth=0, delta=0),
        _profile(start=0x20, instructions=3, entry=2, min_depth=-2, max_depth=1, delta=-1),
        _profile(start=0x30, instructions=2, entry=1, min_depth=-1, max_depth=2, delta=1),
    ]

    filtered = filter_profiles_by_entry_requirement(profiles, minimum=1)
    assert [profile.block_start for profile in filtered] == [0x20, 0x30]

    filtered_strict = filter_profiles_by_entry_requirement(profiles, minimum=3)
    assert filtered_strict == []


def test_filter_profiles_by_delta_direction() -> None:
    profiles = [
        _profile(start=0x40, instructions=1, entry=0, min_depth=0, max_depth=1, delta=1),
        _profile(start=0x50, instructions=2, entry=0, min_depth=-1, max_depth=0, delta=-1),
        _profile(start=0x60, instructions=4, entry=0, min_depth=-1, max_depth=1, delta=0),
    ]

    positives = filter_profiles_by_delta(profiles, positive=True)
    assert [profile.block_start for profile in positives] == [0x40]

    negatives = filter_profiles_by_delta(profiles, positive=False)
    assert [profile.block_start for profile in negatives] == [0x50]

    zeros = filter_profiles_by_delta(profiles, positive=None)
    assert [profile.block_start for profile in zeros] == [0x60]
