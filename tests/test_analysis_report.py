from mbcdisasm.analysis_report import (
    build_analysis,
    build_function_analysis,
    build_module_summary,
)
from mbcdisasm.highlevel import FunctionMetadata, HighLevelFunction, StackEvent
from mbcdisasm.lua_ast import ReturnStatement, wrap_block


def _make_function(name: str, metadata: FunctionMetadata) -> HighLevelFunction:
    return HighLevelFunction(name=name, body=wrap_block([ReturnStatement()]), metadata=metadata)


def test_build_function_analysis_includes_stack_depth_metrics() -> None:
    events = (
        StackEvent(
            action="push",
            value="tmp_0",
            origin=0x10,
            comment=None,
            depth_before=0,
            depth_after=1,
        ),
        StackEvent(
            action="pop",
            value="tmp_0",
            origin=0x14,
            comment="placeholder",
            depth_before=1,
            depth_after=-1,
        ),
    )
    metadata = FunctionMetadata(
        block_count=1,
        instruction_count=2,
        helper_calls=0,
        branch_count=0,
        literal_count=0,
        stack_events=events,
        stack_depth_min=-1,
        stack_depth_max=1,
        stack_depth_final=-1,
        stack_underflows=1,
    )
    function = _make_function("segment_a", metadata)

    analysis = build_function_analysis(function)

    assert analysis.stack_depth_min == -1
    assert analysis.stack_depth_max == 1
    assert analysis.stack_depth_final == -1
    assert analysis.stack_underflows == 1

    payload = analysis.to_dict()
    assert payload["stack_depth"] == {
        "min": -1,
        "max": 1,
        "final": -1,
        "underflows": 1,
    }

    rendered = analysis.render_text()
    assert "stack: min=-1 max=1 final=-1 underflows=1" in rendered


def test_build_module_summary_aggregates_stack_depth() -> None:
    meta_a = FunctionMetadata(
        block_count=1,
        instruction_count=2,
        stack_events=(
            StackEvent(
                action="push",
                value="tmp",
                origin=None,
                comment=None,
                depth_before=0,
                depth_after=1,
            ),
        ),
        stack_depth_min=0,
        stack_depth_max=1,
        stack_depth_final=0,
        stack_underflows=0,
    )
    meta_b = FunctionMetadata(
        block_count=1,
        instruction_count=3,
        stack_events=(
            StackEvent(
                action="pop",
                value="stack",
                origin=None,
                comment="underflow",
                depth_before=0,
                depth_after=-1,
            ),
        ),
        stack_depth_min=-1,
        stack_depth_max=4,
        stack_depth_final=-1,
        stack_underflows=2,
    )
    functions = [_make_function("segment_b", meta_a), _make_function("segment_c", meta_b)]
    analysis = build_analysis(functions)
    summary = build_module_summary(analysis)

    assert summary.max_stack_depth == 4
    assert summary.min_stack_depth == -1
    assert summary.nonzero_final_depths == 1
    assert summary.underflow_functions == 1

    text = "\n".join(summary.render_text())
    assert "stack depth: max=4 min=-1 unfinished=1 underflows=1" in text

    markdown = "\n".join(summary.render_markdown())
    assert "* Stack max depth: 4" in markdown
    assert "* Functions with underflow: 1" in markdown
