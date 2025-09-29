from mbcdisasm.ir import IRInstruction
from mbcdisasm.literal_runs import (
    LiteralRunEntry,
    LiteralRunNote,
    LiteralRunPattern,
    accumulate_statistics,
    group_literal_runs_by_kind,
    group_literal_runs_by_pattern,
    literal_run_histogram,
    literal_run_note_lines,
    literal_run_pattern_histogram,
    longest_literal_runs,
    render_literal_runs_table,
    serialize_literal_run,
    serialize_literal_runs,
    summarize_literal_run,
)
from mbcdisasm.lua_ast import LiteralExpr, NameExpr
from mbcdisasm.lua_literals import LuaLiteral
from mbcdisasm.manual_semantics import InstructionSemantics, StackEffect


def _dummy_semantics() -> InstructionSemantics:
    return InstructionSemantics(
        key="test",
        mnemonic="literal",
        manual_name="literal",
        summary="",
        control_flow=None,
        stack_delta=1.0,
        stack_effect=StackEffect(inputs=0, outputs=1, delta=1.0, source="test"),
        tags=("literal",),
        comparison_operator=None,
        enum_values={},
        enum_namespace=None,
        struct_context=None,
        stack_inputs=0,
        stack_outputs=1,
        uses_operand=False,
        operand_hint=None,
        vm_method="literal",
        vm_call_style="literal",
    )


def _entry(offset: int, expression) -> LiteralRunEntry:
    semantics = _dummy_semantics()
    instruction = IRInstruction(
        offset=offset,
        key=f"{offset:04X}",
        mnemonic="literal",
        operand=0,
        stack_delta=1.0,
        control_flow=None,
        semantics=semantics,
        stack_inputs=0,
        stack_outputs=1,
    )
    return LiteralRunEntry(
        instruction=instruction,
        semantics=semantics,
        expression=expression,
        prefix=None,
    )


def test_numeric_literal_run_summary() -> None:
    entries = [
        _entry(0x0000, LiteralExpr(LuaLiteral("number", 4, "4"))),
        _entry(0x0004, LiteralExpr(LuaLiteral("number", 10, "10"))),
        _entry(0x0008, LiteralExpr(LuaLiteral("number", 7, "7"))),
    ]
    summary = summarize_literal_run(entries)

    assert summary.kind == "number"
    assert summary.numeric_min == 4
    assert summary.numeric_max == 10
    assert summary.numeric_span == 6
    assert "range" in summary.metadata_line()
    detail = summary.detail_lines()
    assert any("range: 4..10" in line for line in detail)
    assert any("note" in line for line in summary.detail_lines()) or summary.notes


def test_string_literal_run_summary() -> None:
    entries = [
        _entry(0x1000, LiteralExpr(LuaLiteral("string", "A", "\"A\""))),
        _entry(0x1004, LiteralExpr(LuaLiteral("string", "B", "\"B\""))),
    ]
    summary = summarize_literal_run(entries)

    assert summary.kind == "string"
    assert summary.string_length == 2
    detail = summary.detail_lines()
    assert any("chunks" in line for line in detail)
    assert any("unique chars" in line for line in detail)
    assert any("Printable ASCII" in line for line in detail)
    assert any(pattern.render_tag() == "string:printable" for pattern in summary.patterns)


def test_enum_literal_run_summary() -> None:
    entries = [
        _entry(0x2000, NameExpr("State.IDLE")),
        _entry(0x2004, NameExpr("State.RUN")),
        _entry(0x2008, NameExpr("State.IDLE")),
    ]
    summary = summarize_literal_run(entries)

    assert summary.kind == "enum"
    assert len(summary.enum_values) == 2
    assert any("values" in line for line in summary.detail_lines())
    assert any("namespace" in pattern.render_tag() for pattern in summary.patterns)
    assert any("enum" in note.render() for note in summary.notes)


def test_string_pattern_annotations() -> None:
    entries = [
        _entry(0x3000, LiteralExpr(LuaLiteral("string", "line1\n", "\"line1\\n\""))),
        _entry(0x3004, LiteralExpr(LuaLiteral("string", "line2", "\"line2\""))),
    ]
    summary = summarize_literal_run(entries)

    rendered_notes = [note.render() for note in summary.notes]
    assert any("multiline" in note for note in rendered_notes)


def test_numeric_pattern_detection_helpers() -> None:
    entries = [
        _entry(0x4000, LiteralExpr(LuaLiteral("number", 1, "1"))),
        _entry(0x4004, LiteralExpr(LuaLiteral("number", 2, "2"))),
        _entry(0x4008, LiteralExpr(LuaLiteral("number", 3, "3"))),
        _entry(0x400C, LiteralExpr(LuaLiteral("number", 4, "4"))),
    ]
    summary = summarize_literal_run(entries)

    tags = [pattern.render_tag() for pattern in summary.patterns]
    assert any(tag.startswith("number:progression") for tag in tags)
    assert any(tag.startswith("number:contiguous") for tag in tags)
    assert any("bitmask" not in tag for tag in tags)
    assert any("values fit within byte range" in note.render() for note in summary.notes)


def test_mixed_pattern_detection() -> None:
    entries = [
        _entry(0x5000, LiteralExpr(LuaLiteral("number", 9, "9"))),
        _entry(0x5004, NameExpr("Enum.Flag")),
        _entry(0x5008, LiteralExpr(LuaLiteral("string", "A", "\"A\""))),
    ]
    summary = summarize_literal_run(entries)

    assert summary.kind == "mixed"
    assert any(pattern.kind == "mixed" for pattern in summary.patterns)
    assert any("distribution" in note.render() for note in summary.notes)


def test_literal_run_statistics_accumulate() -> None:
    numeric_entries = [
        _entry(0x0000, LiteralExpr(LuaLiteral("number", 1, "1"))),
        _entry(0x0004, LiteralExpr(LuaLiteral("number", 5, "5"))),
    ]
    string_entries = [
        _entry(0x1000, LiteralExpr(LuaLiteral("string", "X", "\"X\""))),
        _entry(0x1004, LiteralExpr(LuaLiteral("string", "Y", "\"Y\""))),
        _entry(0x1008, LiteralExpr(LuaLiteral("string", "Z", "\"Z\""))),
    ]
    stats = accumulate_statistics(
        [
            summarize_literal_run(numeric_entries),
            summarize_literal_run(string_entries),
        ]
    )

    lines = stats.summary_lines()
    assert any("literal runs: 2" in line for line in lines)
    assert any("max numeric span" in line for line in lines)
    assert any("longest string payload" in line for line in lines)
    assert any("dominant patterns" in line for line in lines)
    assert any("frequent notes" in line for line in lines)


def test_literal_run_histogram_lines() -> None:
    runs = [
        summarize_literal_run([_entry(0, LiteralExpr(LuaLiteral("number", 1, "1")))]),
        summarize_literal_run([_entry(4, LiteralExpr(LuaLiteral("number", 2, "2")))]),
        summarize_literal_run([_entry(8, NameExpr("Enum.A")), _entry(12, NameExpr("Enum.B"))]),
    ]
    lines = literal_run_histogram(runs)

    assert any("runs single" in line for line in lines)
    assert any("runs small" in line for line in lines)


def test_literal_run_pattern_histogram_lines() -> None:
    runs = [
        summarize_literal_run([_entry(0, LiteralExpr(LuaLiteral("number", 1, "1")))]),
        summarize_literal_run([_entry(4, LiteralExpr(LuaLiteral("number", 2, "2")))]),
        summarize_literal_run(
            [
                _entry(8, LiteralExpr(LuaLiteral("string", "A", "\"A\""))),
                _entry(12, LiteralExpr(LuaLiteral("string", "B", "\"B\""))),
            ]
        ),
    ]
    lines = literal_run_pattern_histogram(runs, limit=4)

    assert any("pattern" in line for line in lines)


def test_literal_run_note_lines_summary() -> None:
    runs = [
        summarize_literal_run(
            [
                _entry(0, LiteralExpr(LuaLiteral("number", value, str(value)))),
                _entry(4, LiteralExpr(LuaLiteral("number", value + 1, str(value + 1)))),
            ]
        )
        for value in (1, 3, 5)
    ]
    lines = literal_run_note_lines(runs, limit=2)

    assert any("note" in line for line in lines)


def test_serialize_literal_run() -> None:
    summary = summarize_literal_run(
        [_entry(0, LiteralExpr(LuaLiteral("number", 9, "9")))]
    )
    data = serialize_literal_run(summary)
    assert data["kind"] == "number"
    assert data["count"] == 1
    assert "patterns" in data and isinstance(data["patterns"], list)
    assert "notes" in data and isinstance(data["notes"], list)
    all_data = serialize_literal_runs([summary])
    assert isinstance(all_data, list) and all_data[0]["kind"] == "number"


def test_group_and_longest_literal_runs() -> None:
    runs = [
        summarize_literal_run([_entry(0x0000, LiteralExpr(LuaLiteral("number", 1, "1")))]),
        summarize_literal_run([_entry(0x0004, NameExpr("Enum.Value"))]),
        summarize_literal_run(
            [
                _entry(0x0008, LiteralExpr(LuaLiteral("string", "A", "\"A\""))),
                _entry(0x000C, LiteralExpr(LuaLiteral("string", "B", "\"B\""))),
            ]
        ),
    ]
    grouped = group_literal_runs_by_kind(runs)
    assert set(grouped) == {"number", "enum", "string"}
    longest = longest_literal_runs(runs, limit=2)
    assert longest and longest[0].count >= longest[-1].count
    table_lines = render_literal_runs_table(longest, limit=2)
    assert table_lines and table_lines[0].startswith("start")
    grouped_patterns = group_literal_runs_by_pattern(runs)
    assert any(tag.startswith("number:") for tag in grouped_patterns)
