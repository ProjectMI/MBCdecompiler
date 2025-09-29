import pytest

from mbcdisasm.literal_analysis import (
    LiteralAnalyzer,
    LiteralCategory,
    LiteralReportBuilder,
    compute_statistics,
    describe_literal,
    diagnostics_table,
    diagnostics_to_json,
    explain_literal,
    iter_literal_fragments,
    render_literal_sequence,
    statistics_to_table,
    summarise_literals,
)
from mbcdisasm.manual_semantics import InstructionSemantics, StackEffect


def _semantics(
    *,
    summary: str = "",
    mnemonic: str = "literal_push",
    tags: tuple[str, ...] = ("literal",),
    enum_values: dict[int, str] | None = None,
    enum_namespace: str | None = None,
    operand_hint: str | None = None,
) -> InstructionSemantics:
    return InstructionSemantics(
        key="00:00",
        mnemonic=mnemonic,
        manual_name=mnemonic,
        summary=summary,
        control_flow=None,
        stack_delta=1,
        stack_effect=StackEffect(inputs=0, outputs=1, delta=1, source="test"),
        tags=tuple(tag.lower() for tag in tags),
        comparison_operator=None,
        enum_values=enum_values or {},
        enum_namespace=enum_namespace,
        struct_context=None,
        stack_inputs=0,
        stack_outputs=1,
        uses_operand=True,
        operand_hint=operand_hint,
        vm_method="literal_push",
        vm_call_style="method",
    )


def test_analyzer_formats_ascii_pair() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="Pushes an ASCII string literal")
    literal = analyzer.analyse(0x6C6F, semantics)
    assert literal.category is LiteralCategory.STRING
    assert literal.render() == '"ol"'


def test_analyzer_handles_trailing_zero_byte() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="Single character literal")
    literal = analyzer.analyse(0x2C00, semantics)
    assert literal.category is LiteralCategory.STRING
    assert literal.render() == '","'


def test_analyzer_formats_control_characters() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="Pushes newline text")
    literal = analyzer.analyse(0x000A, semantics)
    assert literal.category is LiteralCategory.CONTROL
    assert literal.render() == '"\\n"'


def test_analyzer_falls_back_to_numeric() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="Pushes numeric constant", operand_hint="large")
    literal = analyzer.analyse(0xABCD, semantics)
    assert literal.category is LiteralCategory.HEX
    assert literal.render() == "0xABCD"


def test_analyzer_honours_enums() -> None:
    semantics = _semantics(
        summary="Push enum",
        enum_values={1: "IDLE"},
        enum_namespace="State",
    )
    text, category, reason = explain_literal(1, semantics)
    assert text == "State.IDLE"
    assert category is LiteralCategory.ENUM
    assert reason == "enum-annotation"


def test_numeric_defaults_to_decimal_for_small_values() -> None:
    analyzer = LiteralAnalyzer()
    literal = analyzer.analyse(7, _semantics(summary="Small number", operand_hint="tiny"))
    assert literal.category is LiteralCategory.DECIMAL
    assert literal.render() == "7"


def test_boolean_detection_respects_hints() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="Boolean toggle", operand_hint="boolean")
    literal = analyzer.analyse(1, semantics)
    assert literal.category is LiteralCategory.BOOLEAN
    assert literal.render() == "true"
    literal_false = analyzer.analyse(0, semantics)
    assert literal_false.render() == "false"


def test_mask_detection_formats_single_and_multi_bit() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="Bitmask flags", operand_hint="mask")
    single = analyzer.analyse(0x0004, semantics)
    assert single.category is LiteralCategory.MASK
    assert single.render() == "(1 << 2)"
    combined = analyzer.analyse(0x0003, semantics)
    assert combined.category is LiteralCategory.MASK
    assert combined.render().startswith("0b")


def test_analyzer_diagnostics_report_decisions() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="Pushes an ASCII string literal")
    diagnostic = analyzer.analyse_with_diagnostics(0x6C6F, semantics)
    assert "string" in diagnostic.decisions
    assert diagnostic.value.render() == '"ol"'


def test_render_literal_sequence_helper() -> None:
    values = render_literal_sequence([0x6C6F, 7])
    assert values[0] == '"ol"'
    assert values[1] == "7"


def test_describe_literal_includes_category() -> None:
    analyzer = LiteralAnalyzer()
    literal = analyzer.analyse(0x6C6F, _semantics(summary="ASCII"))
    description = describe_literal(literal)
    assert "string" in description.lower()
    assert "ascii-bytes" in description


def test_literal_summary_counts_categories() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="ASCII")
    values = [
        analyzer.analyse(0x6C6F, semantics),
        analyzer.analyse(0x0000, _semantics(summary="zero", operand_hint="tiny")),
        analyzer.analyse(1, _semantics(summary="Boolean", operand_hint="boolean")),
    ]
    summary = summarise_literals(values)
    assert summary["categories"]["STRING"] == 1
    assert summary["categories"]["BOOLEAN"] == 1


def test_compute_statistics_tracks_totals() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="ASCII")
    values = [
        analyzer.analyse(0x6C6F, semantics),
        analyzer.analyse(0x0000, _semantics(summary="zero", operand_hint="tiny")),
    ]
    stats = compute_statistics(values)
    assert stats.total == 2
    assert stats.most_common_category() == "STRING"


def test_statistics_to_table_contains_counts() -> None:
    analyzer = LiteralAnalyzer()
    stats = compute_statistics([analyzer.analyse(0x6C6F, _semantics(summary="ASCII"))])
    table = statistics_to_table(stats)
    assert any("STRING" in line for line in table)


def test_diagnostics_to_json_serialises_payload() -> None:
    analyzer = LiteralAnalyzer()
    diagnostic = analyzer.analyse_with_diagnostics(0x6C6F, _semantics(summary="ASCII"))
    payload = diagnostics_to_json([diagnostic])
    assert "ascii-bytes" in payload


def test_report_builder_collects_operands_and_values() -> None:
    analyzer = LiteralAnalyzer()
    builder = LiteralReportBuilder(analyzer)
    literal = builder.add_operand(0x6C6F, _semantics(summary="ASCII"))
    builder.add_value(literal)
    stats = builder.statistics()
    assert stats.total == 2
    assert stats.by_category["STRING"] == 2


def test_report_builder_summary_lists_categories() -> None:
    analyzer = LiteralAnalyzer()
    builder = LiteralReportBuilder(analyzer)
    builder.add_operand(0x6C6F, _semantics(summary="ASCII"))
    builder.add_operand(1, _semantics(summary="Boolean", operand_hint="boolean"))
    summary = builder.summary_lines()
    assert any("string" in line for line in summary)
    assert any("boolean" in line for line in summary)


def test_report_builder_to_json_serialises_components() -> None:
    analyzer = LiteralAnalyzer()
    builder = LiteralReportBuilder(analyzer)
    builder.add_operand(0x6C6F, _semantics(summary="ASCII"))
    payload = builder.to_json()
    assert "values" in payload
    assert "diagnostics" in payload
    assert "statistics" in payload


def test_literal_fragments_iterator_unpacks_strings() -> None:
    analyzer = LiteralAnalyzer()
    literal = analyzer.analyse(0x6C6F, _semantics(summary="ASCII"))
    fragments = list(iter_literal_fragments([literal]))
    assert fragments == ["o", "l"]


def test_diagnostics_table_renders_rows() -> None:
    analyzer = LiteralAnalyzer()
    diagnostic = analyzer.analyse_with_diagnostics(0x6C6F, _semantics(summary="ASCII"))
    table = diagnostics_table([diagnostic])
    assert any("category" in line for line in table)
    assert any("string" in line.lower() for line in table)


def test_analyse_program_operands_validates_lengths() -> None:
    analyzer = LiteralAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyse_program_operands([1, 2], [None])


def test_clear_cache_resets_cached_objects() -> None:
    analyzer = LiteralAnalyzer()
    semantics = _semantics(summary="ASCII")
    first = analyzer.analyse(0x6C6F, semantics)
    second = analyzer.analyse(0x6C6F, semantics)
    assert first is second  # cached object reused
    analyzer.clear_cache()
    third = analyzer.analyse(0x6C6F, semantics)
    assert third is not first
