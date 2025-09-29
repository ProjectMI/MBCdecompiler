import json

from typing import Optional

from mbcdisasm.lua_ast import Assignment, CommentStatement, LiteralExpr, NameExpr
from mbcdisasm.lua_literals import LuaLiteral
from mbcdisasm.lua_optimizer import (
    LiteralRunAnalyzer,
    LiteralRunCompactor,
    LiteralRunRegistry,
    LiteralRunReporter,
    LiteralRunSummary,
    merge_registries,
    LuaStatementOptimizer,
)


def _literal(value: int, text: Optional[str] = None) -> LiteralExpr:
    literal = LuaLiteral("number", value, text or str(value))
    return LiteralExpr(literal)


def _string_literal(text: str) -> LiteralExpr:
    literal = LuaLiteral("string", text, f'"{text}"')
    return LiteralExpr(literal)


def test_literal_run_compactor_merges_sequences() -> None:
    assignments = [
        Assignment([NameExpr(f"literal_{idx}")], _literal(idx)) for idx in range(6)
    ]
    compactor = LiteralRunCompactor(min_run_length=3, comment_threshold=0)
    rewritten, summaries = compactor.compact(assignments)

    assert len(rewritten) == 1
    multi = rewritten[0]
    assert multi.__class__.__name__ == "MultiAssignment"
    assert len(multi.targets) == 6
    assert summaries[0].count == 6
    assert summaries[0].prefix == "literal"
    assert summaries[0].byte_size == 12


def test_literal_run_compactor_breaks_on_non_literal() -> None:
    assignments = [
        Assignment([NameExpr("literal_0")], _literal(0)),
        CommentStatement("barrier"),
        Assignment([NameExpr("literal_1")], _literal(1)),
    ]
    compactor = LiteralRunCompactor(min_run_length=2)
    rewritten, summaries = compactor.compact(assignments)

    assert len(rewritten) == 3
    assert not summaries


def test_optimizer_emits_comments_and_metadata() -> None:
    assignments = [
        Assignment([NameExpr(f"string_{idx}")], _string_literal("a"))
        for idx in range(5)
    ]
    optimizer = LuaStatementOptimizer()
    rewritten, summaries = optimizer.optimise(assignments)

    assert len(rewritten) == 2  # comment + multi assignment
    assert isinstance(rewritten[0], CommentStatement)
    comment_text = rewritten[0].text
    assert "literal run" in comment_text
    assert "sample:" in comment_text
    assert "bytes" in comment_text
    assert summaries and summaries[0].count == 5
    assert summaries[0].kind_breakdown == [("string", 5)]
    assert summaries[0].contiguous is True
    assert "preview" in summaries[0].to_metadata_line()
    extra = summaries[0].additional_metadata_lines()
    assert any("sample:" in line for line in extra)
    assert any("string preview:" in line for line in extra)
    assert any("estimated bytes:" in line for line in extra)


def test_optimizer_detects_index_gaps() -> None:
    assignments = [
        Assignment([NameExpr("literal_0")], _literal(0)),
        Assignment([NameExpr("literal_2")], _literal(2)),
        Assignment([NameExpr("literal_3")], _literal(3)),
    ]
    optimizer = LuaStatementOptimizer()
    rewritten, summaries = optimizer.optimise(assignments)

    assert summaries and summaries[0].contiguous is False
    assert "gaps present" in summaries[0].to_metadata_line()
    extra = summaries[0].additional_metadata_lines()
    assert any("non-contiguous" in line for line in extra)
    assert any("estimated bytes" in line for line in extra)


def test_literal_run_reporter_generates_text_and_json() -> None:
    registry = LiteralRunRegistry()
    summary = LiteralRunSummary(
        prefix="literal",
        first="literal_0",
        last="literal_2",
        count=3,
        kind_breakdown=[("number", 3)],
        contiguous=False,
        value_sample=["0", "1", "2"],
        numeric_range=(0, 10),
        string_preview=None,
        byte_size=6,
    )
    registry.register(summary)
    reporter = LiteralRunReporter(registry)
    text = reporter.as_text()
    assert "literal runs report" in text
    assert "literal: 1 runs; 6 bytes" in text
    payload = json.loads(reporter.as_json())
    assert payload["total_runs"] == 1
    assert payload["prefix_totals"][0]["prefix"] == "literal"
    assert payload["largest_runs"][0]["contiguous"] is False
    table = reporter.as_markdown_table()
    assert "| prefix | runs | bytes |" in table
    stats = registry.prefix_statistics("literal")
    assert stats.run_count == 1
    assert stats.byte_count == 6
    by_prefix = registry.runs_for_prefix("literal")
    assert len(by_prefix) == 1
    data = registry.to_dict()
    assert data["prefix_totals"][0]["prefix"] == "literal"


def test_literal_run_analyzer_reports_metrics() -> None:
    registry = LiteralRunRegistry()
    registry.register(
        LiteralRunSummary(
            prefix="literal",
            first="literal_0",
            last="literal_1",
            count=2,
            kind_breakdown=[("number", 2)],
            byte_size=4,
        )
    )
    registry.register(
        LiteralRunSummary(
            prefix="string",
            first="string_0",
            last="string_3",
            count=4,
            kind_breakdown=[("string", 4)],
            byte_size=16,
        )
    )
    analyzer = LiteralRunAnalyzer(registry)
    histogram = analyzer.run_length_histogram()
    assert histogram[2] == 1 and histogram[4] == 1
    stats = analyzer.byte_statistics()
    assert stats["min"] == 4.0 and stats["max"] == 16.0
    distribution = analyzer.kind_distribution()
    assert distribution["number"] == 2
    assert distribution["string"] == 4
    prefix_data = analyzer.prefix_summary("literal")
    assert prefix_data["count"] == 1.0
    percentiles = analyzer.byte_percentiles()
    assert percentiles["p25"] == 7.0
    report = analyzer.describe()
    assert "literal run analysis" in report
    assert "byte percentiles" in report


def test_merge_registries_combines_runs() -> None:
    first = LiteralRunRegistry()
    second = LiteralRunRegistry()
    first.register(
        LiteralRunSummary(
            prefix="literal",
            first="literal_0",
            last="literal_0",
            count=1,
            kind_breakdown=[("number", 1)],
            byte_size=2,
        )
    )
    second.register(
        LiteralRunSummary(
            prefix="string",
            first="string_0",
            last="string_1",
            count=2,
            kind_breakdown=[("string", 2)],
            byte_size=8,
        )
    )
    merged = merge_registries([first, second])
    assert merged.total_runs() == 2
    assert merged.total_bytes() == 10
