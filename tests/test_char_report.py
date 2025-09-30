"""Tests for :mod:`mbcdisasm.analyzer.char_report`."""

from pathlib import Path

from mbcdisasm.analyzer.char_report import CharReportBuilder
from mbcdisasm.knowledge import KnowledgeBase


def test_char_report_produces_strings() -> None:
    knowledge = KnowledgeBase.load(Path("knowledge"))
    builder = CharReportBuilder(knowledge)
    report = builder.build(Path("mbc/_char.mbc"), Path("mbc/_char.adb"))

    assert report.segments
    assert report.average_ratio() >= 0.5
    total_strings = sum(len(summary.strings) for summary in report.segments)
    assert total_strings > 10
    assert any(name.startswith("InitChar") for name in report.function_names())
    rendered = report.render()
    assert "Report for" in rendered
    assert report.top_functions()
    assert report.segments_above_threshold(0.5)
    serialised = report.to_dict()
    assert "average_ratio" in serialised
    assert "segment_count" in report.statistics
