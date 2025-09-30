"""Integration tests focused on the ``_char`` container."""

from pathlib import Path

from mbcdisasm.analyzer.pipeline import PipelineAnalyzer
from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.mbc import MbcContainer
from mbcdisasm.instruction import read_instructions


def recognised_counts(report) -> tuple[int, int]:
    recognised = 0
    total = 0
    for block in report.blocks:
        size = len(block.profiles)
        total += size
        if block.category != "unknown":
            recognised += size
    return recognised, total


def test_char_segments_are_half_recognised() -> None:
    knowledge = KnowledgeBase.load(Path("knowledge"))
    analyzer = PipelineAnalyzer(knowledge)
    container = MbcContainer.load(Path("mbc/_char.mbc"), Path("mbc/_char.adb"))

    recognised_total = 0
    total_instructions = 0
    for segment in container.iter_segments():
        instructions, _ = read_instructions(segment.data, segment.descriptor.start)
        report = analyzer.analyse_segment(instructions)
        recognised, total = recognised_counts(report)
        recognised_total += recognised
        total_instructions += total

    assert total_instructions > 0
    assert recognised_total / total_instructions >= 0.5
