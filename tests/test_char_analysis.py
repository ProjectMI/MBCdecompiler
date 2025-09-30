from pathlib import Path

from mbcdisasm import KnowledgeBase, MbcContainer
from mbcdisasm.analyzer import PipelineAnalyzer
from mbcdisasm.instruction import read_instructions


def test_char_script_recognition_ratio():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    analyzer = PipelineAnalyzer(knowledge)
    container = MbcContainer.load(Path("mbc/_char.mbc"), Path("mbc/_char.adb"))

    total_blocks = 0
    recognised = 0
    unknown_instructions = 0

    for segment in container.segments():
        instructions, _ = read_instructions(segment.data, segment.start)
        report = analyzer.analyse_segment(instructions)
        for block in report.blocks:
            total_blocks += 1
            if block.category != "unknown":
                recognised += 1
            unknown_instructions += sum(1 for profile in block.profiles if profile.kind.name == "UNKNOWN")

    assert total_blocks > 0
    ratio = recognised / total_blocks
    assert ratio == 1.0
    assert unknown_instructions == 0


def test_call_helper_wildcard_resolution():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    info = knowledge.lookup("10:84")
    assert info is not None
    assert (info.control_flow or "").lower().startswith("call")
