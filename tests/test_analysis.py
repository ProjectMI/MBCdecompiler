from pathlib import Path

from mbcdisasm.adb import SegmentDescriptor
from mbcdisasm.analysis import Analyzer
from mbcdisasm.disassembler import Disassembler
from mbc_disasm import serialize_analysis
from mbcdisasm.instruction import WORD_SIZE
from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.mbc import MbcContainer, Segment


def _make_word(opcode: int, mode: int, operand: int) -> bytes:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return raw.to_bytes(WORD_SIZE, "big")


def test_analyzer_categorizes_relative_operand_as_small(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")
    key = "10:00"
    knowledge._annotations[key] = {"flow_target": "relative"}

    start = 0x200
    data = _make_word(0x10, 0x00, 0xFFF8)
    descriptor = SegmentDescriptor(index=0, start=start, end=start + len(data))
    segment = Segment(descriptor, data, "code")
    container = MbcContainer(tmp_path / "dummy.mbc", [segment])

    analysis = Analyzer(knowledge).analyze(container)
    profile = analysis.opcode_profiles[key]

    assert profile.count == 1
    assert profile.operand_types["relative_word:small"] == 1


def test_analysis_builds_opcode_mode_matrix(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")

    data = _make_word(0x10, 0x00, 0x0001) + _make_word(0x10, 0x01, 0x0002)
    descriptor = SegmentDescriptor(index=0, start=0x100, end=0x100 + len(data))
    segment = Segment(descriptor, data, "code")
    container = MbcContainer(tmp_path / "matrix.mbc", [segment])

    analysis = Analyzer(knowledge).analyze(container)
    matrix = analysis.opcode_mode_matrix()

    assert matrix["10"] == ["00", "01"]


def test_serialize_analysis_uses_compact_opcode_profiles(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")

    data = _make_word(0x20, 0x03, 0x0001) + _make_word(0x21, 0x04, 0x0002)
    descriptor = SegmentDescriptor(index=0, start=0x200, end=0x200 + len(data))
    segment = Segment(descriptor, data, "code")
    container = MbcContainer(tmp_path / "profiles.mbc", [segment])

    analysis = Analyzer(knowledge).analyze(container)
    serialized = serialize_analysis(analysis)

    profiles = serialized["opcode_profiles"]
    assert profiles["modes"] == ["03", "04"]
    assert profiles["opcodes"] == [
        {"opcode": "20", "modes": ["03"]},
        {"opcode": "21", "modes": ["04"]},
    ]
    assert serialized["stack_observations"]
    quality = serialized["emulation_quality"]
    assert quality["total_instructions"] == 2
    assert quality["determined_instructions"] <= quality["total_instructions"]


def test_analysis_exposes_stack_observations(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")

    data = _make_word(0x30, 0x05, 0x0001)
    descriptor = SegmentDescriptor(index=0, start=0x100, end=0x100 + len(data))
    segment = Segment(descriptor, data, "code")
    container = MbcContainer(tmp_path / "stack.mbc", [segment])

    analysis = Analyzer(knowledge).analyze(container)
    observation = analysis.stack_observations.get("30:05")
    assert observation is not None
    assert observation.total_samples == 1
    assert observation.known_samples == 1
    assert observation.unknown_samples == 0
    unknown = analysis.unknown_stack_profiles()
    assert all(obs.key != "30:05" for obs in unknown)


def test_emulation_quality_tracks_uncertain_instructions(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")

    confident = _make_word(0x40, 0x11, 0x0000)
    uncertain = _make_word(0x41, 0x00, 0x0001)
    data = confident + uncertain
    descriptor = SegmentDescriptor(index=0, start=0x180, end=0x180 + len(data))
    segment = Segment(descriptor, data, "code")
    container = MbcContainer(tmp_path / "quality.mbc", [segment])

    analysis = Analyzer(knowledge).analyze(container)
    metrics = analysis.emulation_quality

    assert metrics.total_instructions == 2
    assert metrics.uncertain_instructions == 1
    assert metrics.unknown_ratio == metrics.uncertain_instructions / metrics.total_instructions
    assert metrics.segment_unknown_ratios[0] == metrics.unknown_ratio


def test_manual_segment_selection_overrides_classification(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")

    # A short blob that heuristics classify as non-code.
    descriptor = SegmentDescriptor(index=3, start=0x40, end=0x44)
    segment = Segment(descriptor, b"\x01\x01\x01\x01", "tables")
    container = MbcContainer(tmp_path / "selection.mbc", [segment])

    disasm = Disassembler(knowledge)

    auto_listing = disasm.generate_listing(container)
    assert "segment 3" not in auto_listing

    forced_listing = disasm.generate_listing(container, segment_indices=[3])
    assert "segment 3" in forced_listing


def test_disassembler_renders_operand_confidence_from_annotations(
    tmp_path: Path,
) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")
    key = "11:01"
    knowledge._annotations[key] = {
        "name": "custom_operand",
        "operand_hint": "table_index",
        "operand_confidence": 0.875,
    }

    data = _make_word(0x11, 0x01, 0x0001)
    descriptor = SegmentDescriptor(index=0, start=0x100, end=0x100 + len(data))
    segment = Segment(descriptor, data, "code")
    container = MbcContainer(tmp_path / "confidence.mbc", [segment])

    disasm = Disassembler(knowledge)
    listing = disasm.generate_listing(container)
    lines = [line for line in listing.splitlines() if line and not line.startswith("; ")]
    assert lines, "expected at least one instruction line"
    instruction_line = lines[0]

    assert "operand≈table_index (88%)" in instruction_line
