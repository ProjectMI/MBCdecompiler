from pathlib import Path

from mbcdisasm.adb import SegmentDescriptor
from mbcdisasm.cfg import ControlFlowGraphBuilder
from mbcdisasm.instruction import WORD_SIZE
from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.mbc import Segment


def _make_word(opcode: int, mode: int, operand: int) -> bytes:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return raw.to_bytes(WORD_SIZE, "big")


def test_cfg_branch_with_negative_relative_offset(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")
    knowledge._annotations["10:00"] = {"control_flow": "branch", "flow_target": "relative"}

    start = 0x200
    data = b"".join(
        [
            _make_word(0x01, 0x00, 0x0000),
            _make_word(0x10, 0x00, 0xFFF8),
            _make_word(0x02, 0x00, 0x0000),
        ]
    )
    descriptor = SegmentDescriptor(index=0, start=start, end=start + len(data))
    segment = Segment(descriptor, data, "code")

    cfg = ControlFlowGraphBuilder(knowledge).build(segment)

    assert start in cfg.blocks
    branch_block = cfg.blocks[start + WORD_SIZE]
    assert start in branch_block.successors


def test_cfg_branch_with_negative_offset_without_hint(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")
    knowledge._annotations["10:00"] = {"control_flow": "branch"}

    start = 0x200
    data = b"".join(
        [
            _make_word(0x01, 0x00, 0x0000),
            _make_word(0x10, 0x00, 0xFFF8),
            _make_word(0x02, 0x00, 0x0000),
        ]
    )
    descriptor = SegmentDescriptor(index=0, start=start, end=start + len(data))
    segment = Segment(descriptor, data, "code")

    cfg = ControlFlowGraphBuilder(knowledge).build(segment)

    assert start in cfg.blocks
    branch_block = cfg.blocks[start + WORD_SIZE]
    assert start in branch_block.successors


def test_cfg_with_positive_and_negative_relative_branches(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")
    knowledge._annotations["10:00"] = {"control_flow": "branch", "flow_target": "relative"}

    start = 0x200
    data = b"".join(
        [
            _make_word(0x01, 0x00, 0x0000),
            _make_word(0x10, 0x00, 0xFFF8),
            _make_word(0x01, 0x00, 0x0000),
            _make_word(0x10, 0x00, 0x0004),
            _make_word(0x01, 0x00, 0x0000),
            _make_word(0x02, 0x00, 0x0000),
        ]
    )
    descriptor = SegmentDescriptor(index=0, start=start, end=start + len(data))
    segment = Segment(descriptor, data, "code")

    cfg = ControlFlowGraphBuilder(knowledge).build(segment)

    assert start in cfg.blocks

    backward_branch_block = cfg.blocks[start + WORD_SIZE]
    assert start in backward_branch_block.successors
    assert backward_branch_block.start in cfg.blocks[start].predecessors

    forward_branch_block = cfg.blocks[start + 3 * WORD_SIZE]
    forward_target = start + 5 * WORD_SIZE
    assert forward_target in cfg.blocks
    assert forward_target in forward_branch_block.successors
    assert forward_branch_block.start in cfg.blocks[forward_target].predecessors
