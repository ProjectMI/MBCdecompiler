import json
from pathlib import Path

from mbcdisasm import IRNormalizer, KnowledgeBase, MbcContainer
from mbcdisasm.adb import SegmentDescriptor
from mbcdisasm.ir import IRTextRenderer
from mbcdisasm.mbc import Segment
from mbcdisasm.instruction import InstructionWord


def build_word(offset: int, opcode: int, mode: int, operand: int) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset=offset, raw=raw)


def encode_instructions(words: list[InstructionWord]) -> bytes:
    return b"".join(word.raw.to_bytes(4, "big") for word in words)


def write_manual(path: Path) -> KnowledgeBase:
    manual = {
        "push_literal": {
            "opcodes": ["00:00"],
            "name": "push_literal",
            "category": "literal",
            "stack_push": 1,
        },
        "reduce_pair": {
            "opcodes": ["04:00"],
            "name": "reduce_pair",
            "category": "reduce",
            "stack_delta": -1,
        },
        "tailcall_dispatch": {
            "opcodes": ["0x29:0x00"],
            "name": "tailcall_dispatch",
            "category": "tailcall_dispatch",
        },
        "return_values": {
            "opcodes": ["0x30:0x00"],
            "name": "return_values",
            "category": "return_values",
        },
        "branch_eq": {
            "opcodes": ["0x23:0x00"],
            "name": "branch_eq",
            "category": "branch_eq",
        },
        "testset_branch": {
            "opcodes": ["0x27:0x00"],
            "name": "testset_branch",
            "category": "testset_branch",
        },
        "indirect_access": {
            "opcodes": ["0x69:0x01"],
            "name": "indirect_access",
            "category": "indirect_access",
        },
        "stack_teardown_1": {
            "opcodes": ["0x01:0x00"],
            "name": "stack_teardown_1",
            "category": "stack_teardown",
        },
    }
    manual_path = path / "manual_annotations.json"
    manual_path.write_text(json.dumps(manual, indent=2), "utf-8")
    return KnowledgeBase.load(manual_path)


def build_container(tmp_path: Path) -> tuple[MbcContainer, KnowledgeBase]:
    knowledge = write_manual(tmp_path)

    seg0_words = [
        build_word(0, 0x00, 0x00, 0x0001),
        build_word(4, 0x00, 0x00, 0x0002),
        build_word(8, 0x29, 0x00, 0x1234),
        build_word(12, 0x30, 0x00, 0x0002),
    ]
    seg1_words = [
        build_word(0, 0x00, 0x00, 0x0003),
        build_word(4, 0x00, 0x00, 0x0004),
        build_word(8, 0x04, 0x00, 0x0000),
        build_word(12, 0x23, 0x00, 0x0010),
        build_word(16, 0x00, 0x00, 0x0005),
        build_word(20, 0x27, 0x00, 0x0008),
        build_word(24, 0x69, 0x01, 0x0005),
        build_word(28, 0x69, 0x01, 0x9000),
        build_word(32, 0x01, 0x00, 0x0000),
    ]

    seg0_bytes = encode_instructions(seg0_words)
    seg1_bytes = encode_instructions(seg1_words)

    segments = [
        Segment(SegmentDescriptor(0, 0, len(seg0_bytes)), seg0_bytes),
        Segment(
            SegmentDescriptor(1, len(seg0_bytes), len(seg0_bytes) + len(seg1_bytes)),
            seg1_bytes,
        ),
    ]

    container = MbcContainer(Path("dummy"), segments)
    return container, knowledge


def test_normalizer_builds_ir(tmp_path: Path) -> None:
    container, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    assert program.metrics.calls == 1
    assert program.metrics.tail_calls == 1
    assert program.metrics.returns >= 1
    assert program.metrics.literals == 5
    assert program.metrics.literal_chunks == 0
    assert program.metrics.aggregates == 1
    assert program.metrics.testset_branches == 1
    assert program.metrics.if_branches == 1
    assert program.metrics.loads == 1
    assert program.metrics.stores == 1
    assert program.metrics.reduce_replaced == 1

    segment = program.segments[1]
    descriptions = [
        getattr(node, "describe", lambda: "")()
        for block in segment.blocks
        for node in block.nodes
    ]
    assert any("map" in text for text in descriptions)
    assert any(text.startswith("if cond") for text in descriptions)
    assert any(text.startswith("testset") for text in descriptions)
    assert any(text.startswith("load") for text in descriptions)
    assert any(text.startswith("store") for text in descriptions)

    renderer = IRTextRenderer()
    text = renderer.render(program)
    assert "normalizer metrics" in text
    assert f"segment {segment.index}" in text
