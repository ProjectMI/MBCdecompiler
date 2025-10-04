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


def build_ascii_word(offset: int, text: str) -> InstructionWord:
    data = text.encode("ascii", "replace")[:4].ljust(4, b" ")
    return InstructionWord(offset=offset, raw=int.from_bytes(data, "big"))


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
        "literal_marker": {
            "opcodes": ["00:38"],
            "name": "literal_marker",
            "category": "literal_marker",
        },
        "reduce_pair": {
            "opcodes": ["04:00"],
            "name": "reduce_pair",
            "category": "reduce",
            "stack_delta": -1,
        },
        "tailcall_dispatch": {
            "opcodes": ["0x2B:0x00"],
            "name": "tailcall_dispatch",
            "category": "call_dispatch",
            "stack_delta": -1,
        },
        "return_values": {
            "opcodes": ["0x30:0x00"],
            "name": "return_values",
            "category": "return_values",
        },
        "call_dispatch": {
            "opcodes": ["0x28:0x00"],
            "name": "call_dispatch",
            "category": "call_dispatch",
            "stack_delta": -1,
        },
        "call_helpers": {
            "opcodes": ["0x10:0x00"],
            "name": "call_helpers",
            "category": "call_helper",
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
        "op_6C_01": {
            "opcodes": ["0x6C:0x01"],
            "name": "op_6C_01",
            "category": "meta",
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
        build_word(8, 0x2B, 0x00, 0x1234),
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


def build_template_container(tmp_path: Path) -> tuple[MbcContainer, KnowledgeBase]:
    knowledge = write_manual(tmp_path)

    segments: list[Segment] = []
    offset = 0

    def add_segment(words: list[InstructionWord]) -> None:
        nonlocal offset
        data = encode_instructions(words)
        descriptor = SegmentDescriptor(len(segments), offset, offset + len(data))
        segments.append(Segment(descriptor, data))
        offset += len(data)

    add_segment(
        [
            build_word(0, 0x04, 0x00, 0x0000),
            build_word(4, 0x00, 0x00, 0x0067),
            build_word(8, 0x00, 0x00, 0x0400),
            build_word(12, 0x00, 0x00, 0x6704),
            build_word(16, 0x00, 0x00, 0x0067),
            build_word(20, 0x00, 0x00, 0x0400),
            build_word(24, 0x00, 0x00, 0x6704),
            build_word(28, 0x00, 0x00, 0x6910),
            build_word(32, 0x30, 0x00, 0x0000),
        ]
    )

    add_segment(
        [
            build_word(0, 0x00, 0x00, 0x0001),
            build_word(4, 0x2B, 0x00, 0x0010),
            build_ascii_word(8, "COND"),
            build_word(12, 0x23, 0x00, 0x0008),
        ]
    )

    add_segment(
        [
            build_ascii_word(0, "HEAD"),
            build_ascii_word(4, "ER00"),
            build_word(8, 0x30, 0x00, 0x0000),
        ]
    )

    add_segment(
        [
            build_word(0, 0x28, 0x00, 0x0020),
            build_word(4, 0x30, 0x00, 0x0001),
        ]
    )

    add_segment(
        [
            build_word(0, 0x27, 0x00, 0x0004),
        ]
    )

    add_segment(
        [
            build_word(0, 0x28, 0x00, 0x0030),
            build_ascii_word(4, "TEXT"),
            build_word(8, 0x30, 0x00, 0x0000),
        ]
    )

    container = MbcContainer(Path("dummy"), segments)
    return container, knowledge


def test_normalizer_builds_ir(tmp_path: Path) -> None:
    container, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    assert program.metrics.calls == 1
    assert program.metrics.tail_calls == 0
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
    assert any(text.startswith("function_prologue") for text in descriptions)
    assert any(text.startswith("load") for text in descriptions)
    assert any(text.startswith("store") for text in descriptions)

    renderer = IRTextRenderer()
    text = renderer.render(program)
    assert "normalizer metrics" in text
    assert f"segment {segment.index}" in text


def test_normalizer_structural_templates(tmp_path: Path) -> None:
    container, knowledge = build_template_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    descriptions = [
        getattr(node, "describe", lambda: "")()
        for segment in program.segments
        for block in segment.blocks
        for node in block.nodes
    ]

    assert any("literal_block" in text and "via reduce_pair" in text for text in descriptions)
    assert any(text.startswith("if cond=ascii(") for text in descriptions)
    assert any(text.startswith("ascii_header[") for text in descriptions)
    assert any(text.startswith("function_prologue") for text in descriptions)
    assert any(text.startswith("call_return") for text in descriptions)
    assert any(text.startswith("ascii_wrapper_call target") for text in descriptions)


def test_normalizer_collapses_ascii_runs_and_literal_hints(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)
    words = [
        InstructionWord(0, int.from_bytes(b"A\x00B\x00", "big")),
        InstructionWord(4, int.from_bytes(b"\x00C\x00D", "big")),
        build_word(8, 0x00, 0x38, 0x6704),
        build_word(12, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    descriptions = [getattr(node, "describe", lambda: "")() for node in block.nodes]

    assert "ascii_header[ascii(A\\x00B\\x00), ascii(\\x00C\\x00D)]" in descriptions
    assert "lit(0x6704)" in descriptions


def test_normalizer_demotes_tailcall_condition(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)
    words = [
        build_word(0, 0x00, 0x00, 0x0001),
        build_word(4, 0x2B, 0x00, 0x0010),
        build_word(8, 0x23, 0x00, 0x000C),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    program = IRNormalizer(knowledge).normalise_container(container)
    descriptions = [
        getattr(node, "describe", lambda: "")()
        for block in program.segments[0].blocks
        for node in block.nodes
    ]

    assert any(text.startswith("call target=0x0010") for text in descriptions)
    assert not any(text.startswith("call tail target=0x0010") for text in descriptions)
    assert any(text.startswith("if cond=call target=0x0010") for text in descriptions)


def test_ascii_finalize_ignored_in_conditions(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)
    words = [
        build_word(0, 0x00, 0x00, 0x0001),
        build_ascii_word(4, "TEXT"),
        build_word(8, 0x10, 0x00, 0xF172),
        build_word(12, 0x23, 0x00, 0x0008),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    program = IRNormalizer(knowledge).normalise_container(container)
    descriptions = [
        getattr(node, "describe", lambda: "")()
        for block in program.segments[0].blocks
        for node in block.nodes
    ]

    assert any(text.startswith("ascii_finalize") for text in descriptions)
    assert any(text.startswith("if cond=ascii(") for text in descriptions)
    assert not any(text.startswith("if cond=ascii_finalize") for text in descriptions)


def test_function_prologue_with_helper_prefix(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)
    words = [
        build_word(0, 0x6C, 0x01, 0xFC03),
        build_word(4, 0x00, 0x00, 0x1400),
        build_word(8, 0x00, 0x00, 0x5E29),
        build_word(12, 0x10, 0x00, 0xF04B),
        build_word(16, 0x27, 0x00, 0x306C),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    program = IRNormalizer(knowledge).normalise_container(container)
    descriptions = [
        getattr(node, "describe", lambda: "")()
        for block in program.segments[0].blocks
        for node in block.nodes
    ]

    assert descriptions == [
        "function_prologue slot(0x306C)=stack_top then=0x306C else=0x0014"
    ]


def test_stack_teardown_nodes_are_canonical(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)
    words = [build_word(0, 0x01, 0x00, 0x0000)]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    program = IRNormalizer(knowledge).normalise_container(container)
    descriptions = [
        getattr(node, "describe", lambda: "")()
        for block in program.segments[0].blocks
        for node in block.nodes
    ]

    assert descriptions == ["stack_teardown count=1 operand=0x0000"]
