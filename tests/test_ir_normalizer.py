import json
from pathlib import Path

from mbcdisasm import IRNormalizer, KnowledgeBase, MbcContainer
from mbcdisasm.adb import SegmentDescriptor
from mbcdisasm.ir import IRTextRenderer
from mbcdisasm.ir.model import (
    IRCallPreparation,
    IRCallReturn,
    IRFunctionEpilogue,
    IRIf,
    IRReturn,
    IRTestSetBranch,
    IRRaw,
)
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
            "opcodes": ["0x10:0xE8"],
            "name": "call_helpers",
            "category": "call_helpers",
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
            "stack_delta": -1,
        },
        "stack_teardown_4": {
            "opcodes": ["0x01:0xF0"],
            "name": "stack_teardown_4",
            "category": "stack_teardown",
            "stack_delta": -4,
        },
        "stack_shuffle": {
            "opcodes": ["0x66:0x15"],
            "name": "stack_shuffle",
            "category": "stack_shuffle",
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

    if_nodes = [
        node
        for block in segment.blocks
        for node in block.nodes
        if isinstance(node, IRIf)
    ]
    assert if_nodes and all(node.condition.startswith("ssa") for node in if_nodes)

    testset_nodes = [
        node
        for block in segment.blocks
        for node in block.nodes
        if isinstance(node, IRTestSetBranch)
    ]
    assert testset_nodes and all(node.expr.startswith("ssa") for node in testset_nodes)


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
    assert any(text.startswith("tailcall_ascii") for text in descriptions)
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


def test_normalizer_groups_call_helper_cleanup(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x66, 0x15, 0x4B08),  # stack_shuffle
        build_word(4, 0x4A, 0x05, 0x0052),  # op_4A_05 helper
        build_word(8, 0x28, 0x00, 0x1234),  # call_dispatch
        build_word(12, 0x10, 0xE8, 0x0001),  # call_helpers
        build_word(16, 0x32, 0x29, 0x1000),  # op_32_29 helper
        build_word(20, 0x01, 0xF0, 0x0000),  # stack_teardown_4
        build_word(24, 0x30, 0x00, 0x0002),  # return_values
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    prep = next(node for node in block.nodes if isinstance(node, IRCallPreparation))
    assert prep.steps == (("stack_shuffle", 0x4B08), ("op_4A_05", 0x0052))

    call_return = next(node for node in block.nodes if isinstance(node, IRCallReturn))
    assert call_return.cleanup == (
        ("call_helpers", 0x0001),
        ("op_32_29", 0x1000),
        ("stack_teardown_4", 0x0000),
    )
    assert call_return.cleanup_popped == 4
    assert call_return.epilogue == tuple()
    assert call_return.epilogue_popped == 0
    assert call_return.target == 0x1234

    assert not any(
        isinstance(node, IRRaw)
        and node.mnemonic in {"op_4A_05", "op_32_29", "stack_teardown_4"}
        for node in block.nodes
    )


def test_normalizer_inlines_stack_shuffle_without_helper(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x66, 0x15, 0x4B08),
        build_word(4, 0x28, 0x00, 0x1111),
        build_word(8, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    prep = next(node for node in block.nodes if isinstance(node, IRCallPreparation))
    assert prep.steps == (("stack_shuffle", 0x4B08),)

    call_return = next(node for node in block.nodes if isinstance(node, IRCallReturn))
    assert call_return.cleanup == tuple()
    assert call_return.epilogue == tuple()

    assert not any(
        isinstance(node, IRRaw) and node.mnemonic == "stack_shuffle" for node in block.nodes
    )


def test_normalizer_attaches_teardown_epilogue_to_return(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x01, 0x00, 0x0000),
        build_word(4, 0x01, 0xF0, 0x0000),
        build_word(8, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    return_node = next(node for node in block.nodes if isinstance(node, IRReturn))
    assert return_node.epilogue == (
        ("stack_teardown_1", 0x0000),
        ("stack_teardown_4", 0x0000),
    )
    assert return_node.epilogue_popped == 5


def test_normalizer_emits_function_epilogue_for_unattached_teardown(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x01, 0xF0, 0x0000),
        build_word(4, 0x01, 0x00, 0x0000),
        build_word(8, 0x2B, 0x00, 0x2222),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    epilogue = next(node for node in block.nodes if isinstance(node, IRFunctionEpilogue))
    assert epilogue.steps == (
        ("stack_teardown_4", 0x0000),
        ("stack_teardown_1", 0x0000),
    )
    assert epilogue.popped == 5

    assert not any(
        isinstance(node, IRRaw)
        and node.mnemonic in {"stack_teardown_4", "stack_teardown_1"}
        for node in block.nodes
    )
