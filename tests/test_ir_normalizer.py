import json
from pathlib import Path

from mbcdisasm import IRNormalizer, KnowledgeBase, MbcContainer
from mbcdisasm.adb import SegmentDescriptor
from mbcdisasm.analyzer.instruction_profile import (
    InstructionKind,
    InstructionProfile,
    StackEffectHint,
)
from mbcdisasm.analyzer.stack import StackEvent
from mbcdisasm.ir import (
    IRAsciiFinalize,
    IRAsciiHeader,
    IRAsciiPreamble,
    IRAsciiWrapperCall,
    IRCall,
    IRCallPreparation,
    IRFunctionPrologue,
    IRLiteralBlock,
    IRLiteralChunk,
    IRTestSetBranch,
    IRTailcallAscii,
    IRTextRenderer,
)
from mbcdisasm.ir.normalizer import _ItemList, RawInstruction
from mbcdisasm.mbc import Segment
from mbcdisasm.instruction import InstructionWord


def build_word(offset: int, opcode: int, mode: int, operand: int) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset=offset, raw=raw)


def encode_instructions(words: list[InstructionWord]) -> bytes:
    return b"".join(word.raw.to_bytes(4, "big") for word in words)


def make_reducer_instruction(mnemonic: str) -> RawInstruction:
    word = InstructionWord(offset=0, raw=0)
    profile = InstructionProfile(
        word=word,
        info=None,
        mnemonic=mnemonic,
        summary=None,
        category=None,
        control_flow=None,
        stack_hint=StackEffectHint(0, 0, 0),
        kind=InstructionKind.REDUCE,
    )
    event = StackEvent(
        profile=profile,
        delta=-1,
        minimum=-1,
        maximum=0,
        confidence=1.0,
        depth_before=2,
        depth_after=1,
        kind=InstructionKind.REDUCE,
    )
    return RawInstruction(profile=profile, event=event, annotations=tuple())


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


def test_literal_block_helper_records_kind(tmp_path: Path) -> None:
    _, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    reducer = make_reducer_instruction("reduce_pair")
    values = (0x1000, 0x2000, 0x3000)

    block = normalizer._build_literal_block(values, [reducer])

    assert isinstance(block, IRLiteralBlock)
    assert block.kind == "reduce_pair"
    assert block.group_size == 2
    assert block.reducers == ("reduce_pair",)
    assert block.values == values


def test_ascii_header_groups_chunks(tmp_path: Path) -> None:
    _, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    chunk_a = IRLiteralChunk(data=b"TEST", source="inline", annotations=tuple())
    chunk_b = IRLiteralChunk(data=b"DATA", source="inline", annotations=tuple())
    items = _ItemList([chunk_a, chunk_b])

    normalizer._pass_ascii_headers(items)

    assert len(items.to_tuple()) == 1
    header = items[0]
    assert isinstance(header, IRAsciiHeader)
    assert header.chunk_count == 2
    assert header.summary.startswith("ascii(")


def test_ascii_wrapper_collapse(tmp_path: Path) -> None:
    _, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    preamble = IRAsciiPreamble(loader_operand=0x1111, mode_operand=0x2222, shuffle_operand=0x4B08)
    chunk = IRLiteralChunk(data=b"ABCD", source="inline", annotations=tuple())
    prep = IRCallPreparation(steps=(("stack_shuffle", 0x4B08),))
    call = IRCall(target=0x4010, args=("arg",), tail=False)
    finalize = IRAsciiFinalize(helper=0xF172, summary="ascii(ABCD)")
    items = _ItemList([preamble, chunk, prep, call, finalize])

    normalizer._pass_ascii_wrappers(items)

    assert len(items.to_tuple()) == 1
    wrapper = items[0]
    assert isinstance(wrapper, IRAsciiWrapperCall)
    assert wrapper.chunk_count == 1
    assert wrapper.preamble == (0x1111, 0x2222, 0x4B08)
    assert wrapper.summary.startswith("ascii(")


def test_tailcall_ascii_wrapper(tmp_path: Path) -> None:
    _, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    preamble = IRAsciiPreamble(loader_operand=0x0101, mode_operand=0x0202, shuffle_operand=0x4B08)
    chunk = IRLiteralChunk(data=b"TAIL", source="inline", annotations=tuple())
    call = IRCall(target=0x7777, args=tuple(), tail=True)
    finalize = IRAsciiFinalize(helper=0x3D30, summary="ascii(TAIL)")
    items = _ItemList([preamble, chunk, call, finalize])

    normalizer._pass_ascii_wrappers(items)

    assert len(items.to_tuple()) == 1
    wrapper = items[0]
    assert isinstance(wrapper, IRTailcallAscii)
    assert wrapper.chunk_count == 1
    assert wrapper.preamble == (0x0101, 0x0202, 0x4B08)


def test_function_prologue_combines_branch(tmp_path: Path) -> None:
    _, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    wrapper = IRAsciiWrapperCall(
        target=0x5555,
        helper=0xF172,
        summary="ascii(BODY)",
        chunk_count=3,
        preamble=(0xAAAA, 0xBBBB, 0xCCCC),
        preparation=(("stack_shuffle", 0x1234),),
    )
    branch = IRTestSetBranch(var="slot(0x0001)", expr="cond", then_target=0x10, else_target=0x20)
    items = _ItemList([branch, wrapper])

    normalizer._pass_function_prologues(items)

    assert len(items.to_tuple()) == 1
    prologue = items[0]
    assert isinstance(prologue, IRFunctionPrologue)
    assert not prologue.tail
    assert prologue.call_target == 0x5555
    assert prologue.summary == "ascii(BODY)"
    assert prologue.chunk_count == 3


def test_function_prologue_preserves_tail_flag(tmp_path: Path) -> None:
    _, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    tail_wrapper = IRTailcallAscii(
        target=0x6000,
        helper=0x7223,
        summary="ascii(TAIL)",
        chunk_count=2,
        preamble=(0x1234, 0x5678, 0x9ABC),
    )
    branch = IRTestSetBranch(var="slot(0x0002)", expr="cond", then_target=0x30, else_target=0x40)
    items = _ItemList([branch, tail_wrapper])

    normalizer._pass_function_prologues(items)

    assert len(items.to_tuple()) == 1
    prologue = items[0]
    assert isinstance(prologue, IRFunctionPrologue)
    assert prologue.tail
    assert prologue.call_target == 0x6000
