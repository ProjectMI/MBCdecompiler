from mbcdisasm import Disassembler, Segment, SegmentDescriptor
from mbcdisasm.analyzer import PipelineAnalyzer
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase, OpcodeInfo


def make_word(offset: int, opcode: int, mode: int = 0, operand: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def encode_word(opcode: int, mode: int = 0, operand: int = 0) -> bytes:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return raw.to_bytes(4, "big")


def build_knowledge() -> KnowledgeBase:
    annotations = {
        "10:00": OpcodeInfo(mnemonic="literal", summary="literal load", category="literal", stack_delta=1),
        "11:00": OpcodeInfo(mnemonic="push", summary="stack push", category="push", stack_delta=1),
        "12:00": OpcodeInfo(
            mnemonic="test_branch",
            summary="test",
            control_flow="branch",
            category="test",
            stack_delta=-1,
        ),
        "20:00": OpcodeInfo(mnemonic="reduce", summary="reduce", category="reduce", stack_delta=-2),
        "21:00": OpcodeInfo(mnemonic="call_helper", summary="call", control_flow="call", category="call"),
        "30:00": OpcodeInfo(
            mnemonic="teardown",
            summary="teardown",
            category="stack_teardown",
            stack_delta=-2,
        ),
        "31:00": OpcodeInfo(
            mnemonic="return",
            summary="return",
            control_flow="return",
            category="return",
            stack_delta=-1,
        ),
    }
    return KnowledgeBase(annotations)


def test_literal_pipeline_detection():
    analyzer = PipelineAnalyzer(build_knowledge())
    instructions = [
        make_word(0, 0x10),
        make_word(4, 0x11),
        make_word(8, 0x12),
    ]
    report = analyzer.analyse_segment(instructions)
    assert report.blocks
    first = report.blocks[0]
    assert first.category == "literal"
    assert first.pattern is not None
    assert first.pattern.pattern.name == "literal_push_test"
    assert report.statistics is not None
    assert report.statistics.block_count == len(report.blocks)


def test_call_pipeline_detection():
    analyzer = PipelineAnalyzer(build_knowledge())
    instructions = [
        make_word(0, 0x10),
        make_word(4, 0x11),
        make_word(8, 0x21),
    ]
    report = analyzer.analyse_segment(instructions)
    assert any(block.category == "call" for block in report.blocks)


def test_return_pipeline_detection():
    analyzer = PipelineAnalyzer(build_knowledge())
    instructions = [
        make_word(0, 0x30),
        make_word(4, 0x31),
    ]
    report = analyzer.analyse_segment(instructions)
    categories = [block.category for block in report.blocks]
    assert "return" in categories
    assert report.total_stack_change() <= 0


class _DummyContainer:
    def __init__(self, segment: Segment) -> None:
        self._segment = segment

    def segments(self):
        return (self._segment,)


def test_listing_embeds_pipeline_blocks():
    knowledge = build_knowledge()
    disassembler = Disassembler(knowledge)

    segment_bytes = b"".join(
        [
            encode_word(0x10),
            encode_word(0x11),
            encode_word(0x12),
        ]
    )
    descriptor = SegmentDescriptor(index=0, start=0, end=len(segment_bytes))
    segment = Segment(descriptor, segment_bytes)
    container = _DummyContainer(segment)

    listing = disassembler.generate_listing(container)

    assert "; pipeline stats:" in listing
    assert "; pipeline block 1:" in listing
    assert "category=literal" in listing
    assert "literal" in listing  # mnemonic still present
