from mbcdisasm import Disassembler, Segment, SegmentDescriptor
from mbcdisasm.analyzer import PipelineAnalyzer
from mbcdisasm.analyzer.context import SegmentContext
from mbcdisasm.analyzer.instruction_profile import InstructionKind, InstructionProfile
from mbcdisasm.analyzer.report import PipelineBlock, PipelineReport
from mbcdisasm.analyzer.stack import StackSummary
from mbcdisasm.analyzer.stats import CategoryStats, KindStats, PipelineStatistics
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


def test_guarded_return_pipeline_detection():
    analyzer = PipelineAnalyzer(build_knowledge())
    instructions = [
        make_word(0, 0x30),
        make_word(4, 0x10),
        make_word(8, 0x12),
        make_word(12, 0x11),
        make_word(16, 0x31),
    ]
    report = analyzer.analyse_segment(instructions)
    assert len(report.blocks) == 1
    block = report.blocks[0]
    assert block.category == "return"
    assert block.pattern is not None
    assert block.pattern.pattern.name == "guarded_return"


def test_segment_context_captures_test_and_return_boundaries():
    knowledge = build_knowledge()
    instructions = [
        make_word(0, 0x10),  # literal
        make_word(4, 0x12),  # test branch
        make_word(8, 0x10),  # literal between control edges
        make_word(12, 0x31),  # return terminator
    ]
    profiles = [InstructionProfile.from_word(word, knowledge) for word in instructions]

    context = SegmentContext(profiles)
    blocks = list(context.iter_blocks())

    assert len(context.boundaries) == 2
    assert len(blocks) == 2

    first = blocks[0]
    assert first.start.ordinal == -1
    assert first.end.profile.mnemonic == "test_branch"
    assert [profile.mnemonic for profile in first.profiles()] == ["literal"]

    second = blocks[1]
    assert second.start.profile.mnemonic == "test_branch"
    assert second.end.profile.mnemonic == "return"
    assert [profile.mnemonic for profile in second.profiles()] == ["literal"]

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


def test_listing_summary_counts_unknowns():
    knowledge = KnowledgeBase({})

    word = InstructionWord(0, 0)
    profile = InstructionProfile.from_word(word, knowledge)
    stack = StackSummary(change=0, minimum=0, maximum=0, uncertain=False, events=tuple())
    block = PipelineBlock(
        profiles=(profile,),
        stack=stack,
        kind=InstructionKind.UNKNOWN,
        category="unknown",
        pattern=None,
        confidence=0.1,
    )
    stats = PipelineStatistics(
        block_count=1,
        instruction_count=1,
        total_stack_delta=0,
        categories={"unknown": CategoryStats(count=1, stack_delta=0, stack_abs=0)},
        kinds=KindStats(counts={InstructionKind.UNKNOWN: 1}),
    )
    report = PipelineReport(blocks=(block,), warnings=("manual warning",), statistics=stats)

    class _SummaryAnalyzer:
        def __init__(self, result: PipelineReport) -> None:
            self._result = result

        def analyse_segment(self, instructions):
            return self._result

    analyzer = _SummaryAnalyzer(report)

    segment_bytes = encode_word(0)
    descriptor = SegmentDescriptor(index=0, start=0, end=len(segment_bytes))
    segment = Segment(descriptor, segment_bytes)
    container = _DummyContainer(segment)

    disassembler = Disassembler(knowledge, analyzer=analyzer)
    listing = disassembler.generate_listing(container)

    assert "; pipeline block 1:" in listing

    summary = disassembler.summary
    assert summary is not None
    assert summary.unknown_kinds == 1
    assert summary.unknown_categories == 1
    assert summary.unknown_patterns == 1
    assert summary.unknown_dominant == 1
    assert summary.warning_count == 1


def test_fanout_return_pattern_detection():
    analyzer = PipelineAnalyzer(KnowledgeBase({}))
    instructions = [
        make_word(0, 0x69, 0x10, 0x0000),
        make_word(4, 0x00, 0x00, 0x0000),
        make_word(8, 0x02, 0x66, 0x0000),
        make_word(12, 0x30, 0x41, 0x0000),
        make_word(16, 0x00, 0x00, 0x0000),
        make_word(20, 0x69, 0x10, 0x0000),
    ]
    report = analyzer.analyse_segment(instructions)
    names = [block.pattern.pattern.name for block in report.blocks if block.pattern]
    assert "fanout_return" in names


def test_fanout_tail_return_pattern_detection():
    analyzer = PipelineAnalyzer(KnowledgeBase({}))
    instructions = [
        make_word(0, 0xFD, 0x4A, 0x9D01),
        make_word(4, 0x30, 0x69, 0x108C),
        make_word(8, 0x09, 0x00, 0x0029),
    ]
    report = analyzer.analyse_segment(instructions)
    names = [block.pattern.pattern.name for block in report.blocks if block.pattern]
    assert "fanout_tail_return" in names
