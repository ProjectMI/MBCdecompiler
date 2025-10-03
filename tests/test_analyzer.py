from mbcdisasm import Disassembler, Segment, SegmentDescriptor
from mbcdisasm.analyzer import PipelineAnalyzer
from mbcdisasm.analyzer.instruction_profile import (
    InstructionKind,
    InstructionProfile,
    StackEffectHint,
)
from mbcdisasm.analyzer.pipeline import AnalyzerSettings
from mbcdisasm.analyzer.report import PipelineBlock, PipelineReport
from mbcdisasm.analyzer.stack import StackSummary, StackTracker
from mbcdisasm.analyzer.stats import CategoryStats, KindStats, PipelineStatistics
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase, OpcodeInfo


def make_word(offset: int, opcode: int, mode: int = 0, operand: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def encode_word(opcode: int, mode: int = 0, operand: int = 0) -> bytes:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return raw.to_bytes(4, "big")


def make_profile(
    offset: int,
    opcode: int,
    mode: int,
    mnemonic: str,
    kind: InstructionKind,
    *,
    stack_delta: int = 0,
) -> InstructionProfile:
    raw = (opcode << 24) | (mode << 16)
    word = InstructionWord(offset, raw)
    hint = StackEffectHint(
        nominal=stack_delta,
        minimum=stack_delta,
        maximum=stack_delta,
        confidence=1.0,
    )
    return InstructionProfile(
        word=word,
        info=None,
        mnemonic=mnemonic,
        summary=None,
        category=None,
        control_flow=None,
        stack_hint=hint,
        kind=kind,
        traits={},
    )


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


def test_pipeline_analyzer_normalises_marker_runs():
    knowledge = KnowledgeBase({})
    analyzer = PipelineAnalyzer(knowledge)

    profiles = [
        make_profile(0, 0x29, 0x10, "tailcall_dispatch", InstructionKind.TAILCALL),
        make_profile(4, 0x41, 0x00, "inline_ascii_chunk", InstructionKind.ASCII_CHUNK),
        make_profile(8, 0x40, 0x00, "literal_marker", InstructionKind.LITERAL),
        make_profile(12, 0x67, 0x00, "literal_marker", InstructionKind.LITERAL),
        make_profile(16, 0x30, 0x69, "return_values", InstructionKind.RETURN, stack_delta=-1),
    ]

    tracker = StackTracker()
    events = tuple(tracker.process(profile) for profile in profiles)
    normalised, spans = analyzer._normalise_events(events)

    assert len(events) == 5
    assert len(normalised) == 4
    assert spans[2] == (2, 4)
    assert normalised[2].delta == 2
    assert normalised[2].profile.label == "40:00"


def test_pipeline_analyzer_groups_tailcall_ascii_return_with_markers():
    annotations = {
        "29:10": OpcodeInfo(
            mnemonic="tailcall_dispatch",
            summary="tailcall",
            control_flow="call",
            category="tailcall",
            stack_delta=0,
        ),
        "30:69": OpcodeInfo(
            mnemonic="return_values",
            summary="return",
            control_flow="return",
            category="return",
            stack_delta=-1,
        ),
        "40:00": OpcodeInfo(
            mnemonic="literal_marker",
            summary="marker",
            category="literal_marker",
            stack_delta=0,
        ),
        "67:00": OpcodeInfo(
            mnemonic="literal_marker",
            summary="marker",
            category="literal_marker",
            stack_delta=0,
        ),
    }

    knowledge = KnowledgeBase(annotations)
    analyzer = PipelineAnalyzer(knowledge, settings=AnalyzerSettings(max_window=4))

    instructions = [
        InstructionWord(0, (0x29 << 24) | (0x10 << 16)),
        InstructionWord(4, int.from_bytes(b"ASCI", "big")),
        InstructionWord(8, (0x40 << 24)),
        InstructionWord(12, (0x67 << 24)),
        InstructionWord(16, (0x30 << 24) | (0x69 << 16)),
    ]

    report = analyzer.analyse_segment(instructions)
    assert len(report.blocks) == 1
    block = report.blocks[0]
    assert [profile.word.offset for profile in block.profiles] == [0, 4, 8, 12, 16]
