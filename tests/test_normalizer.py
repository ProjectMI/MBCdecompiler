from mbcdisasm.analyzer import (
    IntermediateSummary,
    MacroNormalizer,
    PipelineBlock,
    PipelineReport,
)
from mbcdisasm.analyzer.instruction_profile import InstructionProfile
from mbcdisasm.analyzer.report import build_block as build_pipeline_block
from mbcdisasm.analyzer.stack import StackTracker
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase, OpcodeInfo


def make_word(offset: int, opcode: int, mode: int = 0, operand: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def build_knowledge() -> KnowledgeBase:
    annotations = {
        "10:00": OpcodeInfo(
            mnemonic="literal",
            summary="literal",
            category="literal",
            stack_delta=1,
        ),
        "11:00": OpcodeInfo(
            mnemonic="push_slot",
            summary="push",
            category="push",
            stack_delta=1,
        ),
        "12:00": OpcodeInfo(
            mnemonic="testset",
            summary="test",
            category="test",
            stack_delta=-1,
        ),
        "20:00": OpcodeInfo(
            mnemonic="reduce",
            summary="reduce",
            category="reduce",
            stack_delta=-1,
        ),
        "21:00": OpcodeInfo(
            mnemonic="indirect",
            summary="indirect",
            category="indirect",
            stack_delta=0,
        ),
        "21:10": OpcodeInfo(
            mnemonic="indirect",
            summary="indirect",
            category="indirect",
            stack_delta=0,
        ),
        "21:20": OpcodeInfo(
            mnemonic="indirect",
            summary="indirect",
            category="indirect",
            stack_delta=0,
        ),
        "29:00": OpcodeInfo(
            mnemonic="tailcall_dispatch",
            summary="tailcall",
            control_flow="call",
            category="tailcall",
            stack_delta=0,
        ),
        "30:00": OpcodeInfo(
            mnemonic="return",
            summary="return",
            control_flow="return",
            category="return",
            stack_delta=-1,
        ),
    }
    return KnowledgeBase(annotations)


def make_block(words, knowledge: KnowledgeBase, category: str, confidence: float = 0.6) -> PipelineBlock:
    profiles = tuple(InstructionProfile.from_word(word, knowledge) for word in words)
    tracker = StackTracker()
    stack = tracker.process_block(profiles)
    return build_pipeline_block(profiles, stack, pattern=None, category=category, confidence=confidence)


def test_tail_dispatch_collapses_to_macro():
    knowledge = build_knowledge()
    block = make_block([make_word(0, 0x29)], knowledge, category="call")
    normalizer = MacroNormalizer()

    normalized = normalizer.normalize_block(block)

    assert len(normalized.operations) == 1
    op = normalized.operations[0]
    assert op.macro == "tail_dispatch"
    assert op.sources == ("29:00",)


def test_return_collapses_to_macro():
    knowledge = build_knowledge()
    block = make_block([make_word(0, 0x30)], knowledge, category="return")
    normalizer = MacroNormalizer()

    normalized = normalizer.normalize_block(block)

    assert len(normalized.operations) == 1
    op = normalized.operations[0]
    assert op.macro == "frame_return"
    assert op.sources == ("30:00",)


def test_literal_reduce_forms_tuple_macro():
    knowledge = build_knowledge()
    words = [make_word(0, 0x10), make_word(4, 0x10), make_word(8, 0x20)]
    block = make_block(words, knowledge, category="compute")
    normalizer = MacroNormalizer()

    normalized = normalizer.normalize_block(block)

    assert normalized.operations[0].macro == "tuple_build"
    assert normalized.operations[0].operands[0] == "width=2"
    assert normalized.operations[0].sources == ("10:00", "10:00", "20:00")


def test_literal_reduce_with_push_forms_table_macro():
    knowledge = build_knowledge()
    words = [make_word(0, 0x11), make_word(4, 0x10), make_word(8, 0x20)]
    block = make_block(words, knowledge, category="literal")
    normalizer = MacroNormalizer()

    normalized = normalizer.normalize_block(block)

    assert normalized.operations[0].macro == "table_build"
    assert "width=2" in normalized.operations[0].operands


def test_predicate_assignment_macro():
    knowledge = build_knowledge()
    words = [make_word(0, 0x11), make_word(4, 0x12)]
    block = make_block(words, knowledge, category="test")
    normalizer = MacroNormalizer()

    normalized = normalizer.normalize_block(block)

    assert any(op.macro == "predicate_assign" for op in normalized.operations)
    predicate = next(op for op in normalized.operations if op.macro == "predicate_assign")
    assert "tests=1" in predicate.operands
    assert "target=11:00" in predicate.operands


def test_indirect_access_maps_to_zones():
    knowledge = build_knowledge()
    words = [make_word(0, 0x21, mode) for mode in (0x10, 0x20)]
    block = make_block(words, knowledge, category="indirect")
    normalizer = MacroNormalizer()

    normalized = normalizer.normalize_block(block)

    macros = {op.macro for op in normalized.operations}
    assert macros == {"frame_access", "global_access"}
    slots = {operand for op in normalized.operations for operand in op.operands}
    assert "slot=0x10" in slots
    assert "slot=0x20" in slots


def test_summary_collects_macro_counts():
    knowledge = build_knowledge()
    tail_block = make_block([make_word(0, 0x29)], knowledge, category="call")
    return_block = make_block([make_word(0, 0x30)], knowledge, category="return")
    literal_block = make_block([make_word(0, 0x10), make_word(4, 0x10), make_word(8, 0x20)], knowledge, category="literal")

    report = PipelineReport(blocks=(tail_block, return_block, literal_block))

    normalizer = MacroNormalizer()
    normalized_blocks = normalizer.normalize_report(report)
    summary = normalizer.summarise(normalized_blocks)

    assert isinstance(summary, IntermediateSummary)
    assert summary.operation_counts["tail_dispatch"] == 1
    assert summary.operation_counts["frame_return"] == 1
    assert summary.operation_counts["tuple_build"] == 1
    assert summary.block_count == 3
    description = summary.describe()
    assert "blocks=3" in description
    assert "tail_dispatch:1" in description
