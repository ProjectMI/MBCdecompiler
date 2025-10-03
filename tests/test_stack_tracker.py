from pathlib import Path

from mbcdisasm import KnowledgeBase
from mbcdisasm.analyzer.instruction_profile import InstructionKind, InstructionProfile
from mbcdisasm.analyzer.stack import StackTracker, StackValueType
from mbcdisasm.instruction import InstructionWord


def make_word(opcode: int, mode: int, operand: int = 0, offset: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def load_profiles(words):
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    return [InstructionProfile.from_word(word, knowledge) for word in words]


def test_literal_marker_is_stack_neutral():
    words = [
        make_word(0x00, 0x26, 0x0000, 0),  # literal_marker according to knowledge
        make_word(0x00, 0x00, 0x0001, 4),
    ]
    profiles = load_profiles(words)
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    marker_event = events[0]
    literal_event = events[1]

    assert marker_event.delta == 0
    assert marker_event.pushed_types == (StackValueType.MARKER,)
    assert literal_event.delta > 0
    assert literal_event.depth_after == marker_event.depth_after + literal_event.delta


def test_indirect_access_load_variant():
    words = [
        make_word(0x02, 0x00, 0x0000, 0),  # stack push helper
        make_word(0x69, 0x10, 0x0000, 4),  # indirect access (load)
    ]
    profiles = load_profiles(words)
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    indirect_event = events[-1]
    assert indirect_event.kind is InstructionKind.INDIRECT_LOAD
    assert indirect_event.delta == 1
    assert indirect_event.pushed_types == (StackValueType.NUMBER,)


def test_indirect_access_store_variant_detects_cleanup():
    words = [
        make_word(0x02, 0x00, 0x0000, 0),
        make_word(0x69, 0x10, 0x0000, 4),
        make_word(0x01, 0xF0, 0x0000, 8),  # stack teardown helper
    ]
    profiles = load_profiles(words)
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    indirect_event = events[1]
    assert indirect_event.kind is InstructionKind.INDIRECT_STORE
    assert indirect_event.delta == 0
    assert indirect_event.pushed_types == tuple()
    assert StackValueType.SLOT in indirect_event.popped_types
