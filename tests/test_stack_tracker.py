from pathlib import Path

from mbcdisasm.analyzer.instruction_profile import InstructionProfile
from mbcdisasm.analyzer.stack import StackTracker, StackValueType
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase


def make_word(opcode: int, mode: int, operand: int = 0, offset: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def load_profiles(*words: InstructionWord):
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    return [InstructionProfile.from_word(word, knowledge) for word in words]


def test_literal_marker_has_zero_stack_weight():
    tracker = StackTracker()
    (profile,) = load_profiles(make_word(0x40, 0x00, 0x0000))
    event = tracker.process(profile)
    assert event.delta == 0
    assert event.pushed_types == tuple()
    assert event.token == "marker"


def test_indirect_load_classification_raises_stack_type():
    tracker = StackTracker()
    profiles = load_profiles(
        make_word(0x00, 0x00, 0x0700),
        make_word(0x69, 0x10, 0x0000),
    )
    events = [tracker.process(profile) for profile in profiles]
    first, second = events
    assert first.pushed_types == (StackValueType.SLOT,)
    assert second.delta == 1
    assert second.pushed_types == (StackValueType.NUMBER,)
    assert second.token == "indirect_load"


def test_indirect_store_classification_marks_token():
    tracker = StackTracker()
    profiles = load_profiles(
        make_word(0x00, 0x00, 0x0700),
        make_word(0x00, 0x00, 0x1234),
        make_word(0x69, 0x10, 0x0000),
    )
    first = tracker.process(profiles[0])
    second = tracker.process(profiles[1])
    third = tracker.process(profiles[2])

    assert first.pushed_types == (StackValueType.SLOT,)
    assert second.pushed_types == (StackValueType.NUMBER,)
    assert third.delta == 0
    assert third.token == "indirect_store"
