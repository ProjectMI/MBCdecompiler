from pathlib import Path

from mbcdisasm.analyzer.instruction_profile import InstructionProfile
from mbcdisasm.analyzer.stack import StackTracker, StackValueKind
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase


def make_word(opcode: int, mode: int, operand: int = 0, offset: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def profiles_from_words(words, knowledge):
    return [InstructionProfile.from_word(word, knowledge) for word in words]


def test_marker_has_zero_stack_weight():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    marker_word = make_word(0x00, 0x26, 0x0000, 0)
    profile = InstructionProfile.from_word(marker_word, knowledge)

    tracker = StackTracker()
    event = tracker.process(profile)

    assert event.delta == 0
    assert event.ignore_for_tokens
    assert event.pushed and event.pushed[0].kind is StackValueKind.MARKER


def test_indirect_access_load_classification():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x26, 0x0000, 0),
        make_word(0x69, 0x10, 0x0000, 4),
    ]
    profiles = profiles_from_words(words, knowledge)

    tracker = StackTracker()
    summary = tracker.run(profiles)
    events = summary.events

    assert events[-1].tag == "indirect_load"
    assert events[-1].delta == 1
    assert events[-1].pushed and events[-1].pushed[0].kind is StackValueKind.NUMBER


def test_indirect_access_store_classification():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x00, 0x1234, 0),
        make_word(0x00, 0x26, 0x0000, 4),
        make_word(0x69, 0x10, 0x0000, 8),
    ]
    profiles = profiles_from_words(words, knowledge)

    tracker = StackTracker()
    summary = tracker.run(profiles)
    events = summary.events

    assert events[-1].tag == "indirect_store"
    assert events[-1].delta == 0
    assert all(value.kind is StackValueKind.MARKER for value in events[-1].popped)
