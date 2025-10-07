from pathlib import Path

from mbcdisasm import KnowledgeBase
from mbcdisasm.analyzer.instruction_profile import InstructionKind, InstructionProfile
from mbcdisasm.analyzer.stack import StackTracker, StackValueType
from mbcdisasm.constants import CALL_SHUFFLE_STANDARD, RET_MASK
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


def test_literal_marker_mode_72_is_stack_neutral():
    words = [
        make_word(0x00, 0x72, 0x0000, 0),
        make_word(0x00, 0x00, 0x0001, 4),
    ]

    profiles = load_profiles(words)
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    marker_event = events[0]
    literal_event = events[1]

    assert marker_event.delta == 0
    assert marker_event.pushed_types == (StackValueType.MARKER,)
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


def test_call_helper_wrapper_has_stable_stack_model():
    words = [
        make_word(0x66, 0x15, CALL_SHUFFLE_STANDARD, 0),
        make_word(0x4A, 0x05, 0x0052, 4),
        make_word(0x28, 0x00, 0x1234, 8),
        make_word(0x10, 0xE8, 0x0001, 12),
        make_word(0x32, 0x29, 0x1000, 16),
        make_word(0x29, 0x10, RET_MASK, 20),
        make_word(0x01, 0xF0, 0x0000, 24),
        make_word(0x30, 0x00, 0x0002, 28),
    ]

    profiles = load_profiles(words)
    tracker = StackTracker()
    summary = tracker.run(profiles)

    assert summary.change == -4
    assert not summary.uncertain
    assert all(not event.uncertain for event in summary.events)

    call_dispatch = summary.events[2]
    helper_call = summary.events[3]
    tail_mask = summary.events[5]
    teardown = summary.events[6]

    assert call_dispatch.kind is InstructionKind.CALL
    assert helper_call.kind is InstructionKind.CALL
    assert tail_mask.kind is InstructionKind.TAILCALL
    assert teardown.delta == -4


def test_literal_reduce_chain_is_tracked_with_high_confidence():
    words = [
        make_word(0x00, 0x00, 0x0001, 0),
        make_word(0x00, 0x00, 0x0002, 4),
        make_word(0x04, 0x00, 0x0000, 8),
    ]

    profiles = load_profiles(words)
    tracker = StackTracker()
    summary = tracker.run(profiles)

    assert summary.change == 1
    assert not summary.uncertain

    deltas = [1, 1, -1]
    for event, expected in zip(summary.events, deltas):
        assert event.delta == expected
        assert not event.uncertain

    reducer = summary.events[-1]
    assert reducer.popped_types == (
        StackValueType.NUMBER,
        StackValueType.NUMBER,
    )
    assert reducer.pushed_types == (StackValueType.NUMBER,)


def test_tail_mask_is_stack_neutral():
    words = [make_word(0x29, 0x10, RET_MASK, 0)]

    profiles = load_profiles(words)
    tracker = StackTracker()
    summary = tracker.run(profiles)

    assert not summary.uncertain
    assert len(summary.events) == 1
    event = summary.events[0]
    assert event.kind is InstructionKind.TAILCALL
    assert event.delta == 0
    assert not event.uncertain
