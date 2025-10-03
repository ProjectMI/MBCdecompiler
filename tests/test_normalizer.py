from pathlib import Path

from mbcdisasm.analyzer.normalizer import MacroNormalizer
from mbcdisasm.analyzer.instruction_profile import InstructionKind, InstructionProfile
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase


def make_word(opcode: int, mode: int = 0, operand: int = 0, offset: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def build_profiles(words):
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    normalizer = MacroNormalizer()
    normalizer.apply(profiles)
    return profiles


def test_tail_dispatch_and_frame_return_macros():
    words = [
        make_word(0x16, 0x01, 0, 0),  # call helper
        make_word(0x29, 0x10, 0, 4),  # tail dispatch
        make_word(0x01, 0x08, 0, 8),  # frame teardown
        make_word(0x30, 0x00, 0, 12),  # return
    ]
    profiles = build_profiles(words)

    leader = profiles[0]
    assert leader.traits.get("macro") == "tail_dispatch"
    assert leader.traits.get("macro_kind") is InstructionKind.MACRO_CALL
    assert profiles[1].traits.get("macro_member") == "tail_dispatch"

    frame_return = profiles[2]
    assert frame_return.traits.get("macro") == "frame_return"
    assert frame_return.traits.get("macro_kind") is InstructionKind.MACRO_FRAME_END
    assert profiles[3].traits.get("macro_member") == "frame_return"


def test_literal_reduce_chain_collapses_into_array_builder():
    words = [
        make_word(0x00, 0x00, 0x0010, 0),
        make_word(0x00, 0x00, 0x0011, 4),
        make_word(0x00, 0x00, 0x0012, 8),
        make_word(0x04, 0x00, 0x0000, 12),
    ]
    profiles = build_profiles(words)

    leader = profiles[0]
    assert leader.traits.get("macro") == "literal_array_builder"
    assert leader.traits.get("macro_kind") is InstructionKind.MACRO_LITERAL_ARRAY
    assert leader.traits.get("literal_count") == 3
    assert profiles[1].traits.get("macro_member") == "literal_array_builder"


def test_predicate_branch_promoted_to_macro():
    words = [make_word(0x26, 0x00, 0x0002, 0)]
    profiles = build_profiles(words)

    leader = profiles[0]
    assert leader.traits.get("macro") == "predicate_assign"
    assert leader.traits.get("macro_kind") is InstructionKind.MACRO_PREDICATE


def test_indirect_access_zones_are_tagged():
    words = [
        make_word(0x69, 0x01, 0x0010, 0),  # frame slot
        make_word(0x69, 0x20, 0x4000, 4),  # global slot
    ]
    profiles = build_profiles(words)

    frame = profiles[0]
    assert frame.traits.get("macro") == "frame_slot_access"
    assert frame.traits.get("zone") == "frame"
    assert frame.traits.get("macro_kind") is InstructionKind.MACRO_FRAME_SLOT

    glob = profiles[1]
    assert glob.traits.get("macro") == "global_slot_access"
    assert glob.traits.get("zone") == "global"
    assert glob.traits.get("macro_kind") is InstructionKind.MACRO_GLOBAL_SLOT
