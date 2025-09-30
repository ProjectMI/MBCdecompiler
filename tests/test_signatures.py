from pathlib import Path

from mbcdisasm import KnowledgeBase
from mbcdisasm.analyzer.instruction_profile import (
    InstructionKind,
    InstructionProfile,
    looks_like_ascii_chunk,
)
from mbcdisasm.analyzer.signatures import SignatureDetector
from mbcdisasm.analyzer.stack import StackTracker
from mbcdisasm.instruction import InstructionWord


def make_word(opcode: int, mode: int, operand: int = 0, offset: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def profiles_from_words(words, knowledge):
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    stack = StackTracker()
    summary = stack.run(profiles)
    return profiles, summary


def test_ascii_detection_marks_inline_chunk():
    knowledge = KnowledgeBase({})
    word = InstructionWord(0, int.from_bytes(b"test", "big"))
    assert looks_like_ascii_chunk(word)
    profile = InstructionProfile.from_word(word, knowledge)
    assert profile.mnemonic == "inline_ascii_chunk"
    assert profile.kind is InstructionKind.ASCII_CHUNK
    assert profile.traits.get("heuristic")


def test_signature_detector_matches_ascii_run():
    knowledge = KnowledgeBase({})
    words = [
        InstructionWord(i * 4, int.from_bytes(text.encode("ascii"), "big"))
        for i, text in enumerate(["char", "name", "data"])
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "ascii_run"
    assert match.category == "literal"


def test_signature_detector_matches_literal_run():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [make_word(0x00, 0x00, operand, idx * 4) for idx, operand in enumerate([0x10, 0x11, 0x12, 0x13, 0x14])]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name in {"literal_run", "literal_bulk"}


def test_signature_detector_matches_table_store_pattern():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x66, 0x75, 0x3029, 0),
        make_word(0x10, 0x0E, 0x41E1, 4),
        make_word(0x03, 0x00, 0x0006, 8),
        make_word(0x00, 0x6C, 0x11F4, 12),
        make_word(0x00, 0x00, 0x0090, 16),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "table_store"
