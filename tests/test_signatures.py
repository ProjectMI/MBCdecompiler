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


def test_signature_detector_matches_literal_reduce_chain_ex():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x00, 0x0010, 0),
        make_word(0x40, 0x00, 0x0000, 4),
        make_word(0x00, 0x01, 0x0020, 8),
        make_word(0x04, 0x00, 0x0000, 12),
        make_word(0x00, 0x02, 0x0030, 16),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "literal_reduce_chain_ex"


def test_signature_detector_matches_tailcall_return_combo():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x52, 0x0000, 0),
        make_word(0x29, 0x10, 0x003D, 4),
        make_word(0x30, 0x69, 0x0000, 8),
        InstructionWord(12, int.from_bytes(b"exit", "big")),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "tailcall_return_combo"


def test_signature_detector_matches_indirect_call_ex():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x10, 0xE8, 0x0000, 0),
        make_word(0x69, 0x00, 0x0000, 4),
        make_word(0x00, 0x01, 0x1234, 8),
        InstructionWord(12, int.from_bytes(b"ID__", "big")),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "indirect_call_ex"


def test_signature_detector_matches_indirect_return_ex():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x10, 0x00, 0x0000, 0),
        make_word(0x69, 0x00, 0x0000, 4),
        make_word(0x00, 0x20, 0x0001, 8),
        make_word(0x2C, 0x01, 0x0000, 12),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "indirect_return_ex"


def test_signature_detector_matches_header_ascii_ctrl_seq():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        InstructionWord(0, int.from_bytes(b"scpt", "big")),
        InstructionWord(4, int.from_bytes(b"pt01", "big")),
        make_word(0x34, 0x2E, 0x0000, 8),
        make_word(0x33, 0xFF, 0x0000, 12),
        make_word(0xEB, 0x0B, 0x0000, 16),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "header_ascii_ctrl_seq"


def test_signature_detector_matches_literal_run_with_markers():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x40, 0x00, 0x0000, 0),
        make_word(0x40, 0x00, 0x0000, 4),
        make_word(0x00, 0x02, 0x2000, 8),
        InstructionWord(12, int.from_bytes(b"text", "big")),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "literal_run_with_markers"


def test_signature_detector_matches_ascii_tailcall_pattern():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x52, 0x0000, 0),
        make_word(0x03, 0x00, 0x3032, 4),
        make_word(0x29, 0x10, 0x003D, 8),
        InstructionWord(12, int.from_bytes(b"NAME", "big")),
        InstructionWord(16, int.from_bytes(b"tail", "big")),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "ascii_tailcall_pattern"


def test_signature_detector_matches_fanout_teardown_seq():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x10, 0x00, 0x0000, 0),
        make_word(0x66, 0x15, 0x0000, 4),
        make_word(0x01, 0xF0, 0x0000, 8),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "fanout_teardown_seq"


def test_signature_detector_matches_callprep_ascii_dispatch():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x10, 0x04, 0x0000, 0),
        InstructionWord(4, int.from_bytes(b"CALL", "big")),
        make_word(0x00, 0x05, 0x1000, 8),
        InstructionWord(12, int.from_bytes(b"DISP", "big")),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "callprep_ascii_dispatch"
