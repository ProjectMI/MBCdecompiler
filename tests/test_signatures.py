import struct
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


def test_ascii_detection_accepts_wide_pairs():
    wide_be = InstructionWord(0, int.from_bytes(b"\x00A\x00B", "big"))
    wide_le = InstructionWord(4, int.from_bytes(b"A\x00B\x00", "big"))
    pair_le = InstructionWord(8, int.from_bytes(struct.pack("<HH", 0x4142, 0x4344), "big"))

    assert looks_like_ascii_chunk(wide_be)
    assert looks_like_ascii_chunk(wide_le)
    assert looks_like_ascii_chunk(pair_le)


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


def test_signature_detector_matches_literal_mirror_reduce_loop():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x00, 0x6704, 0),
        make_word(0x00, 0x00, 0x0067, 4),
        make_word(0x04, 0x00, 0x0000, 8),
        make_word(0x67, 0x04, 0x0000, 12),
        make_word(0x00, 0x00, 0x6704, 16),
        make_word(0x00, 0x00, 0x0067, 20),
        make_word(0x04, 0x00, 0x0000, 24),
        make_word(0x67, 0x04, 0x0000, 28),
        make_word(0x00, 0x00, 0x6704, 32),
        make_word(0x00, 0x00, 0x0067, 36),
        make_word(0x04, 0x00, 0x0000, 40),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "literal_mirror_reduce_loop"


def test_signature_detector_matches_mode_sweep_block():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0xAF, 0x4E, 0x0001, 0),
        make_word(0xC6, 0x4E, 0x0002, 4),
        make_word(0xD7, 0x4E, 0x0003, 8),
        make_word(0xE0, 0x4E, 0x0004, 12),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "mode_sweep_block"


def test_signature_detector_matches_extended_mode_sweep_block():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x80, 0x2A, 0x0001, 0),
        make_word(0x81, 0x2A, 0x0002, 4),
        make_word(0x82, 0x2A, 0x0003, 8),
        make_word(0x83, 0x2A, 0x0004, 12),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "mode_sweep_block"


def test_signature_detector_matches_stack_lift_pair():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x30, 0x0700, 0),
        make_word(0x00, 0x48, 0x0000, 4),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "stack_lift_pair"


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


def test_signature_detector_matches_tailcall_return_marker():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x52, 0x0000, 0),
        make_word(0x29, 0x10, 0x0001, 4),
        make_word(0x30, 0x69, 0x0000, 8),
        make_word(0x00, 0x00, 0x6704, 12),
        make_word(0x00, 0x00, 0x0067, 16),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "tailcall_return_marker"


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


def test_signature_detector_matches_indirect_call_mini():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x10, 0xE8, 0x0000, 0),
        make_word(0x69, 0x10, 0x0000, 4),
        make_word(0x00, 0x00, 0x0067, 8),
        make_word(0x01, 0x3D, 0x0000, 12),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "indirect_call_mini"


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


def test_signature_detector_matches_ascii_header():
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
    assert match.name == "ascii_header"


def test_signature_detector_matches_ascii_control_cluster():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        InstructionWord(0, int.from_bytes(b"char", "big")),
        InstructionWord(4, int.from_bytes(b"name", "big")),
        make_word(0x34, 0x2E, 0x0000, 8),
        make_word(0x33, 0xFF, 0x0000, 12),
        make_word(0x00, 0x40, 0x0000, 16),
        make_word(0x67, 0x04, 0x0000, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "ascii_control_cluster"


def test_signature_detector_matches_script_header_prolog():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        InstructionWord(0, int.from_bytes(b"scri", "big")),
        InstructionWord(4, int.from_bytes(b"pt v", "big")),
        make_word(0x34, 0x2E, 0x0000, 8),
        make_word(0x58, 0x26, 0x0000, 12),
        make_word(0x06, 0x00, 0x0000, 16),
        make_word(0x63, 0x00, 0x0000, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "script_header_prolog"


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


def test_signature_detector_matches_ascii_block():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x04, 0x00, 0x0000, 0),
        InstructionWord(4, int.from_bytes(b"head", "big")),
        make_word(0x66, 0x1B, 0x0000, 8),
        make_word(0x00, 0x52, 0x0000, 12),
        make_word(0x4A, 0x05, 0x0000, 16),
        make_word(0x3D, 0x00, 0x0000, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "ascii_block"


def test_signature_detector_matches_ascii_inline():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x04, 0x00, 0x0000, 0),
        make_word(0x00, 0x4F, 0x0110, 4),
        InstructionWord(8, int.from_bytes(b"text", "big")),
        InstructionWord(12, int.from_bytes(b"DATA", "big")),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "ascii_inline"


def test_signature_detector_matches_literal_zero_init():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x00, 0x1000, 0),
        make_word(0x00, 0x01, 0x2000, 4),
        make_word(0xDE, 0xED, 0x0000, 8),
        make_word(0x00, 0x00, 0x0000, 12),
        make_word(0x00, 0x00, 0x0000, 16),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "literal_zero_init"


def test_signature_detector_matches_ascii_marker():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x40, 0x00, 0x0000, 0),
        make_word(0x40, 0x00, 0x0001, 4),
        make_word(0x41, 0xB4, 0x0003, 8),
        make_word(0x08, 0x00, 0x0008, 12),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "ascii_marker"


def test_signature_detector_matches_marker_fence_reduce():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x3D, 0x30, 0x3069, 0),
        make_word(0x01, 0x90, 0x0000, 4),
        make_word(0x5E, 0x29, 0x2910, 8),
        make_word(0xED, 0x4D, 0x4D0E, 12),
        make_word(0x00, 0x69, 0x0190, 16),
        make_word(0x04, 0x10, 0x0000, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "marker_fence_reduce"


def test_signature_detector_matches_ascii_marker_combo():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x05, 0x00, 0x0000, 0),
        InstructionWord(4, int.from_bytes(b"DATA", "big")),
        make_word(0x10, 0x0F, 0x0000, 8),
        make_word(0x40, 0x00, 0x0000, 12),
        make_word(0x40, 0x00, 0x0001, 16),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "ascii_marker_combo"


def test_signature_detector_matches_ascii_wrapper():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0xEF, 0x28, 0x0000, 0),
        InstructionWord(4, int.from_bytes(b"wrap", "big")),
        make_word(0x48, 0x00, 0x0000, 8),
        make_word(0x00, 0x00, 0x0001, 12),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "ascii_wrapper"


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


def test_signature_detector_matches_tailcall_ascii_wrapper():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x03, 0x00, 0x0000, 0),
        make_word(0x29, 0x10, 0x0072, 4),
        InstructionWord(8, int.from_bytes(b"#HO ", "big")),
        make_word(0x23, 0x4F, 0x0000, 12),
        make_word(0x52, 0x05, 0x0000, 16),
        make_word(0x32, 0x29, 0x0000, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "tailcall_ascii_wrapper"


def test_signature_detector_matches_jump_ascii_tailcall():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x22, 0x00, 0x0002, 0),
        InstructionWord(4, int.from_bytes(b"pre_", "big")),
        make_word(0x29, 0x10, 0x0000, 8),
        InstructionWord(12, int.from_bytes(b"tail", "big")),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "jump_ascii_tailcall"


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


def test_signature_detector_matches_fanout_teardown_ext():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x1A, 0x21, 0x0000, 0),
        make_word(0x04, 0x00, 0x0000, 4),
        make_word(0x66, 0x20, 0x0000, 8),
        make_word(0x01, 0x69, 0x0000, 12),
        make_word(0x27, 0x00, 0x0000, 16),
        make_word(0x04, 0x66, 0x0000, 20),
        make_word(0x10, 0x00, 0x0000, 24),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "fanout_teardown_ext"


def test_signature_detector_matches_double_tailcall_branch():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x29, 0x10, 0x0000, 0),
        make_word(0x29, 0x20, 0x0000, 4),
        make_word(0x28, 0x10, 0x0000, 8),
        make_word(0x2F, 0x29, 0x0000, 12),
        make_word(0x2F, 0x2C, 0x0000, 16),
        make_word(0x26, 0x30, 0x0000, 20),
        make_word(0xBD, 0x00, 0x0000, 24),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "double_tailcall_branch"


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


def test_signature_detector_matches_ascii_indirect_tailcall():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x00, 0x38, 0x0000, 0),
        make_word(0x29, 0x10, 0x0000, 4),
        make_word(0x4B, 0x30, 0x0000, 8),
        make_word(0x69, 0x10, 0x0000, 12),
        make_word(0x00, 0x00, 0x0001, 16),
        InstructionWord(20, int.from_bytes(b"ID__", "big")),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "ascii_indirect_tailcall"


def test_signature_detector_matches_tailcall_post_jump():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x22, 0x30, 0x0002, 0),
        make_word(0x29, 0x10, 0x0000, 4),
        make_word(0x69, 0x00, 0x0000, 8),
        InstructionWord(12, int.from_bytes(b"jmp_", "big")),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "tailcall_post_jump"


def test_signature_detector_matches_tailcall_return_indirect():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x4C, 0x00, 0x0000, 0),
        make_word(0x29, 0x10, 0x0000, 4),
        make_word(0x30, 0x00, 0x0000, 8),
        make_word(0x00, 0x00, 0x0001, 12),
        make_word(0x10, 0x01, 0x0000, 16),
        make_word(0x69, 0x00, 0x0000, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "tailcall_return_indirect"


def test_signature_detector_matches_return_teardown_marker():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x01, 0xEC, 0x0000, 0),
        make_word(0x30, 0x10, 0x0000, 4),
        make_word(0x00, 0x00, 0x6704, 8),
        make_word(0x00, 0x00, 0x0067, 12),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "return_teardown_marker"


def test_signature_detector_matches_return_mode_ribbon():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x27, 0x5B, 0x0000, 0),
        make_word(0x2A, 0x5B, 0x0000, 4),
        make_word(0x30, 0x5B, 0x0000, 8),
        make_word(0x3F, 0x5B, 0x0000, 12),
        make_word(0x4A, 0x5B, 0x0000, 16),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "return_mode_ribbon"


def test_signature_detector_matches_return_stack_marker_seq():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x5E, 0x29, 0x0000, 0),
        make_word(0xF0, 0x4B, 0x0000, 4),
        make_word(0x30, 0x6C, 0x0000, 8),
        make_word(0x01, 0xF0, 0x0000, 12),
        make_word(0x00, 0x00, 0x0000, 16),
        make_word(0x00, 0x00, 0x0000, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "return_stack_marker_seq"


def test_signature_detector_matches_return_bd_capsule():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0xC5, 0xBD, 0x0000, 0),
        make_word(0x30, 0x69, 0x10A8, 4),
        make_word(0x10, 0x37, 0x7DFA, 8),
        make_word(0x0B, 0x3D, 0x3041, 12),
        make_word(0xAC, 0xBD, 0x0000, 16),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "return_bd_capsule"


def test_signature_detector_matches_poison_return_prolog():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0xFA, 0xFF, 0xFFFF, 0),
        make_word(0x41, 0xDD, 0x0E00, 4),
        make_word(0x00, 0x09, 0x003D, 8),
        make_word(0x30, 0x29, 0x1002, 12),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "poison_return_prolog"


def test_signature_detector_matches_return_ascii_epilogue():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x04, 0x00, 0x0000, 0),
        make_word(0x10, 0x00, 0x0000, 4),
        make_word(0x77, 0x00, 0x3069, 8),
        make_word(0x01, 0x84, 0x0000, 12),
        InstructionWord(16, int.from_bytes(b"tail", "big")),
        make_word(0x00, 0x00, 0x8300, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "return_ascii_epilogue"


def test_signature_detector_matches_b4_slot_return():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x20, 0x00, 0x1234, 0),
        make_word(0x01, 0xB4, 0x0001, 4),
        make_word(0x00, 0x69, 0x0190, 8),
        make_word(0x27, 0x00, 0x0000, 12),
        make_word(0x02, 0x66, 0x0000, 16),
        make_word(0x30, 0x00, 0x0000, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "b4_slot_return"


def test_signature_detector_matches_indirect_call_dual_literal():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    words = [
        make_word(0x10, 0x38, 0x0000, 0),
        make_word(0x00, 0x00, 0x0001, 4),
        make_word(0xF1, 0x3D, 0x0000, 8),
        make_word(0x10, 0x48, 0x0000, 12),
        make_word(0x00, 0x01, 0x0002, 16),
        make_word(0x69, 0x10, 0x0000, 20),
    ]
    profiles, summary = profiles_from_words(words, knowledge)
    detector = SignatureDetector()
    match = detector.detect(profiles, summary)
    assert match is not None
    assert match.name == "indirect_call_dual_literal"
