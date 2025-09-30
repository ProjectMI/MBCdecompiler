"""Tests for :mod:`mbcdisasm.analyzer.char_patterns`."""

from pathlib import Path

from mbcdisasm.analyzer.char_patterns import (
    find_patterns,
    summarise_patterns,
    pattern_summary_text,
    list_pattern_names,
    contains_pattern,
)
from mbcdisasm.analyzer.instruction_profile import InstructionProfile
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase


KNOWLEDGE = KnowledgeBase.load(Path("knowledge"))


def make_profile(raw: int, offset: int = 0) -> InstructionProfile:
    word = InstructionWord(offset=offset, raw=raw)
    return InstructionProfile.from_word(word, KNOWLEDGE)


def test_zero_pattern_detected() -> None:
    profiles = [make_profile(0x00000000) for _ in range(4)]
    matches = find_patterns(profiles)
    assert "zero_block" in matches


def test_config_header_detected() -> None:
    text1 = int.from_bytes(b"para", "big")
    text2 = int.from_bytes(b"ms\\\x00", "big")
    profiles = [make_profile(text1), make_profile(text2)]
    matches = find_patterns(profiles)
    assert "config_header" in matches
    summary = summarise_patterns(profiles)
    assert summary["config_header"] == 1
    text = pattern_summary_text(profiles)
    assert "config_header" in text
    assert "config_header" in list_pattern_names()
    assert contains_pattern(profiles, "config_header")
