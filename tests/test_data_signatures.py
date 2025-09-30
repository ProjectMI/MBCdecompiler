"""Unit tests for :mod:`mbcdisasm.analyzer.data_signatures`."""

from mbcdisasm.analyzer.data_signatures import DataClassifier
from mbcdisasm.instruction import InstructionWord


def build_word(value: int) -> InstructionWord:
    return InstructionWord(offset=0, raw=value)


def test_ascii_detector_extracts_text() -> None:
    classifier = DataClassifier()
    word = build_word(0x47657450)  # "GetP"
    info = classifier.classify(word)
    assert info is not None
    assert "ascii" in (info.category or "")
    assert info.attributes["ascii_text"] == "GetP"


def test_opcode_zero_literal_annotation() -> None:
    classifier = DataClassifier()
    word = build_word(0x00670400)
    info = classifier.classify(word)
    assert info is not None
    assert info.attributes["detector"] == "literal_opcode00"
    assert info.attributes["operand"] == 0x0400


def test_repeated_halfword_detection() -> None:
    classifier = DataClassifier()
    word = build_word(0x12341234)
    info = classifier.classify(word)
    assert info is not None
    assert info.attributes["detector"] == "repeat16"
    assert info.attributes["value"] == 0x1234


def test_classifier_ignores_regular_opcode() -> None:
    classifier = DataClassifier()
    word = build_word(0x291000F0)
    assert classifier.classify(word) is None


def test_zero_word_detector() -> None:
    classifier = DataClassifier()
    word = build_word(0x00000000)
    info = classifier.classify(word)
    assert info is not None
    assert info.attributes["detector"] == "zero_word"


def test_palindrome_detector() -> None:
    classifier = DataClassifier()
    word = build_word(0x12343412)
    info = classifier.classify(word)
    assert info is not None
    assert info.attributes["detector"] == "palindrome"


def test_sentinel_detector() -> None:
    classifier = DataClassifier()
    word = build_word(0x00EDDEED)
    info = classifier.classify(word)
    assert info is not None
    assert info.attributes["detector"] == "sentinel"
