"""Tests for :mod:`mbcdisasm.analyzer.string_classifier`."""

from mbcdisasm.analyzer.string_classifier import GLOBAL_STRING_CLASSIFIER


def test_classifies_function_name() -> None:
    result = GLOBAL_STRING_CLASSIFIER.classify("InitCharOwn")
    assert result.category == "function_name"
    assert "camel" in result.reason


def test_classifies_config_path() -> None:
    result = GLOBAL_STRING_CLASSIFIER.classify("params\\turnskin.cfg")
    assert result.category == "config_path"


def test_animation_state_detection() -> None:
    result = GLOBAL_STRING_CLASSIFIER.classify("WALK RUN SHOOT")
    assert result.category == "animation_state"


def test_log_message_detection() -> None:
    result = GLOBAL_STRING_CLASSIFIER.classify("ERROR: Can't open file")
    assert result.category == "log_message"
