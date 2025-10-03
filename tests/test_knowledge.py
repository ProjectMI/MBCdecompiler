from pathlib import Path

from mbcdisasm import KnowledgeBase


def test_wildcard_lookup_for_tailcall():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    info = knowledge.lookup("29:10")
    assert info is not None
    assert info.control_flow and "call" in info.control_flow.lower()


def test_wildcard_lookup_for_call_helpers():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    info = knowledge.lookup("16:84")
    assert info is not None
    assert info.control_flow and "call" in info.control_flow.lower()


def test_wildcard_lookup_for_indirect_stack_access():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    info = knowledge.lookup("69:10")
    assert info is not None
    assert info.category and "indirect" in info.category
    assert info.stack_push == 1
    assert info.stack_pop == 1


def test_wildcard_lookup_for_stack_shuffle():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    info = knowledge.lookup("66:20")
    assert info is not None
    assert info.category and "copy" in info.category
    assert info.stack_effect() == 0
