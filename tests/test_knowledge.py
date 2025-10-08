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


def test_call_signature_includes_fanout_helpers():
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    for operand in (0x4B76, 0x9D01, 0xDAFF):
        signature = knowledge.call_signature(operand)
        assert signature is not None
        assert any(pattern.mnemonic for pattern in signature.prelude)
