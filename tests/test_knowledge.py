import json
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


def test_address_symbol_lookup(tmp_path):
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(json.dumps({}, indent=2), "utf-8")
    address_path = tmp_path / "address_labels.json"
    address_path.write_text(json.dumps({"0x10": "helper_0010"}, indent=2), "utf-8")

    knowledge = KnowledgeBase.load(manual_path)
    assert knowledge.address_symbol(0x0010) == "helper_0010"
    assert knowledge.address_symbol(0x0020) is None
