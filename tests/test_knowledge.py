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


def test_address_table_lookup(tmp_path: Path) -> None:
    manual = {"noop": {"opcodes": ["00:00"], "name": "noop"}}
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(json.dumps(manual), "utf-8")
    address_table = {"0x1234": "helper"}
    (tmp_path / "address_table.json").write_text(json.dumps(address_table), "utf-8")

    knowledge = KnowledgeBase.load(tmp_path)
    assert knowledge.call_target_symbol(0x1234) == "helper"
    assert knowledge.call_target_symbol(0x4321) is None
