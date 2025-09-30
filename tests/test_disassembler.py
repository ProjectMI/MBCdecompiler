import json
from pathlib import Path

from mbcdisasm import Disassembler, KnowledgeBase, MbcContainer


def _write_container(base: Path) -> tuple[Path, Path]:
    instructions = [
        0x01000010,
        0x02010020,
    ]
    data = b"".join(value.to_bytes(4, "big") for value in instructions)

    mbc_path = base / "sample.mbc"
    mbc_path.write_bytes(data)

    adb_path = base / "sample.adb"
    adb_path.write_bytes((0).to_bytes(4, "little"))

    return adb_path, mbc_path


def _write_knowledge(base: Path) -> Path:
    manual_path = base / "manual_annotations.json"
    manual_payload = {
        "test_category": {
            "opcodes": ["01:00"],
            "name": "manual_push",
            "summary": "Manual override for testing.",
            "stack_delta": 1,
            "operand_hint": "literal",
        }
    }
    manual_path.write_text(json.dumps(manual_payload, indent=2), "utf-8")

    kb_path = base / "kb.json"
    kb_path.write_text(json.dumps({"schema": 1, "opcode_modes": {}}), "utf-8")
    return kb_path


def test_listing_includes_manual_annotations(tmp_path: Path) -> None:
    adb_path, mbc_path = _write_container(tmp_path)
    kb_path = _write_knowledge(tmp_path)

    knowledge = KnowledgeBase.load(kb_path)
    container = MbcContainer.load(mbc_path, adb_path)

    listing = Disassembler(knowledge).generate_listing(container)

    assert "stackΔ=+1" in listing
    assert "Manual override for testing." in listing
