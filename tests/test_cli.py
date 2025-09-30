import json
import subprocess
import sys
from pathlib import Path


def _write_container(base: Path) -> tuple[Path, Path]:
    instructions = [
        0x01000010,
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
            "summary": "Manual override for CLI test.",
            "stack_delta": 2,
        }
    }
    manual_path.write_text(json.dumps(manual_payload, indent=2), "utf-8")

    kb_path = base / "kb.json"
    kb_path.write_text(json.dumps({"schema": 1, "opcode_modes": {}}), "utf-8")
    return kb_path


def test_cli_generates_listing(tmp_path: Path) -> None:
    adb_path, mbc_path = _write_container(tmp_path)
    kb_path = _write_knowledge(tmp_path)
    output_path = tmp_path / "out.txt"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "mbc_disasm.py"),
            str(adb_path),
            str(mbc_path),
            "--knowledge-base",
            str(kb_path),
            "--disasm-out",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    listing = output_path.read_text("utf-8")
    assert "stackΔ=+2" in listing
    assert "Manual override for CLI test." in listing
    assert "disassembly written" in result.stdout
