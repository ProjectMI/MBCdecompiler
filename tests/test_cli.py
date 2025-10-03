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
        }
    }
    manual_path.write_text(json.dumps(manual_payload, indent=2), "utf-8")
    return manual_path


def test_cli_generates_listing(tmp_path: Path) -> None:
    adb_path, mbc_path = _write_container(tmp_path)
    manual_path = _write_knowledge(tmp_path)
    output_path = tmp_path / "out.txt"
    ir_path = tmp_path / "out.ir.json"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "mbc_disasm.py"),
            str(adb_path),
            str(mbc_path),
            "--knowledge-base",
            str(manual_path),
            "--disasm-out",
            str(output_path),
            "--ir-out",
            str(ir_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    listing = output_path.read_text("utf-8")
    assert "manual_push" in listing
    assert "Manual override for CLI test." in listing
    assert "disassembly written" in result.stdout
    assert "ir written" in result.stdout
    assert "normalizer metrics:" in result.stdout

    assert ir_path.exists()
    payload = json.loads(ir_path.read_text("utf-8"))
    assert payload["container"].endswith("sample.mbc")
    assert "metrics" in payload
    assert "segments" in payload and len(payload["segments"]) == 1
    segment_entry = payload["segments"][0]
    assert segment_entry["segment_index"] == 0
    assert "blocks" in segment_entry
    assert "metrics" in segment_entry
