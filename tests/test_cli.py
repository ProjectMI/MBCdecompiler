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
    ast_path = tmp_path / "out.ast"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "mbc_disasm.py"),
            str(adb_path),
            str(mbc_path),
            "--knowledge-base",
            str(manual_path),
            "--ast-out",
            str(ast_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "disassembly generation completed" in result.stdout
    assert not mbc_path.with_suffix(".disasm.txt").exists()
    ir_output = mbc_path.with_suffix(".ir.txt")
    assert ir_output.exists()
    ir_text = ir_output.read_text("utf-8")
    assert "normalizer metrics" in ir_text
    assert "segment" in ir_text
    assert ast_path.exists()
    ast_text = ast_path.read_text("utf-8")
    assert "function" in ast_text
    assert "block" in ast_text
