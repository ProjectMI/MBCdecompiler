from __future__ import annotations

import sys
from pathlib import Path

import pytest

import mbc_lua_reconstruct


class _StubClassifier:
    def __init__(self, *_, **__):
        pass

    def classify(self, descriptor, data):  # pragma: no cover - tiny shim
        return "code" if descriptor.index == 0 else "strings"


def _write_container(base: Path) -> tuple[Path, Path]:
    segments = [
        b"\x00" * 8,
        b"alpha\x00alpha\x00",
        b"\x00" * 10 + b"beta\x00" + b"\xFF" * 5,
    ]
    mbc_path = base / "sample.mbc"
    adb_path = base / "sample.adb"
    mbc_path.write_bytes(b"".join(segments))
    offsets = []
    cursor = 0
    for chunk in segments:
        offsets.append(cursor)
        cursor += len(chunk)
    adb_path.write_bytes(b"".join(offset.to_bytes(4, "little") for offset in offsets))
    return adb_path, mbc_path


def test_cli_emits_enhanced_data_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mbcdisasm.segment_classifier.SegmentClassifier", _StubClassifier)
    monkeypatch.setattr("mbcdisasm.SegmentClassifier", _StubClassifier)
    monkeypatch.setattr(mbc_lua_reconstruct, "SegmentClassifier", _StubClassifier)

    adb_path, mbc_path = _write_container(tmp_path)
    kb_path = tmp_path / "kb.json"
    output_path = tmp_path / "out.lua"

    argv = [
        "mbc_lua_reconstruct",
        str(adb_path),
        str(mbc_path),
        "--knowledge-base",
        str(kb_path),
        "--output",
        str(output_path),
        "--data-stats",
        "--string-table",
        "--emit-data-table",
        "--data-run-threshold",
        "4",
        "--data-max-runs",
        "5",
    ]

    monkeypatch.setattr(sys, "argv", argv)
    mbc_lua_reconstruct.main()

    module_text = output_path.read_text("utf-8")
    assert "data segments:" in module_text
    assert "string table:" in module_text
    assert "data segment statistics:" in module_text
    assert "most common bytes" in module_text
    assert "entropy range" in module_text
    assert "local __data_segments = {" in module_text
    assert "longest zero run" in module_text
