from __future__ import annotations

"""Persistent editor project files for the MBC source editor.

An editor project is intentionally a sidecar JSON document, not an MBC file.
It stores the internal lossless IR plus UI/editor metadata such as display
renames, bookmarks, and an optional unapplied source draft.  Exporting back to
MBC remains an explicit compiler action handled by :mod:`compile.lossless_ir`.
"""

from copy import deepcopy
from pathlib import Path
import json
from typing import Any, Mapping

from compile.lossless_ir import IR_FORMAT

PROJECT_FORMAT = "mbc_editor_project_v1"

JsonDict = dict[str, Any]


def normalize_editor_state(state: Mapping[str, Any] | None = None) -> JsonDict:
    state = dict(state or {})
    renames = state.get("renames") or {}
    if not isinstance(renames, Mapping):
        renames = {}
    scoped_renames = state.get("scoped_renames") or {}
    if not isinstance(scoped_renames, Mapping):
        scoped_renames = {}
    source_cache = state.get("source_cache") or {}
    if not isinstance(source_cache, Mapping):
        source_cache = {}
    bookmarks_raw = state.get("bookmarks") or []
    bookmarks: list[int] = []
    if isinstance(bookmarks_raw, list):
        for item in bookmarks_raw:
            try:
                bookmarks.append(int(item))
            except Exception:
                continue
    value_patch_offsets_raw = state.get("value_patch_offsets") or []
    value_patch_offsets: list[int] = []
    if isinstance(value_patch_offsets_raw, list):
        for item in value_patch_offsets_raw:
            try:
                value_patch_offsets.append(int(item))
            except Exception:
                continue
    value_patch_data_offsets_raw = state.get("value_patch_data_offsets") or []
    value_patch_data_offsets: list[int] = []
    if isinstance(value_patch_data_offsets_raw, list):
        for item in value_patch_data_offsets_raw:
            try:
                value_patch_data_offsets.append(int(item))
            except Exception:
                continue
    return {
        "renames": {str(k): str(v) for k, v in renames.items() if str(k) and str(v)},
        "scoped_renames": {str(k): str(v) for k, v in scoped_renames.items() if str(k) and str(v)},
        "notes": str(state.get("notes", "")),
        "draft_source": str(state.get("draft_source", "")),
        "bookmarks": sorted(set(bookmarks)),
        "value_patch_offsets": sorted(set(value_patch_offsets)),
        "value_patch_data_offsets": sorted(set(value_patch_data_offsets)),
        "source_cache": dict(source_cache),
    }


def make_editor_project(
    ir: Mapping[str, Any],
    *,
    source_mbc_path: str | Path | None = None,
    editor_state: Mapping[str, Any] | None = None,
) -> JsonDict:
    if ir.get("format") != IR_FORMAT:
        raise ValueError(f"Unsupported lossless IR format: {ir.get('format')!r}")
    return {
        "format": PROJECT_FORMAT,
        "source_mbc_path": str(source_mbc_path) if source_mbc_path is not None else "",
        "lossless_ir": deepcopy(dict(ir)),
        "editor": normalize_editor_state(editor_state),
    }


def save_editor_project(
    path: str | Path,
    ir: Mapping[str, Any],
    *,
    source_mbc_path: str | Path | None = None,
    editor_state: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = make_editor_project(ir, source_mbc_path=source_mbc_path, editor_state=editor_state)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_editor_project(path: str | Path) -> tuple[JsonDict, JsonDict, Path | None]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != PROJECT_FORMAT:
        raise ValueError(f"Unsupported editor project format: {payload.get('format')!r}")
    ir = payload.get("lossless_ir")
    if not isinstance(ir, dict) or ir.get("format") != IR_FORMAT:
        raise ValueError("Editor project does not contain a supported lossless IR payload")
    editor = normalize_editor_state(payload.get("editor") if isinstance(payload.get("editor"), Mapping) else None)
    raw_source = str(payload.get("source_mbc_path") or "")
    source_path = Path(raw_source) if raw_source else None
    return ir, editor, source_path
