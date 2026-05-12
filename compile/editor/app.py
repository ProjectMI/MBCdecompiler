from __future__ import annotations
import argparse
from pathlib import Path
import tkinter as tk
from typing import Any, Mapping
from decompile.decompiler import DecompileDocument
from compile.project_file import normalize_editor_state
from compile.editor.tokens import PROJECT_SUFFIX
from .editor_ui import EditorUIMixin
from .editor_io import EditorIOMixin
from .source_view import SourceViewMixin
from .edit_actions import EditActionsMixin


class MbcEditor(
    EditorUIMixin,
    EditorIOMixin,
    SourceViewMixin,
    EditActionsMixin,
    tk.Tk,
):
    def __init__(self, input_path: Path):
        super().__init__()
        self.geometry("1380x880")
        self.minsize(960, 620)

        self.mbc_path: Path | None = None
        self.project_path: Path | None = None
        self.ir: dict[str, Any] = {}
        self.editor_state: dict[str, Any] = normalize_editor_state()
        self.rename_map: dict[str, str] = {}
        self.scoped_rename_map: dict[str, str] = {}
        self.selected_offset: int | None = None

        self.project_dirty = False
        self.bytecode_dirty = False
        self.source_dirty = False
        self._loading_text = False
        self._source_raw_text = ""
        self._source_baseline_text = ""
        self._pretty_document: DecompileDocument | None = None
        self._line_offsets: dict[int, int] = {}
        self._offset_to_line: dict[int, int] = {}
        self._line_offsets_sorted: list[int] = []
        self._offsets_sorted: list[int] = []
        self._function_ranges: list[tuple[int, int, int, int, str]] = []
        self._function_items: dict[str, tuple[int, int, int, int, str]] = {}
        self._bookmarks: set[int] = set()
        self._value_patch_offsets: set[int] = set()
        self._value_patch_data_offsets: set[int] = set()

        self._project_scripts_cache: list[object] | None = None
        self._project_process_plan: object | None = None
        self._full_project_links_loaded = False
        self._pending_highlight_job: str | None = None
        self._pending_dirty_job: str | None = None
        self._pending_double_click_job: str | None = None
        self._modal_action_open = False
        self._search_hits: list[tuple[str, str]] = []
        self._search_index = -1
        self._last_pointer_index: str | None = None
        self._search_anchor_index: str | None = None

        self.instruction_by_offset: dict[int, Mapping[str, Any]] = {}
        self.instruction_offsets: list[int] = []

        self._build_widgets()
        self._load_input(input_path)


def _resolve_input_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    script_dir = Path(__file__).resolve().parent
    if value.endswith(PROJECT_SUFFIX):
        candidate = script_dir / value
        if candidate.exists():
            return candidate
    name = path.name
    if not name.lower().endswith(".mbc"):
        name += ".mbc"
    candidate = script_dir / "mbc" / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Input file not found: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the single-pane MBC AST/source editor.")
    parser.add_argument("script", nargs="?", default="mbc/entry.mbc", help="MBC path or .mbcproj.json project to open")
    args = parser.parse_args()
    path = _resolve_input_path(args.script)
    app = MbcEditor(path)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
