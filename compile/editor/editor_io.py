from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Iterable, Mapping

from decompile.decompiler import DecompileDocument, decompile_to_document
from decompile.linker import MbcProjectLinker
from mbc_format.common import TYPE_CHAR, TYPE_FLOAT, TYPE_INT
from mbc_format.loader import MbcProject
from compile.lossless_ir import (
    assemble_lossless_ir,
    load_mbc_as_lossless_ir,
    patch_data_c_string,
    patch_data_scalar,
    patch_fixed_integer_operand,
    patch_inline_span_text,
    patch_typed_immediate,
    read_data_c_string,
    read_data_scalar,
    render_lossless_instruction,
    script_from_lossless_ir,
    write_mbc_from_lossless_ir,
)
from compile.project_file import load_editor_project, normalize_editor_state, save_editor_project
from compile.edit_model import (
    DEFAULT_POLICY,
    capabilities_for_offset,
    concise_status,
    is_display_symbol,
    validate_identifier,
)
from compile.editor.tokens import (
    LOC_RE,
    FUNCTION_RE,
    LOCAL_HELPER_RE,
    IDENT_RE,
    VARIABLE_RE,
    RESERVED_WORDS,
    KEYWORD_RE,
    STRING_RE,
    NUMBER_RE,
    SIGNED_NUMBER_RE,
    COMMENT_RE,
    CALL_RE,
    DATA_COMMENT_RE,
    DATA_SYMBOL_RE,
    LINE_STRING_RE,
    PROJECT_SUFFIX,
    DEFAULT_DECOMPILED_DIR,
    TYPED_IMMEDIATE_OPCODES,
    FIXED_SPAN_OPCODES,
    FIXED_INTEGER_OPERAND_OPCODES,
    SCOPED_RENAME_SYMBOL_RE,
    GLOBAL_RENAME_SYMBOL_RE,
    GLOBAL_CANON_SYMBOL_RE,
)
from compile.editor.widgets import TextLineNumbers


class EditorIOMixin:
    def _load_input(self, path: Path) -> None:
        if path.suffix.lower() == ".json" and path.name.endswith(PROJECT_SUFFIX):
            self._load_project_file(path)
        else:
            self._load_mbc_file(path)

    def _load_mbc_file(self, path: Path) -> None:
        if not self._confirm_discard_draft():
            return
        managed_project = self._managed_project_path_for_source(path)
        if managed_project.exists():
            self._load_project_file(managed_project, confirm=False)
            self._set_status(f"Loaded saved project for {path.name}: {managed_project}")
            return
        self.mbc_path = path
        self.project_path = None
        self.ir = load_mbc_as_lossless_ir(path)
        self.editor_state = normalize_editor_state()
        self.rename_map = dict(self.editor_state.get("renames") or {})
        self.scoped_rename_map = dict(self.editor_state.get("scoped_renames") or {})
        self._bookmarks = set(int(x) for x in self.editor_state.get("bookmarks", []) if isinstance(x, int))
        self._value_patch_offsets = set(int(x) for x in self.editor_state.get("value_patch_offsets", []) if isinstance(x, int))
        self._value_patch_data_offsets = set(int(x) for x in self.editor_state.get("value_patch_data_offsets", []) if isinstance(x, int))
        self.selected_offset = None
        self.project_dirty = False
        self.bytecode_dirty = False
        self.source_dirty = False
        self._clear_project_context()
        self._rebuild_instruction_index()
        self._render_source_from_ir(center_selected=False)
        self._update_title()
        self._set_status(f"Loaded MBC: {path}")

    def _load_project_file(self, path: Path, *, confirm: bool = True) -> None:
        if confirm and not self._confirm_discard_draft():
            return
        ir, editor_state, source_path = load_editor_project(path)
        self.project_path = path
        self.mbc_path = source_path if source_path is not None else Path(str(ir.get("source_name") or path.with_suffix(".mbc")))
        self.ir = ir
        self.editor_state = editor_state
        self.rename_map = dict(editor_state.get("renames") or {})
        self.scoped_rename_map = dict(editor_state.get("scoped_renames") or {})
        self._bookmarks = set(int(x) for x in editor_state.get("bookmarks", []) if isinstance(x, int))
        self._value_patch_offsets = set(int(x) for x in editor_state.get("value_patch_offsets", []) if isinstance(x, int))
        self._value_patch_data_offsets = set(int(x) for x in editor_state.get("value_patch_data_offsets", []) if isinstance(x, int))
        self.selected_offset = None
        self.project_dirty = False
        self.bytecode_dirty = False
        self.source_dirty = False
        self._clear_project_context()
        self._rebuild_instruction_index()
        self._render_source_from_ir(center_selected=False)
        draft = str(editor_state.get("draft_source") or "")
        if draft:
            self._replace_source_text(draft, baseline=False)
            self.source_dirty = True
            self._mark_changed_lines()
            self._set_status("Loaded project with unapplied source draft")
        else:
            self._set_status(f"Loaded project: {path}")
        self._update_title()

    def _clear_project_context(self) -> None:
        self._project_scripts_cache = None
        self._project_process_plan = None
        self._full_project_links_loaded = False

    def _prime_project_context(self) -> None:
        self._clear_project_context()
        if self.mbc_path is None or not self.mbc_path.exists():
            return
        project = MbcProject.load_for_script(self.mbc_path)
        linker = MbcProjectLinker.from_ffprc_plan(project.scripts)
        self._project_scripts_cache = list(project.scripts)
        self._project_process_plan = linker.process_plan
        self._full_project_links_loaded = True

    def load_project_links(self) -> None:
        try:
            self._prime_project_context()
        except Exception as exc:
            messagebox.showerror("Load project links failed", str(exc))
            return
        if not self._full_project_links_loaded:
            messagebox.showinfo("Project links", "No project context was loaded for this MBC path.")
            return
        self.refresh_source_now(force=True)
        count = len(self._project_scripts_cache or [])
        self._set_status(f"Loaded project links for {count} MBC modules")

    def open_mbc(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("MBC files", "*.mbc"), ("All files", "*.*")])
        if not filename:
            return
        try:
            self._load_mbc_file(Path(filename))
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))

    def open_project(self) -> None:
        filename = filedialog.askopenfilename(
            filetypes=[("MBC editor projects", f"*{PROJECT_SUFFIX}"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filename:
            return
        try:
            self._load_project_file(Path(filename))
        except Exception as exc:
            messagebox.showerror("Open project failed", str(exc))

    def _default_project_name(self) -> str:
        source = self.mbc_path or Path(str(self.ir.get("source_name") or "script.mbc"))
        stem = source.name[:-4] if source.name.lower().endswith(".mbc") else source.stem
        return f"{stem}{PROJECT_SUFFIX}"

    def _managed_project_root_for_source(self, source: Path) -> Path:
        if source.parent.name.lower() == "mbc":
            return source.parent.parent / DEFAULT_DECOMPILED_DIR
        return source.parent / DEFAULT_DECOMPILED_DIR

    def _managed_project_path_for_source(self, source: Path) -> Path:
        stem = source.name[:-4] if source.name.lower().endswith(".mbc") else source.stem
        return self._managed_project_root_for_source(source) / stem / f"{stem}{PROJECT_SUFFIX}"

    def _managed_project_root(self) -> Path:
        if self.mbc_path is not None:
            return self._managed_project_root_for_source(self.mbc_path)
        return Path(__file__).resolve().parent / DEFAULT_DECOMPILED_DIR

    def _default_project_path(self) -> Path:
        source = self.mbc_path or Path(str(self.ir.get("source_name") or "script.mbc"))
        return self._managed_project_path_for_source(source)

    def save_project(self) -> None:
        if self.project_path is None:
            self.project_path = self._default_project_path()
        self._write_project(self.project_path)
        self._update_title()

    def save_project_as(self) -> None:
        default_path = self.project_path or self._default_project_path()
        filename = filedialog.asksaveasfilename(
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            defaultextension=PROJECT_SUFFIX,
            filetypes=[("MBC editor projects", f"*{PROJECT_SUFFIX}"), ("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return
        self.project_path = Path(filename)
        self._write_project(self.project_path)
        self._update_title()

    def _source_cache_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "text": self._source_raw_text,
            "line_offsets": {str(k): int(v) for k, v in self._line_offsets.items()},
            "function_ranges": [list(item) for item in self._function_ranges],
        }

    def _write_project(self, path: Path) -> None:
        try:
            state = {
                **self.editor_state,
                "renames": self.rename_map,
                "scoped_renames": self.scoped_rename_map,
                "bookmarks": sorted(self._bookmarks),
                "value_patch_offsets": sorted(self._value_patch_offsets),
                "value_patch_data_offsets": sorted(self._value_patch_data_offsets),
                "draft_source": self._current_source_text() if self.source_dirty else "",
                "source_cache": self._source_cache_payload(),
            }
            self.editor_state = normalize_editor_state(state)
            save_editor_project(
                path,
                self.ir,
                source_mbc_path=self.mbc_path,
                editor_state=self.editor_state,
            )
        except Exception as exc:
            messagebox.showerror("Save project failed", str(exc))
            return
        self.project_dirty = False
        self._set_status(f"Saved project: {path}")

    def save_mbc(self) -> None:
        if not self._offer_apply_before_export():
            return
        initial = (self.mbc_path.name if self.mbc_path is not None else str(self.ir.get("source_name") or "script.mbc"))
        filename = filedialog.asksaveasfilename(
            initialfile=initial,
            defaultextension=".mbc",
            filetypes=[("MBC files", "*.mbc"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            write_mbc_from_lossless_ir(self.ir, filename)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.bytecode_dirty = False
        self._set_status(f"Exported MBC: {filename}")

    def _confirm_discard_draft(self) -> bool:
        if not (self.source_dirty or self.project_dirty or self.bytecode_dirty):
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved changes",
            "There are unsaved project/source changes. Save project before continuing?",
        )
        if answer is None:
            return False
        if answer:
            self.save_project()
            if self.project_dirty:
                return False
        return True

    def _offer_apply_before_export(self) -> bool:
        if not self.source_dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Unapplied source edits",
            "The source editor contains unapplied edits. Apply safe edits before exporting MBC?\n\n"
            "Yes = apply then export. No = export current internal IR. Cancel = stop.",
        )
        if answer is None:
            return False
        if answer:
            return self.apply_source_edits(silent=False)
        return True

    def _cancel_pending_jobs(self) -> None:
        for attr in ("_pending_highlight_job", "_pending_dirty_job", "_pending_double_click_job"):
            job = getattr(self, attr, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                setattr(self, attr, None)

    def destroy(self) -> None:  # type: ignore[override]
        self._cancel_pending_jobs()
        try:
            self.unbind_all("<Control-KeyPress>")
        except Exception:
            pass
        super().destroy()

    def _on_close(self) -> None:
        if self._confirm_discard_draft():
            self.destroy()

    def verify_roundtrip(self) -> None:
        try:
            rebuilt = assemble_lossless_ir(self.ir)
            original = self.mbc_path.read_bytes() if self.mbc_path is not None and self.mbc_path.exists() else None
        except Exception as exc:
            messagebox.showerror("Verify failed", str(exc))
            return
        if original is None:
            messagebox.showinfo("Round-trip", "The current IR assembles successfully. No original MBC path is available for byte comparison.")
        elif rebuilt == original:
            messagebox.showinfo("Round-trip", "Current internal IR rebuilds byte-for-byte identical MBC.")
        else:
            messagebox.showwarning("Round-trip", "Current internal IR assembles, but differs from the source MBC.")
