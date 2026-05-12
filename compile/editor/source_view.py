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


class SourceViewMixin:
    def _rebuild_instruction_index(self) -> None:
        self.instruction_by_offset = {
            int(row.get("offset", -1)): row
            for row in self.ir.get("instructions") or []
            if int(row.get("offset", -1)) >= 0
        }
        self.instruction_offsets = sorted(self.instruction_by_offset)

    def _render_current_document(self) -> DecompileDocument:
        try:
            script = script_from_lossless_ir(self.ir, path=self.mbc_path)
            project_linker = self._project_linker_with_current_script(script)
            return decompile_to_document(script, project_linker=project_linker)
        except Exception as exc:
            text = f"// Pretty decompilation failed; internal lossless IR is still available.\n// {exc}\n"
            return DecompileDocument(text=text, line_offsets={}, function_ranges=[])

    def _project_linker_with_current_script(self, script: object) -> MbcProjectLinker | None:
        cached_scripts = self._project_scripts_cache
        if not cached_scripts:
            return None
        try:
            current_stem = getattr(getattr(script, "path"), "stem")
            scripts = [script if getattr(existing.path, "stem", None) == current_stem else existing for existing in cached_scripts]
            if not any(getattr(existing.path, "stem", None) == current_stem for existing in cached_scripts):
                scripts.append(script)  # type: ignore[arg-type]
            if self._project_process_plan is not None:
                return MbcProjectLinker(scripts, process_plan=self._project_process_plan, include_unplanned=True)  # type: ignore[arg-type]
            return MbcProjectLinker.from_ffprc_plan(scripts)  # type: ignore[arg-type]
        except Exception:
            return None

    def _cached_document_from_state(self) -> DecompileDocument | None:
        cache = self.editor_state.get("source_cache") if isinstance(self.editor_state, Mapping) else None
        if not isinstance(cache, Mapping) or int(cache.get("version", 0) or 0) != 1:
            return None
        text = cache.get("text")
        line_offsets_raw = cache.get("line_offsets")
        ranges_raw = cache.get("function_ranges")
        if not isinstance(text, str) or not isinstance(line_offsets_raw, Mapping) or not isinstance(ranges_raw, list):
            return None
        line_offsets: dict[int, int] = {}
        try:
            for key, value in line_offsets_raw.items():
                line_offsets[int(key)] = int(value)
            function_ranges = [
                (int(item[0]), int(item[1]), int(item[2]), int(item[3]), str(item[4]))
                for item in ranges_raw
                if isinstance(item, (list, tuple)) and len(item) == 5
            ]
        except Exception:
            return None
        return DecompileDocument(text=text, line_offsets=line_offsets, function_ranges=function_ranges)

    def _render_source_from_ir(self, *, center_selected: bool, preserve_view: bool = False, use_cache: bool = True) -> None:
        view = self.source.yview() if preserve_view else None
        insert = self.source.index(tk.INSERT) if preserve_view else None
        document = self._cached_document_from_state() if use_cache and not self.bytecode_dirty else None
        if document is None:
            document = self._render_current_document()
            self.editor_state = normalize_editor_state({**self.editor_state, "source_cache": {
                "version": 1,
                "text": document.text,
                "line_offsets": {str(k): int(v) for k, v in document.line_offsets.items()},
                "function_ranges": [list(item) for item in document.function_ranges],
            }})
        self._pretty_document = document
        self._source_raw_text = document.text
        self._line_offsets = dict(document.line_offsets)
        self._function_ranges = list(document.function_ranges)
        self._source_baseline_text = self._apply_renames(document.text)
        self._rebuild_source_navigation_maps()
        self._replace_source_text(self._source_baseline_text, baseline=True)
        self._apply_source_syntax(visible_only=True)
        self._rebuild_function_outline()
        self._refresh_bookmarks()
        self.source_dirty = False
        if view is not None:
            self.source.yview_moveto(view[0])
        if insert is not None:
            try:
                self.source.mark_set(tk.INSERT, insert)
            except tk.TclError:
                pass
        if self.selected_offset is not None:
            self._select_offset(self.selected_offset, center=center_selected)
        else:
            self._update_inspector(None)
        self._refresh_line_numbers_later()

    def _replace_source_text(self, value: str, *, baseline: bool) -> None:
        self._loading_text = True
        try:
            self.source.delete("1.0", tk.END)
            self.source.insert("1.0", value)
            self.source.edit_reset()
            self.source.edit_modified(False)
        finally:
            self._loading_text = False
        if baseline:
            self._source_baseline_text = value
        self._mark_search_dirty()
        self._refresh_line_numbers_later()

    def _replace_source_text_incremental(self, value: str, *, baseline: bool) -> None:
        """Replace only changed lines to avoid visible rename flicker."""
        old_lines = self._current_source_text().splitlines()
        new_lines = value.splitlines()
        if len(old_lines) != len(new_lines):
            self._replace_source_text(value, baseline=baseline)
            return
        self._loading_text = True
        try:
            for line_no in range(len(old_lines), 0, -1):
                old = old_lines[line_no - 1]
                new = new_lines[line_no - 1]
                if old == new:
                    continue
                self.source.delete(f"{line_no}.0", f"{line_no}.end")
                self.source.insert(f"{line_no}.0", new)
            self.source.edit_modified(False)
        finally:
            self._loading_text = False
        if baseline:
            self._source_baseline_text = value
        self._mark_search_dirty()
        self._refresh_line_numbers_later()

    def refresh_source_now(self, *, force: bool = False) -> None:
        if self.source_dirty and not force:
            answer = messagebox.askyesnocancel(
                "Refresh source",
                "The source draft has unapplied edits. Apply safe edits before refreshing?",
            )
            if answer is None:
                return
            if answer:
                if not self.apply_source_edits(silent=False):
                    return
            else:
                self.source_dirty = False
        self._render_source_from_ir(center_selected=False, preserve_view=True, use_cache=not force)
        self._set_status("Source refreshed from internal IR")

    def revert_source_draft(self) -> None:
        if not self.source_dirty:
            self._set_status("No source draft to revert")
            return
        if not messagebox.askyesno("Revert draft", "Discard unapplied source edits and reload from internal IR?"):
            return
        self.source_dirty = False
        self._replace_source_text(self._source_baseline_text, baseline=False)
        self._apply_source_syntax()
        self._highlight_renamed_symbols()
        self._mark_changed_lines()
        self._set_status("Source draft reverted")

    def _rebuild_source_navigation_maps(self) -> None:
        self._offset_to_line.clear()
        if self._line_offsets:
            for line_no, offset in sorted(self._line_offsets.items()):
                self._offset_to_line.setdefault(offset, line_no)
        else:
            self._line_offsets = {}
            current_offset: int | None = None
            program_offsets = self._program_offsets_by_name()
            for line_no, line in enumerate(self._source_baseline_text.splitlines(), start=1):
                offset: int | None = None
                loc_match = LOC_RE.search(line)
                function_match = FUNCTION_RE.search(line)
                if loc_match:
                    offset = int(loc_match.group(1), 16)
                elif function_match:
                    name = function_match.group(1).split(".")[-1]
                    helper_match = LOCAL_HELPER_RE.match(name)
                    if helper_match:
                        offset = int(helper_match.group(1), 16)
                    elif name in program_offsets:
                        offset = program_offsets[name]
                if offset is not None:
                    current_offset = offset
                    self._offset_to_line.setdefault(offset, line_no)
                if current_offset is not None:
                    self._line_offsets[line_no] = current_offset
        self._line_offsets_sorted = sorted(self._line_offsets)
        self._offsets_sorted = sorted(self._offset_to_line)

    def _program_offsets_by_name(self) -> dict[str, int]:
        names: dict[str, int] = {}
        for item in self.ir.get("programs") or []:
            raw_name = str(item.get("name", ""))
            if not raw_name:
                continue
            start = int(item.get("start", 0))
            names[raw_name] = start
            renamed = self.rename_map.get(raw_name)
            if renamed:
                names[renamed] = start
        return names

    def _rebuild_function_outline(self) -> None:
        filter_text = self.function_filter.get().strip().lower() if hasattr(self, "function_filter") else ""
        self.function_tree.delete(*self.function_tree.get_children())
        self._function_items.clear()
        source_lines = self._current_source_text().splitlines() if hasattr(self, "source") else []
        items: list[tuple[int, int, int, int, str, str]] = []
        if self._function_ranges:
            for start_line, end_line, start, end, name in self._function_ranges:
                display = name
                if 1 <= start_line <= len(source_lines):
                    match = FUNCTION_RE.search(source_lines[start_line - 1])
                    if match:
                        display = match.group(1)
                items.append((start_line, end_line, start, end, name, display))
        else:
            for line_no, line in enumerate(source_lines, start=1):
                match = FUNCTION_RE.search(line)
                if not match:
                    continue
                display = match.group(1)
                offset = self._line_offsets.get(line_no, 0)
                items.append((line_no, line_no, offset, offset, display, display))
        for start_line, end_line, start, end, canonical, display in items:
            hay = f"{display} {canonical} {start:08X}".lower()
            if filter_text and filter_text not in hay:
                continue
            label = f"{display}   @ {start:08X}"
            iid = self.function_tree.insert("", tk.END, text=label)
            self._function_items[iid] = (start_line, end_line, start, end, canonical)

    def _function_selected(self, _event: object | None = None) -> None:
        selection = self.function_tree.selection()
        if not selection:
            return
        item = self._function_items.get(selection[0])
        if item is None:
            return
        start_line, _end_line, start, _end, _name = item
        self._goto_line(start_line, center=True)
        self._select_offset(start, center=False)

    def _current_source_text(self) -> str:
        return self.source.get("1.0", "end-1c") if hasattr(self, "source") else ""

    def _current_line_number(self) -> int:
        return int(self.source.index(tk.INSERT).split(".", 1)[0])

    def _insert_line_col(self) -> tuple[int, int]:
        line_no, col_s = self.source.index(tk.INSERT).split(".", 1)
        return int(line_no), int(col_s)

    def _offset_for_line(self, line_no: int) -> int | None:
        if line_no in self._line_offsets:
            return self._line_offsets[line_no]
        pos = bisect_right(self._line_offsets_sorted, line_no) - 1
        if pos < 0:
            return None
        return self._line_offsets.get(self._line_offsets_sorted[pos])

    def _best_line_for_offset(self, offset: int) -> int | None:
        if offset in self._offset_to_line:
            return self._offset_to_line[offset]
        pos = bisect_right(self._offsets_sorted, offset) - 1
        if pos < 0:
            return None
        return self._offset_to_line.get(self._offsets_sorted[pos])

    def _function_block_for_offset(self, offset: int) -> tuple[int, int, int, int, str] | None:
        for item in self._function_ranges:
            start_line, end_line, start, end, name = item
            if start <= offset <= end:
                return item
        for item in self._function_ranges:
            start_line, end_line, start, _end, name = item
            if start == offset:
                return item
        return None

    def _select_offset(self, offset: int, *, center: bool) -> None:
        self.selected_offset = offset
        self.source.tag_remove("selected_line", "1.0", tk.END)
        self.source.tag_remove("paired_block", "1.0", tk.END)
        block = self._function_block_for_offset(offset)
        if block is not None:
            self.source.tag_add("paired_block", f"{block[0]}.0", f"{block[1]}.end")
            self.source.tag_lower("paired_block")
        line_no = self._best_line_for_offset(offset)
        if line_no is not None:
            self.source.tag_add("selected_line", f"{line_no}.0", f"{line_no}.end")
            self.source.tag_raise("selected_line")
            if center:
                self._goto_line(line_no, center=True)
        self._update_inspector(offset)
        identifier = self._selected_or_current_identifier()
        status = concise_status(capabilities_for_offset(self.ir, offset, identifier=identifier))
        self._set_status(f"Selected loc_{offset:08X}; {status}")

    def _goto_line(self, line_no: int, *, center: bool) -> None:
        total = max(1, int(self.source.index("end-1c").split(".", 1)[0]))
        if center:
            self.source.update_idletasks()
            font = tkfont.Font(self.source, self.source.cget("font"))
            visible = max(1, self.source.winfo_height() // max(1, font.metrics("linespace")))
            top_line = max(0, line_no - visible // 2)
            self.source.yview_moveto(top_line / max(total, 1))
        self.source.see(f"{line_no}.0")
        self.source.mark_set(tk.INSERT, f"{line_no}.0")
        self._refresh_line_numbers_later()

    def _source_clicked(self, _event: object) -> None:
        line_no = self._current_line_number()
        offset = self._offset_for_line(line_no)
        if offset is not None:
            self._select_offset(offset, center=False)
        else:
            self._update_inspector(None)

    def _source_motion(self, event: tk.Event) -> None:
        try:
            self._last_pointer_index = self.source.index(f"@{event.x},{event.y}")
        except Exception:
            self._last_pointer_index = None

    def _current_search_anchor_index(self) -> str:
        if self._last_pointer_index:
            try:
                # Normalize and ensure the remembered mouse position still points
                # into the source widget.
                return self.source.index(self._last_pointer_index)
            except tk.TclError:
                pass
        try:
            return self.source.index(tk.INSERT)
        except tk.TclError:
            return "1.0"

    def _token_at_index(self, index: str) -> tuple[str, str, str, str] | None:
        line_no_s, col_s = self.source.index(index).split(".", 1)
        line_no = int(line_no_s)
        col = int(col_s)
        line = self.source.get(f"{line_no}.0", f"{line_no}.end")
        for kind, pattern in (("string", LINE_STRING_RE), ("number", SIGNED_NUMBER_RE), ("identifier", IDENT_RE)):
            search_line = line if kind == "string" else line.split("//", 1)[0]
            for match in pattern.finditer(search_line):
                if match.start() <= col <= match.end():
                    return kind, match.group(0), f"{line_no}.{match.start()}", f"{line_no}.{match.end()}"
        return None

    def _source_double_clicked(self, event: tk.Event) -> str:
        if self._modal_action_open:
            return "break"
        index = self.source.index(f"@{event.x},{event.y}")
        token = self._token_at_index(index)
        if token is None:
            return "break"
        kind, text, start, end = token
        self.source.tag_remove(tk.SEL, "1.0", tk.END)
        self.source.tag_add(tk.SEL, start, end)
        self.source.mark_set(tk.INSERT, start)
        self.source.see(start)
        self._source_clicked(None)
        if self._pending_double_click_job is not None:
            try:
                self.after_cancel(self._pending_double_click_job)
            except Exception:
                pass
        if kind == "identifier":
            if self._rename_target_for_symbol(text) is None:
                self._set_status(f"Symbol {text!r} is not renameable here")
                return "break"
            self._pending_double_click_job = self.after(35, self.rename_symbol)
        elif kind in {"number", "string"}:
            self._pending_double_click_job = self.after(35, self.edit_value)
        return "break"

    def _source_key_released(self, _event: object) -> None:
        if self._loading_text:
            return
        line_no = self._current_line_number()
        offset = self._offset_for_line(line_no)
        if offset is not None:
            self.selected_offset = offset
            self._update_inspector(offset)
        self._schedule_highlight()
        self._schedule_changed_line_marking()
        self._refresh_line_numbers_later()

    def _source_modified(self, _event: object) -> None:
        if self._loading_text:
            self.source.edit_modified(False)
            return
        if self.source.edit_modified():
            self.source_dirty = True
            self.project_dirty = True
            self.source.edit_modified(False)
            self._set_status("Source draft changed; Ctrl+Enter applies safe edits")
            self._schedule_changed_line_marking()
            self._schedule_highlight()

    def _schedule_highlight(self) -> None:
        if self._pending_highlight_job is not None:
            try:
                self.after_cancel(self._pending_highlight_job)
            except Exception:
                pass
        self._pending_highlight_job = self.after(180, lambda: self._apply_source_syntax(visible_only=True))

    def _schedule_changed_line_marking(self) -> None:
        if self._pending_dirty_job is not None:
            try:
                self.after_cancel(self._pending_dirty_job)
            except Exception:
                pass
        self._pending_dirty_job = self.after(220, self._mark_changed_lines)

    def _visible_text_range(self, *, margin: int = 120) -> tuple[str, str]:
        try:
            start_line = int(self.source.index("@0,0").split(".", 1)[0])
            end_line = int(self.source.index(f"@0,{max(1, self.source.winfo_height())}").split(".", 1)[0])
        except Exception:
            return "1.0", tk.END
        start_line = max(1, start_line - margin)
        end_line = end_line + margin
        return f"{start_line}.0", f"{end_line}.end"

    def _apply_source_syntax(self, *, visible_only: bool = True) -> None:
        self._pending_highlight_job = None
        widget = self.source
        start_index, end_index = self._visible_text_range() if visible_only else ("1.0", tk.END)
        for tag in ("comment", "label", "keyword", "function_name", "call", "variable", "string", "number", "edited_symbol", "edited_literal"):
            widget.tag_remove(tag, start_index, end_index)
        text = widget.get(start_index, end_index)
        self._tag_regex(COMMENT_RE, "comment", text=text, base_index=start_index)
        self._tag_regex(STRING_RE, "string", text=text, base_index=start_index)
        self._tag_regex(LOC_RE, "label", text=text, base_index=start_index)
        self._tag_regex(KEYWORD_RE, "keyword", group=1, text=text, base_index=start_index)
        self._tag_regex(VARIABLE_RE, "variable", text=text, base_index=start_index)
        self._tag_regex(NUMBER_RE, "number", text=text, base_index=start_index)
        self._tag_regex(CALL_RE, "call", group=1, text=text, base_index=start_index)
        self._tag_regex(re.compile(r"^\s*function\s+([A-Za-z_][\w.]*)", re.MULTILINE), "function_name", group=1, text=text, base_index=start_index)
        self._highlight_renamed_symbols(start_index=start_index, end_index=end_index, text=text)
        self._highlight_value_patches(start_index=start_index, end_index=end_index)
        self._refresh_bookmarks()
        self._refresh_search_highlights()

    def _tag_regex(self, pattern: re.Pattern[str], tag: str, *, group: int = 0, text: str | None = None, base_index: str = "1.0") -> None:
        if text is None:
            text = self._current_source_text()
            base_index = "1.0"
        for match in pattern.finditer(text):
            start, end = match.span(group)
            if start >= 0 and end >= start:
                self.source.tag_add(tag, f"{base_index}+{start}c", f"{base_index}+{end}c")

    def _highlight_renamed_symbols(self, *, start_index: str = "1.0", end_index: str | object = tk.END, text: str | None = None) -> None:
        self.source.tag_remove("edited_symbol", start_index, end_index)
        global_names = {name for pair in self.rename_map.items() for name in pair if name}
        if global_names:
            pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in sorted(global_names, key=len, reverse=True)) + r")\b")
            self._tag_regex(pattern, "edited_symbol", group=1, text=text, base_index=start_index)

        try:
            visible_start = int(str(start_index).split(".", 1)[0])
            visible_end = int(str(end_index).split(".", 1)[0]) if isinstance(end_index, str) else int(self.source.index("end-1c").split(".", 1)[0])
        except Exception:
            visible_start, visible_end = 1, int(self.source.index("end-1c").split(".", 1)[0])

        for key, new_name in self.scoped_rename_map.items():
            scope, old_name = self._split_scoped_rename_key(key)
            if scope is None or not old_name:
                continue
            line_range = self._line_range_for_function_start(scope)
            if line_range is None:
                continue
            start_line, end_line = line_range
            start_line = max(start_line, visible_start)
            end_line = min(end_line, visible_end)
            if start_line > end_line:
                continue
            names = {old_name, new_name} - {""}
            pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True)) + r")\b")
            scoped_text = self.source.get(f"{start_line}.0", f"{end_line}.end")
            self._tag_regex(pattern, "edited_symbol", group=1, text=scoped_text, base_index=f"{start_line}.0")

    def _line_range_for_function_start(self, function_start: int) -> tuple[int, int] | None:
        for start_line, end_line, start, _end, _name in self._function_ranges:
            if int(start) == int(function_start):
                return start_line, end_line
        return None

    def _highlight_value_patches(self, *, start_index: str = "1.0", end_index: str | object = tk.END) -> None:
        self.source.tag_remove("edited_literal", start_index, end_index)
        if not self._value_patch_offsets and not self._value_patch_data_offsets:
            return
        try:
            visible_start = int(str(start_index).split(".", 1)[0])
            visible_end = int(str(end_index).split(".", 1)[0]) if isinstance(end_index, str) else int(self.source.index("end-1c").split(".", 1)[0])
        except Exception:
            visible_start, visible_end = 1, int(self.source.index("end-1c").split(".", 1)[0])
        for offset in sorted(self._value_patch_offsets):
            line = self._best_line_for_offset(offset)
            if line is not None and visible_start <= line <= visible_end:
                self.source.tag_add("edited_literal", f"{line}.0", f"{line}.end")
        if self._value_patch_data_offsets:
            data_hexes = {f"0x{off:04X}" for off in self._value_patch_data_offsets}
            for line_no in range(visible_start, visible_end + 1):
                line = self.source.get(f"{line_no}.0", f"{line_no}.end")
                if any(token in line for token in data_hexes):
                    self.source.tag_add("edited_literal", f"{line_no}.0", f"{line_no}.end")

    def _mark_changed_lines(self) -> None:
        self._pending_dirty_job = None
        start_index, end_index = self._visible_text_range(margin=160)
        self.source.tag_remove("changed_line", start_index, end_index)
        if not self.source_dirty:
            return
        old_lines = self._source_baseline_text.splitlines()
        new_lines = self._current_source_text().splitlines()
        try:
            start_line = int(str(start_index).split(".", 1)[0])
            end_line = int(str(end_index).split(".", 1)[0])
        except Exception:
            start_line, end_line = 1, len(new_lines)
        for line_no in range(max(1, start_line), min(end_line, len(new_lines)) + 1):
            old = old_lines[line_no - 1] if line_no - 1 < len(old_lines) else None
            new = new_lines[line_no - 1] if line_no - 1 < len(new_lines) else None
            if old != new:
                self.source.tag_add("changed_line", f"{line_no}.0", f"{line_no}.end")
        self.source.tag_lower("changed_line")

    def _mark_search_dirty(self) -> None:
        self._search_hits = []
        self._search_index = -1
        self._search_anchor_index = None
        if hasattr(self, "source"):
            self._refresh_search_highlights()

    def _refresh_search_highlights(self) -> None:
        if not hasattr(self, "source"):
            return
        self.source.tag_remove("search_hit", "1.0", tk.END)
        self.source.tag_remove("search_current", "1.0", tk.END)
        query = self.search_var.get() if hasattr(self, "search_var") else ""
        if not query:
            return
        start = "1.0"
        hits: list[tuple[str, str]] = []
        while True:
            pos = self.source.search(query, start, nocase=True, stopindex=tk.END)
            if not pos:
                break
            end = f"{pos}+{len(query)}c"
            hits.append((pos, end))
            self.source.tag_add("search_hit", pos, end)
            start = end
        self._search_hits = hits
        if hits and 0 <= self._search_index < len(hits):
            self.source.tag_add("search_current", hits[self._search_index][0], hits[self._search_index][1])

    def _search_index_from_anchor(self, anchor: str, *, forward: bool) -> int:
        try:
            insert = self.source.index(anchor)
        except tk.TclError:
            insert = self.source.index(tk.INSERT)
        if forward:
            for idx, (start, _end) in enumerate(self._search_hits):
                if self.source.compare(start, ">=", insert):
                    return idx
            return 0
        for idx in range(len(self._search_hits) - 1, -1, -1):
            start, _end = self._search_hits[idx]
            if self.source.compare(start, "<=", insert):
                return idx
        return len(self._search_hits) - 1

    def find_next(self) -> None:
        self._refresh_search_highlights()
        if not self._search_hits:
            self._set_status("No search hits")
            return
        anchor = self._current_search_anchor_index()
        if self._search_index < 0 or anchor != self._search_anchor_index:
            self._search_anchor_index = anchor
            self._search_index = self._search_index_from_anchor(anchor, forward=True)
        else:
            self._search_index = (self._search_index + 1) % len(self._search_hits)
        self._jump_to_search_hit()

    def find_prev(self) -> None:
        self._refresh_search_highlights()
        if not self._search_hits:
            self._set_status("No search hits")
            return
        anchor = self._current_search_anchor_index()
        if self._search_index < 0 or anchor != self._search_anchor_index:
            self._search_anchor_index = anchor
            self._search_index = self._search_index_from_anchor(anchor, forward=False)
        else:
            self._search_index = (self._search_index - 1) % len(self._search_hits)
        self._jump_to_search_hit()

    def _jump_to_search_hit(self) -> None:
        self.source.tag_remove("search_current", "1.0", tk.END)
        start, end = self._search_hits[self._search_index]
        self.source.tag_add("search_current", start, end)
        self.source.mark_set(tk.INSERT, start)
        self.source.see(start)
        self._search_anchor_index = self._current_search_anchor_index()
        self._source_clicked(None)
        self._set_status(f"Search hit {self._search_index + 1}/{len(self._search_hits)}")

    def _refresh_line_numbers_later(self) -> None:
        if hasattr(self, "line_numbers"):
            self.after_idle(self.line_numbers.redraw)

    def toggle_bookmark(self) -> None:
        if self.selected_offset is None:
            line_no = self._current_line_number()
            self.selected_offset = self._offset_for_line(line_no)
        if self.selected_offset is None:
            self._set_status("No loc selected for bookmark")
            return
        if self.selected_offset in self._bookmarks:
            self._bookmarks.remove(self.selected_offset)
            self._set_status(f"Removed bookmark loc_{self.selected_offset:08X}")
        else:
            self._bookmarks.add(self.selected_offset)
            self._set_status(f"Bookmarked loc_{self.selected_offset:08X}")
        self.project_dirty = True
        self._refresh_bookmarks()

    def _refresh_bookmarks(self) -> None:
        self.source.tag_remove("bookmark", "1.0", tk.END)
        self.bookmark_list.delete(0, tk.END)
        for offset in sorted(self._bookmarks):
            line = self._best_line_for_offset(offset)
            if line is not None:
                self.source.tag_add("bookmark", f"{line}.0", f"{line}.end")
                self.bookmark_list.insert(tk.END, f"loc_{offset:08X}  line {line}")
            else:
                self.bookmark_list.insert(tk.END, f"loc_{offset:08X}")
        self.source.tag_lower("bookmark")

    def next_bookmark(self) -> None:
        if not self._bookmarks:
            self._set_status("No bookmarks")
            return
        current = self.selected_offset if self.selected_offset is not None else -1
        ordered = sorted(self._bookmarks)
        pos = bisect_right(ordered, current)
        offset = ordered[pos if pos < len(ordered) else 0]
        self._select_offset(offset, center=True)

    def _bookmark_selected(self, _event: object | None = None) -> None:
        selection = self.bookmark_list.curselection()
        if not selection:
            return
        ordered = sorted(self._bookmarks)
        idx = selection[0]
        if 0 <= idx < len(ordered):
            self._select_offset(ordered[idx], center=True)

    def goto_loc_dialog(self) -> None:
        initial = f"loc_{self.selected_offset:08X}" if self.selected_offset is not None else "loc_00000000"
        value = simpledialog.askstring("Goto loc", "Code offset / loc label:", initialvalue=initial)
        if value is None:
            return
        value = value.strip()
        match = LOC_RE.search(value)
        try:
            offset = int(match.group(1), 16) if match else int(value, 0)
        except ValueError:
            messagebox.showerror("Goto loc", f"Not a valid offset: {value}")
            return
        self._select_offset(offset, center=True)
