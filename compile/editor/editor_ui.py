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


class EditorUIMixin:
    def _build_widgets(self) -> None:
        self.title("MBC AST source editor")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open MBC", command=self.open_mbc).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Open Project", command=self.open_project).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(toolbar, text="Save Project", command=self.save_project).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Save As", command=self.save_project_as).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Export MBC", command=self.save_mbc).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Verify", command=self.verify_roundtrip).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(toolbar, text="Apply Source Edits", command=self.apply_source_edits).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Revert Draft", command=self.revert_source_draft).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Refresh Source", command=self.refresh_source_now).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(toolbar, text="Rename", command=self.rename_symbol).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Edit Literal", command=self.edit_value).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Goto loc", command=self.goto_loc_dialog).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Rules", command=self.show_edit_rules).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Load Project Links", command=self.load_project_links).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(toolbar, text="Inspector", command=self.toggle_inspector).pack(side=tk.LEFT, padx=3, pady=4)

        self.status = tk.StringVar(value="")
        ttk.Label(toolbar, textvariable=self.status).pack(side=tk.LEFT, padx=10)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(main, width=260)
        editor_area = ttk.Frame(main)
        main.add(sidebar, weight=0)
        main.add(editor_area, weight=1)

        ttk.Label(sidebar, text="Functions").pack(anchor=tk.W, padx=6, pady=(6, 2))
        filter_row = ttk.Frame(sidebar)
        filter_row.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.function_filter = tk.StringVar(value="")
        filter_entry = ttk.Entry(filter_row, textvariable=self.function_filter)
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(filter_row, text="×", width=3, command=lambda: self.function_filter.set("")).pack(side=tk.LEFT)
        self.function_filter.trace_add("write", lambda *_args: self._rebuild_function_outline())

        self.function_tree = ttk.Treeview(sidebar, columns=("line",), show="tree", height=18)
        self.function_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.function_tree.bind("<<TreeviewSelect>>", self._function_selected)
        self.function_tree.bind("<Double-1>", self._function_selected)

        ttk.Label(sidebar, text="Search").pack(anchor=tk.W, padx=6)
        search_row = ttk.Frame(sidebar)
        search_row.pack(fill=tk.X, padx=6, pady=(2, 2))
        self.search_var = tk.StringVar(value="")
        search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        self.search_entry = search_entry
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        search_entry.bind("<Return>", lambda _e: self.find_next())
        ttk.Button(search_row, text="Next", command=self.find_next).pack(side=tk.LEFT, padx=(3, 0))
        ttk.Button(search_row, text="Prev", command=self.find_prev).pack(side=tk.LEFT, padx=(3, 0))
        self.search_var.trace_add("write", lambda *_args: self._mark_search_dirty())

        bookmark_row = ttk.Frame(sidebar)
        bookmark_row.pack(fill=tk.X, padx=6, pady=(8, 4))
        ttk.Button(bookmark_row, text="★ Bookmark", command=self.toggle_bookmark).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bookmark_row, text="Next ★", command=self.next_bookmark).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        self.bookmark_list = tk.Listbox(sidebar, height=6, activestyle="dotbox")
        self.bookmark_list.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.bookmark_list.bind("<<ListboxSelect>>", self._bookmark_selected)

        editor_header = ttk.Frame(editor_area)
        editor_header.pack(side=tk.TOP, fill=tk.X)
        self.source_label = tk.StringVar(value="Editable pseudo-source")
        ttk.Label(editor_header, textvariable=self.source_label).pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Label(editor_header, text="Ctrl+Enter applies safe edits; F2 rename; F4 edit literal; Ctrl+F search").pack(side=tk.RIGHT, padx=6)

        text_frame = ttk.Frame(editor_area)
        text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        mono = tkfont.Font(family="Consolas", size=10)
        self.source = tk.Text(text_frame, wrap=tk.NONE, undo=True, maxundo=500, autoseparators=True, font=mono, padx=8, pady=6)
        self.line_numbers = TextLineNumbers(text_frame, self.source, background="#f4f4f4")
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        self.source.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self._on_y_scroll)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.source.configure(yscrollcommand=lambda first, last: self._on_text_yscroll(scroll_y, first, last))
        scroll_x = ttk.Scrollbar(editor_area, orient=tk.HORIZONTAL, command=self.source.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.source.configure(xscrollcommand=scroll_x.set)

        self.inspector_frame = ttk.Frame(editor_area)
        self.inspector_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(self.inspector_frame, text="Inspector").pack(anchor=tk.W, padx=6, pady=(4, 0))
        self.inspector = tk.Text(self.inspector_frame, height=7, wrap=tk.WORD, font=mono, padx=8, pady=4)
        self.inspector.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.inspector.configure(state=tk.DISABLED)

        self._configure_text_tags(self.source)
        self.source.bind("<<Modified>>", self._source_modified)
        self.source.bind("<ButtonRelease-1>", self._source_clicked)
        self.source.bind("<Motion>", self._source_motion)
        self.source.bind("<Double-Button-1>", self._source_double_clicked)
        self.source.bind("<KeyPress>", self._source_key_pressed, add="+")
        self.source.bind("<KeyRelease>", self._source_key_released)
        self.source.bind("<Control-s>", lambda _event: self._key(self.save_project))
        self.source.bind("<Control-S>", lambda _event: self._key(self.save_project))
        self.source.bind("<Control-Return>", lambda _event: self._key(self.apply_source_edits))
        self.source.bind("<F2>", lambda _event: self._key(self.rename_symbol))
        self.source.bind("<F4>", lambda _event: self._key(self.edit_value))
        self.source.bind("<F5>", lambda _event: self._key(self.refresh_source_now))
        self.source.bind("<Control-f>", lambda _event: self._focus_search(search_entry))
        self.source.bind("<Control-F>", lambda _event: self._focus_search(search_entry))
        self.source.bind("<Control-g>", lambda _event: self._key(self.goto_loc_dialog))
        self.source.bind("<Control-G>", lambda _event: self._key(self.goto_loc_dialog))
        self.bind_all("<Control-KeyPress>", self._control_key_pressed, add="+")
        for sequence in (
            "<Control-KeyPress-f>", "<Control-KeyPress-F>",
            "<Control-KeyPress-g>", "<Control-KeyPress-G>",
            "<Control-KeyPress-s>", "<Control-KeyPress-S>",
            "<Control-KeyPress-Cyrillic_a>", "<Control-KeyPress-Cyrillic_A>",
            "<Control-KeyPress-Cyrillic_pe>", "<Control-KeyPress-Cyrillic_PE>",
            "<Control-KeyPress-Cyrillic_yeru>", "<Control-KeyPress-Cyrillic_YERU>",
        ):
            try:
                self.bind_all(sequence, self._control_key_pressed, add="+")
            except tk.TclError:
                pass
        self.source.bind("<Button-3>", self._show_context_menu)

        self.context_menu = tk.Menu(self, tearoff=False)
        self.context_menu.add_command(label="Apply source edits", command=self.apply_source_edits)
        self.context_menu.add_command(label="Edit literal/value", command=self.edit_value)
        self.context_menu.add_command(label="Rename symbol", command=self.rename_symbol)
        self.context_menu.add_command(label="Save project to mbc_decompiled/", command=self.save_project)
        self.context_menu.add_command(label="Toggle bookmark", command=self.toggle_bookmark)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Show edit rules", command=self.show_edit_rules)
        self.context_menu.add_command(label="Copy loc", command=self.copy_current_loc)
        self.context_menu.add_command(label="Show internal IR row", command=self.show_ir_row_popup)

    def _key(self, func: object) -> str:
        try:
            func()  # type: ignore[misc]
        finally:
            return "break"

    def _focus_search(self, entry: ttk.Entry) -> str:
        self._search_anchor_index = self._current_search_anchor_index()
        self._search_index = -1
        entry.focus_set()
        entry.select_range(0, tk.END)
        return "break"

    def _is_control_down(self, event: tk.Event) -> bool:
        try:
            return bool(int(getattr(event, "state", 0)) & 0x0004)
        except Exception:
            return False

    def _shortcut_tokens(self, event: tk.Event) -> set[str]:
        tokens: set[str] = set()
        for attr in ("keysym", "char"):
            value = str(getattr(event, attr, "") or "").strip()
            if value:
                tokens.add(value.lower())
        try:
            tokens.add(f"keycode:{int(getattr(event, 'keycode'))}")
        except Exception:
            pass
        return tokens

    def _control_key_pressed(self, event: tk.Event) -> str | None:
        """Handle shortcuts that Tk misses on non-Latin keyboard layouts."""
        if not self._is_control_down(event):
            return None
        tokens = self._shortcut_tokens(event)
        # Russian layout: physical Ctrl+F -> Cyrillic 'а', Ctrl+G -> Cyrillic 'п', Ctrl+S -> Cyrillic 'ы'.
        # Keycode fallbacks cover common Windows / X11 / macOS Tk mappings.
        if tokens & {"f", "cyrillic_a", "а", "keycode:70", "keycode:41", "keycode:3"}:
            return self._focus_search(self.search_entry)
        if tokens & {"g", "cyrillic_pe", "п", "keycode:71", "keycode:42", "keycode:5"}:
            return self._key(self.goto_loc_dialog)
        if tokens & {"s", "cyrillic_yeru", "ы", "keycode:83", "keycode:39", "keycode:1"}:
            return self._key(self.save_project)
        return None

    def _source_key_pressed(self, event: tk.Event) -> str | None:
        # Tk can miss <Control-f> style bindings on Cyrillic layouts.  Handling
        # every keypress here keeps the shortcut local to the editor even if the
        # generic bind_all path is skipped by the platform.
        return self._control_key_pressed(event)

    def _show_context_menu(self, event: tk.Event) -> str:
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
        return "break"

    def _on_y_scroll(self, *args: object) -> None:
        self.source.yview(*args)
        self.line_numbers.redraw()

    def _on_text_yscroll(self, scrollbar: ttk.Scrollbar, first: str, last: str) -> None:
        scrollbar.set(first, last)
        self.line_numbers.redraw()
        self._schedule_highlight()

    def _configure_text_tags(self, widget: tk.Text) -> None:
        bold = tkfont.Font(widget, widget.cget("font"))
        bold.configure(weight="bold")
        italic = tkfont.Font(widget, widget.cget("font"))
        italic.configure(slant="italic")

        widget.tag_configure("selected_line", background="#fff2a8")
        widget.tag_configure("paired_block", background="#eef6ff")
        widget.tag_configure("changed_line", background="#fff6dc")
        widget.tag_configure("search_hit", background="#dcecff")
        widget.tag_configure("search_current", background="#ffd987")
        widget.tag_configure("bookmark", background="#f0e5ff")
        widget.tag_configure("comment", foreground="#777777", font=italic)
        widget.tag_configure("label", foreground="#1f5fbf", font=bold)
        widget.tag_configure("keyword", foreground="#6b2fa3", font=bold)
        widget.tag_configure("function_name", foreground="#0c6f6f", font=bold)
        widget.tag_configure("call", foreground="#1f6b4d")
        widget.tag_configure("variable", foreground="#8a4b08")
        widget.tag_configure("string", foreground="#9b3a2d")
        widget.tag_configure("number", foreground="#2454a6")
        widget.tag_configure("edited_symbol", background="#edf7d2")
        widget.tag_configure("edited_literal", background="#ffe8d6")

    def _update_inspector(self, offset: int | None) -> None:
        lines: list[str] = []
        if offset is None:
            lines.append("No mapped bytecode location at cursor.")
        else:
            row = self.instruction_by_offset.get(offset)
            lines.append(f"loc_{offset:08X}")
            if row is not None:
                lines.append(render_lossless_instruction(row))
            data_offset = self._data_offset_from_current_line()
            if data_offset is not None:
                lines.append(f"data[0x{data_offset:04X}] selected/near cursor")
                lines.extend(self._describe_data_at_current_line(data_offset))
            identifier = self._selected_or_current_identifier()
            if identifier is not None and self._rename_target_for_symbol(identifier) is None:
                identifier = None
            caps = capabilities_for_offset(self.ir, offset, identifier=identifier)
            lines.append("")
            lines.append("Safe actions:")
            for cap in caps:
                marker = "✓" if cap.supported else "·"
                lines.append(f"  {marker} {cap.title}: {cap.status}")
        self.inspector.configure(state=tk.NORMAL)
        self.inspector.delete("1.0", tk.END)
        self.inspector.insert("1.0", "\n".join(lines))
        self.inspector.configure(state=tk.DISABLED)

    def _describe_data_at_current_line(self, data_offset: int) -> list[str]:
        line = self.source.get(f"{self._current_line_number()}.0", f"{self._current_line_number()}.end")
        out: list[str] = []
        try:
            if "string" in line or "buf_" in line:
                capacity = self._capacity_from_line(line) or 256
                out.append(f"  string: {read_data_c_string(self.ir, data_offset, max_length=capacity)!r}")
            elif "float" in line:
                out.append(f"  float32: {read_data_scalar(self.ir, data_offset, type_id=TYPE_FLOAT)}")
            elif "char" in line and "string" not in line:
                out.append(f"  int8: {read_data_scalar(self.ir, data_offset, type_id=TYPE_CHAR)}")
            else:
                out.append(f"  int32: {read_data_scalar(self.ir, data_offset, type_id=TYPE_INT)}")
        except Exception as exc:
            out.append(f"  value read failed: {exc}")
        return out

    def toggle_inspector(self) -> None:
        if self.inspector_frame.winfo_ismapped():
            self.inspector_frame.pack_forget()
        else:
            self.inspector_frame.pack(side=tk.BOTTOM, fill=tk.X)

    def copy_current_loc(self) -> None:
        if self.selected_offset is None:
            return
        value = f"loc_{self.selected_offset:08X}"
        self.clipboard_clear()
        self.clipboard_append(value)
        self._set_status(f"Copied {value}")

    def show_ir_row_popup(self) -> None:
        if self.selected_offset is None:
            messagebox.showinfo("Internal IR row", "No bytecode location selected.")
            return
        row = self.instruction_by_offset.get(self.selected_offset)
        if row is None:
            messagebox.showinfo("Internal IR row", f"No instruction at loc_{self.selected_offset:08X}.")
            return
        messagebox.showinfo("Internal IR row", render_lossless_instruction(row))

    def show_edit_rules(self) -> None:
        lines = DEFAULT_POLICY.summary_lines()
        lines.extend([
            "",
            "Source editor mode:",
            "  - Direct typing is accepted as a draft first.",
            "  - Ctrl+Enter applies only safe line-local edits to the internal IR.",
            "  - Unsupported edits stay in the draft/project; they are not exported to MBC.",
        ])
        if self.selected_offset is not None:
            lines.extend(["", f"Selection loc_{self.selected_offset:08X}:"])
            identifier = self._selected_or_current_identifier()
            if identifier is not None and self._rename_target_for_symbol(identifier) is None:
                identifier = None
            caps = capabilities_for_offset(self.ir, self.selected_offset, identifier=identifier)
            for cap in caps:
                marker = "OK" if cap.supported else "--"
                lines.append(f"  [{marker}] {cap.title}: {cap.status}. {cap.reason}")
        messagebox.showinfo("Safe edit model", "\n".join(lines))

    def _update_title(self) -> None:
        name = self.project_path.name if self.project_path is not None else (self.mbc_path.name if self.mbc_path is not None else "<unnamed>")
        flags = []
        if self.source_dirty:
            flags.append("source draft*")
        if self.project_dirty:
            flags.append("project*")
        if self.bytecode_dirty:
            flags.append("mbc*")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        self.title(f"MBC AST source editor - {name}{suffix}")
        self.source_label.set(f"Editable pseudo-source — {name}{suffix}")

    def _set_status(self, text: str) -> None:
        flags = []
        if self.source_dirty:
            flags.append("source draft")
        if self.project_dirty:
            flags.append("project unsaved")
        if self.bytecode_dirty:
            flags.append("MBC export dirty")
        full = text if not flags else f"{text} | {', '.join(flags)}"
        self.status.set(full)
        self._update_title()
