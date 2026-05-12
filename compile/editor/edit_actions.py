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


@dataclass(frozen=True)
class PlannedTextEdit:
    line_no: int
    kind: str
    description: str
    offset: int | None = None
    data_offset: int | None = None
    old_text: str | None = None
    new_text: str | None = None
    aux: Mapping[str, Any] | None = None


class EditActionsMixin:
    def _apply_renames(self, text: str) -> str:
        """Apply display renames.

        Global names are replaced everywhere.  Local/argument names are replaced
        only inside the function range where the rename was created, so arg0 in
        one function no longer renames arg0 across the whole module.
        """
        global_renames = {old: new for old, new in self.rename_map.items() if old and new and old != new}
        global_pattern = self._rename_pattern(global_renames)
        scoped_by_function: dict[int, dict[str, str]] = {}
        for key, new_name in self.scoped_rename_map.items():
            scope, old_name = self._split_scoped_rename_key(key)
            if scope is None or not old_name or not new_name or old_name == new_name:
                continue
            scoped_by_function.setdefault(scope, {})[old_name] = new_name

        out: list[str] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if global_pattern is not None:
                line = global_pattern.sub(lambda match: global_renames.get(match.group(1), match.group(1)), line)
            scope = self._function_start_for_line(line_no)
            scoped = scoped_by_function.get(scope or -1)
            if scoped:
                pattern = self._rename_pattern(scoped)
                if pattern is not None:
                    line = pattern.sub(lambda match, scoped=scoped: scoped.get(match.group(1), match.group(1)), line)
            out.append(line)
        return "\n".join(out) + ("\n" if text.endswith("\n") else "")

    def _rename_pattern(self, renames: Mapping[str, str]) -> re.Pattern[str] | None:
        keys = [key for key in renames if key]
        if not keys:
            return None
        return re.compile(r"\b(" + "|".join(re.escape(key) for key in sorted(keys, key=len, reverse=True)) + r")\b")

    def _function_start_for_line(self, line_no: int) -> int | None:
        for start_line, end_line, start, _end, _name in self._function_ranges:
            if start_line <= line_no <= end_line:
                return int(start)
        return None

    def _scope_key(self, function_start: int | None, symbol: str) -> str:
        return f"{int(function_start or 0):08X}:{symbol}"

    def _split_scoped_rename_key(self, key: str) -> tuple[int | None, str]:
        if ":" not in key:
            return None, key
        scope_s, symbol = key.split(":", 1)
        try:
            return int(scope_s, 16), symbol
        except ValueError:
            return None, symbol

    def _is_scoped_rename_symbol(self, name: str) -> bool:
        return bool(SCOPED_RENAME_SYMBOL_RE.fullmatch(name))

    def apply_source_edits(self, *, silent: bool = False) -> bool:
        current = self._current_source_text()
        baseline = self._source_baseline_text
        if current == baseline:
            self.source_dirty = False
            self._mark_changed_lines()
            if not silent:
                self._set_status("No source edits to apply")
            return True

        try:
            plans = self._plan_source_edits(baseline, current)
        except Exception as exc:
            if not silent:
                messagebox.showerror("Apply source edits", str(exc))
            return False

        if not plans:
            self.source_dirty = False
            self._replace_source_text(baseline, baseline=False)
            if not silent:
                self._set_status("Only cosmetic source edits were discarded")
            return True

        new_ir = deepcopy(self.ir)
        new_renames = dict(self.rename_map)
        new_scoped_renames = dict(self.scoped_rename_map)
        applied: list[str] = []
        try:
            for plan in plans:
                self._apply_text_edit_plan(plan, new_ir, new_renames, new_scoped_renames)
                applied.append(f"line {plan.line_no}: {plan.description}")
        except Exception as exc:
            if not silent:
                messagebox.showerror("Apply source edits failed", str(exc))
            return False

        self.ir = new_ir
        self.rename_map = new_renames
        self.scoped_rename_map = new_scoped_renames
        self._rebuild_instruction_index()
        self.project_dirty = True
        if any(plan.kind not in {"rename", "cosmetic"} for plan in plans):
            self.bytecode_dirty = True
        for plan in plans:
            if plan.kind in {"number", "string", "fixed_integer"} and plan.offset is not None:
                self._value_patch_offsets.add(int(plan.offset))
            if plan.kind in {"data_number", "data_string"} and plan.data_offset is not None:
                self._value_patch_data_offsets.add(int(plan.data_offset))
        self.source_dirty = False
        selected = self.selected_offset
        self._render_source_from_ir(center_selected=False, preserve_view=True)
        if selected is not None:
            self.selected_offset = selected
            self._select_offset(selected, center=False)
        msg = "Applied source edits: " + "; ".join(applied[:4])
        if len(applied) > 4:
            msg += f"; +{len(applied) - 4} more"
        self._set_status(msg)
        return True

    def _plan_source_edits(self, baseline: str, current: str) -> list[PlannedTextEdit]:
        old_lines = baseline.splitlines()
        new_lines = current.splitlines()
        if len(old_lines) != len(new_lines):
            raise ValueError(
                "Line insert/delete is not supported yet. Keep edits line-local: rename symbols or replace existing numbers/strings."
            )
        plans: list[PlannedTextEdit] = []
        unsupported: list[str] = []
        for idx, (old, new) in enumerate(zip(old_lines, new_lines, strict=True), start=1):
            if old == new:
                continue
            if old.strip() == new.strip():
                plans.append(PlannedTextEdit(idx, "cosmetic", "cosmetic whitespace ignored"))
                continue
            plan = self._classify_line_edit(idx, old, new)
            if plan is None:
                unsupported.append(f"line {idx}: {old.strip()}  ->  {new.strip()}")
            else:
                plans.append(plan)
        if unsupported:
            preview = "\n".join(unsupported[:10])
            extra = "" if len(unsupported) <= 10 else f"\n... and {len(unsupported) - 10} more"
            raise ValueError(
                "Unsupported source edits detected. Current compiler accepts only line-local symbol renames, "
                "single number replacements, and single string replacements.\n\n" + preview + extra
            )
        return [plan for plan in plans if plan.kind != "cosmetic"]

    def _classify_line_edit(self, line_no: int, old: str, new: str) -> PlannedTextEdit | None:
        old_code = old.split("//", 1)[0]
        new_code = new.split("//", 1)[0]
        data_offset = self._data_offset_from_line(old) or self._data_offset_from_line(new)
        offset = self._offset_for_line(line_no)

        string_change = self._single_token_change(old_code, new_code, LINE_STRING_RE)
        if string_change is not None:
            old_s, new_s = string_change
            if data_offset is not None and self._looks_like_data_declaration(old):
                return PlannedTextEdit(line_no, "data_string", f"data string {old_s} -> {new_s}", data_offset=data_offset, old_text=old_s, new_text=new_s, aux={"line": old})
            if offset is not None:
                return PlannedTextEdit(line_no, "string", f"string {old_s} -> {new_s}", offset=offset, old_text=old_s, new_text=new_s)

        number_change = self._single_token_change(old_code, new_code, SIGNED_NUMBER_RE)
        if number_change is not None:
            old_n, new_n = number_change
            if data_offset is not None and self._looks_like_data_declaration(old):
                return PlannedTextEdit(line_no, "data_number", f"data value {old_n} -> {new_n}", data_offset=data_offset, old_text=old_n, new_text=new_n, aux={"line": old})
            if offset is not None:
                return PlannedTextEdit(line_no, "number", f"number {old_n} -> {new_n}", offset=offset, old_text=old_n, new_text=new_n)

        ident_change = self._single_token_change(old_code, new_code, IDENT_RE)
        if ident_change is not None:
            old_name, new_name = ident_change
            target = self._rename_target_for_line_symbol(line_no, old_name)
            if target is not None and validate_identifier(new_name) and new_name not in RESERVED_WORDS:
                return PlannedTextEdit(line_no, "rename", f"rename {old_name} -> {new_name}", old_text=old_name, new_text=new_name, aux={"target": target})
        return None

    def _single_token_change(self, old: str, new: str, pattern: re.Pattern[str]) -> tuple[str, str] | None:
        old_parts = self._tokenize_for_change(old, pattern)
        new_parts = self._tokenize_for_change(new, pattern)
        if len(old_parts) != len(new_parts):
            return None
        changed: list[tuple[str, str]] = []
        for (old_kind, old_text), (new_kind, new_text) in zip(old_parts, new_parts, strict=True):
            if old_kind != new_kind:
                return None
            if old_kind == "text":
                if old_text != new_text:
                    return None
            elif old_text != new_text:
                changed.append((old_text, new_text))
        if len(changed) == 1:
            return changed[0]
        return None

    def _tokenize_for_change(self, text: str, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        pos = 0
        for match in pattern.finditer(text):
            if match.start() > pos:
                parts.append(("text", text[pos:match.start()]))
            parts.append(("token", match.group(0)))
            pos = match.end()
        if pos < len(text):
            parts.append(("text", text[pos:]))
        return parts

    def _apply_text_edit_plan(self, plan: PlannedTextEdit, ir: dict[str, Any], renames: dict[str, str], scoped_renames: dict[str, str]) -> None:
        if plan.kind == "rename":
            old = str(plan.old_text or "")
            new = str(plan.new_text or "")
            target = (plan.aux or {}).get("target") if isinstance(plan.aux, Mapping) else None
            if isinstance(target, (list, tuple)) and len(target) == 3:
                scope_kind, key, canonical = str(target[0]), str(target[1]), str(target[2])
            else:
                target_now = self._rename_target_for_line_symbol(plan.line_no, old)
                if target_now is None:
                    raise ValueError(f"line {plan.line_no}: {old!r} is not renameable here")
                scope_kind, key, canonical = target_now
            if not validate_identifier(new) or new in RESERVED_WORDS:
                raise ValueError(f"line {plan.line_no}: invalid identifier {new!r}")
            conflict = self._rename_conflict_message(
                scope_kind=scope_kind,
                key=key,
                canonical=canonical,
                new_name=new,
                renames=renames,
                scoped_renames=scoped_renames,
            )
            if conflict:
                raise ValueError(f"line {plan.line_no}: {conflict}")
            if scope_kind == "scoped":
                if new == canonical:
                    scoped_renames.pop(key, None)
                else:
                    scoped_renames[key] = new
            else:
                if new == canonical:
                    renames.pop(key, None)
                else:
                    renames[key] = new
            return

        if plan.kind == "number":
            if plan.offset is None or plan.old_text is None or plan.new_text is None:
                raise ValueError(f"line {plan.line_no}: incomplete numeric edit")
            old_value = self._parse_number(plan.old_text)
            new_value = self._parse_number(plan.new_text)
            candidates = self._nearby_immediate_candidates(plan.offset, old_value)
            if candidates:
                patch_typed_immediate(ir, candidates[0][0], new_value)
                return
            fixed_candidates = self._nearby_fixed_integer_candidates(plan.offset, old_value)
            if fixed_candidates:
                if isinstance(new_value, float) and not float(new_value).is_integer():
                    raise ValueError(f"line {plan.line_no}: fixed integer operand cannot store non-integer value {new_value!r}")
                patch_fixed_integer_operand(ir, fixed_candidates[0][0], int(new_value))
                return
            raise ValueError(f"line {plan.line_no}: no matching fixed-width numeric operand near loc_{plan.offset:08X}")
            return

        if plan.kind == "string":
            if plan.offset is None or plan.old_text is None or plan.new_text is None:
                raise ValueError(f"line {plan.line_no}: incomplete string edit")
            old_value = self._unquote_string_literal(plan.old_text)
            new_value = self._unquote_string_literal(plan.new_text)
            candidates = self._nearby_span_candidates(plan.offset, old_value)
            if not candidates:
                raise ValueError(f"line {plan.line_no}: no matching fixed-capacity string/span near loc_{plan.offset:08X}")
            patch_inline_span_text(ir, candidates[0][0], new_value)
            return

        if plan.kind == "data_number":
            if plan.data_offset is None or plan.new_text is None:
                raise ValueError(f"line {plan.line_no}: incomplete data numeric edit")
            line = str((plan.aux or {}).get("line", ""))
            type_id = self._type_id_from_line(line)
            value = float(plan.new_text) if type_id == TYPE_FLOAT else int(plan.new_text, 0)
            patch_data_scalar(ir, plan.data_offset, value, type_id=type_id)
            return

        if plan.kind == "data_string":
            if plan.data_offset is None or plan.new_text is None:
                raise ValueError(f"line {plan.line_no}: incomplete data string edit")
            line = str((plan.aux or {}).get("line", ""))
            capacity = self._capacity_from_line(line)
            if capacity is None:
                raise ValueError(f"line {plan.line_no}: cannot infer fixed capacity for data string")
            patch_data_c_string(ir, plan.data_offset, self._unquote_string_literal(plan.new_text), max_length=capacity)
            return

        raise ValueError(f"line {plan.line_no}: unsupported planned edit kind {plan.kind}")

    def _canonical_symbol_for_display(self, display: str, renames: Mapping[str, str]) -> str:
        for old, new in renames.items():
            if new == display:
                return old
        return display

    def _program_names(self) -> set[str]:
        return {str(item.get("name", "")) for item in self.ir.get("programs") or [] if str(item.get("name", ""))}

    def _global_canonical_symbols(self) -> set[str]:
        names = set(self._program_names())
        names.update(GLOBAL_CANON_SYMBOL_RE.findall(self._source_raw_text))
        names.update(key for key in self.rename_map if key)
        return {name for name in names if name}

    def _scoped_canonical_symbols_for_scope(self, function_start: int) -> set[str]:
        line_range = self._line_range_for_function_start(function_start)
        if line_range is None:
            return set()
        start_line, end_line = line_range
        lines = self._source_raw_text.splitlines()
        text = "\n".join(lines[start_line - 1:end_line]) if lines else ""
        names = set(SCOPED_RENAME_SYMBOL_RE.findall(text))
        for key in self.scoped_rename_map:
            scope, old_name = self._split_scoped_rename_key(key)
            if scope == function_start and old_name:
                names.add(old_name)
        return {name for name in names if name}

    def _rename_conflict_message(
        self,
        *,
        scope_kind: str,
        key: str,
        canonical: str,
        new_name: str,
        renames: Mapping[str, str] | None = None,
        scoped_renames: Mapping[str, str] | None = None,
    ) -> str | None:
        renames = renames if renames is not None else self.rename_map
        scoped_renames = scoped_renames if scoped_renames is not None else self.scoped_rename_map
        if new_name == canonical:
            return None

        global_displays: dict[str, str] = {}
        for canon in self._global_canonical_symbols():
            global_displays[canon] = str(renames.get(canon, canon))

        if scope_kind == "global":
            for other_key, display in global_displays.items():
                if other_key != key and display == new_name:
                    return f"Name {new_name!r} already exists in global scope as {other_key!r}."
            # A global name colliding with an explicit local rename is confusing
            # in the single-pane editor, so block it even though bytecode would
            # technically survive.
            for other_key, display in scoped_renames.items():
                if display == new_name:
                    return f"Name {new_name!r} already exists as a local display name ({other_key})."
            return None

        scope, _old = self._split_scoped_rename_key(key)
        if scope is None:
            return None
        for canon in self._scoped_canonical_symbols_for_scope(scope):
            other_key = self._scope_key(scope, canon)
            display = str(scoped_renames.get(other_key, canon))
            if other_key != key and display == new_name:
                return f"Name {new_name!r} already exists in this function as {canon!r}."
        for other_key, display in global_displays.items():
            if display == new_name:
                return f"Name {new_name!r} already exists in global scope as {other_key!r}."
        return None

    def _symbol_is_function_header_name(self, name: str) -> bool:
        line_no, col = self._insert_line_col()
        line = self.source.get(f"{line_no}.0", f"{line_no}.end")
        match = FUNCTION_RE.search(line)
        if not match:
            return False
        start, end = match.span(1)
        return match.group(1) == name and start <= col <= end

    def _rename_target_for_line_symbol(self, line_no: int, symbol: str) -> tuple[str, str, str] | None:
        """Return (scope_kind, key, canonical_symbol) for a safe display rename on a specific line."""
        if not symbol or not validate_identifier(symbol) or symbol in RESERVED_WORDS:
            return None
        function_start = self._function_start_for_line(line_no)

        if function_start is not None:
            for key, new_name in self.scoped_rename_map.items():
                scope, old_name = self._split_scoped_rename_key(key)
                if scope == function_start and new_name == symbol and old_name:
                    return "scoped", key, old_name

        for old_name, new_name in self.rename_map.items():
            if new_name == symbol:
                return "global", old_name, old_name

        if self._is_scoped_rename_symbol(symbol):
            if function_start is None:
                return None
            return "scoped", self._scope_key(function_start, symbol), symbol

        if GLOBAL_RENAME_SYMBOL_RE.fullmatch(symbol):
            return "global", symbol, symbol

        line = self.source.get(f"{line_no}.0", f"{line_no}.end") if hasattr(self, "source") else ""
        header_match = FUNCTION_RE.search(line)
        if header_match and header_match.group(1) == symbol:
            canonical = self._canonical_symbol_for_display(symbol, self.rename_map)
            program_names = self._program_names()
            if canonical in program_names or symbol in program_names or LOCAL_HELPER_RE.match(canonical):
                return "global", canonical, canonical
        return None

    def _rename_target_for_symbol(self, symbol: str) -> tuple[str, str, str] | None:
        return self._rename_target_for_line_symbol(self._current_line_number(), symbol)

    def _is_renameable_symbol(self, name: str) -> bool:
        return self._rename_target_for_symbol(name) is not None

    def rename_symbol(self) -> None:
        if self._modal_action_open:
            return
        symbol = self._selected_or_current_identifier()
        if not symbol:
            self._set_status("No renameable symbol at cursor")
            return
        target = self._rename_target_for_symbol(symbol)
        if target is None:
            self._set_status(f"Symbol {symbol!r} is not renameable here")
            return
        scope_kind, key, canonical = target
        current_display = self.scoped_rename_map.get(key, symbol) if scope_kind == "scoped" else self.rename_map.get(key, symbol)
        self._modal_action_open = True
        try:
            new_name = simpledialog.askstring("Rename symbol", f"New display name for {symbol}:", initialvalue=current_display)
        finally:
            self._modal_action_open = False
        if new_name is None:
            return
        new_name = new_name.strip()
        if not validate_identifier(new_name) or new_name in RESERVED_WORDS:
            messagebox.showwarning("Rename symbol", f"Not a valid identifier: {new_name}")
            return
        conflict = self._rename_conflict_message(scope_kind=scope_kind, key=key, canonical=canonical, new_name=new_name)
        if conflict:
            messagebox.showwarning("Rename symbol", conflict)
            return
        if scope_kind == "scoped":
            if new_name == canonical:
                self.scoped_rename_map.pop(key, None)
            else:
                self.scoped_rename_map[key] = new_name
        else:
            if new_name == canonical:
                self.rename_map.pop(key, None)
            else:
                self.rename_map[key] = new_name

        view = self.source.yview()
        insert = self.source.index(tk.INSERT)
        selected = self.selected_offset
        self.project_dirty = True
        new_baseline = self._apply_renames(self._source_raw_text)
        self._replace_source_text_incremental(new_baseline, baseline=True)
        self._apply_source_syntax(visible_only=True)
        self._rebuild_function_outline()
        self.source.yview_moveto(view[0])
        try:
            self.source.mark_set(tk.INSERT, insert)
        except tk.TclError:
            pass
        if selected is not None:
            self.selected_offset = selected
            self._select_offset(selected, center=False)
        self._set_status(f"Renamed {scope_kind} display symbol {canonical} -> {new_name}")

    def edit_value(self) -> None:
        if self._modal_action_open:
            return
        self._modal_action_open = True
        try:
            self._edit_value_impl()
        finally:
            self._modal_action_open = False

    def _edit_value_impl(self) -> None:
        line_no = self._current_line_number()
        line = self.source.get(f"{line_no}.0", f"{line_no}.end")
        offset = self._offset_for_line(line_no) or self.selected_offset
        data_offset = self._data_offset_from_line(line)

        # Exact token under cursor/selection wins.  This prevents double-clicking
        # buf_XXXX/arr_XXXX identifiers from accidentally editing the whole data
        # declaration or some unrelated literal on the same line.
        string_literal = self._selected_or_current_string()
        if string_literal is not None and offset is not None and self._edit_string_literal_near(offset, string_literal):
            return

        number_literal = self._selected_or_current_number()
        if number_literal is not None and offset is not None and self._edit_numeric_literal_near(offset, number_literal):
            return

        if data_offset is not None and self._looks_like_data_declaration(line):
            self._edit_data_value(data_offset, line)
            return

        if offset is None:
            self._set_status("No editable literal/value at cursor")
            return
        row = self.instruction_by_offset.get(offset)
        if row is not None and int(row.get("opcode", -1)) in TYPED_IMMEDIATE_OPCODES:
            self._edit_typed_immediate_row(offset, row)
            return
        if row is not None and int(row.get("opcode", -1)) in FIXED_INTEGER_OPERAND_OPCODES:
            self._edit_fixed_integer_row(offset, row)
            return
        if row is not None and int(row.get("opcode", -1)) in FIXED_SPAN_OPCODES:
            self._edit_inline_span_row(offset, row)
            return
        self._set_status("No safe value edit at cursor; select an exact number/string or a data declaration")

    def _edit_typed_immediate_row(self, offset: int, row: Mapping[str, Any], *, initial_override: str | None = None) -> bool:
        operands = row.get("operands") or {}
        initial = initial_override if initial_override is not None else self._current_immediate_text(row)
        type_name = str(operands.get("type_name", ""))
        prompt = f"New value for loc_{offset:08X}"
        if type_name:
            prompt += f" ({type_name})"
        value_text = simpledialog.askstring("Edit literal", prompt + ":", initialvalue=initial)
        if value_text is None:
            return False
        value_text = value_text.strip()
        try:
            value = float(value_text) if self._row_is_float(row) else int(value_text, 0)
            patch_typed_immediate(self.ir, offset, value)
        except Exception as exc:
            messagebox.showerror("Edit literal failed", str(exc))
            return False
        self._after_ir_patch(offset, f"Patched immediate loc_{offset:08X}")
        return True

    def _edit_inline_span_row(self, offset: int, row: Mapping[str, Any], *, initial_override: str | None = None) -> bool:
        operands = row.get("operands") or {}
        data_offset = operands.get("data_offset")
        length = operands.get("length")
        if not isinstance(data_offset, int) or not isinstance(length, int) or length <= 0:
            messagebox.showinfo("Edit literal", "This span row does not expose a fixed data offset and capacity.")
            return False
        try:
            initial = initial_override if initial_override is not None else read_data_c_string(self.ir, data_offset, max_length=length)
        except Exception:
            initial = initial_override or ""
        value = simpledialog.askstring(
            "Edit string/span data",
            f"New text for data[0x{data_offset:04X}] (capacity {length} bytes including NUL):",
            initialvalue=initial,
        )
        if value is None:
            return False
        try:
            patch_inline_span_text(self.ir, offset, value)
        except Exception as exc:
            messagebox.showerror("Edit string failed", str(exc))
            return False
        self._after_ir_patch(offset, f"Patched fixed span at loc_{offset:08X}")
        return True

    def _edit_fixed_integer_row(self, offset: int, row: Mapping[str, Any], *, initial_override: str | None = None) -> bool:
        operands = row.get("operands") or {}
        initial = initial_override
        if initial is None:
            if "program_index" in operands:
                initial = str(operands.get("program_index"))
            else:
                initial = str(operands.get("value", operands.get("data_offset", "0")))
        value_text = simpledialog.askstring(
            "Edit fixed integer operand",
            f"New same-width integer value for loc_{offset:08X} ({row.get('mnemonic', '?')}):",
            initialvalue=initial,
        )
        if value_text is None:
            return False
        try:
            value = int(value_text.strip(), 0)
            patch_fixed_integer_operand(self.ir, offset, value)
        except Exception as exc:
            messagebox.showerror("Edit fixed integer failed", str(exc))
            return False
        self._after_ir_patch(offset, f"Patched fixed integer operand loc_{offset:08X}")
        return True

    def _edit_data_value(self, data_offset: int, line: str) -> bool:
        is_string = "string" in line or "buf_" in line or self._selected_or_current_string() is not None
        type_id = self._type_id_from_line(line)
        if is_string:
            capacity = self._capacity_from_line(line)
            if capacity is None:
                capacity_text = simpledialog.askstring("Edit data string", f"Capacity for data[0x{data_offset:04X}] in bytes:", initialvalue="256")
                if capacity_text is None:
                    return False
                try:
                    capacity = int(capacity_text, 0)
                except ValueError:
                    messagebox.showerror("Edit data string", "Capacity must be an integer.")
                    return False
            try:
                initial = read_data_c_string(self.ir, data_offset, max_length=capacity)
            except Exception:
                initial = ""
            value = simpledialog.askstring(
                "Edit data string",
                f"New text for data[0x{data_offset:04X}] (capacity {capacity} bytes including NUL):",
                initialvalue=initial,
            )
            if value is None:
                return False
            try:
                patch_data_c_string(self.ir, data_offset, value, max_length=capacity)
            except Exception as exc:
                messagebox.showerror("Edit data string failed", str(exc))
                return False
            self._after_data_patch(f"Patched data string at 0x{data_offset:04X}", data_offset=data_offset)
            return True

        try:
            initial = str(read_data_scalar(self.ir, data_offset, type_id=type_id))
        except Exception:
            initial = "0"
        label = "float32" if type_id == TYPE_FLOAT else ("int8" if type_id == TYPE_CHAR else "int32")
        value_text = simpledialog.askstring("Edit data value", f"New {label} value for data[0x{data_offset:04X}]:", initialvalue=initial)
        if value_text is None:
            return False
        try:
            value = float(value_text) if type_id == TYPE_FLOAT else int(value_text, 0)
            patch_data_scalar(self.ir, data_offset, value, type_id=type_id)
        except Exception as exc:
            messagebox.showerror("Edit data value failed", str(exc))
            return False
        self._after_data_patch(f"Patched data value at 0x{data_offset:04X}", data_offset=data_offset)
        return True

    def _edit_numeric_literal_near(self, offset: int, literal: str) -> bool:
        try:
            wanted = self._parse_number(literal)
        except ValueError:
            return False
        candidates = self._nearby_immediate_candidates(offset, wanted)
        if candidates:
            candidate_offset, row = candidates[0]
            return self._edit_typed_immediate_row(candidate_offset, row, initial_override=literal)
        fixed_candidates = self._nearby_fixed_integer_candidates(offset, wanted)
        if fixed_candidates:
            candidate_offset, row = fixed_candidates[0]
            return self._edit_fixed_integer_row(candidate_offset, row, initial_override=literal)
        return False

    def _edit_string_literal_near(self, offset: int, literal: str) -> bool:
        unquoted = self._unquote_string_literal(literal)
        candidates = self._nearby_span_candidates(offset, unquoted)
        if not candidates:
            # If the literal was already edited once, the pretty string can be
            # escaped/rendered slightly differently.  Fall back to a unique near
            # fixed span so a previously patched string remains editable.
            candidates = self._nearby_span_candidates(offset, None)
            if len(candidates) != 1:
                return False
        candidate_offset, row = candidates[0]
        return self._edit_inline_span_row(candidate_offset, row, initial_override=unquoted)

    def _after_ir_patch(self, offset: int, reason: str) -> None:
        self._rebuild_instruction_index()
        self.project_dirty = True
        self.bytecode_dirty = True
        self.source_dirty = False
        self._value_patch_offsets.add(int(offset))
        self.selected_offset = offset
        self._render_source_from_ir(center_selected=False, preserve_view=True)
        self._select_offset(offset, center=False)
        self._set_status(reason)

    def _after_data_patch(self, reason: str, *, data_offset: int | None = None) -> None:
        self.project_dirty = True
        self.bytecode_dirty = True
        self.source_dirty = False
        if data_offset is not None:
            self._value_patch_data_offsets.add(int(data_offset))
        self._render_source_from_ir(center_selected=False, preserve_view=True)
        self._set_status(reason)

    def _nearby_immediate_candidates(self, offset: int, wanted: int | float | None = None) -> list[tuple[int, Mapping[str, Any]]]:
        if not self.instruction_offsets:
            return []
        pos = bisect_right(self.instruction_offsets, offset)
        window_offsets = list(reversed(self.instruction_offsets[max(0, pos - 24):pos + 4]))
        candidates: list[tuple[int, Mapping[str, Any]]] = []
        for ins_offset in window_offsets:
            row = self.instruction_by_offset[ins_offset]
            if int(row.get("opcode", -1)) not in TYPED_IMMEDIATE_OPCODES:
                continue
            if wanted is not None and not self._immediate_matches(row, wanted):
                continue
            candidates.append((ins_offset, row))
        return sorted(candidates, key=lambda item: abs(item[0] - offset))

    def _nearby_fixed_integer_candidates(self, offset: int, wanted: int | float | None = None) -> list[tuple[int, Mapping[str, Any]]]:
        if not self.instruction_offsets:
            return []
        pos = bisect_right(self.instruction_offsets, offset)
        window_offsets = list(reversed(self.instruction_offsets[max(0, pos - 16):pos + 8]))
        candidates: list[tuple[int, Mapping[str, Any]]] = []
        for ins_offset in window_offsets:
            row = self.instruction_by_offset[ins_offset]
            if int(row.get("opcode", -1)) not in FIXED_INTEGER_OPERAND_OPCODES:
                continue
            if wanted is not None and not self._fixed_integer_matches(row, wanted):
                continue
            candidates.append((ins_offset, row))
        return sorted(candidates, key=lambda item: abs(item[0] - offset))

    def _nearby_span_candidates(self, offset: int, literal: str | None = None) -> list[tuple[int, Mapping[str, Any]]]:
        if not self.instruction_offsets:
            return []
        pos = bisect_right(self.instruction_offsets, offset)
        window_offsets = self.instruction_offsets[max(0, pos - 16):pos + 32]
        candidates: list[tuple[int, Mapping[str, Any]]] = []
        for ins_offset in window_offsets:
            row = self.instruction_by_offset[ins_offset]
            if int(row.get("opcode", -1)) not in FIXED_SPAN_OPCODES:
                continue
            operands = row.get("operands") or {}
            data_offset = operands.get("data_offset")
            length = operands.get("length")
            if literal is not None and isinstance(data_offset, int) and isinstance(length, int):
                try:
                    current = read_data_c_string(self.ir, data_offset, max_length=length)
                except Exception:
                    current = None
                if current != literal:
                    continue
            candidates.append((ins_offset, row))
        return sorted(candidates, key=lambda item: abs(item[0] - offset))

    def _immediate_matches(self, row: Mapping[str, Any], wanted: int | float) -> bool:
        operands = row.get("operands") or {}
        values = []
        for key in ("value_i32", "value", "value_float"):
            if key in operands:
                values.append(operands[key])
        for value in values:
            try:
                if isinstance(wanted, float) and not isinstance(wanted, int):
                    if abs(float(value) - float(wanted)) < 1e-6:
                        return True
                elif int(float(value)) == int(wanted):
                    return True
            except Exception:
                continue
        return False

    def _fixed_integer_matches(self, row: Mapping[str, Any], wanted: int | float) -> bool:
        if isinstance(wanted, float) and not float(wanted).is_integer():
            return False
        operands = row.get("operands") or {}
        for key in ("value", "program_index", "program_index_u16", "data_offset"):
            if key not in operands:
                continue
            try:
                if int(operands[key]) == int(wanted):
                    return True
            except Exception:
                continue
        return False

    def _parse_number(self, literal: str) -> int | float:
        return float(literal) if "." in literal else int(literal, 0)

    def _data_offset_from_current_line(self) -> int | None:
        line = self.source.get(f"{self._current_line_number()}.0", f"{self._current_line_number()}.end")
        return self._data_offset_from_line(line)

    def _data_offset_from_line(self, line: str) -> int | None:
        match = DATA_COMMENT_RE.search(line)
        if match:
            return int(match.group(1), 16)
        match = DATA_SYMBOL_RE.search(line)
        if match:
            return int(match.group(1) or match.group(2), 16)
        return None

    def _looks_like_data_declaration(self, line: str) -> bool:
        stripped = line.strip()
        return bool(DATA_COMMENT_RE.search(line)) and stripped.startswith(("char ", "int ", "float ", "string ", "record ", "int_ref ", "float_ref "))

    def _capacity_from_line(self, line: str) -> int | None:
        match = re.search(r"\[(\d+)\]", line)
        if match:
            return int(match.group(1))
        length_match = re.search(r"length\s*=\s*(\d+)", line)
        if length_match:
            return int(length_match.group(1))
        bytes_match = re.search(r",\s*(\d+)\s+bytes", line)
        if bytes_match:
            return int(bytes_match.group(1))
        return None

    def _type_id_from_line(self, line: str) -> int:
        stripped = line.strip()
        if stripped.startswith("float") or "float32" in line:
            return TYPE_FLOAT
        if stripped.startswith("char") and "string" not in stripped:
            return TYPE_CHAR
        return TYPE_INT

    def _selection_contains_insert(self) -> bool:
        try:
            first = self.source.index(tk.SEL_FIRST)
            last = self.source.index(tk.SEL_LAST)
        except tk.TclError:
            return False
        insert = self.source.index(tk.INSERT)
        return bool(self.source.compare(first, "<=", insert) and self.source.compare(insert, "<=", last))

    def _selected_text_if(self, pattern: re.Pattern[str]) -> str | None:
        if not self._selection_contains_insert():
            return None
        try:
            selected = self.source.get(tk.SEL_FIRST, tk.SEL_LAST).strip()
        except tk.TclError:
            return None
        return selected if pattern.fullmatch(selected) else None

    def _selected_or_current_number(self) -> str | None:
        selected = self._selected_text_if(SIGNED_NUMBER_RE)
        if selected is not None:
            return selected
        line_no, col = self._insert_line_col()
        line = self.source.get(f"{line_no}.0", f"{line_no}.end")
        code_part = line.split("//", 1)[0]
        for match in SIGNED_NUMBER_RE.finditer(code_part):
            if match.start() <= col <= match.end():
                return match.group(0)
        return None

    def _selected_or_current_string(self) -> str | None:
        selected = self._selected_text_if(LINE_STRING_RE)
        if selected is not None:
            return selected
        line_no, col = self._insert_line_col()
        line = self.source.get(f"{line_no}.0", f"{line_no}.end")
        for match in LINE_STRING_RE.finditer(line):
            if match.start() <= col <= match.end():
                return match.group(0)
        return None

    def _selected_or_current_identifier(self) -> str | None:
        selected = self._selected_text_if(IDENT_RE)
        if selected is not None:
            return selected
        line_no, col = self._insert_line_col()
        line = self.source.get(f"{line_no}.0", f"{line_no}.end")
        for match in IDENT_RE.finditer(line):
            if match.start() <= col <= match.end():
                return match.group(0)
        return None

    def _unquote_string_literal(self, literal: str) -> str:
        body = literal[1:-1]
        try:
            return bytes(body, "utf-8").decode("unicode_escape")
        except Exception:
            return body.replace('\\"', '"').replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t').replace('\\\\', '\\')

    def _row_is_float(self, row: Mapping[str, Any]) -> bool:
        operands = row.get("operands") or {}
        return str(operands.get("type_name", "")).lower().startswith("float") or "value_float" in operands

    def _current_immediate_text(self, row: Mapping[str, Any]) -> str:
        operands = row.get("operands") or {}
        if self._row_is_float(row) and "value_float" in operands:
            return str(operands.get("value_float"))
        if "value_i32" in operands:
            return str(operands.get("value_i32"))
        if "value" in operands:
            return str(operands.get("value"))
        return "0"
