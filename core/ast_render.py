from __future__ import annotations

"""Rendering helpers for the experimental stack-AST layer.

The symbolic stack pass decides *what* VM operations mean.  This module only
formats bytecode operands and symbolic references into readable pseudo-source
fragments.
"""

from typing import Any, Callable
import math

from .loader import MbcProgram, MbcScript


class AstExpressionRenderer:
    def __init__(self, script: MbcScript, program: MbcProgram):
        self.script = script
        self.program = program

    def push_value(self, ins: Any) -> str:
        operands = ins.operands or {}
        m = ins.mnemonic
        typ = operands.get("type")
        tname = operands.get("type_name") or (f"type_{typ}" if typ is not None else None)

        if m == "push_data_ref":
            off = operands.get("data_offset", 0)
            preview = self.data_preview(off)
            extra = f", {preview!r}" if preview else ""
            return f"data[{off:#x}:{tname}{extra}]"

        if m in {"push_imm32", "push_imm_u16", "push_imm_i8"}:
            if typ == 32 and "value_float" in operands:
                value = operands["value_float"]
                if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
                    value_text = repr(value)
                else:
                    value_text = f"{value:.9g}"
            elif "value_i32" in operands:
                value_text = str(operands["value_i32"])
            else:
                value_text = str(operands.get("value", operands.get("value_u32", "?")))
            return f"{value_text} /* {tname} */"

        if m == "push_inline_span":
            off = operands.get("data_offset", 0)
            length = operands.get("length", 0)
            preview = self.data_preview(off, length)
            return f"span[{off:#x}, {length}]" + (f" /* {preview!r} */" if preview else "")

        if m in {"push_typed_span_ref", "push_inline_typed_span"}:
            off = operands.get("data_offset", 0)
            length = operands.get("length", 0)
            preview = self.data_preview(off, length)
            return f"span[{off:#x}, {length}:{tname}]" + (f" /* {preview!r} */" if preview else "")

        return f"{m}(...)"

    def index_expr(self, ins: Any, pop: Callable[[str], str]) -> str:
        operands = ins.operands or {}
        typ = operands.get("type_name", "?")
        if ins.mnemonic == "array_index_abs":
            idx = pop("index")
            return f"array[{operands.get('base', 0):#x} + {idx}*{operands.get('element_size', '?')}:{typ}]"
        if ins.mnemonic in {"array2_index", "array2_index_checked"}:
            idx = pop("index")
            base = pop("base")
            return f"{base}[{idx}*{operands.get('element_size', '?')}:{typ}]"
        if ins.mnemonic == "slice_offset_ref":
            base = pop("base")
            return f"{base}+{operands.get('offset', '?')}:{typ}"
        if ins.mnemonic == "slice_offset_span":
            base = pop("base")
            return f"span({base}+{operands.get('offset', '?')}, {operands.get('length', '?')}:{typ})"
        return f"{ins.mnemonic}(...)"

    def program_arg(self, idx: Any) -> str:
        pname = self.program_name(idx)
        return f"{idx}" if pname is None else f"{idx} /* {pname} */"

    def program_name(self, idx: Any) -> str | None:
        if not isinstance(idx, int):
            return None
        if 0 <= idx < len(self.script.programs):
            return self.script.programs[idx].name
        return None

    def data_preview(self, off: Any, max_len: int | None = None) -> str:
        if not isinstance(off, int) or off < 0 or off >= len(self.script.data):
            return ""
        limit = len(self.script.data) if max_len is None or max_len <= 0 else min(len(self.script.data), off + max_len)
        end = self.script.data.find(b"\x00", off, limit)
        if end < 0:
            end = min(limit, off + 48)
        raw = self.script.data[off:end]
        if not raw:
            return ""
        text = raw.decode("cp1251", errors="replace")
        if any(ord(ch) < 32 and ch not in "\t\r\n" for ch in text):
            return ""
        return text[:80]
