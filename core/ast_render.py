from __future__ import annotations

"""Formatting helpers for pseudo-source fragments.

The VM pass owns values, references and memory locations.  This module only
formats immediates, spans and program-table references.
"""

from typing import Any
import math

from .loader import MbcProgram, MbcScript
from .vm_stack import SymbolicDataMemory, TYPE_FLOAT, TYPE_INT, TYPE_CHAR


class AstExpressionRenderer:
    def __init__(self, script: MbcScript, program: MbcProgram, *, memory: SymbolicDataMemory | None = None):
        self.script = script
        self.program = program
        self.memory = memory or SymbolicDataMemory(module_name=script.path.stem, data=script.data)

    def push_value(self, ins: Any) -> str:
        operands = ins.operands or {}
        m = ins.mnemonic
        typ = operands.get("type")
        tname = operands.get("type_name") or (f"type_{typ}" if typ is not None else None)

        if m == "push_data_ref":
            off = operands.get("data_offset", 0)
            return self.memory.location(offset=off, type_id=typ).render()

        if m in {"push_imm32", "push_imm_u16", "push_imm_i8"}:
            if typ == TYPE_FLOAT and "value_float" in operands:
                value = operands["value_float"]
                if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
                    return repr(value)
                return f"{value:.9g}"
            if "value_i32" in operands:
                return str(operands["value_i32"])
            return str(operands.get("value", operands.get("value_u32", "?")))

        if m == "push_inline_span":
            off = operands.get("data_offset", 0)
            length = operands.get("length", 0)
            return self.span_expr(off, length)

        if m in {"push_typed_span_ref", "push_inline_typed_span"}:
            off = operands.get("data_offset", 0)
            length = operands.get("length", 0)
            return self.span_expr(off, length, tname)

        return f"{m}(...)"

    def span_expr(self, off: Any, length: Any, type_name: str | None = None) -> str:
        if isinstance(off, int):
            loc = self.memory.location(offset=off, type_id=None, role="span", length=length if isinstance(length, int) else None)
            base = loc.render()
        else:
            base = f"span[{off!r}]"
        suffix = "" if type_name is None else f":{type_name}"
        preview = self.data_preview(off, length if isinstance(length, int) else None)
        return f"{base}[{length}{suffix}]" + (f" /* {preview!r} */" if preview else "")

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
        return self.memory.preview(off, max_len)
