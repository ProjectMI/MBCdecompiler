from __future__ import annotations

"""A small, deliberately conservative AST seed for MBC bytecode.

The real VM is stack based and many opcodes carry reference metadata, not just a
plain value.  This builder therefore does not pretend to be a finished source
reconstructor.  It keeps a symbolic expression stack, emits pseudo-statements for
obvious operations, and preserves instruction offsets so later structural passes
can fold gotos into if/while blocks.
"""

from dataclasses import dataclass, asdict
from typing import Any, Iterable, List, Optional
import math

from .loader import MbcProgram, MbcScript
from .opcodes import BINARY_AST_OPS, UNARY_AST_OPS, safe_chr


@dataclass
class AstStatement:
    offset: int
    file_offset: int
    kind: str
    text: str
    opcode: int
    mnemonic: str
    operands: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StackAstBuilder:
    def __init__(self, script: MbcScript, program: MbcProgram):
        self.script = script
        self.program = program
        self.stack: list[str] = []
        self.statements: list[AstStatement] = []
        self.pending_arg_count: Optional[int] = None
        self.temp_index = 0
        self.underflows = 0

    def build(self, instructions: Iterable[Any]) -> dict[str, Any]:
        for ins in instructions:
            self._visit(ins)
        return {
            "format": "experimental_stack_ast_v0",
            "warning": (
                "This is a symbolic stack AST seed, not a final decompilation. "
                "It is meant to preserve expression intent and control-flow anchors for later structuring."
            ),
            "statement_count": len(self.statements),
            "underflow_placeholders": self.underflows,
            "residual_stack": list(self.stack[-16:]),
            "statements": [stmt.to_dict() for stmt in self.statements],
            "source": "\n".join(stmt.text for stmt in self.statements),
        }

    def _visit(self, ins: Any) -> None:
        op = ins.opcode
        m = ins.mnemonic
        operands = ins.operands or {}

        if m == "set_arg_count":
            self.pending_arg_count = int(operands.get("value", 0))
            self._emit(ins, "meta", f"// argc = {self.pending_arg_count}")
            return

        if op in (0x69, 0x39, 0x28, 0x29, 0x41, 0x65, 0x6C):
            self._push(self._render_push_value(ins))
            return

        if op in (0x61, 0x62, 0x6D, 0x64, 0x68):
            self._push(self._render_index_expr(ins))
            return

        if op in BINARY_AST_OPS:
            rhs = self._pop(ins, "rhs")
            lhs = self._pop(ins, "lhs")
            self._push(f"({lhs} {BINARY_AST_OPS[op]} {rhs})")
            return

        if op in UNARY_AST_OPS:
            value = self._pop(ins, "value")
            symbol = UNARY_AST_OPS[op]
            self._push(f"({symbol}{value})")
            return

        if m in {"to_float", "to_float_prev"}:
            value = self._pop(ins, "value")
            self._push(f"float({value})")
            return

        if m in {"to_int", "to_int_prev"}:
            value = self._pop(ins, "value")
            self._push(f"int({value})")
            return

        if m == "swap":
            if len(self.stack) >= 2:
                self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
            else:
                self._emit(ins, "warning", "// swap skipped: symbolic stack underflow")
            return

        if m in {"force_int_type", "force_int_type_alt", "force_two_ints"}:
            # Runtime metadata normalization; it does not change the high-level expression.
            return

        if m == "address_of":
            value = self._pop(ins, "value")
            self._push(f"&{value}")
            return

        if m == "deref":
            value = self._pop(ins, "ptr")
            self._push(f"*{value}")
            return

        if m == "store":
            value = self._pop(ins, "value")
            target = self._pop(ins, "target")
            self._emit(ins, "assign", f"{target} = {value};")
            # Assignment handlers keep useful value/reference metadata on the VM stack.
            self._push(target)
            return

        if m in {"pre_inc", "post_inc", "pre_dec", "post_dec"}:
            target = self._pop(ins, "target")
            symbol = "++" if "inc" in m else "--"
            text = f"{symbol}{target}" if m.startswith("pre") else f"{target}{symbol}"
            self._emit(ins, "expr", f"{text};")
            self._push(text)
            return

        if m.endswith("assign_u16") or m in {"ptr_add_assign_u16", "ptr_sub_assign_u16"}:
            target = self._pop(ins, "target")
            value = operands.get("value", "?")
            op_symbol = "+=" if "add" in m else "-="
            self._emit(ins, "assign", f"{target} {op_symbol} {value};")
            self._push(target)
            return

        if m in {"ptr_add_scaled_u16", "ptr_sub_scaled_u16"}:
            rhs = self._pop(ins, "rhs")
            lhs = self._pop(ins, "lhs")
            op_symbol = "+" if "add" in m else "-"
            self._push(f"({lhs} {op_symbol} {operands.get('value', '?')} * {rhs})")
            return

        if m in {"jmp_rel16", "jmp_rel32"}:
            self._emit(ins, "goto", f"goto loc_{operands.get('target', 0):08X};")
            return

        if m in {"jfalse_rel16", "jfalse_rel32"}:
            cond = self._pop(ins, "cond")
            self._emit(ins, "if_goto", f"if (!({cond})) goto loc_{operands.get('target', 0):08X};")
            return

        if m == "logical_or_rel16":
            cond = self._pop(ins, "lhs")
            self._emit(ins, "if_goto", f"if ({cond}) goto loc_{operands.get('target', 0):08X}; // || short-circuit")
            return

        if m == "logical_and_rel16":
            cond = self._pop(ins, "lhs")
            self._emit(ins, "if_goto", f"if (!({cond})) goto loc_{operands.get('target', 0):08X}; // && short-circuit")
            return

        if m == "call_rel32":
            args = self._pop_args(ins)
            call = f"sub_{operands.get('target', 0):08X}({', '.join(args)})"
            self._emit(ins, "call", f"{call};")
            self._push(f"ret_{ins.offset:08X}")
            return

        if m == "call_linked_function":
            args = self._pop_args(ins)
            self._emit(ins, "call", f"linked_call({', '.join(args)});")
            self._push(f"ret_{ins.offset:08X}")
            return

        if operands.get("subopcode") is not None:
            args = self._pop_args(ins)
            expr = f"{m}({', '.join(args)})"
            # Builtins vary between void and returning.  Keep both a statement and
            # a symbolic result so later passes can decide whether it is consumed.
            self._emit(ins, "builtin", f"{expr};")
            self._push(f"ret_{m}_{ins.offset:08X}")
            return

        if m in {"return", "return_local"}:
            value = self.stack[-1] if self.stack else ""
            suffix = f" {value}" if value else ""
            self._emit(ins, "return", f"return{suffix};")
            return

        if m == "halt_interpreter":
            self._emit(ins, "halt", "halt_interpreter();")
            return

        if m in {"program_restart", "program_restart_child", "program_activate", "program_reset_alt_pc", "program_stop"}:
            idx = operands.get("program_index", "?")
            pname = self._program_name(idx)
            arg = f"{idx}" if pname is None else f"{idx} /* {pname} */"
            self._emit(ins, "program", f"{m}({arg});")
            return

        if m == "program_prologue":
            descriptors = operands.get("descriptors", [])
            self._emit(ins, "prologue", f"prologue(argc={operands.get('signed_count')}, descriptors={len(descriptors)});")
            return

        if not ins.known:
            self._emit(ins, "unknown", f"// unknown/truncated opcode 0x{ins.opcode:02X} at 0x{ins.offset:08X}")
            return

        # Preserve no-op/runtime metadata instructions as sparse comments only
        # when they are likely useful for later passes.
        if m not in {"stack_frame_reset", "push_stack_frame", "pop_stack_frame"}:
            self._emit(ins, "op", f"// {m}")

    def _render_push_value(self, ins: Any) -> str:
        operands = ins.operands or {}
        m = ins.mnemonic
        typ = operands.get("type")
        tname = operands.get("type_name") or (f"type_{typ}" if typ is not None else None)

        if m == "push_data_ref":
            off = operands.get("data_offset", 0)
            preview = self._data_preview(off)
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
            preview = self._data_preview(off, length)
            return f"span[{off:#x}, {length}]" + (f" /* {preview!r} */" if preview else "")

        if m in {"push_typed_span_ref", "push_inline_typed_span"}:
            off = operands.get("data_offset", 0)
            length = operands.get("length", 0)
            preview = self._data_preview(off, length)
            return f"span[{off:#x}, {length}:{tname}]" + (f" /* {preview!r} */" if preview else "")

        return f"{m}(...)"

    def _render_index_expr(self, ins: Any) -> str:
        operands = ins.operands or {}
        typ = operands.get("type_name", "?")
        if ins.mnemonic == "array_index_abs":
            idx = self._pop(ins, "index")
            return f"array[{operands.get('base', 0):#x} + {idx}*{operands.get('element_size', '?')}:{typ}]"
        if ins.mnemonic in {"array2_index", "array2_index_checked"}:
            idx = self._pop(ins, "index")
            base = self._pop(ins, "base")
            return f"{base}[{idx}*{operands.get('element_size', '?')}:{typ}]"
        if ins.mnemonic == "slice_offset_ref":
            base = self._pop(ins, "base")
            return f"{base}+{operands.get('offset', '?')}:{typ}"
        if ins.mnemonic == "slice_offset_span":
            base = self._pop(ins, "base")
            return f"span({base}+{operands.get('offset', '?')}, {operands.get('length', '?')}:{typ})"
        return f"{ins.mnemonic}(...)"

    def _pop_args(self, ins: Any) -> list[str]:
        count = self.pending_arg_count
        self.pending_arg_count = None
        if count is None:
            return []
        args = [self._pop(ins, f"arg{idx}") for idx in range(count)]
        args.reverse()
        return args

    def _pop(self, ins: Any, name: str) -> str:
        if self.stack:
            return self.stack.pop()
        self.underflows += 1
        return f"<{name}@0x{ins.offset:08X}>"

    def _push(self, expr: str) -> None:
        self.stack.append(expr)

    def _emit(self, ins: Any, kind: str, text: str) -> None:
        self.statements.append(
            AstStatement(
                offset=ins.offset,
                file_offset=ins.file_offset,
                kind=kind,
                text=text,
                opcode=ins.opcode,
                mnemonic=ins.mnemonic,
                operands=dict(ins.operands or {}),
            )
        )

    def _program_name(self, idx: Any) -> Optional[str]:
        if not isinstance(idx, int):
            return None
        if 0 <= idx < len(self.script.programs):
            return self.script.programs[idx].name
        return None

    def _data_preview(self, off: Any, max_len: int | None = None) -> str:
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


def build_program_ast(script: MbcScript, program: MbcProgram, instructions: Iterable[Any]) -> dict[str, Any]:
    return StackAstBuilder(script, program).build(instructions)
