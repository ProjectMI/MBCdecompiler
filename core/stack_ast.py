from __future__ import annotations

"""Symbolic stack-machine pass for the experimental pseudo-AST.

The real VM is stack based and many opcodes carry reference metadata, not just a
plain value.  This pass therefore does not pretend to be a finished source
reconstructor.  It keeps a symbolic expression stack, emits pseudo-statements for
obvious operations, and preserves instruction offsets so later structural passes
can fold gotos into if/while blocks.
"""

from typing import Any, Iterable, Optional

from .ast_model import AstStatement, ast_payload
from .ast_render import AstExpressionRenderer
from .loader import MbcProgram, MbcScript
from .opcodes import BINARY_AST_OPS, UNARY_AST_OPS


class StackAstBuilder:
    """Interpret decoded instructions as a conservative symbolic stack AST seed."""

    def __init__(self, script: MbcScript, program: MbcProgram):
        self.script = script
        self.program = program
        self.renderer = AstExpressionRenderer(script, program)
        self.stack: list[str] = []
        self.statements: list[AstStatement] = []
        self.pending_arg_count: Optional[int] = None
        self.temp_index = 0
        self.underflows = 0

    def build(self, instructions: Iterable[Any]) -> dict[str, Any]:
        for ins in instructions:
            self._visit(ins)
        return ast_payload(
            statements=self.statements,
            residual_stack=self.stack,
            underflows=self.underflows,
        )

    def _visit(self, ins: Any) -> None:
        op = ins.opcode
        m = ins.mnemonic
        operands = ins.operands or {}

        if m == "set_arg_count":
            self.pending_arg_count = int(operands.get("value", 0))
            self._emit(ins, "meta", f"// argc = {self.pending_arg_count}")
            return

        if op in (0x69, 0x39, 0x28, 0x29, 0x41, 0x65, 0x6C):
            self._push(self.renderer.push_value(ins))
            return

        if op in (0x61, 0x62, 0x6D, 0x64, 0x68):
            self._push(self.renderer.index_expr(ins, lambda name: self._pop(ins, name)))
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
            self._emit(ins, "program", f"{m}({self.renderer.program_arg(idx)});")
            return

        if m == "program_prologue":
            descriptors = operands.get("descriptors", [])
            self._emit(ins, "prologue", f"prologue(argc={operands.get('signed_count')}, descriptors={len(descriptors)});")
            return

        if m == "import_stub_u32":
            self._emit(ins, "import_stub", f"// import stub payload={operands.get('value', '?')}")
            return

        if not ins.known:
            self._emit(ins, "unknown", f"// unknown/truncated opcode 0x{ins.opcode:02X} at 0x{ins.offset:08X}")
            return

        # Preserve no-op/runtime metadata instructions as sparse comments only
        # when they are likely useful for later passes.
        if m not in {"stack_frame_reset", "push_stack_frame", "pop_stack_frame"}:
            self._emit(ins, "op", f"// {m}")

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


def build_program_ast(script: MbcScript, program: MbcProgram, instructions: Iterable[Any]) -> dict[str, Any]:
    return StackAstBuilder(script, program).build(instructions)
