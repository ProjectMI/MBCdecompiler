from __future__ import annotations

"""Symbolic VM-stack pass for the experimental pseudo-AST.

This layer now models the native stack helpers instead of keeping a bare list of
strings.  Slots carry MBC type tags, scalar/reference/slice intent and enough
range metadata to render array/slice operations conservatively.
"""

from typing import Any, Iterable, Optional

from .ast_model import AstStatement, ast_payload
from .ast_render import AstExpressionRenderer
from .linker import MbcStaticLinker
from .loader import MbcProgram, MbcScript
from .opcodes import BINARY_AST_OPS, UNARY_AST_OPS
from .vm_stack import (
    TYPE_CHAR,
    TYPE_FLOAT,
    TYPE_FLOAT_REF,
    TYPE_INT,
    TYPE_INT_REF,
    TYPE_SLICE,
    TYPE_STRING,
    VMStackMachine,
    VMValue,
)


COMPARISON_OPS = {0xF0, 0xED, 0x3E, 0x3C, 0xE1, 0xEC}
REF_TYPES = {TYPE_STRING, TYPE_INT_REF, TYPE_FLOAT_REF, TYPE_SLICE}


class StackAstBuilder:
    """Interpret decoded instructions as a conservative symbolic stack AST seed."""

    def __init__(self, script: MbcScript, program: MbcProgram, *, linker: MbcStaticLinker | None = None):
        self.script = script
        self.program = program
        self.linker = linker or MbcStaticLinker(script)
        self.renderer = AstExpressionRenderer(script, program)
        self.vm = VMStackMachine()
        self.statements: list[AstStatement] = []
        self.pending_arg_count: Optional[int] = None

    def build(self, instructions: Iterable[Any]) -> dict[str, Any]:
        for ins in instructions:
            self._visit(ins)
        return ast_payload(
            statements=self.statements,
            residual_stack=self.vm.residual(),
            underflows=self.vm.underflows,
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
            self._push_decoded_value(ins)
            return

        if op in (0x61, 0x62, 0x6D, 0x64, 0x68):
            self._push_index_or_slice(ins)
            return

        if op in BINARY_AST_OPS:
            rhs = self._pop_value(ins, "rhs")
            lhs = self._pop_value(ins, "lhs")
            expr = f"({lhs.render()} {BINARY_AST_OPS[op]} {rhs.render()})"
            if op in COMPARISON_OPS:
                self.vm.push_int(expr)
            elif lhs.is_float or rhs.is_float or op == 0x2F:
                self.vm.push_float(expr)
            else:
                self.vm.push_int(expr)
            return

        if op in UNARY_AST_OPS:
            value = self._pop_value(ins, "value")
            symbol = UNARY_AST_OPS[op]
            result_type = TYPE_INT if op == 0x21 else value.type_id
            self.vm.push(VMValue(expr=f"({symbol}{value.render()})", type_id=result_type, kind="value"))
            return

        if m in {"to_float", "to_float_prev"}:
            self.vm.push(self.vm.coerce_float(self._pop_value(ins, "value")))
            return

        if m in {"to_int", "to_int_prev"}:
            self.vm.push(self.vm.coerce_int(self._pop_value(ins, "value")))
            return

        if m == "swap":
            if not self.vm.swap_top_two():
                self._emit(ins, "warning", "// swap skipped: symbolic stack underflow")
            return

        if m in {"force_int_type", "force_int_type_alt"}:
            self.vm.force_int_top()
            return

        if m == "force_two_ints":
            self.vm.force_two_ints()
            return

        if m == "address_of":
            value = self._pop_value(ins, "value")
            self.vm.push(value.clone(expr=f"&{value.render()}", kind="pointer", is_lvalue=True))
            return

        if m == "deref":
            value = self.vm.get_pointer_or_slice(self._pop_value(ins, "ptr"))
            self.vm.push(value.clone(expr=f"*{value.render()}", kind="value", is_lvalue=False))
            return

        if m == "store":
            value = self._pop_value(ins, "value")
            target = self._pop_value(ins, "target")
            self._emit(ins, "assign", f"{target.render()} = {value.render()};")
            # Native assignment handlers leave useful lvalue/value metadata on the stack.
            self.vm.push(target.clone(expr=target.render()))
            return

        if m in {"pre_inc", "post_inc", "pre_dec", "post_dec"}:
            target = self._pop_value(ins, "target")
            symbol = "++" if "inc" in m else "--"
            text = f"{symbol}{target.render()}" if m.startswith("pre") else f"{target.render()}{symbol}"
            self._emit(ins, "expr", f"{text};")
            self.vm.push(target.clone(expr=text, kind="value", is_lvalue=False))
            return

        if m.endswith("assign_u16") or m in {"ptr_add_assign_u16", "ptr_sub_assign_u16"}:
            target = self._pop_value(ins, "target")
            value = operands.get("value", "?")
            op_symbol = "+=" if "add" in m else "-="
            self._emit(ins, "assign", f"{target.render()} {op_symbol} {value};")
            self.vm.push(target)
            return

        if m in {"ptr_add_scaled_u16", "ptr_sub_scaled_u16"}:
            rhs = self._pop_value(ins, "rhs")
            lhs = self._pop_value(ins, "lhs")
            op_symbol = "+" if "add" in m else "-"
            expr = f"({lhs.render()} {op_symbol} {operands.get('value', '?')} * {rhs.render()})"
            self.vm.push(lhs.clone(expr=expr, kind="pointer", is_lvalue=True))
            return

        if m in {"jmp_rel16", "jmp_rel32"}:
            target = operands.get("target", 0)
            label = operands.get("target_name") or f"loc_{target:08X}"
            self._emit(ins, "goto", f"goto {label};")
            return

        if m in {"jfalse_rel16", "jfalse_rel32"}:
            cond = self._pop_value(ins, "cond")
            self._emit(ins, "if_goto", f"if (!({cond.render()})) goto loc_{operands.get('target', 0):08X};")
            return

        if m == "logical_or_rel16":
            cond = self._pop_value(ins, "lhs")
            self._emit(ins, "if_goto", f"if ({cond.render()}) goto loc_{operands.get('target', 0):08X}; // || short-circuit")
            return

        if m == "logical_and_rel16":
            cond = self._pop_value(ins, "lhs")
            self._emit(ins, "if_goto", f"if (!({cond.render()})) goto loc_{operands.get('target', 0):08X}; // && short-circuit")
            return

        if m == "call_rel32":
            args = self._pop_args(ins)
            target = operands.get("target", 0)
            call_name = operands.get("target_name") or self.linker.callable_name_for_offset(target) or f"sub_{target:08X}"
            self._emit(ins, "call", f"{call_name}({', '.join(args)});")
            self.vm.push_unknown(f"ret_{call_name}_{ins.offset:08X}")
            return

        if m == "call_linked_function":
            args = self._pop_args(ins)
            # The runtime dispatcher searches the loaded function table by the pending
            # linked-call name and only enters records whose +0x28 program index is resolved.
            self._emit(ins, "call", f"linked_call({', '.join(args)});")
            self.vm.push_unknown(f"ret_linked_call_{ins.offset:08X}")
            return

        if operands.get("subopcode") is not None:
            args = self._pop_args(ins)
            expr = f"{m}({', '.join(args)})"
            self._emit(ins, "builtin", f"{expr};")
            # Until builtin-specific stack effects are modelled, preserve a possible result
            # so subsequent bytecode does not lose expression flow.
            self.vm.push_unknown(f"ret_{m}_{ins.offset:08X}")
            return

        if m in {"return", "return_local"}:
            top = self.vm.peek()
            suffix = f" {top.render()}" if top is not None else ""
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
            name = operands.get("link_name") or operands.get("function_name")
            if name:
                self._emit(ins, "import_stub", f"extern {name}; // import stub payload={operands.get('value', '?')}")
            else:
                self._emit(ins, "import_stub", f"// import stub payload={operands.get('value', '?')}")
            return

        if not ins.known:
            self._emit(ins, "unknown", f"// unknown/truncated opcode 0x{ins.opcode:02X} at 0x{ins.offset:08X}")
            return

        # Preserve no-op/runtime metadata instructions as sparse comments only
        # when they are likely useful for later passes.
        if m not in {"stack_frame_reset", "push_stack_frame", "pop_stack_frame"}:
            self._emit(ins, "op", f"// {m}")

    def _push_decoded_value(self, ins: Any) -> None:
        operands = ins.operands or {}
        expr = self.renderer.push_value(ins)
        typ = operands.get("type")
        m = ins.mnemonic

        if m == "push_data_ref":
            self.vm.push_ref(expr, type_id=typ, ptr=operands.get("data_offset"), storage_size=self._storage_size(typ), note="data_ref")
            return

        if m in {"push_inline_span", "push_typed_span_ref", "push_inline_typed_span"}:
            begin = operands.get("data_offset")
            length = operands.get("length")
            end = begin + length - 1 if isinstance(begin, int) and isinstance(length, int) and length > 0 else None
            self.vm.push_slice(expr, type_id=typ if typ is not None else TYPE_STRING, begin=begin, end=end, note=m)
            return

        if typ == TYPE_FLOAT and "value_float" in operands:
            self.vm.push_float(expr, value=operands.get("value_float"))
        elif typ in {TYPE_CHAR, TYPE_INT} or typ is None:
            self.vm.push_int(expr, value=operands.get("value", operands.get("value_i32", operands.get("value_u32"))))
        elif typ in REF_TYPES:
            self.vm.push_ref(expr, type_id=typ, storage_size=self._storage_size(typ))
        else:
            self.vm.push_unknown(expr, type_id=typ)

    def _push_index_or_slice(self, ins: Any) -> None:
        operands = ins.operands or {}
        typ = operands.get("type")
        typ_name = operands.get("type_name", "?")
        m = ins.mnemonic

        if m == "array_index_abs":
            index = self.vm.coerce_int(self._pop_value(ins, "index"))
            expr = f"array[{operands.get('base', 0):#x} + {index.render()}*{operands.get('element_size', '?')}:{typ_name}]"
            self._push_typed_index_result(expr, typ, operands)
            return

        if m in {"array2_index", "array2_index_checked"}:
            index = self.vm.coerce_int(self._pop_value(ins, "index"))
            base = self.vm.get_pointer_or_slice(self._pop_value(ins, "base"))
            expr = f"{base.render()}[{index.render()}*{operands.get('element_size', '?')}:{typ_name}]"
            self._push_typed_index_result(expr, typ, operands)
            return

        if m == "slice_offset_ref":
            base = self.vm.get_pointer_or_slice(self._pop_value(ins, "base"))
            expr = f"{base.render()}+{operands.get('offset', '?')}:{typ_name}"
            self._push_typed_index_result(expr, typ, operands)
            return

        if m == "slice_offset_span":
            base = self.vm.get_pointer_or_slice(self._pop_value(ins, "base"))
            expr = f"span({base.render()}+{operands.get('offset', '?')}, {operands.get('length', '?')}:{typ_name})"
            self.vm.push_slice(expr, type_id=typ, note="slice_offset_span")
            return

        self.vm.push_unknown(f"{m}(...)", type_id=typ)

    def _push_typed_index_result(self, expr: str, typ: Any, operands: dict[str, Any]) -> None:
        if typ == TYPE_FLOAT:
            self.vm.push_float(expr, note="indexed float load")
        elif typ in {TYPE_CHAR, TYPE_INT}:
            self.vm.push_int(expr, note="indexed scalar load")
        elif typ == TYPE_SLICE:
            self.vm.push_slice(expr, type_id=typ, note="indexed descriptor")
        elif typ in REF_TYPES:
            self.vm.push_ref(expr, type_id=typ, storage_size=self._storage_size(typ), note="indexed ref")
        else:
            self.vm.push_unknown(expr, type_id=typ, kind="indexed")

    def _pop_args(self, ins: Any) -> list[str]:
        count = self.pending_arg_count
        self.pending_arg_count = None
        if count is None:
            return []
        args = [self._pop_value(ins, f"arg{idx}").render() for idx in range(count)]
        args.reverse()
        return args

    def _pop_value(self, ins: Any, name: str) -> VMValue:
        return self.vm.pop(f"<{name}@0x{ins.offset:08X}>")

    def _storage_size(self, typ: Any) -> int:
        if typ in {TYPE_CHAR, TYPE_STRING}:
            return 1
        if typ in {TYPE_INT, TYPE_INT_REF, TYPE_FLOAT, TYPE_FLOAT_REF}:
            return 4
        if typ == TYPE_SLICE:
            return 0x0C
        return 1

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


def build_program_ast(
    script: MbcScript,
    program: MbcProgram,
    instructions: Iterable[Any],
    *,
    linker: MbcStaticLinker | None = None,
) -> dict[str, Any]:
    return StackAstBuilder(script, program, linker=linker).build(instructions)
