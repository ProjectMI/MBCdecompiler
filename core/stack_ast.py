from __future__ import annotations

"""Symbolic VM interpreter that builds the low-level pseudo-AST.

This pass treats the VM stack as the execution core.  VMSlots carry value/ref /
slice/pointer class, type tags and symbolic memory locations.  Function prologue
instructions bind incoming arguments to data-section locations before later
instructions read those locations.
"""

import re
from typing import Any, Iterable, Optional

from .ast_model import AstStatement, ast_payload
from .ast_normalize import label_for_offset, normalize_ast_statements
from .ast_render import AstExpressionRenderer
from .effects import builtin_effect, function_effect, native_import_effect, specialize_builtin_call, unresolved_import_effect, CallEffect
from .linker import FunctionSignature, MbcStaticLinker
from .loader import MbcProgram, MbcScript
from .opcodes import BINARY_AST_OPS, UNARY_AST_OPS
from .vm_stack import (
    SymbolicDataMemory,
    TYPE_CHAR,
    TYPE_FLOAT,
    TYPE_FLOAT_REF,
    TYPE_INT,
    TYPE_INT_REF,
    TYPE_SLICE,
    TYPE_STRING,
    VMSlot,
    VMStackMachine,
)


COMPARISON_OPS = {0xF0, 0xED, 0x3E, 0x3C, 0xE1, 0xEC}
REF_TYPES = {TYPE_STRING, TYPE_INT_REF, TYPE_FLOAT_REF, TYPE_SLICE}


class StackAstBuilder:
    """Interpret decoded instructions as a conservative symbolic stack AST seed."""

    def __init__(self, script: MbcScript, program: MbcProgram, *, linker: MbcStaticLinker | None = None):
        self.script = script
        self.program = program
        self.linker = linker or MbcStaticLinker(script)
        self.memory = SymbolicDataMemory(module_name=script.path.stem, data=script.data)
        self.renderer = AstExpressionRenderer(script, program, memory=self.memory)
        self.vm = VMStackMachine(memory=self.memory)
        self.statements: list[AstStatement] = []
        self.pending_arg_count: Optional[int] = None

    def build(self, instructions: Iterable[Any]) -> dict[str, Any]:
        for ins in instructions:
            self._visit(ins)
        payload = ast_payload(
            statements=normalize_ast_statements(self.statements),
            residual_stack=self.vm.residual(),
            underflows=self.vm.underflows,
        )
        payload["memory_bindings"] = self.memory.bindings()
        return payload

    def _visit(self, ins: Any) -> None:
        op = ins.opcode
        m = ins.mnemonic
        operands = ins.operands or {}

        if m == "set_arg_count":
            self.pending_arg_count = int(operands.get("value", 0))
            return

        if m == "stack_frame_reset":
            self.vm.reset_frame()
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
            self.vm.push(VMSlot(expr=f"({symbol}{value.render()})", type_id=result_type, kind="value"))
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
            self.vm.push(self.vm.address_of(self._pop_value(ins, "value")))
            return

        if m == "deref":
            self.vm.push(self.vm.deref(self._pop_value(ins, "ptr")))
            return

        if m == "store":
            value = self._pop_value(ins, "value")
            target = self._pop_value(ins, "target")
            version = self.vm.store(target, value)
            self._emit(ins, "assign", f"{target.render()} = {value.render()};", extra={"defs": [version], "uses": self._uses(target, value)})
            self.vm.push(target.clone(expr=target.render()))
            return

        if m in {"pre_inc", "post_inc", "pre_dec", "post_dec"}:
            target = self._pop_value(ins, "target")
            symbol = "++" if "inc" in m else "--"
            text = f"{symbol}{target.render()}" if m.startswith("pre") else f"{target.render()}{symbol}"
            version = self.vm.store(target, VMSlot(expr=text, type_id=target.type_id, kind="value"))
            self._emit(ins, "expr", f"{text};", extra={"defs": [version], "uses": self._uses(target)})
            self.vm.push(target.clone(expr=text, kind="value", is_lvalue=False))
            return

        if m.endswith("assign_u16") or m in {"ptr_add_assign_u16", "ptr_sub_assign_u16"}:
            target = self._pop_value(ins, "target")
            value = operands.get("value", "?")
            op_symbol = "+=" if "add" in m else "-="
            version = self.vm.store(target, VMSlot(expr=str(value), type_id=target.type_id, kind="value"))
            self._emit(ins, "assign", f"{target.render()} {op_symbol} {value};", extra={"defs": [version], "uses": self._uses(target)})
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
            self._emit(ins, "goto", f"goto {label_for_offset(target)};")
            return

        if m in {"jfalse_rel16", "jfalse_rel32"}:
            cond = self._pop_value(ins, "cond")
            self._emit(ins, "if_goto", f"if (!({cond.render()})) goto {label_for_offset(operands.get('target'))};", extra={"uses": self._uses(cond)})
            return

        if m == "logical_or_rel16":
            cond = self._pop_value(ins, "lhs")
            self._emit(ins, "if_goto", f"if ({cond.render()}) goto {label_for_offset(operands.get('target'))}; // || short-circuit", extra={"uses": self._uses(cond)})
            return

        if m == "logical_and_rel16":
            cond = self._pop_value(ins, "lhs")
            self._emit(ins, "if_goto", f"if (!({cond.render()})) goto {label_for_offset(operands.get('target'))}; // && short-circuit", extra={"uses": self._uses(cond)})
            return

        if m == "call_rel32":
            self._handle_function_call(ins)
            return

        if m == "call_linked_function":
            argc = self._consume_arg_count()
            args = self._pop_arg_slots(ins, argc or 0)
            effect = unresolved_import_effect("linked_call", argc=argc)
            self._apply_call(ins, "call", "linked_call", args, effect)
            return

        if operands.get("subopcode") is not None:
            argc = self._consume_arg_count()
            subopcode = int(operands["subopcode"])
            base_effect = builtin_effect(subopcode, argc=argc)
            count = base_effect.consumes if base_effect.consumes is not None else (argc or 0)
            args = self._pop_arg_slots(ins, count)
            call_name, call_args, effect = specialize_builtin_call(subopcode, args, base_effect)
            self._apply_call(ins, "builtin", call_name, call_args, effect)
            return

        if m in {"return", "return_local"}:
            top = self.vm.peek()
            suffix = f" {top.render()}" if top is not None else ""
            self._emit(ins, "return", f"return{suffix};", extra={"uses": self._uses(top) if top is not None else []})
            return

        if m == "halt_interpreter":
            self._emit(ins, "halt", "halt_interpreter();")
            return

        if m in {"program_restart", "program_restart_child", "program_activate", "program_reset_alt_pc", "program_stop"}:
            idx = operands.get("program_index", "?")
            self._emit(ins, "program", f"{m}({self.renderer.program_arg(idx)});")
            return

        if m == "program_prologue":
            self._bind_program_prologue(operands)
            return

        if m == "import_stub_u32":
            # Normally skipped by CFG.  If a user manually decodes a stub as a
            # program entry, show the virtual link/native endpoint instead of a fake body.
            link = operands.get("resolved_import")
            native = operands.get("resolved_native_import")
            if link:
                target = link["target"]["qualified_name"]
                self._emit(ins, "runtime_link", f"// runtime link stub -> {target}")
            elif native:
                target = native["native"]["name"]
                self._emit(ins, "native_link", f"// engine-native import stub -> {target}")
            else:
                name = operands.get("link_name") or operands.get("function_name")
                self._emit(ins, "import_stub", f"// unresolved import {name or '<unknown>'} payload={operands.get('value', '?')}")
            return

        if not ins.known:
            self._emit(ins, "unknown", f"// unknown/truncated opcode 0x{ins.opcode:02X} at 0x{ins.offset:08X}")
            return

        if m not in {"stack_frame_reset", "push_stack_frame", "pop_stack_frame"}:
            self._emit(ins, "op", f"// {m}")

    def _handle_function_call(self, ins: Any) -> None:
        operands = ins.operands or {}
        target = operands.get("target", 0)
        call_name = operands.get("target_name") or self.linker.callable_name_for_offset(target) or f"sub_{target:08X}"
        argc = self._consume_arg_count()
        target_symbol = self.linker.symbol_at(target) if isinstance(target, int) else None
        if target_symbol is not None and target_symbol.is_import and not operands.get("resolved_call"):
            native_effect = native_import_effect(target_symbol.name, argc=argc)
            if native_effect is not None:
                effect = native_effect
                call_name = effect.name
            else:
                effect = unresolved_import_effect(call_name, argc=argc)
        else:
            signature = self.linker.signature_for_offset(target) if isinstance(target, int) else FunctionSignature.unknown()
            effect = function_effect(call_name, signature, argc=argc, linked=bool(operands.get("resolved_call")))
        count = effect.consumes if effect.consumes is not None else 0
        args = self._pop_arg_slots(ins, count)
        self._apply_call(ins, "call", call_name, args, effect)

    def _apply_call(self, ins: Any, kind: str, name: str, args: list[VMSlot], effect: CallEffect) -> None:
        rendered_args = [arg.render() for arg in args]
        call_expr = f"{name}({', '.join(rendered_args)})"
        extra = {"call_effect": effect.to_dict(), "uses": self._uses(*args)}

        if effect.returns_value:
            if name in {"push_zero", "push_zero_alias"}:
                self.vm.push_int("0", value=0, metadata={"call_effect": effect.to_dict()})
                return
            if name == "push_minus_one":
                self.vm.push_int("(-1)", value=-1, metadata={"call_effect": effect.to_dict()})
                return
            if effect.statement:
                self._emit(ins, kind, f"{call_expr};", extra=extra)
                ret_name = f"ret_{self._safe_name(name)}_{ins.offset:08X}"
                self.vm.push_unknown(ret_name, type_id=effect.return_type_id, metadata={"call_effect": effect.to_dict(), "call": call_expr})
                return
            slot_kind = "slice" if effect.return_type_id == TYPE_SLICE else "value"
            self.vm.push(VMSlot(expr=call_expr, type_id=effect.return_type_id, kind=slot_kind, metadata={"call_effect": effect.to_dict()}))
            return

        self._emit(ins, kind, f"{call_expr};", extra=extra)

    def _bind_program_prologue(self, operands: dict[str, Any]) -> None:
        signature = FunctionSignature.from_prologue(
            operands.get("descriptors", []),
            signed_count=int(operands.get("signed_count", 0)),
        )
        for arg in signature.args:
            self.memory.bind_argument(index=arg.index, type_id=arg.type_id, data_offset=arg.data_offset, name=arg.name)

    def _push_decoded_value(self, ins: Any) -> None:
        operands = ins.operands or {}
        typ = operands.get("type")
        m = ins.mnemonic

        if m == "push_data_ref":
            self.vm.push_data_ref(offset=int(operands.get("data_offset", 0)), type_id=typ, storage_size=self._storage_size(typ))
            return

        expr = self.renderer.push_value(ins)
        if m in {"push_inline_span", "push_typed_span_ref", "push_inline_typed_span"}:
            begin = operands.get("data_offset")
            length = operands.get("length")
            end = begin + length - 1 if isinstance(begin, int) and isinstance(length, int) and length > 0 else None
            loc = self.memory.location(offset=begin, type_id=typ if typ is not None else TYPE_STRING, role="span", length=length) if isinstance(begin, int) else None
            self.vm.push_slice(expr, type_id=typ if typ is not None else TYPE_STRING, begin=begin, end=end, note=m, location=loc)
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
        m = ins.mnemonic

        if m == "array_index_abs":
            index = self.vm.coerce_int(self._pop_value(ins, "index"))
            base = int(operands.get("base", 0))
            expr = f"array_{base:04X}[{index.render()}]"
            self._push_typed_index_result(expr, typ, operands)
            return

        if m in {"array2_index", "array2_index_checked"}:
            index = self.vm.coerce_int(self._pop_value(ins, "index"))
            base = self.vm.get_pointer_or_slice(self._pop_value(ins, "base"))
            expr = f"{base.render()}[{index.render()}]"
            self._push_typed_index_result(expr, typ, operands)
            return

        if m == "slice_offset_ref":
            base = self.vm.get_pointer_or_slice(self._pop_value(ins, "base"))
            offset = operands.get("offset", "?")
            expr = f"{base.render()}+{offset}"
            self._push_typed_index_result(expr, typ, operands)
            return

        if m == "slice_offset_span":
            base = self.vm.get_pointer_or_slice(self._pop_value(ins, "base"))
            expr = f"span({base.render()}+{operands.get('offset', '?')}, {operands.get('length', '?')})"
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

    def _consume_arg_count(self) -> int | None:
        count = self.pending_arg_count
        self.pending_arg_count = None
        return count

    def _pop_arg_slots(self, ins: Any, count: int) -> list[VMSlot]:
        args = [self._pop_value(ins, f"arg{idx}") for idx in range(max(0, count))]
        args.reverse()
        return args

    def _pop_value(self, ins: Any, name: str) -> VMSlot:
        return self.vm.pop(f"<{name}@0x{ins.offset:08X}>")

    def _storage_size(self, typ: Any) -> int:
        if typ in {TYPE_CHAR, TYPE_STRING}:
            return 1
        if typ in {TYPE_INT, TYPE_INT_REF, TYPE_FLOAT, TYPE_FLOAT_REF}:
            return 4
        if typ == TYPE_SLICE:
            return 0x0C
        return 1

    def _uses(self, *slots: VMSlot) -> list[str]:
        uses: list[str] = []
        for slot in slots:
            if slot.location is not None:
                uses.append(slot.location.render())
            elif slot.expr and not slot.expr.startswith("<"):
                uses.append(slot.expr)
        return uses

    def _emit(self, ins: Any, kind: str, text: str, *, extra: dict[str, Any] | None = None) -> None:
        operands = dict(ins.operands or {})
        if extra:
            operands.update(extra)
        self.statements.append(
            AstStatement(
                offset=ins.offset,
                file_offset=ins.file_offset,
                kind=kind,
                text=text,
                opcode=ins.opcode,
                mnemonic=ins.mnemonic,
                operands=operands,
            )
        )

    def _safe_name(self, name: str) -> str:
        return re.sub(r"\W+", "_", name).strip("_") or "call"


def build_program_ast(
    script: MbcScript,
    program: MbcProgram,
    instructions: Iterable[Any],
    *,
    linker: MbcStaticLinker | None = None,
) -> dict[str, Any]:
    return StackAstBuilder(script, program, linker=linker).build(instructions)
