from __future__ import annotations

"""Symbolic model of the MBC VM value stack.

The native stack slots are 0x20 bytes wide.  The asm helpers exposed here model
the parts that matter for decompilation: type tags, scalar values, references,
and slice/range descriptors.  This is intentionally symbolic; it does not read
or write process memory.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


STACK_LIMIT = 0x100
SLOT_STRIDE = 0x20

TYPE_CHAR = 0x00
TYPE_STRING = 0x01
TYPE_INT = 0x10
TYPE_INT_REF = 0x11
TYPE_FLOAT = 0x20
TYPE_FLOAT_REF = 0x21
TYPE_SLICE = 0x30

SCALAR_TYPES = {TYPE_CHAR, TYPE_INT, TYPE_FLOAT}
FLOAT_TYPES = {TYPE_FLOAT, TYPE_FLOAT_REF}
INT_TYPES = {TYPE_CHAR, TYPE_INT, TYPE_INT_REF}
REFERENCE_TYPES = {TYPE_STRING, TYPE_INT_REF, TYPE_FLOAT_REF, TYPE_SLICE}



@dataclass
class VMValue:
    expr: str
    type_id: Optional[int] = None
    kind: str = "value"
    value: Any = None
    ptr: Optional[int] = None
    begin: Optional[int] = None
    end: Optional[int] = None
    storage_size: int = 1
    is_lvalue: bool = False
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_float(self) -> bool:
        return self.type_id in FLOAT_TYPES

    @property
    def is_int_like(self) -> bool:
        return self.type_id in INT_TYPES

    @property
    def is_reference(self) -> bool:
        return self.kind in {"ref", "slice", "pointer"} or self.is_lvalue or self.type_id in REFERENCE_TYPES

    def render(self) -> str:
        return self.expr

    def clone(self, **changes: Any) -> "VMValue":
        data = {
            "expr": self.expr,
            "type_id": self.type_id,
            "kind": self.kind,
            "value": self.value,
            "ptr": self.ptr,
            "begin": self.begin,
            "end": self.end,
            "storage_size": self.storage_size,
            "is_lvalue": self.is_lvalue,
            "note": self.note,
            "metadata": dict(self.metadata),
        }
        data.update(changes)
        return VMValue(**data)


class VMStackMachine:
    def __init__(self) -> None:
        self.stack: list[VMValue] = []
        self.underflows = 0
        self.overflow_warnings = 0

    def __len__(self) -> int:
        return len(self.stack)

    def residual(self) -> list[str]:
        return [value.render() for value in self.stack[-16:]]

    def push(self, value: VMValue) -> VMValue:
        if len(self.stack) >= STACK_LIMIT:
            self.overflow_warnings += 1
            self.stack.pop(0)
        self.stack.append(value)
        return value

    def push_unknown(self, expr: str, *, type_id: Optional[int] = None, kind: str = "unknown") -> VMValue:
        return self.push(VMValue(expr=expr, type_id=type_id, kind=kind))

    def push_int(self, expr: str, *, value: Any = None, note: str = "") -> VMValue:
        return self.push(VMValue(expr=expr, type_id=TYPE_INT, kind="value", value=value, storage_size=4, note=note))

    def push_float(self, expr: str, *, value: Any = None, note: str = "") -> VMValue:
        return self.push(VMValue(expr=expr, type_id=TYPE_FLOAT, kind="value", value=value, storage_size=4, note=note))

    def push_ref(self, expr: str, *, type_id: Optional[int], ptr: Optional[int] = None, storage_size: int = 4, note: str = "") -> VMValue:
        return self.push(
            VMValue(
                expr=expr,
                type_id=type_id,
                kind="ref",
                ptr=ptr,
                begin=ptr,
                storage_size=storage_size,
                is_lvalue=True,
                note=note,
            )
        )

    def push_slice(self, expr: str, *, type_id: Optional[int], begin: Optional[int] = None, end: Optional[int] = None, note: str = "") -> VMValue:
        return self.push(
            VMValue(
                expr=expr,
                type_id=type_id,
                kind="slice",
                ptr=begin,
                begin=begin,
                end=end,
                storage_size=0x0C,
                is_lvalue=True,
                note=note,
            )
        )

    def pop(self, placeholder: str) -> VMValue:
        if self.stack:
            return self.stack.pop()
        self.underflows += 1
        return VMValue(expr=placeholder, type_id=None, kind="underflow", note="symbolic stack underflow")

    def peek(self) -> Optional[VMValue]:
        return self.stack[-1] if self.stack else None

    def swap_top_two(self) -> bool:
        if len(self.stack) < 2:
            self.underflows += 1
            return False
        self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
        return True

    def coerce_int(self, value: VMValue) -> VMValue:
        if value.type_id == TYPE_INT:
            return value.clone(type_id=TYPE_INT, kind="value", storage_size=4)
        if value.type_id == TYPE_CHAR:
            return value.clone(expr=f"int8({value.expr})", type_id=TYPE_INT, kind="value", storage_size=4)
        if value.type_id == TYPE_FLOAT:
            return value.clone(expr=f"int({value.expr})", type_id=TYPE_INT, kind="value", storage_size=4)
        return value.clone(expr=f"int({value.expr})", type_id=TYPE_INT, kind="value", storage_size=4)

    def coerce_float(self, value: VMValue) -> VMValue:
        if value.type_id == TYPE_FLOAT:
            return value.clone(type_id=TYPE_FLOAT, kind="value", storage_size=4)
        return value.clone(expr=f"float({value.expr})", type_id=TYPE_FLOAT, kind="value", storage_size=4)

    def get_pointer_or_slice(self, value: VMValue) -> VMValue:
        # sub_47AC70 returns a pointer to slot+0x14.  If the low nibble of the
        # type is zero, the native helper clears the range fields, so scalar
        # values are modelled as a direct pointer without begin/end bounds.
        if value.type_id is not None and (value.type_id & 0x0F) == 0:
            return value.clone(kind="pointer", begin=None, end=None, is_lvalue=True)
        return value.clone(kind="slice" if value.end is not None else "pointer", is_lvalue=True)

    def force_int_top(self) -> None:
        if self.stack:
            self.stack[-1] = self.stack[-1].clone(type_id=TYPE_INT)

    def force_two_ints(self) -> None:
        for idx in range(1, min(2, len(self.stack)) + 1):
            self.stack[-idx] = self.stack[-idx].clone(type_id=TYPE_INT)
