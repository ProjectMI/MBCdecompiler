from __future__ import annotations

"""Symbolic model of the MBC VM stack and data memory.

The native stack uses 0x20-byte cells, but a decompiler needs a semantic view:
values, references, slices and pointers are different things.  The classes here
keep that distinction so AST construction no longer treats every stack item as a
pre-rendered string.
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

TYPE_NAMES = {
    TYPE_CHAR: "i8/char",
    TYPE_STRING: "span/string",
    TYPE_INT: "int32",
    TYPE_INT_REF: "int32_ref_or_span",
    TYPE_FLOAT: "float32",
    TYPE_FLOAT_REF: "float32_ref_or_span",
    TYPE_SLICE: "slice_descriptor",
}

SCALAR_TYPES = {TYPE_CHAR, TYPE_INT, TYPE_FLOAT}
FLOAT_TYPES = {TYPE_FLOAT, TYPE_FLOAT_REF}
INT_TYPES = {TYPE_CHAR, TYPE_INT, TYPE_INT_REF}
REFERENCE_TYPES = {TYPE_STRING, TYPE_INT_REF, TYPE_FLOAT_REF, TYPE_SLICE}


def type_name(type_id: int | None) -> str:
    return "unknown" if type_id is None else TYPE_NAMES.get(type_id, f"type_{type_id}")


@dataclass(frozen=True)
class MemoryLocation:
    module: str
    section: str
    offset: int
    type_id: int | None = None
    name: str | None = None
    role: str = "data"
    length: int | None = None

    @property
    def type_name(self) -> str:
        return type_name(self.type_id)

    def render(self) -> str:
        if self.name:
            return self.name
        return f"{self.section}_{self.offset:04X}"

    def to_dict(self) -> dict[str, object]:
        return {
            "module": self.module,
            "section": self.section,
            "offset": self.offset,
            "type_id": self.type_id,
            "type_name": self.type_name,
            "name": self.name or self.render(),
            "role": self.role,
            "length": self.length,
        }


class SymbolicDataMemory:
    """Symbolic view over an MBC data section.

    Program prologues bind argument descriptors to concrete data offsets.  Later
    ``push_data_ref`` operations for those offsets render as ``argN`` instead of
    anonymous data cells.  Other data offsets are stable symbolic variables.
    """

    def __init__(self, *, module_name: str, data: bytes):
        self.module_name = module_name
        self.data = data
        self._bindings: dict[int, MemoryLocation] = {}
        self._versions: dict[str, int] = {}

    def bind_argument(self, *, index: int, type_id: int, data_offset: int, name: str | None = None) -> MemoryLocation:
        loc = MemoryLocation(
            module=self.module_name,
            section="data",
            offset=data_offset,
            type_id=type_id,
            name=name or f"arg{index}",
            role="arg",
        )
        self._bindings[data_offset] = loc
        return loc

    def location(self, *, offset: int, type_id: int | None = None, role: str = "data", length: int | None = None) -> MemoryLocation:
        bound = self._bindings.get(offset)
        if bound is not None:
            if type_id is None or bound.type_id == type_id:
                return bound
            return MemoryLocation(
                module=bound.module,
                section=bound.section,
                offset=bound.offset,
                type_id=type_id,
                name=bound.name,
                role=bound.role,
                length=length or bound.length,
            )
        prefix = "span" if role == "span" or type_id in {TYPE_STRING, TYPE_SLICE} else "var"
        return MemoryLocation(
            module=self.module_name,
            section="data",
            offset=offset,
            type_id=type_id,
            name=f"{prefix}_{offset:04X}",
            role=role,
            length=length,
        )

    def define(self, location: MemoryLocation | None, fallback: str) -> str:
        key = location.render() if location is not None else fallback
        self._versions[key] = self._versions.get(key, 0) + 1
        return f"{key}#{self._versions[key]}"

    def preview(self, off: Any, max_len: int | None = None) -> str:
        if not isinstance(off, int) or off < 0 or off >= len(self.data):
            return ""
        limit = len(self.data) if max_len is None or max_len <= 0 else min(len(self.data), off + max_len)
        end = self.data.find(b"\x00", off, limit)
        if end < 0:
            end = min(limit, off + 48)
        raw = self.data[off:end]
        if not raw:
            return ""
        # Most MBC data sections use 0xFF fill bytes; cp1251 renders those as a
        # wall of 'я', which is not a useful source preview.
        if raw.count(0xFF) > max(1, len(raw) // 2):
            return ""
        text = raw.decode("cp1251", errors="replace")
        if any(ord(ch) < 32 and ch not in "\t\r\n" for ch in text):
            return ""
        return text[:80]

    def bindings(self) -> list[dict[str, object]]:
        return [loc.to_dict() for _, loc in sorted(self._bindings.items())]


@dataclass
class VMSlot:
    expr: str
    type_id: Optional[int] = None
    kind: str = "value"  # value | ref | slice | pointer | unknown | underflow
    value: Any = None
    location: MemoryLocation | None = None
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

    @property
    def type_name(self) -> str:
        return type_name(self.type_id)

    def render(self) -> str:
        return self.expr

    def to_dict(self) -> dict[str, object]:
        return {
            "expr": self.expr,
            "type_id": self.type_id,
            "type_name": self.type_name,
            "kind": self.kind,
            "value": self.value,
            "location": None if self.location is None else self.location.to_dict(),
            "ptr": self.ptr,
            "begin": self.begin,
            "end": self.end,
            "storage_size": self.storage_size,
            "is_lvalue": self.is_lvalue,
            "note": self.note,
            "metadata": self.metadata,
        }

    def clone(self, **changes: Any) -> "VMSlot":
        data = {
            "expr": self.expr,
            "type_id": self.type_id,
            "kind": self.kind,
            "value": self.value,
            "location": self.location,
            "ptr": self.ptr,
            "begin": self.begin,
            "end": self.end,
            "storage_size": self.storage_size,
            "is_lvalue": self.is_lvalue,
            "note": self.note,
            "metadata": dict(self.metadata),
        }
        data.update(changes)
        return VMSlot(**data)


# Backwards-compatible alias for older imports inside local experiments.  The
# canonical name in the VM model is VMSlot.
VMValue = VMSlot


class VMStackMachine:
    def __init__(self, *, memory: SymbolicDataMemory | None = None) -> None:
        self.memory = memory or SymbolicDataMemory(module_name="<unknown>", data=b"")
        self.stack: list[VMSlot] = []
        self.underflows = 0
        self.overflow_warnings = 0

    def __len__(self) -> int:
        return len(self.stack)

    def residual(self) -> list[str]:
        return [value.render() for value in self.stack[-16:]]

    def reset_frame(self) -> None:
        self.stack.clear()

    def push(self, value: VMSlot) -> VMSlot:
        if len(self.stack) >= STACK_LIMIT:
            self.overflow_warnings += 1
            self.stack.pop(0)
        self.stack.append(value)
        return value

    def push_unknown(self, expr: str, *, type_id: Optional[int] = None, kind: str = "unknown", metadata: dict[str, Any] | None = None) -> VMSlot:
        return self.push(VMSlot(expr=expr, type_id=type_id, kind=kind, metadata=metadata or {}))

    def push_int(self, expr: str, *, value: Any = None, note: str = "", metadata: dict[str, Any] | None = None) -> VMSlot:
        return self.push(VMSlot(expr=expr, type_id=TYPE_INT, kind="value", value=value, storage_size=4, note=note, metadata=metadata or {}))

    def push_float(self, expr: str, *, value: Any = None, note: str = "", metadata: dict[str, Any] | None = None) -> VMSlot:
        return self.push(VMSlot(expr=expr, type_id=TYPE_FLOAT, kind="value", value=value, storage_size=4, note=note, metadata=metadata or {}))

    def push_ref(self, expr: str, *, type_id: Optional[int], ptr: Optional[int] = None, storage_size: int = 4, note: str = "", location: MemoryLocation | None = None, metadata: dict[str, Any] | None = None) -> VMSlot:
        return self.push(
            VMSlot(
                expr=expr,
                type_id=type_id,
                kind="ref",
                location=location,
                ptr=ptr,
                begin=ptr,
                storage_size=storage_size,
                is_lvalue=True,
                note=note,
                metadata=metadata or {},
            )
        )

    def push_data_ref(self, *, offset: int, type_id: int | None, storage_size: int = 4, note: str = "data_ref") -> VMSlot:
        loc = self.memory.location(offset=offset, type_id=type_id)
        return self.push_ref(loc.render(), type_id=type_id, ptr=offset, storage_size=storage_size, note=note, location=loc)

    def push_slice(self, expr: str, *, type_id: Optional[int], begin: Optional[int] = None, end: Optional[int] = None, note: str = "", location: MemoryLocation | None = None, metadata: dict[str, Any] | None = None) -> VMSlot:
        return self.push(
            VMSlot(
                expr=expr,
                type_id=type_id,
                kind="slice",
                location=location,
                ptr=begin,
                begin=begin,
                end=end,
                storage_size=0x0C,
                is_lvalue=True,
                note=note,
                metadata=metadata or {},
            )
        )

    def pop(self, placeholder: str) -> VMSlot:
        if self.stack:
            return self.stack.pop()
        self.underflows += 1
        return VMSlot(expr=placeholder, type_id=None, kind="underflow", note="symbolic stack underflow")

    def peek(self) -> Optional[VMSlot]:
        return self.stack[-1] if self.stack else None

    def swap_top_two(self) -> bool:
        if len(self.stack) < 2:
            self.underflows += 1
            return False
        self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
        return True

    def coerce_int(self, value: VMSlot) -> VMSlot:
        if value.type_id == TYPE_INT:
            return value.clone(type_id=TYPE_INT, kind="value", storage_size=4)
        if value.type_id == TYPE_CHAR:
            return value.clone(expr=f"int8({value.expr})", type_id=TYPE_INT, kind="value", storage_size=4, is_lvalue=False)
        if value.type_id == TYPE_FLOAT:
            return value.clone(expr=f"int({value.expr})", type_id=TYPE_INT, kind="value", storage_size=4, is_lvalue=False)
        return value.clone(expr=f"int({value.expr})", type_id=TYPE_INT, kind="value", storage_size=4, is_lvalue=False)

    def coerce_float(self, value: VMSlot) -> VMSlot:
        if value.type_id == TYPE_FLOAT:
            return value.clone(type_id=TYPE_FLOAT, kind="value", storage_size=4, is_lvalue=False)
        return value.clone(expr=f"float({value.expr})", type_id=TYPE_FLOAT, kind="value", storage_size=4, is_lvalue=False)

    def address_of(self, value: VMSlot) -> VMSlot:
        return value.clone(expr=f"&{value.render()}", kind="pointer", is_lvalue=True, metadata={**value.metadata, "address_of": True})

    def deref(self, value: VMSlot) -> VMSlot:
        ptr = self.get_pointer_or_slice(value)
        return ptr.clone(expr=f"*{ptr.render()}", kind="value", is_lvalue=False, metadata={**ptr.metadata, "deref": True})

    def store(self, target: VMSlot, value: VMSlot) -> str:
        return self.memory.define(target.location, target.render())

    def get_pointer_or_slice(self, value: VMSlot) -> VMSlot:
        # sub_47AC70 returns a pointer to slot+0x14. If the low nibble of the
        # type is zero, the native helper clears range fields, so scalar values
        # are modelled as direct pointers without begin/end bounds.
        if value.type_id is not None and (value.type_id & 0x0F) == 0:
            return value.clone(kind="pointer", begin=None, end=None, is_lvalue=True)
        return value.clone(kind="slice" if value.end is not None else "pointer", is_lvalue=True)

    def force_int_top(self) -> None:
        if self.stack:
            self.stack[-1] = self.stack[-1].clone(type_id=TYPE_INT)

    def force_two_ints(self) -> None:
        for idx in range(1, min(2, len(self.stack)) + 1):
            self.stack[-idx] = self.stack[-idx].clone(type_id=TYPE_INT)
