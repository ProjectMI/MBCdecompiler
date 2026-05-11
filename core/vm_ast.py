from __future__ import annotations

"""Symbolic VM stack, data memory and low-level pseudo-AST builder.

The previous split had several tiny files that all described adjacent parts of
one pass: VM slots, AST records, expression rendering, normalization and the
actual stack-AST interpreter.  They live here now so the symbolic execution
layer can be read as one unit.
"""

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Iterable, Optional

from .calls import (
    CallEffect,
    builtin_effect,
    function_effect,
    native_import_effect,
    specialize_builtin_call,
    unresolved_import_effect,
)
from .common import (
    CODE_FILE_OFFSET,
    TYPE_BASE_NAMES,
    TYPE_CHAR,
    TYPE_FLOAT,
    TYPE_FLOAT_REF,
    TYPE_FLOAT_REF_REF,
    TYPE_INT,
    TYPE_INT_REF,
    TYPE_INT_REF_REF,
    TYPE_NAMES,
    TYPE_SLICE,
    TYPE_SLICE_REF,
    TYPE_STRING,
    TYPE_STRING_REF,
    deref_storage_size_for_type,
    dereferenced_type,
    is_reference_type,
    reference_type,
    storage_size_for_type,
    type_name,
)
from .linker import FunctionSignature, MbcStaticLinker
from .loader import MbcProgram, MbcScript
from .opcodes import BINARY_AST_OPS, UNARY_AST_OPS

STACK_LIMIT = 0x100
SLOT_STRIDE = 0x20

# Recovered from the big IDA ASM slice around sub_477500/sub_47AA30..sub_47AF10.
# This is documentation data used by the linker/VM layer to keep the symbolic
# model aligned with the real interpreter contract, without introducing another
# tiny module just for notes.
VM_HELPER_CONTRACTS: dict[str, str] = {
    "sub_477500": "builtin dispatcher: reads subopcode byte, stores old stack top in dword_9C6430, subtracts argc (dword_86232C) from dword_474C684, and starts the argument cursor dword_3E6D19C at the first argument slot",
    "sub_47AA30": "popint from normal VM stack top; coerces char/int/float slot to int32 and decrements dword_474C684",
    "sub_47AA90": "popsliceref from normal VM stack top; returns pointer to slot descriptor and normalizes empty scalar-ish descriptors",
    "sub_47AAF0": "read next builtin argument as int32 through dword_3E6D19C; advances the argument cursor but does not alter the normal stack top",
    "sub_47AB60": "read next builtin argument as float32 through dword_3E6D19C; advances the argument cursor; sub_47AF10 is an alias jump to this helper",
    "sub_47AC00": "read next builtin argument as slice descriptor by value into a caller buffer; advances dword_3E6D19C",
    "sub_47AC70": "read next builtin argument as pointer/ref/slice descriptor and return a pointer to slot+0x14; advances dword_3E6D19C",
    "sub_47ACD0": "push int32 slot with type 0x10 and one-cell storage markers",
    "sub_47AD30": "push float32 slot with type 0x20 and one-cell storage markers",
    "sub_47AD90": "push descriptor/ref/slice slot from ecx[0..8] with type in edx",
    "sub_47AE00": "push int32 like sub_47ACD0 and also store the value in dword_9C643C as the last process/native result",
    "loc_47AE60": "push pointer/string descriptor: type 1, base-relative begin/end; null pointer becomes an empty descriptor",
    "sub_47AF10": "alias jump to sub_47AB60",
}

SCALAR_TYPES = {TYPE_CHAR, TYPE_INT, TYPE_FLOAT}
FLOAT_TYPES = {TYPE_FLOAT, TYPE_FLOAT_REF, TYPE_FLOAT_REF_REF}
INT_TYPES = {TYPE_CHAR, TYPE_INT, TYPE_INT_REF, TYPE_INT_REF_REF}



REFERENCE_TYPES = {tid for tid in TYPE_NAMES if is_reference_type(tid)}

PSEUDO_TYPE_NAMES: dict[int, str] = {
    TYPE_CHAR: "char",
    TYPE_STRING: "string",
    TYPE_STRING_REF: "string_ref",
    TYPE_INT: "int",
    TYPE_INT_REF: "int_ref",
    TYPE_INT_REF_REF: "int_ref_ref",
    TYPE_FLOAT: "float",
    TYPE_FLOAT_REF: "float_ref",
    TYPE_FLOAT_REF_REF: "float_ref_ref",
    TYPE_SLICE: "record",
    TYPE_SLICE_REF: "record_ref",
}


def pseudo_type_name(type_id: int | None) -> str:
    if type_id in PSEUDO_TYPE_NAMES:
        return PSEUDO_TYPE_NAMES[type_id]  # type: ignore[index]
    return type_name(type_id).replace("span/", "").replace("_or_span", "")


def _escape_string_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{escaped}"'


@dataclass(frozen=True)
class MemoryLocation:
    module: str
    section: str
    offset: int
    type_id: int | None = None
    name: str | None = None
    role: str = "data"
    length: int | None = None
    scope: str = "global"

    @property
    def type_name(self) -> str:
        return type_name(self.type_id)

    @property
    def pseudo_type_name(self) -> str:
        return pseudo_type_name(self.type_id)

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
            "scope": self.scope,
        }


class SymbolicDataMemory:
    """Symbolic view over an MBC data section.

    Program prologues bind argument descriptors to concrete data offsets.  Later
    ``push_data_ref`` operations for those offsets render as ``argN`` instead of
    anonymous data cells.  Other data offsets are named with recovered scope:
    ``local_*`` for program-private slots and ``global_*`` for slots shared by
    several programs.
    """

    def __init__(self, *, module_name: str, data: bytes, scope_map: dict[int, str] | None = None):
        self.module_name = module_name
        self.data = data
        self.scope_map = scope_map or {}
        self._bindings: dict[int, MemoryLocation] = {}
        self._locations: dict[tuple[int, str], MemoryLocation] = {}
        self._versions: dict[str, int] = {}

    def bind_argument(self, *, index: int, type_id: int, data_offset: int, name: str | None = None) -> MemoryLocation:
        loc = MemoryLocation(
            module=self.module_name,
            section="data",
            offset=data_offset,
            type_id=type_id,
            name=name or f"arg{index}",
            role="arg",
            scope="arg",
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
                scope=bound.scope,
            )

        category = self._category_for(role, type_id)
        key = (offset, category)
        cached = self._locations.get(key)
        if cached is not None:
            new_type_id = type_id if type_id is not None and cached.type_id is None else cached.type_id
            new_length = cached.length
            if length is not None and (cached.length is None or length > cached.length):
                new_length = length
            if new_type_id != cached.type_id or new_length != cached.length:
                cached = MemoryLocation(
                    module=cached.module,
                    section=cached.section,
                    offset=cached.offset,
                    type_id=new_type_id,
                    name=cached.name,
                    role=cached.role,
                    length=new_length,
                    scope=cached.scope,
                )
                self._locations[key] = cached
            return cached

        scope = "const" if role == "const_span" else self.scope_map.get(offset, "global")
        loc = MemoryLocation(
            module=self.module_name,
            section="data",
            offset=offset,
            type_id=type_id,
            name=self._name_for(offset=offset, category=category, scope=scope, type_id=type_id),
            role=role,
            length=length,
            scope=scope,
        )
        self._locations[key] = loc
        return loc

    def declarations(self, *, scope: str | None = None) -> list[MemoryLocation]:
        locations = [loc for loc in self._locations.values() if loc.scope != "const" and loc.role != "arg"]
        if scope is not None:
            locations = [loc for loc in locations if loc.scope == scope]
        return sorted(locations, key=lambda loc: (loc.offset, loc.role, loc.name or ""))

    @staticmethod
    def declaration_text(location: MemoryLocation) -> str:
        typ = location.pseudo_type_name
        name = location.render()
        extent, note = SymbolicDataMemory._declaration_extent(location)
        if extent is not None and extent > 1:
            name = f"{name}[{extent}]"
        suffix_parts = [f"data[0x{location.offset:04X}]"]
        if note:
            suffix_parts.append(note)
        return f"{typ} {name}; // {', '.join(suffix_parts)}"

    @staticmethod
    def _declaration_extent(location: MemoryLocation) -> tuple[int | None, str]:
        length = location.length
        if not isinstance(length, int) or length <= 1:
            return None, ""
        if location.role == "array":
            return length, ""
        if location.type_id == TYPE_STRING:
            return length, ""
        if location.type_id == TYPE_SLICE and length % 12 == 0:
            return length // 12, f"{length} bytes"
        if location.type_id in {TYPE_INT, TYPE_FLOAT} and length % 4 == 0:
            return length // 4, f"{length} bytes"
        return length, f"{length} bytes"

    @staticmethod
    def _category_for(role: str, type_id: int | None) -> str:
        if role == "array":
            return "array"
        if role in {"span", "const_span"}:
            return "span"
        if isinstance(type_id, int) and type_id not in {TYPE_CHAR, TYPE_INT, TYPE_FLOAT}:
            return "span"
        return "var"

    @staticmethod
    def _name_for(*, offset: int, category: str, scope: str, type_id: int | None = None) -> str:
        if scope == "const":
            return f"str_{offset:04X}"
        local = scope == "local"
        prefix = "" if local else "g_"
        if category == "array":
            return f"{prefix}arr_{offset:04X}"
        if category == "span":
            if type_id == TYPE_STRING:
                stem = "buf"
            elif type_id == TYPE_SLICE:
                stem = "rec"
            else:
                stem = "span"
            return f"{prefix}{stem}_{offset:04X}"
        return f"v_{offset:04X}" if local else f"g_{offset:04X}"

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
        return self.kind in {"ref", "slice", "pointer"} or self.is_lvalue or is_reference_type(self.type_id)

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
        new_type = reference_type(value.type_id)
        return value.clone(
            expr=f"&{value.render()}",
            type_id=new_type,
            kind="pointer",
            storage_size=0x0C,
            is_lvalue=True,
            metadata={**value.metadata, "address_of": True},
        )

    def deref(self, value: VMSlot) -> VMSlot:
        ptr = self.get_pointer_or_slice(value)
        old_type = ptr.type_id
        new_type = dereferenced_type(old_type)
        storage_size = 1 if old_type == TYPE_STRING else deref_storage_size_for_type(new_type)
        return ptr.clone(
            expr=f"*{ptr.render()}",
            type_id=new_type,
            kind="value",
            storage_size=storage_size,
            is_lvalue=False,
            metadata={**ptr.metadata, "deref": True},
        )

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


def ast_payload(
    *,
    statements: list[AstStatement],
    residual_stack: list[str],
    underflows: int,
    warning: str | None = None,
) -> dict[str, Any]:
    return {
        "format": "experimental_stack_ast_v0",
        "warning": warning or (
            "This is a symbolic stack AST seed, not a final decompilation. "
            "It is meant to preserve expression intent and control-flow anchors for later structuring."
        ),
        "statement_count": len(statements),
        "underflow_placeholders": underflows,
        "residual_stack": list(residual_stack[-16:]),
        "statements": [stmt.to_dict() for stmt in statements],
        "source": "\n".join(stmt.text for stmt in statements),
    }

_BRANCH_KINDS = {"goto", "if_goto", "yield"}
_ARGC_META_PREFIX = "// argc ="
_TYPE_COMMENT_RE = re.compile(
    r"\s*/\*\s*(?:char|string|int32|int32_ref_or_span|float|float_ref|span/string|slice_descriptor|type_\d+)\s*\*/"
)


def label_for_offset(offset: Any) -> str:
    """Return the canonical pseudo-source label for a code offset."""
    if isinstance(offset, int):
        return f"loc_{offset:08X}"
    return "loc_UNKNOWN"


def normalize_ast_statements(statements: Iterable[AstStatement]) -> list[AstStatement]:
    """Return a source-oriented AST statement list.

    Normalization currently does four small things:
    * drops transient `set_arg_count` comments (`// argc = N`);
    * removes inline comments that only repeat scalar MBC type names;
    * collapses linear runs of coroutine yields into one visible wait marker;
    * inserts `loc_XXXXXXXX:` labels before visible branch targets.

    Labels are inserted before the first emitted statement at the target offset.
    If the exact target instruction was suppressed as metadata/no-op, the label
    is placed before the next visible statement, which keeps the generated text
    navigable without reintroducing low-level VM noise.
    """
    cleaned = [_normalize_statement(stmt) for stmt in statements if not _is_argc_meta(stmt)]
    cleaned = _collapse_consecutive_yields(cleaned)
    return _insert_branch_labels(cleaned)


def _is_argc_meta(stmt: AstStatement) -> bool:
    return stmt.kind == "meta" and stmt.text.strip().startswith(_ARGC_META_PREFIX)


def _normalize_statement(stmt: AstStatement) -> AstStatement:
    text = _TYPE_COMMENT_RE.sub("", stmt.text)
    if text == stmt.text:
        return stmt
    return AstStatement(
        offset=stmt.offset,
        file_offset=stmt.file_offset,
        kind=stmt.kind,
        text=text,
        opcode=stmt.opcode,
        mnemonic=stmt.mnemonic,
        operands=dict(stmt.operands or {}),
    )


def _collapse_consecutive_yields(statements: list[AstStatement]) -> list[AstStatement]:
    """Coalesce linear `yield_program` chains without losing wait count.

    The runtime saves PC after each `|`, so `|||| work();` means four scheduler
    slices pass before `work()` runs.  Emitting all intermediate resume labels is
    noisy, but treating the chain as a single ordinary yield is wrong.  This
    pass replaces only unbranched, adjacent yield-resume chains with one
    pseudo-source marker carrying the original count and final resume target.

    If a non-yield branch targets an intermediate yield, the chain is split at
    that target because callers entering the middle observe a different delay.
    """
    if not statements:
        return statements

    by_offset = {stmt.offset: stmt for stmt in statements}
    protected_targets = {
        target
        for stmt in statements
        if stmt.kind in _BRANCH_KINDS - {"yield"}
        for target in [stmt.operands.get("target")]
        if isinstance(target, int)
    }

    result: list[AstStatement] = []
    i = 0
    while i < len(statements):
        first = statements[i]
        if first.kind != "yield":
            result.append(first)
            i += 1
            continue

        chain = [first]
        final_target = first.operands.get("target")
        j = i + 1
        while isinstance(final_target, int):
            if final_target in protected_targets:
                break
            next_stmt = by_offset.get(final_target)
            if next_stmt is None or next_stmt.kind != "yield":
                break
            if j >= len(statements) or statements[j] is not next_stmt:
                # Do not fold across reordered output or hidden visible statements.
                break
            chain.append(next_stmt)
            final_target = next_stmt.operands.get("target")
            j += 1

        if len(chain) == 1:
            result.append(first)
            i += 1
            continue

        result.append(_make_coalesced_yield_statement(chain, final_target))
        i += len(chain)

    return result


def _make_coalesced_yield_statement(chain: list[AstStatement], final_target: object) -> AstStatement:
    first = chain[0]
    count = len(chain)
    operands = dict(first.operands or {})
    operands["target"] = final_target
    operands["yield_count"] = count
    operands["coalesced_offsets"] = [stmt.offset for stmt in chain]
    if isinstance(final_target, int):
        suffix = f" // suspend x{count}; resumes at {label_for_offset(final_target)}"
    else:
        suffix = f" // suspend x{count}; resumes from saved PC"
    return AstStatement(
        offset=first.offset,
        file_offset=first.file_offset,
        kind="yield",
        text=f"yield_program();{suffix}",
        opcode=first.opcode,
        mnemonic=first.mnemonic,
        operands=operands,
    )


def _insert_branch_labels(statements: list[AstStatement]) -> list[AstStatement]:
    targets = sorted(
        {
            stmt.operands.get("target")
            for stmt in statements
            if stmt.kind in _BRANCH_KINDS and isinstance(stmt.operands.get("target"), int)
        }
    )
    if not targets:
        return statements

    result: list[AstStatement] = []
    emitted: set[int] = set()

    for stmt in statements:
        for target in targets:
            if target in emitted:
                continue
            if target <= stmt.offset:
                result.append(_make_label_statement(target))
                emitted.add(target)
        result.append(stmt)

    for target in targets:
        if target not in emitted:
            result.append(_make_label_statement(target, unresolved=True))
            emitted.add(target)

    return result


def _make_label_statement(offset: int, *, unresolved: bool = False) -> AstStatement:
    suffix = " // target outside emitted statement stream" if unresolved else ""
    return AstStatement(
        offset=offset,
        file_offset=offset + CODE_FILE_OFFSET,
        kind="label",
        text=f"{label_for_offset(offset)}:{suffix}",
        opcode=-1,
        mnemonic="label",
        operands={"target": offset},
    )

class AstExpressionRenderer:
    def __init__(self, script: MbcScript, program: MbcProgram, *, memory: SymbolicDataMemory | None = None, scope_map: dict[int, str] | None = None):
        self.script = script
        self.program = program
        self.memory = memory or SymbolicDataMemory(module_name=script.path.stem, data=script.data, scope_map=scope_map)

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
            return self.span_expr(off, length, type_id=TYPE_STRING, role="const_span")

        if m in {"push_typed_span_ref", "push_inline_typed_span"}:
            off = operands.get("data_offset", 0)
            length = operands.get("length", 0)
            return self.span_expr(off, length, type_id=typ, role="span")

        return f"{m}(...)"

    def span_expr(self, off: Any, length: Any, type_id: int | None = None, *, role: str = "span") -> str:
        preview = self.data_preview(off, length if isinstance(length, int) else None)
        if role == "const_span" and preview:
            return _escape_string_literal(preview)
        if isinstance(off, int):
            loc = self.memory.location(offset=off, type_id=type_id, role=role, length=length if isinstance(length, int) else None)
            return loc.render()
        return f"span[{off!r}]"

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

COMPARISON_OPS = {0xF0, 0xED, 0x3E, 0x3C, 0xE1, 0xEC}
REF_TYPES = {TYPE_STRING, TYPE_INT_REF, TYPE_FLOAT_REF, TYPE_SLICE}  # legacy; use is_reference_type() for recovered high-depth refs


class StackAstBuilder:
    """Interpret decoded instructions as a conservative symbolic stack AST seed."""

    def __init__(self, script: MbcScript, program: MbcProgram, *, linker: MbcStaticLinker | None = None, scope_map: dict[int, str] | None = None):
        self.script = script
        self.program = program
        self.linker = linker or MbcStaticLinker(script)
        self.memory = SymbolicDataMemory(module_name=script.path.stem, data=script.data, scope_map=scope_map)
        self.renderer = AstExpressionRenderer(script, program, memory=self.memory, scope_map=scope_map)
        self.vm = VMStackMachine(memory=self.memory)
        self.statements: list[AstStatement] = []
        self.pending_arg_count: Optional[int] = None

    def build(self, instructions: Iterable[Any]) -> dict[str, Any]:
        for ins in instructions:
            self._visit(ins)
        self._flush_pending_side_effect_slots()
        normalized = normalize_ast_statements(self.statements)
        declarations = self._declaration_statements(scope="local")
        payload = ast_payload(
            statements=declarations + normalized,
            residual_stack=self.vm.residual(),
            underflows=self.vm.underflows,
        )
        payload["memory_bindings"] = self.memory.bindings()
        payload["declarations"] = [loc.to_dict() for loc in self.memory.declarations()]
        return payload

    def _declaration_statements(self, *, scope: str) -> list[AstStatement]:
        locations = self.memory.declarations(scope=scope)
        if not locations:
            return []
        statements = [
            AstStatement(
                offset=self.program.start,
                file_offset=self.program.file_start,
                kind="decl_header",
                text=f"// {scope}s",
                opcode=-1,
                mnemonic="declaration",
                operands={},
            )
        ]
        for loc in locations:
            statements.append(
                AstStatement(
                    offset=self.program.start,
                    file_offset=self.program.file_start,
                    kind="decl",
                    text=SymbolicDataMemory.declaration_text(loc),
                    opcode=-1,
                    mnemonic="declaration",
                    operands={"declaration": loc.to_dict()},
                )
            )
        return statements

    def _visit(self, ins: Any) -> None:
        op = ins.opcode
        m = ins.mnemonic
        operands = ins.operands or {}

        if m == "set_arg_count":
            self.pending_arg_count = int(operands.get("value", 0))
            return

        if m == "stack_frame_reset":
            self._flush_pending_side_effect_slots(fallback_ins=ins)
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
            self._flush_pending_side_effect_slots(fallback_ins=ins)
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

        if m == "discard_value":
            discarded = self._pop_value(ins, "discard")
            self._emit_side_effect_call_if_discarded(discarded, fallback_ins=ins)
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
            self._flush_pending_side_effect_slots(except_slot=top, fallback_ins=ins)
            return_expr = self._return_expr_for_slot(top)
            suffix = f" {return_expr}" if return_expr else ""
            self._emit(ins, "return", f"return{suffix};", extra={"uses": self._uses(top) if top is not None else []})
            return

        if m == "halt_interpreter":
            self._flush_pending_side_effect_slots(fallback_ins=ins)
            self._emit(ins, "halt", "halt_interpreter();")
            return

        if m == "yield_program":
            self._flush_pending_side_effect_slots(fallback_ins=ins)
            self.vm.reset_frame()
            self.pending_arg_count = None
            resume_target = operands.get("resume_target")
            suffix = f" // suspend; resumes at {label_for_offset(resume_target)}" if isinstance(resume_target, int) else " // suspend; resumes from saved PC"
            self._emit(ins, "yield", f"yield_program();{suffix}", extra={"target": resume_target} if isinstance(resume_target, int) else None)
            return

        if m == "end_program":
            self._flush_pending_side_effect_slots(fallback_ins=ins)
            self._emit(ins, "op", "// end_program")
            return

        if m in {"program_restart", "program_restart_child", "program_activate", "program_reset_alt_pc", "program_stop"}:
            self._flush_pending_side_effect_slots(fallback_ins=ins)
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
        uses = self._uses(*args)
        extra = {"call_effect": effect.to_dict(), "uses": uses}

        if not effect.returns_value and not effect.side_effects:
            # Recovered no-op/default native dispatch branch: it consumes its
            # selector arguments, pushes nothing and has no VM-visible effect.
            return

        if effect.returns_value:
            if name in {"push_zero", "push_zero_alias"}:
                self.vm.push_int("0", value=0, metadata={"call_effect": effect.to_dict()})
                return
            if name == "push_minus_one":
                self.vm.push_int("(-1)", value=-1, metadata={"call_effect": effect.to_dict()})
                return

            embedded_side_effect = any(arg.metadata.get("emit_on_discard") for arg in args)
            metadata = {
                "call_effect": effect.to_dict(),
                "call": call_expr,
                "call_kind": kind,
                "call_offset": ins.offset,
                "call_file_offset": ins.file_offset,
                "call_opcode": ins.opcode,
                "call_mnemonic": ins.mnemonic,
                "uses": uses,
                # Inline when consumed, but emit as a standalone statement if the
                # returned VM slot is later discarded by stack_frame_reset / discard_value.
                "emit_on_discard": bool(effect.statement or embedded_side_effect),
            }
            slot_kind = "slice" if effect.return_type_id == TYPE_SLICE else "value"
            self.vm.push(VMSlot(expr=call_expr, type_id=effect.return_type_id, kind=slot_kind, metadata=metadata))
            return

        self._emit(ins, kind, f"{call_expr};", extra=extra)

    def _flush_pending_side_effect_slots(self, *, except_slot: VMSlot | None = None, fallback_ins: Any | None = None) -> None:
        for slot in list(self.vm.stack):
            if except_slot is not None and slot is except_slot:
                continue
            self._emit_side_effect_call_if_discarded(slot, fallback_ins=fallback_ins)

    def _return_expr_for_slot(self, slot: VMSlot | None) -> str:
        if slot is None:
            return ""

        metadata = slot.metadata
        if metadata.get("emit_on_discard"):
            # Returning the slot is a real use of the value produced by the call.
            # The call is already represented in the return expression, so the
            # final end-of-program stack flush must not preserve it again as a
            # discarded-return side-effect statement.
            metadata["emitted_on_discard"] = True
        return slot.render()

    def _emit_side_effect_call_if_discarded(self, slot: VMSlot | None, *, fallback_ins: Any | None = None) -> bool:
        if slot is None:
            return False
        metadata = slot.metadata
        if not metadata.get("emit_on_discard") or metadata.get("emitted_on_discard"):
            return False
        call_expr = metadata.get("call") or slot.render()
        operands = {
            "discarded_return": True,
            "call_effect": metadata.get("call_effect", {}),
            "uses": list(metadata.get("uses", [])),
        }
        self.statements.append(
            AstStatement(
                offset=int(metadata.get("call_offset", getattr(fallback_ins, "offset", self.program.start))),
                file_offset=int(metadata.get("call_file_offset", getattr(fallback_ins, "file_offset", 0))),
                kind=str(metadata.get("call_kind", "call")),
                text=f"{call_expr};",
                opcode=int(metadata.get("call_opcode", getattr(fallback_ins, "opcode", 0))),
                mnemonic=str(metadata.get("call_mnemonic", getattr(fallback_ins, "mnemonic", "call"))),
                operands=operands,
            )
        )
        metadata["emitted_on_discard"] = True
        return True

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
            role = "const_span" if m == "push_inline_span" else "span"
            slot_type = typ if typ is not None else TYPE_STRING
            loc = self.memory.location(offset=begin, type_id=slot_type, role=role, length=length) if isinstance(begin, int) else None
            self.vm.push_slice(
                expr,
                type_id=slot_type,
                begin=begin,
                end=end,
                note=m,
                location=loc,
                metadata={
                    "span_base": loc.render() if loc is not None else expr,
                    "span_data_offset": begin,
                    "span_length": length,
                    "span_type_id": slot_type,
                    "span_role": role,
                },
            )
            return

        if typ == TYPE_FLOAT and "value_float" in operands:
            self.vm.push_float(expr, value=operands.get("value_float"))
        elif typ in {TYPE_CHAR, TYPE_INT} or typ is None:
            self.vm.push_int(expr, value=operands.get("value", operands.get("value_i32", operands.get("value_u32"))))
        elif is_reference_type(typ) or (isinstance(typ, int) and typ not in {TYPE_CHAR, TYPE_INT, TYPE_FLOAT}):
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
            count = operands.get("count")
            loc = self.memory.location(
                offset=base,
                type_id=typ,
                role="array",
                length=abs(count) if isinstance(count, int) and count != 0 else None,
            )
            expr = f"{loc.render()}[{index.render()}]"
            self._push_typed_index_result(
                expr,
                typ,
                operands,
                metadata={
                    "array_base": loc.render(),
                    "array_index": index.render(),
                    "array_data_offset": base,
                    "array_element_size": operands.get("element_size"),
                    "array_type_id": typ,
                },
            )
            return

        if m in {"array2_index", "array2_index_checked"}:
            base = self.vm.get_pointer_or_slice(self._pop_value(ins, "base"))
            index = self.vm.coerce_int(self._pop_value(ins, "index"))
            expr = f"{base.render()}[{index.render()}]"
            self._push_typed_index_result(
                expr,
                typ,
                operands,
                metadata={
                    "array_base": base.render(),
                    "array_index": index.render(),
                    "array_element_size": operands.get("element_size"),
                    "array_type_id": typ,
                    "base_metadata": dict(base.metadata),
                },
            )
            return

        if m == "slice_offset_ref":
            base = self.vm.get_pointer_or_slice(self._pop_value(ins, "base"))
            expr, metadata = self._slice_offset_expr(base, operands, typ)
            self._push_typed_index_result(expr, typ, operands, metadata=metadata)
            return

        if m == "slice_offset_span":
            base = self.vm.get_pointer_or_slice(self._pop_value(ins, "base"))
            offset = operands.get("offset", "?")
            length = operands.get("length", "?")
            expr = f"{base.render()}.span_{offset:02X}_{length}" if isinstance(offset, int) else f"span({base.render()}+{offset}, {length})"
            self.vm.push_slice(expr, type_id=typ, note="slice_offset_span", metadata={"base_metadata": dict(base.metadata), "offset": offset, "length": length})
            return

        self.vm.push_unknown(f"{m}(...)", type_id=typ)

    def _slice_offset_expr(self, base: VMSlot, operands: dict[str, Any], typ: Any) -> tuple[str, dict[str, Any]]:
        offset = operands.get("offset", "?")
        metadata = {"base_metadata": dict(base.metadata), "offset": offset, "field_type_id": typ}

        if typ == TYPE_SLICE and isinstance(offset, int):
            stride = operands.get("length")
            if isinstance(stride, int) and stride > 0 and offset % stride == 0:
                index = offset // stride
                expr = f"{base.render()}[{index}]"
                metadata.update({"record_base": base.render(), "record_index": index, "record_stride": stride})
                return expr, metadata

        if isinstance(offset, int):
            return self._field_expr(base.render(), offset, typ), metadata
        return f"{base.render()}+{offset}", metadata

    def _field_expr(self, base_expr: str, offset: int, typ: Any = None) -> str:
        return f"{base_expr}.{self._field_name(offset, typ)}"

    @staticmethod
    def _field_name(offset: int, typ: Any = None) -> str:
        if typ == TYPE_FLOAT and offset % 4 == 0:
            return f"f{offset // 4}"
        if typ == TYPE_INT and offset % 4 == 0:
            return f"i{offset // 4}"
        if typ == TYPE_CHAR:
            return f"b{offset:02X}"
        return f"field_{offset:02X}"

    def _push_typed_index_result(self, expr: str, typ: Any, operands: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        if typ == TYPE_FLOAT:
            self.vm.push_float(expr, note="indexed float load", metadata=metadata)
        elif typ in {TYPE_CHAR, TYPE_INT}:
            self.vm.push_int(expr, note="indexed scalar load", metadata=metadata)
        elif typ == TYPE_SLICE:
            self.vm.push_slice(expr, type_id=typ, note="indexed descriptor", metadata=metadata)
        elif is_reference_type(typ) or (isinstance(typ, int) and typ not in {TYPE_CHAR, TYPE_INT, TYPE_FLOAT}):
            self.vm.push_ref(expr, type_id=typ, storage_size=self._storage_size(typ), note="indexed ref", metadata=metadata)
        else:
            self.vm.push_unknown(expr, type_id=typ, kind="indexed", metadata=metadata)

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
        return storage_size_for_type(typ)

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
    scope_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    return StackAstBuilder(script, program, linker=linker, scope_map=scope_map).build(instructions)
