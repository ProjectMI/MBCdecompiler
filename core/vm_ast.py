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
    cleaned = _fold_short_circuit_conditions(cleaned)
    cleaned = _rewrite_residual_short_circuit_guards(cleaned)
    cleaned = _structure_control_flow(cleaned)
    cleaned = _retarget_branches_to_visible_offsets(cleaned)
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


def _retarget_branches_to_visible_offsets(statements: list[AstStatement]) -> list[AstStatement]:
    """Move branch labels from suppressed VM offsets to the next visible statement.

    Several bytecode targets land on instructions that do not produce source
    statements (`stack_frame_reset`, argc metadata, and other VM book-keeping).
    `_insert_branch_labels` used to preserve those offsets verbatim, which could
    produce adjacent naked labels such as `loc_A:` immediately followed by
    `loc_B:` before real code.  At source level both labels enter the same
    visible statement, so rewrite remaining branch targets to that visible
    offset before labels are inserted.

    Exact targets are left unchanged.  A coroutine resume that lands on a real
    visible `goto` also remains unchanged; that jump is executable and must stay
    visible.
    """
    if not statements:
        return statements

    visible_offsets = sorted({stmt.offset for stmt in statements})
    if not visible_offsets:
        return statements

    def resolve(target: int) -> int:
        for offset in visible_offsets:
            if offset >= target:
                return offset
        return target

    aliases: dict[int, int] = {}
    for stmt in statements:
        if stmt.kind not in _BRANCH_KINDS:
            continue
        target = stmt.operands.get("target") if isinstance(stmt.operands, dict) else None
        if not isinstance(target, int):
            continue
        resolved = resolve(target)
        if resolved != target:
            aliases[target] = resolved

    if not aliases:
        return statements

    label_aliases = {label_for_offset(old): label_for_offset(new) for old, new in aliases.items()}

    def rewrite_text(text: str) -> str:
        for old, new in label_aliases.items():
            text = text.replace(old, new)
        return text

    rewritten: list[AstStatement] = []
    for stmt in statements:
        operands = dict(stmt.operands or {})
        for key in ("target", "resume_target", "fallthrough"):
            value = operands.get(key)
            if isinstance(value, int) and value in aliases:
                operands[key] = aliases[value]
        rewritten.append(
            AstStatement(
                offset=stmt.offset,
                file_offset=stmt.file_offset,
                kind=stmt.kind,
                text=rewrite_text(stmt.text),
                opcode=stmt.opcode,
                mnemonic=stmt.mnemonic,
                operands=operands,
            )
        )
    return rewritten


def _fold_short_circuit_conditions(statements: list[AstStatement]) -> list[AstStatement]:
    """Fold VM short-circuit helper branches into source-level boolean uses.

    Opcodes ``logical_or_rel16`` and ``logical_and_rel16`` are not source-level
    branches.  They are VM helpers used while constructing a boolean value on
    the stack.  Emitting them as visible ``goto`` statements makes many ordinary
    conditions look like tangled control flow and prevents later if/switch
    recovery.

    Two consumer shapes are recognized:
    * helper chain + real ``jfalse`` consumer -> one source-level conditional;
    * helper chain + expression consumer (return/assignment) -> the consumer's
      final value expression becomes the full short-circuit expression.
    """
    if not statements:
        return statements

    result: list[AstStatement] = []
    i = 0
    while i < len(statements):
        if not _is_short_circuit_branch(statements[i]):
            result.append(statements[i])
            i += 1
            continue

        start = i
        while i < len(statements) and _is_short_circuit_branch(statements[i]):
            i += 1

        if i < len(statements) and _is_boolean_consumer_branch(statements[i]):
            folded = _make_folded_short_circuit_statement(statements[start:i], statements[i])
            if folded is not None:
                result.append(folded)
                i += 1
                continue

        if i < len(statements):
            folded_consumer = _make_folded_short_circuit_consumer_statement(statements[start:i], statements[i])
            if folded_consumer is not None:
                result.append(folded_consumer)
                i += 1
                continue

        # The run was not a complete condition, so keep it unchanged.  A later
        # guard-rewrite pass may still convert single helper branches that guard
        # a visible side-effect block.
        result.extend(statements[start:i])

    return result


def _is_short_circuit_branch(stmt: AstStatement) -> bool:
    return (
        stmt.kind == "if_goto"
        and stmt.operands.get("short_circuit") in {"and", "or"}
        and isinstance(stmt.operands.get("condition"), str)
        and isinstance(stmt.operands.get("target"), int)
    )


def _is_boolean_consumer_branch(stmt: AstStatement) -> bool:
    return (
        stmt.kind == "if_goto"
        and stmt.operands.get("branch_when") == "false"
        and not stmt.operands.get("short_circuit")
        and isinstance(stmt.operands.get("condition"), str)
        and isinstance(stmt.operands.get("target"), int)
    )


def _make_folded_short_circuit_statement(chain: list[AstStatement], final_branch: AstStatement) -> AstStatement | None:
    if not chain:
        return None

    condition_stmts = list(chain) + [final_branch]
    offset_to_pos = {stmt.offset: idx for idx, stmt in enumerate(condition_stmts)}

    ops: list[str] = []
    target_positions: list[int | None] = []
    for stmt in chain:
        op_name = stmt.operands.get("short_circuit")
        ops.append("&&" if op_name == "and" else "||")
        target_positions.append(_target_position_for_short_circuit(stmt, condition_stmts, offset_to_pos))

    # Do not fold malformed or backward helper branches.  Those are probably
    # real control flow, or at least not a safe source-level boolean expression.
    for idx, target_pos in enumerate(target_positions):
        if target_pos is None or target_pos <= idx:
            return None

    conditions: list[str] = []
    for stmt in condition_stmts:
        cond = stmt.operands.get("condition")
        if not isinstance(cond, str) or not cond.strip():
            return None
        conditions.append(_ControlStructurer._clean_condition(cond))

    combined = _render_short_circuit_expression(conditions, ops, target_positions, 0, len(conditions) - 1)
    if combined is None:
        return None

    first = chain[0]
    operands = dict(final_branch.operands or {})
    operands["condition"] = combined
    operands["branch_when"] = "false"
    operands["short_circuit_folded"] = True
    operands["folded_offsets"] = [stmt.offset for stmt in condition_stmts]
    operands["folded_ops"] = list(ops)

    return AstStatement(
        offset=first.offset,
        file_offset=first.file_offset,
        kind="if_goto",
        text=f"if (!({combined})) goto {label_for_offset(operands.get('target'))};",
        opcode=final_branch.opcode,
        mnemonic="folded_short_circuit",
        operands=operands,
    )


def _make_folded_short_circuit_consumer_statement(chain: list[AstStatement], consumer: AstStatement) -> AstStatement | None:
    """Fold helper chain into a value consumer such as ``return rhs`` or ``x = rhs``."""
    if not chain:
        return None

    final_expr = _consumer_value_expression(consumer)
    if final_expr is None:
        return None

    final_condition = AstStatement(
        offset=consumer.offset,
        file_offset=consumer.file_offset,
        kind=consumer.kind,
        text=consumer.text,
        opcode=consumer.opcode,
        mnemonic=consumer.mnemonic,
        operands={"condition": final_expr},
    )
    condition_stmts = list(chain) + [final_condition]
    offset_to_pos = {stmt.offset: idx for idx, stmt in enumerate(condition_stmts)}

    ops: list[str] = []
    target_positions: list[int | None] = []
    for stmt in chain:
        op_name = stmt.operands.get("short_circuit")
        ops.append("&&" if op_name == "and" else "||")
        target_positions.append(_target_position_for_short_circuit(stmt, condition_stmts, offset_to_pos))

    for idx, target_pos in enumerate(target_positions):
        if target_pos is None or target_pos <= idx:
            return None

    conditions = [_ControlStructurer._clean_condition(str(stmt.operands.get("condition", ""))) for stmt in condition_stmts]
    if any(not cond for cond in conditions):
        return None

    combined = _render_short_circuit_expression(conditions, ops, target_positions, 0, len(conditions) - 1)
    if combined is None:
        return None

    rewritten_text = _rewrite_consumer_value_expression(consumer.text, final_expr, combined)
    if rewritten_text is None:
        return None

    operands = dict(consumer.operands or {})
    uses = list(operands.get("uses") or [])
    if uses:
        uses[-1] = combined
        operands["uses"] = uses
    operands["short_circuit_folded"] = True
    operands["folded_offsets"] = [stmt.offset for stmt in condition_stmts]
    operands["folded_ops"] = list(ops)
    operands["folded_value"] = combined

    first = chain[0]
    return AstStatement(
        offset=first.offset,
        file_offset=first.file_offset,
        kind=consumer.kind,
        text=rewritten_text,
        opcode=consumer.opcode,
        mnemonic=consumer.mnemonic,
        operands=operands,
    )


def _consumer_value_expression(stmt: AstStatement) -> str | None:
    uses = stmt.operands.get("uses") if isinstance(stmt.operands, dict) else None
    if isinstance(uses, list) and uses:
        candidate = uses[-1]
        if isinstance(candidate, str) and candidate.strip():
            return _ControlStructurer._clean_condition(candidate)

    text = stmt.text.strip()
    if stmt.kind == "return" and text.startswith("return ") and text.endswith(";"):
        return _ControlStructurer._clean_condition(text[len("return "):-1])
    if stmt.kind == "assign" and "=" in text and text.endswith(";"):
        return _ControlStructurer._clean_condition(text.split("=", 1)[1][:-1])
    return None


def _rewrite_consumer_value_expression(text: str, old_expr: str, new_expr: str) -> str | None:
    old_expr = old_expr.strip()
    if not old_expr:
        return None

    # Prefer the exact rendered expression, then common parenthesized variants.
    candidates = [old_expr]
    cleaned = _ControlStructurer._clean_condition(old_expr)
    if cleaned != old_expr:
        candidates.append(cleaned)
    candidates.extend([f"({cleaned})", f"(({cleaned}))"])

    for candidate in candidates:
        if candidate and candidate in text:
            return text.replace(candidate, new_expr, 1)

    stripped = text.strip()
    if stripped.startswith("return ") and stripped.endswith(";"):
        return f"return {new_expr};"
    if "=" in stripped and stripped.endswith(";"):
        lhs = stripped.split("=", 1)[0].rstrip()
        return f"{lhs} = {new_expr};"
    return None


def _rewrite_residual_short_circuit_guards(statements: list[AstStatement]) -> list[AstStatement]:
    """Convert leftover helper branches that guard visible side-effect blocks.

    Some bytecode uses a short-circuit helper to skip a small RHS side-effect
    block before a later consumer, e.g. ``a || ++i > n``.  The symbolic stack may
    have already emitted the side effect as a normal statement, so full boolean
    folding is unsafe.  Rewriting the helper to an ordinary guard preserves the
    visible control flow and removes the misleading ``// short-circuit`` pseudo
    goto.
    """
    if not statements:
        return statements

    offset_to_index = {stmt.offset: idx for idx, stmt in enumerate(statements)}
    result: list[AstStatement] = []
    for i, stmt in enumerate(statements):
        if not _is_short_circuit_branch(stmt):
            result.append(stmt)
            continue

        target = stmt.operands.get("target") if isinstance(stmt.operands, dict) else None
        target_idx = offset_to_index.get(target) if isinstance(target, int) else None
        if target_idx is None or target_idx <= i + 1:
            result.append(stmt)
            continue

        cond = stmt.operands.get("condition")
        if not isinstance(cond, str) or not cond.strip():
            result.append(stmt)
            continue

        op_name = stmt.operands.get("short_circuit")
        guarded_cond = _ControlStructurer._clean_condition(cond)
        if op_name == "or":
            guarded_cond = _ControlStructurer._negated_condition(guarded_cond)
        elif op_name != "and":
            result.append(stmt)
            continue

        operands = dict(stmt.operands or {})
        operands.pop("short_circuit", None)
        operands["condition"] = guarded_cond
        operands["branch_when"] = "false"
        operands["short_circuit_guard_rewrite"] = True
        result.append(
            AstStatement(
                offset=stmt.offset,
                file_offset=stmt.file_offset,
                kind="if_goto",
                text=f"if (!({guarded_cond})) goto {label_for_offset(target)};",
                opcode=stmt.opcode,
                mnemonic="short_circuit_guard",
                operands=operands,
            )
        )
    return result


def _target_position_for_short_circuit(stmt: AstStatement, condition_stmts: list[AstStatement], offset_to_pos: dict[int, int]) -> int | None:
    target = stmt.operands.get("target")
    if not isinstance(target, int):
        return None
    exact = offset_to_pos.get(target)
    if exact is not None:
        return exact
    for idx, candidate in enumerate(condition_stmts):
        if candidate.offset >= target:
            return idx
    return None


def _render_short_circuit_expression(
    conditions: list[str],
    ops: list[str],
    target_positions: list[int | None],
    start: int,
    end: int,
) -> str | None:
    """Render a folded boolean expression for condition indexes [start, end].

    A short-circuit helper whose jump skips over later helper tests marks a
    source-level grouping boundary.  This recovers common shapes like
    ``(a || b || c) && (d || e)`` and ``a || (b && c)`` instead of flattening
    them into precedence-changing C syntax.
    """
    if start > end:
        return None
    if start == end:
        return _maybe_parenthesize_condition(conditions[start])

    split: int | None = None
    for idx in range(start, end):
        target_pos = target_positions[idx] if idx < len(target_positions) else None
        if target_pos == end and target_pos > idx + 1:
            split = idx
            break

    if split is not None:
        left = _render_short_circuit_expression(conditions, ops, target_positions, start, split)
        right = _render_short_circuit_expression(conditions, ops, target_positions, split + 1, end)
        if left is None or right is None:
            return None
        return f"({_parenthesize_boolean_side(left)} {ops[split]} {_parenthesize_boolean_side(right)})"

    # No target-delimited subexpression.  Render flat runs of the same operator
    # and parenthesize at operator changes so C precedence cannot rewrite the VM
    # short-circuit order.
    expr = _maybe_parenthesize_condition(conditions[start])
    current_op = ops[start]
    parts = [expr]
    for idx in range(start, end):
        op = ops[idx]
        rhs = _maybe_parenthesize_condition(conditions[idx + 1])
        if op == current_op:
            parts.append(rhs)
        else:
            expr = f"({f' {current_op} '.join(parts)})"
            parts = [expr, rhs]
            current_op = op
    if len(parts) == 1:
        return parts[0]
    return f" {current_op} ".join(parts)


def _maybe_parenthesize_condition(cond: str) -> str:
    cond = _ControlStructurer._clean_condition(cond)
    if re.search(r"\s(&&|\|\|)\s", cond):
        return _parenthesize_boolean_side(cond)
    return cond


def _parenthesize_boolean_side(expr: str) -> str:
    expr = expr.strip()
    if not re.search(r"\s(&&|\|\|)\s", expr):
        return expr
    if expr.startswith("(") and expr.endswith(")") and _outer_parens_wrap(expr):
        return expr
    return f"({expr})"


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

# ---------------------------------------------------------------------------
# Conservative source-level control-flow structuring.
#
# This pass intentionally sits on top of the low-level stack AST.  It only
# rewrites branch shapes that are fully contained in a linear statement range and
# have no external jumps into the would-be structured region.  Coroutine yields
# are treated as scheduler boundaries: regions that contain a yield are left in
# label/goto form so the saved-PC resume labels stay visible and truthful.

_TERMINAL_STATEMENT_KINDS = {"return", "halt", "yield"}
_SWITCH_MIN_CASES = 2
_STRUCTURER_MAX_RECURSION_DEPTH = 64
_STRUCTURER_MAX_CACHE_ITEMS = 20000



def _structure_control_flow(statements: list[AstStatement]) -> list[AstStatement]:
    if not statements:
        return statements
    return _ControlStructurer(statements).structure()


@dataclass
class _StructuredMatch:
    end: int
    lines: list[str]
    kind: str


@dataclass
class _SwitchCase:
    value: str
    body: list[AstStatement]
    exit_kind: str  # "break", "terminal", or "fallthrough"


class _ControlStructurer:
    def __init__(
        self,
        statements: list[AstStatement],
        *,
        depth: int = 0,
        memo: dict[tuple[int, ...], list[AstStatement]] | None = None,
        active: set[tuple[int, ...]] | None = None,
    ):
        self.statements = statements
        self._depth = depth
        self._memo = memo if memo is not None else {}
        self._active = active if active is not None else set()
        self._offset_to_index: dict[int, int] = {}
        for idx, stmt in enumerate(statements):
            self._offset_to_index.setdefault(stmt.offset, idx)

    def structure(self) -> list[AstStatement]:
        if not self.statements:
            return []
        if self._depth > _STRUCTURER_MAX_RECURSION_DEPTH:
            return self.statements

        key = tuple(id(stmt) for stmt in self.statements)
        cached = self._memo.get(key)
        if cached is not None:
            return list(cached)
        if key in self._active:
            # Defensive recursion fuse.  Some large dispatcher functions contain
            # overlapping if/else/switch candidate regions; if a candidate ever
            # re-enters the exact same visible statement slice, preserve the
            # low-level form instead of spinning.
            return self.statements

        self._active.add(key)
        try:
            result = self._structure_range(0, len(self.statements))
        finally:
            self._active.discard(key)

        if len(self._memo) < _STRUCTURER_MAX_CACHE_ITEMS:
            self._memo[key] = list(result)
        return result

    def _structure_range(self, start: int, end: int) -> list[AstStatement]:
        result: list[AstStatement] = []
        i = start
        while i < end:
            if self._is_noop_goto(i, end) or self._is_noop_branch(i, end):
                i += 1
                continue
            match = self._match_switch(i, end) or self._match_while(i, end) or self._match_if(i, end)
            if match is not None:
                first = self.statements[i]
                if match.lines:
                    result.append(
                        AstStatement(
                            offset=first.offset,
                            file_offset=first.file_offset,
                            kind=match.kind,
                            text="\n".join(match.lines),
                            opcode=first.opcode,
                            mnemonic=first.mnemonic,
                            operands={"structured_range": [first.offset, self.statements[match.end - 1].offset if match.end > i else first.offset]},
                        )
                    )
                i = match.end
                continue
            result.append(self.statements[i])
            i += 1
        return result

    def _match_if(self, i: int, end: int) -> _StructuredMatch | None:
        stmt = self.statements[i]
        if not self._is_plain_false_branch(stmt):
            return None
        target = self._target(stmt)
        target_idx = self._index_at_or_after(target, lower=i + 1, upper=end)
        if target_idx is None or target_idx <= i + 1:
            return None

        cond = self._positive_condition(stmt)
        if cond is None:
            return None

        # Inverted guard shape:
        #   if (!cond) goto body_start;
        #   goto after;
        # body_start:
        #   body...
        # after:
        # This comes from source like `if (!cond) { ... }` when the true side
        # is only a skip over the real body.
        if i + 1 < target_idx and self.statements[i + 1].kind == "goto":
            after_target = self._target(self.statements[i + 1])
            after_idx = self._index_at_or_after(after_target, lower=target_idx + 1, upper=end)
            if after_idx is not None and after_idx > target_idx:
                if self._safe_region(i, after_idx) and not self._range_has_yield(target_idx, after_idx):
                    body_chunks = self._structure_range_by_statements(self.statements[target_idx:after_idx])
                    if not self._has_unstructured_branch(body_chunks):
                        return _StructuredMatch(end=after_idx, lines=self._render_if(self._negated_condition(cond), body_chunks), kind="if")

        # if/else shape:
        #   if (!cond) goto else_start;
        #   then...
        #   goto after;
        # else_start:
        #   else...
        # after:
        maybe_goto_idx = target_idx - 1
        if maybe_goto_idx > i and self.statements[maybe_goto_idx].kind == "goto":
            after_target = self._target(self.statements[maybe_goto_idx])
            after_idx = self._index_at_or_after(after_target, lower=target_idx + 1, upper=end)
            if after_idx is not None and after_idx > target_idx:
                region_end = after_idx
                if self._safe_region(i, region_end) and not self._range_has_yield(i + 1, region_end):
                    then_chunks = self._structure_range_by_statements(self.statements[i + 1:maybe_goto_idx])
                    else_chunks = self._structure_range_by_statements(self.statements[target_idx:after_idx])
                    if not self._has_unstructured_branch(then_chunks) and not self._has_unstructured_branch(else_chunks):
                        lines = self._render_if_else(cond, then_chunks, else_chunks)
                        return _StructuredMatch(end=after_idx, lines=lines, kind="if_else")

        # Simple if / guard shape.  This handles early exits naturally:
        #   if (!cond) goto after;
        #   return ...;
        # after:
        if self._safe_region(i, target_idx) and not self._range_has_yield(i + 1, target_idx):
            body_chunks = self._structure_range_by_statements(self.statements[i + 1:target_idx])
            if not self._has_unstructured_branch(body_chunks):
                return _StructuredMatch(end=target_idx, lines=self._render_if(cond, body_chunks), kind="if")
        return None

    def _match_while(self, i: int, end: int) -> _StructuredMatch | None:
        stmt = self.statements[i]
        if not self._is_plain_false_branch(stmt):
            return None
        target = self._target(stmt)
        after_idx = self._index_at_or_after(target, lower=i + 1, upper=end)
        if after_idx is None or after_idx <= i + 1:
            return None
        back_idx = after_idx - 1
        if back_idx <= i or self.statements[back_idx].kind != "goto":
            return None
        back_target = self._target(self.statements[back_idx])
        back_target_idx = self._index_at_or_after(back_target, lower=0, upper=end)
        if back_target_idx != i:
            return None
        if not self._safe_region(i, after_idx) or self._range_has_yield(i + 1, back_idx):
            return None
        cond = self._positive_condition(stmt)
        if cond is None:
            return None
        body_chunks = self._structure_range_by_statements(self.statements[i + 1:back_idx])
        if self._has_unstructured_branch(body_chunks):
            return None
        lines = [f"while ({cond}) {{"]
        lines.extend(self._indent(self._chunks_to_lines(body_chunks)))
        lines.append("}")
        return _StructuredMatch(end=after_idx, lines=lines, kind="while")

    def _match_switch(self, i: int, end: int) -> _StructuredMatch | None:
        """Recover equality-dispatch ladders as switch statements.

        The bytecode compiler often emits switches as a chain of
        ``if (!(selector == value)) goto next_case`` tests.  Cases may either
        jump to a shared epilogue, return/halt/end immediately, or intentionally
        fall through with an empty body.  The previous pass required a shared
        ``goto after`` and therefore missed very common terminal switch shapes
        such as ``case: return`` / ``default: return``.
        """
        cases: list[_SwitchCase] = []
        selector: str | None = None
        common_end_idx: int | None = None
        saw_terminal_case = False
        saw_break_case = False
        cur = i

        while cur < end:
            stmt = self.statements[cur]
            if not self._is_plain_false_branch(stmt):
                break
            parsed = self._parse_equality_case(self._positive_condition(stmt))
            if parsed is None:
                break
            case_selector, case_value = parsed
            if selector is None:
                selector = case_selector
            elif selector != case_selector:
                break

            next_idx = self._index_at_or_after(self._target(stmt), lower=cur + 1, upper=end)
            if next_idx is None or next_idx < cur + 1:
                break

            # If true and false both resume at the same visible statement, the
            # test is an empty case label falling through into the next/default
            # body.  This is common for the last explicit case before default.
            if self._branch_targets_visible_fallthrough(stmt, cur, next_idx):
                cases.append(_SwitchCase(case_value, [], "fallthrough"))
                cur = next_idx
                continue

            case_body_end = next_idx
            exit_kind = "fallthrough"
            trailing_goto = next_idx - 1
            if trailing_goto > cur and self.statements[trailing_goto].kind == "goto":
                goto_target = self._target(self.statements[trailing_goto])
                goto_idx = self._index_at_or_after(goto_target, lower=next_idx, upper=end)
                if goto_idx is None:
                    break
                if common_end_idx is None:
                    common_end_idx = goto_idx
                elif common_end_idx != goto_idx:
                    break
                case_body_end = trailing_goto
                exit_kind = "break"
                saw_break_case = True
            elif self._range_ends_with_terminal(cur + 1, next_idx):
                exit_kind = "terminal"
                saw_terminal_case = True
            elif cur + 1 == next_idx:
                exit_kind = "fallthrough"
            else:
                # Non-empty, non-terminal fallthrough is legal C, but the MBC
                # AST is not path-sensitive enough yet to prove it is intended.
                break

            if self._range_has_yield(cur + 1, case_body_end):
                return None
            cases.append(_SwitchCase(case_value, self.statements[cur + 1:case_body_end], exit_kind))
            cur = next_idx

        if selector is None or len(cases) < _SWITCH_MIN_CASES:
            return None

        if common_end_idx is not None:
            if common_end_idx < cur or common_end_idx > end:
                return None
            region_end = common_end_idx
            default_body = self.statements[cur:common_end_idx]
            if self._range_has_yield(cur, common_end_idx):
                return None
        else:
            # Terminal switches have no shared epilogue.  The false path after
            # the last comparison is the default body; accept it only when it
            # reaches an obvious terminal so we do not swallow unrelated code.
            default_end = self._default_terminal_end(cur, end)
            if default_end is None:
                # Empty default is still safe when every explicit non-empty case
                # exits terminally and there are no fallthrough-only labels.
                if cur != end or not saw_terminal_case or any(case.exit_kind == "fallthrough" for case in cases):
                    return None
                default_end = cur
            region_end = default_end
            default_body = self.statements[cur:default_end]
            if self._range_has_yield(cur, default_end):
                return None

        if not self._safe_region(i, region_end):
            return None

        rendered_cases: list[_SwitchCase] = []
        for case in cases:
            body_chunks = self._structure_range_by_statements(case.body)
            if self._has_unstructured_branch(body_chunks):
                return None
            rendered_cases.append(_SwitchCase(case.value, body_chunks, case.exit_kind))

        default_chunks = self._structure_range_by_statements(default_body)
        if self._has_unstructured_branch(default_chunks):
            return None

        lines = [f"switch ({selector}) {{"]
        grouped_cases = self._group_equivalent_switch_cases(rendered_cases)
        for group_idx, (values, body, exit_kind) in enumerate(grouped_cases):
            for value in values:
                lines.append(f"case {value}:")
            body_lines = self._chunks_to_lines(body)
            if body_lines:
                lines.extend(self._indent(body_lines, 1))
                if exit_kind == "break" and not self._chunks_end_with_terminal(body):
                    lines.append("    break;")
            elif group_idx == len(grouped_cases) - 1 and not default_chunks:
                lines.append("    break;")
            # Empty fallthrough cases intentionally render as label-only.

        if default_chunks:
            lines.append("default:")
            lines.extend(self._indent(self._chunks_to_lines(default_chunks), 1))
            if common_end_idx is not None and not self._chunks_end_with_terminal(default_chunks):
                lines.append("    break;")
        lines.append("}")
        return _StructuredMatch(end=region_end, lines=lines, kind="switch")

    def _group_equivalent_switch_cases(self, cases: list[_SwitchCase]) -> list[tuple[list[str], list[AstStatement], str]]:
        """Group adjacent cases with identical exiting bodies.

        A ladder like ``if (x == 0) return 0; if (x == 1) return 0;`` is
        usually original switch syntax with stacked case labels.  Grouping only
        adjacent, non-empty, non-fallthrough bodies is semantics-preserving and
        avoids inventing cross-case fallthrough where the bytecode had work.
        """
        groups: list[tuple[list[str], list[AstStatement], str]] = []
        i = 0
        while i < len(cases):
            case = cases[i]
            values = [case.value]
            body_key = tuple(self._chunks_to_lines(case.body))
            if body_key and case.exit_kind != "fallthrough":
                j = i + 1
                while j < len(cases):
                    other = cases[j]
                    if other.exit_kind != case.exit_kind:
                        break
                    if tuple(self._chunks_to_lines(other.body)) != body_key:
                        break
                    values.append(other.value)
                    j += 1
                groups.append((values, case.body, case.exit_kind))
                i = j
            else:
                groups.append((values, case.body, case.exit_kind))
                i += 1
        return groups

    def _default_terminal_end(self, start: int, end: int) -> int | None:
        if start >= end:
            return start
        if self._range_has_yield(start, end):
            return None
        for idx in range(start, end):
            if self._is_terminal_statement(self.statements[idx]):
                return idx + 1
            # A new unrelated branch before a terminal makes the default body
            # ambiguous; leave it in low-level form.
            if idx > start and self.statements[idx].kind in {"if_goto", "goto"}:
                return None
        return None

    def _branch_targets_visible_fallthrough(self, stmt: AstStatement, cur: int, target_idx: int) -> bool:
        fallthrough = stmt.operands.get("fallthrough") if isinstance(stmt.operands, dict) else None
        target = self._target(stmt)
        if isinstance(fallthrough, int) and isinstance(target, int) and fallthrough == target:
            return True
        return target_idx == cur + 1 and isinstance(fallthrough, int) and self._index_at_or_after(fallthrough, lower=cur + 1, upper=len(self.statements)) == target_idx

    def _is_noop_goto(self, i: int, end: int) -> bool:
        if i >= end:
            return False
        stmt = self.statements[i]
        if stmt.kind != "goto":
            return False
        target = self._target(stmt)
        if target is None or i + 1 >= end:
            return False
        # Only remove a goto when its target resolves to the very next visible
        # statement in this same statement stream.  The old check used
        # ``target <= next_stmt.offset`` so any backward jump also looked like a
        # no-op after metadata suppression.  That erased coroutine-loop resumes
        # such as ``yield; goto loop_head`` and left naked resume labels behind.
        target_idx = self._index_at_or_after(target, lower=0, upper=end)
        return target_idx == i + 1

    def _is_noop_branch(self, i: int, end: int) -> bool:
        if i >= end:
            return False
        stmt = self.statements[i]
        if stmt.kind != "if_goto":
            return False
        target = self._target(stmt)
        fallthrough = stmt.operands.get("fallthrough") if isinstance(stmt.operands, dict) else None
        if isinstance(target, int) and isinstance(fallthrough, int) and target == fallthrough:
            return True
        target_idx = self._index_at_or_after(target, lower=i + 1, upper=end)
        fallthrough_idx = self._index_at_or_after(fallthrough if isinstance(fallthrough, int) else None, lower=i + 1, upper=end)
        return target_idx is not None and target_idx == fallthrough_idx == i + 1

    def _structure_range_by_statements(self, stmts: list[AstStatement]) -> list[AstStatement]:
        if not stmts:
            return []
        sub = _ControlStructurer(
            stmts,
            depth=self._depth + 1,
            memo=self._memo,
            active=self._active,
        )
        return sub.structure()

    def _safe_region(self, start: int, end: int) -> bool:
        if end <= start:
            return False
        for idx, stmt in enumerate(self.statements):
            if start <= idx < end:
                continue
            target = self._target(stmt)
            if target is None:
                continue
            target_idx = self._index_at_or_after(target, lower=0, upper=len(self.statements))
            # Incoming edges to the first statement of the candidate region are
            # safe: they are the normal continuation of a previously structured
            # guard/ladder arm.  Reject only jumps into the interior.  The older
            # `start <= target_idx < end` test made guard chains look unsafe and
            # prevented deep nested if/terminal-ladder recovery.
            if target_idx is not None and start < target_idx < end:
                return False
        return True

    def _range_has_yield(self, start: int, end: int) -> bool:
        return any(stmt.kind == "yield" for stmt in self.statements[start:end])

    def _range_ends_with_terminal(self, start: int, end: int) -> bool:
        if start >= end:
            return False
        return self._is_terminal_statement(self.statements[end - 1])

    def _chunks_end_with_terminal(self, chunks: list[AstStatement]) -> bool:
        if not chunks:
            return False
        last = chunks[-1]
        if last.kind in {"if", "if_else", "while", "switch"}:
            return False
        return self._is_terminal_statement(last)

    @staticmethod
    def _is_terminal_statement(stmt: AstStatement) -> bool:
        if stmt.kind in _TERMINAL_STATEMENT_KINDS:
            return True
        if stmt.mnemonic == "end_program" or stmt.text.strip() == "// end_program":
            return True
        return False

    @staticmethod
    def _has_unstructured_branch(chunks: list[AstStatement]) -> bool:
        return any(chunk.kind in {"goto", "if_goto"} for chunk in chunks)

    @staticmethod
    def _target(stmt: AstStatement) -> int | None:
        target = stmt.operands.get("target") if isinstance(stmt.operands, dict) else None
        return target if isinstance(target, int) else None

    def _index_at_or_after(self, target: int | None, *, lower: int, upper: int) -> int | None:
        if target is None:
            return None
        for idx in range(max(0, lower), min(upper, len(self.statements))):
            if self.statements[idx].offset >= target:
                return idx
        if target >= (self.statements[upper - 1].offset if upper > 0 else 0):
            return upper
        return None

    @staticmethod
    def _is_plain_false_branch(stmt: AstStatement) -> bool:
        return (
            stmt.kind == "if_goto"
            and stmt.operands.get("branch_when") == "false"
            and not stmt.operands.get("short_circuit")
            and isinstance(stmt.operands.get("target"), int)
        )

    def _positive_condition(self, stmt: AstStatement) -> str | None:
        cond = stmt.operands.get("condition") if isinstance(stmt.operands, dict) else None
        if not isinstance(cond, str) or not cond.strip():
            return None
        return self._clean_condition(cond)

    @classmethod
    def _parse_equality_case(cls, cond: str | None) -> tuple[str, str] | None:
        if not cond:
            return None
        cond = cls._clean_condition(cond)
        match = re.match(r"^(?P<a>.+?)\s*==\s*(?P<b>.+)$", cond)
        if not match:
            return None
        a = cls._clean_condition(match.group("a"))
        b = cls._clean_condition(match.group("b"))
        if cls._looks_like_case_value(a) and not cls._looks_like_case_value(b):
            return b, a
        if cls._looks_like_case_value(b) and not cls._looks_like_case_value(a):
            return a, b
        return None

    @staticmethod
    def _looks_like_case_value(value: str) -> bool:
        value = value.strip()
        return bool(
            re.fullmatch(r"-?\d+", value)
            or re.fullmatch(r"\(-?\d+\)", value)
            or re.fullmatch(r"0x[0-9A-Fa-f]+", value)
            or (len(value) >= 2 and value[0] == value[-1] == '"')
        )

    @staticmethod
    def _clean_condition(cond: str) -> str:
        cond = cond.strip()
        changed = True
        while changed:
            changed = False
            if cond.startswith("(") and cond.endswith(")") and _outer_parens_wrap(cond):
                cond = cond[1:-1].strip()
                changed = True
        return cond

    @classmethod
    def _negated_condition(cls, cond: str) -> str:
        cond = cls._clean_condition(cond)
        if cond.startswith("!"):
            return cls._clean_condition(cond[1:])
        return f"!({cond})"

    def _render_if(self, cond: str, body_chunks: list[AstStatement]) -> list[str]:
        lines = [f"if ({cond}) {{"]
        lines.extend(self._indent(self._chunks_to_lines(body_chunks) or ["// empty"]))
        lines.append("}")
        return lines

    def _render_if_else(self, cond: str, then_chunks: list[AstStatement], else_chunks: list[AstStatement]) -> list[str]:
        lines = [f"if ({cond}) {{"]
        lines.extend(self._indent(self._chunks_to_lines(then_chunks) or ["// empty"]))
        else_lines = self._chunks_to_lines(else_chunks) or ["// empty"]
        if else_lines and else_lines[0].startswith("if "):
            lines.append(f"}} else {else_lines[0]}")
            lines.extend(else_lines[1:])
        else:
            lines.append("} else {")
            lines.extend(self._indent(else_lines))
            lines.append("}")
        return lines

    @staticmethod
    def _chunks_to_lines(chunks: list[AstStatement]) -> list[str]:
        lines: list[str] = []
        for chunk in chunks:
            text = chunk.text.rstrip()
            if not text:
                continue
            lines.extend(text.splitlines())
        return lines

    @staticmethod
    def _indent(lines: list[str], levels: int = 1) -> list[str]:
        prefix = "    " * levels
        return [prefix + line if line else line for line in lines]


def _outer_parens_wrap(text: str) -> bool:
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and idx != len(text) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0

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
            self._emit(
                ins,
                "if_goto",
                f"if (!({cond.render()})) goto {label_for_offset(operands.get('target'))};",
                extra={"condition": cond.render(), "branch_when": "false", "uses": self._uses(cond)},
            )
            return

        if m == "logical_or_rel16":
            cond = self._pop_value(ins, "lhs")
            self._emit(
                ins,
                "if_goto",
                f"if ({cond.render()}) goto {label_for_offset(operands.get('target'))}; // || short-circuit",
                extra={"condition": cond.render(), "branch_when": "true", "short_circuit": "or", "uses": self._uses(cond)},
            )
            return

        if m == "logical_and_rel16":
            cond = self._pop_value(ins, "lhs")
            self._emit(
                ins,
                "if_goto",
                f"if (!({cond.render()})) goto {label_for_offset(operands.get('target'))}; // && short-circuit",
                extra={"condition": cond.render(), "branch_when": "false", "short_circuit": "and", "uses": self._uses(cond)},
            )
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
