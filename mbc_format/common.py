from __future__ import annotations

"""Shared bytecode constants, primitive readers and VM type helpers."""

import struct

MAGIC = b"MBL script v4.0\x00"
CODE_FILE_OFFSET = 0x20

TYPE_CHAR = 0x00
TYPE_STRING = 0x01
TYPE_STRING_REF = 0x02
TYPE_INT = 0x10
TYPE_INT_REF = 0x11
TYPE_INT_REF_REF = 0x12
TYPE_FLOAT = 0x20
TYPE_FLOAT_REF = 0x21
TYPE_FLOAT_REF_REF = 0x22
TYPE_SLICE = 0x30
TYPE_SLICE_REF = 0x31

TYPE_BASE_NAMES: dict[int, str] = {
    0x00: "span/string",
    0x10: "int32",
    0x20: "float32",
    0x30: "slice_descriptor",
}

TYPE_NAMES: dict[int, str] = {
    TYPE_CHAR: "i8/char",
    TYPE_STRING: "span/string",
    TYPE_STRING_REF: "span/string_ref",
    TYPE_INT: "int32",
    TYPE_INT_REF: "int32_ref_or_span",
    TYPE_INT_REF_REF: "int32_ref_or_span_ref",
    TYPE_FLOAT: "float32",
    TYPE_FLOAT_REF: "float32_ref_or_span",
    TYPE_FLOAT_REF_REF: "float32_ref_or_span_ref",
    TYPE_SLICE: "slice_descriptor",
    TYPE_SLICE_REF: "slice_descriptor_ref",
}


def s8(value: int) -> int:
    return value - 0x100 if value >= 0x80 else value


def s16(buf: bytes, off: int) -> int:
    return struct.unpack_from("<h", buf, off)[0]


def u16(buf: bytes, off: int) -> int:
    return struct.unpack_from("<H", buf, off)[0]


def i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def s32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<i", buf, off)[0]


def u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def f32_from_u32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def read_c_string(buf: bytes, off: int, encoding: str = "cp1251") -> tuple[str, int]:
    end = buf.find(b"\x00", off)
    if end < 0:
        raise ValueError(f"Unterminated string at 0x{off:X}")
    raw = buf[off:end]
    return raw.decode(encoding, errors="replace"), end + 1


def safe_chr(value: int) -> str:
    if 32 <= value < 127:
        return chr(value)
    return "."


def type_name(type_id: int | None) -> str:
    if type_id is None:
        return "unknown"
    if type_id in TYPE_NAMES:
        return TYPE_NAMES[type_id]
    if isinstance(type_id, int):
        base = type_id & 0xF0
        depth = type_id & 0x0F
        base_name = TYPE_BASE_NAMES.get(base, f"unknown_0x{base:02X}")
        if base == 0x00 and type_id == TYPE_CHAR:
            return TYPE_NAMES[TYPE_CHAR]
        return base_name + ("_ref" * depth if depth else "")
    return "unknown"


def is_reference_type(type_id: int | None) -> bool:
    return isinstance(type_id, int) and (type_id & 0x0F) != 0


def reference_type(type_id: int | None) -> int | None:
    return type_id + 1 if isinstance(type_id, int) and type_id < 0xFF else type_id


def dereferenced_type(type_id: int | None) -> int | None:
    if type_id == TYPE_STRING:
        # Native deref of type 1 reads one byte and normalizes the slot to int32.
        return TYPE_INT
    return type_id - 1 if isinstance(type_id, int) and type_id > 0 else type_id


def storage_size_for_type(type_id: int | None) -> int:
    # Program prologue / push_data_ref contract: 0 => byte, 16/32 => dword,
    # every other non-zero type is a 3-dword descriptor in data memory.
    if type_id == TYPE_CHAR:
        return 1
    if type_id in {TYPE_INT, TYPE_FLOAT}:
        return 4
    if isinstance(type_id, int) and type_id != 0:
        return 0x0C
    return 1


def deref_storage_size_for_type(type_id: int | None) -> int:
    # sub_476CD0 uses the *post-decrement* low nibble: non-zero => descriptor,
    # zero => scalar dword, except old type 1 which is handled above as byte->int.
    if isinstance(type_id, int) and (type_id & 0x0F) != 0:
        return 0x0C
    if type_id == TYPE_CHAR:
        return 1
    return 4
