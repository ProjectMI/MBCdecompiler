from __future__ import annotations

"""Lossless MBC image writer.

This module is intentionally lower-level than the pseudo-source decompiler.  It
serializes the exact container sections parsed by :mod:`mbc_format.loader`:
header, code bytes, data bytes, program table, function table and trailing
metadata.  No structuring or pretty-source decisions are involved here; this is
what makes byte-identical round-trips possible.
"""

from pathlib import Path
import struct
from typing import Any, Iterable, Mapping

from mbc_format.common import MAGIC
from mbc_format.loader import MbcFunction, MbcProgram, MbcScript


CP1251 = "cp1251"


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _encode_c_string(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = value.encode(CP1251)
    if b"\x00" in raw:
        raise ValueError("MBC table names cannot contain NUL bytes")
    return raw + b"\x00"


def emit_mbc_image(
    *,
    checksum_or_tag: int,
    module_tag: int,
    code: bytes | bytearray,
    data: bytes | bytearray,
    programs: Iterable[MbcProgram | Mapping[str, Any]],
    functions: Iterable[MbcFunction | Mapping[str, Any]],
    metadata: bytes | bytearray = b"",
) -> bytes:
    """Serialize a complete MBC image.

    ``code_size`` and ``data_size`` are deliberately derived from the supplied
    byte arrays so edits cannot accidentally leave stale section sizes in the
    header.
    """

    code_bytes = bytes(code)
    data_bytes = bytes(data)
    metadata_bytes = bytes(metadata)
    program_items = list(programs)
    function_items = list(functions)

    out = bytearray()
    out += MAGIC
    out += struct.pack(
        "<IIII",
        checksum_or_tag & 0xFFFFFFFF,
        module_tag & 0xFFFFFFFF,
        len(code_bytes),
        len(data_bytes),
    )
    out += code_bytes
    out += data_bytes

    out += struct.pack("<I", len(program_items))
    for program in program_items:
        out += _encode_c_string(_field(program, "name", ""))
        start = int(_field(program, "start", 0))
        end = int(_field(program, "end", 0))
        state_raw = int(_field(program, "state_raw", 0)) & 0xFF
        queue_id = int(_field(program, "queue_id", 0)) & 0xFF
        unknown_48 = int(_field(program, "unknown_48", 0)) & 0xFFFFFFFF
        out += struct.pack("<II", start & 0xFFFFFFFF, end & 0xFFFFFFFF)
        out += bytes((state_raw, queue_id))
        out += struct.pack("<I", unknown_48)

    out += struct.pack("<I", len(function_items))
    for function in function_items:
        out += _encode_c_string(_field(function, "name", ""))
        code_offset = int(_field(function, "code_offset", 0))
        program_index_raw = int(_field(function, "program_index_raw", 0))
        flags_or_module = int(_field(function, "flags_or_module", 0))
        out += struct.pack(
            "<III",
            code_offset & 0xFFFFFFFF,
            program_index_raw & 0xFFFFFFFF,
            flags_or_module & 0xFFFFFFFF,
        )

    out += metadata_bytes
    return bytes(out)


def script_to_bytes(script: MbcScript) -> bytes:
    """Serialize a loaded :class:`MbcScript` back into an MBC byte image."""

    return emit_mbc_image(
        checksum_or_tag=script.header.checksum_or_tag,
        module_tag=script.header.module_tag,
        code=script.code,
        data=script.data,
        programs=script.programs,
        functions=script.functions,
        metadata=script.metadata,
    )


def write_script(script: MbcScript, path: str | Path) -> Path:
    """Write ``script`` to ``path`` and return the resulting path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(script_to_bytes(script))
    return path
