from __future__ import annotations

"""Lossless editable IR for MBC bytecode.

The pretty decompiler is allowed to lose information while it structures code.
This module is the opposite: it stores every byte needed to rebuild the MBC file
and annotates those bytes with decoded instruction metadata for navigation and
safe patching.
"""

from copy import deepcopy
from pathlib import Path
import json
import struct
from typing import Any, Iterable, Mapping

from mbc_format.bytecode import MbcDecoder
from mbc_format.common import MAGIC, TYPE_FLOAT, TYPE_INT, f32_from_u32, type_name
from mbc_format.loader import MbcFunction, MbcHeader, MbcLoader, MbcProgram, MbcScript
from mbc_format.opcodes import OPCODES, decode_opcode
from compile.writer import emit_mbc_image

IR_FORMAT = "mbc_lossless_ir_v1"


JsonDict = dict[str, Any]


def _hex(data: bytes | bytearray) -> str:
    return bytes(data).hex(" ")


def _unhex(value: str | bytes | bytearray) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return bytes.fromhex(value)


def _program_to_dict(program: Any) -> JsonDict:
    return {
        "index": int(program.index),
        "name": program.name,
        "start": int(program.start),
        "end": int(program.end),
        "state_raw": int(program.state_raw),
        "queue_id": int(program.queue_id),
        "unknown_48": int(program.unknown_48),
    }


def _function_to_dict(function: Any) -> JsonDict:
    return {
        "index": int(function.index),
        "name": function.name,
        "code_offset": int(function.code_offset),
        "program_index_raw": int(function.program_index_raw),
        "flags_or_module": int(function.flags_or_module),
    }


def _instruction_to_dict(ins: Any) -> JsonDict:
    return {
        "offset": int(ins.offset),
        "file_offset": int(ins.file_offset),
        "opcode": int(ins.opcode),
        "mnemonic": ins.mnemonic,
        "length": int(ins.length),
        "raw_hex": ins.raw,
        "known": bool(ins.known),
        "terminal": bool(ins.terminal),
        "operands": deepcopy(ins.operands),
        "edges": [
            {
                "kind": edge.kind,
                "src": int(edge.src),
                "dst": edge.dst,
                "dst_program": edge.dst_program,
                "note": edge.note,
            }
            for edge in ins.edges
        ],
    }


def decode_linear_instructions(script: MbcScript, *, rich: bool = True) -> list[JsonDict]:
    """Decode the whole code section linearly, preserving raw bytes per row.

    ``rich=False`` still decodes instruction lengths and mnemonics, but skips
    bulky operand/edge annotations.  It is useful for corpus-wide round-trip
    verification where the raw bytes are the real payload.
    """

    instructions: list[JsonDict] = []
    off = 0
    if rich:
        decoder = MbcDecoder(script, annotate_linkage=False, cache_decodes=False)
        while off < len(script.code):
            ins = decoder.decode_at(off)
            length = max(int(ins.length), 1)
            row = _instruction_to_dict(ins)
            # Keep the raw bytes from the source image as the final authority.  This
            # matters if a decoder fallback reports length 1 for an unknown/trap row.
            row["length"] = length
            row["raw_hex"] = _hex(script.code[off: off + length])
            instructions.append(row)
            off += length
        return instructions

    while off < len(script.code):
        decoded = decode_opcode(script.code, off)
        length = max(int(decoded.length), 1)
        raw = script.code[off: off + length]
        instructions.append({
            "offset": off,
            "file_offset": off + 0x20,
            "opcode": raw[0] if raw else -1,
            "mnemonic": decoded.mnemonic,
            "length": length,
            "raw_hex": _hex(raw),
            "known": bool(decoded.known),
            "terminal": bool(decoded.terminal),
            "operands": {},
            "edges": [],
        })
        off += length
    return instructions


def script_to_lossless_ir(script: MbcScript, *, rich: bool = True) -> JsonDict:
    """Convert a loaded MBC script into JSON-serializable lossless IR."""

    return {
        "format": IR_FORMAT,
        "source_name": script.path.name,
        "header": {
            "magic": script.header.magic,
            "checksum_or_tag": int(script.header.checksum_or_tag),
            "module_tag": int(script.header.module_tag),
            "code_size": len(script.code),
            "data_size": len(script.data),
        },
        "code_hex": _hex(script.code),
        "data_hex": _hex(script.data),
        "programs": [_program_to_dict(program) for program in script.programs],
        "functions": [_function_to_dict(function) for function in script.functions],
        "metadata_hex": _hex(script.metadata),
        "instructions": decode_linear_instructions(script, rich=rich),
    }



def script_from_lossless_ir(ir: Mapping[str, Any], *, path: str | Path | None = None) -> MbcScript:
    """Create an in-memory :class:`MbcScript` from lossless IR.

    This is used by the editor so safe IR patches can be reflected in the pretty
    projection immediately, without first writing a temporary MBC file.
    """

    if ir.get("format") != IR_FORMAT:
        raise ValueError(f"Unsupported lossless IR format: {ir.get('format')!r}")

    header_data = ir.get("header") or {}
    script_path = Path(path) if path is not None else Path(str(ir.get("source_name") or "edited.mbc"))
    code = code_bytes_from_ir(ir)
    data = _unhex(str(ir.get("data_hex", "")))
    metadata = _unhex(str(ir.get("metadata_hex", "")))

    header = MbcHeader(
        magic=MAGIC.rstrip(b"\x00").decode("ascii", errors="replace"),
        checksum_or_tag=int(header_data.get("checksum_or_tag", 0)),
        module_tag=int(header_data.get("module_tag", 0)),
        code_size=len(code),
        data_size=len(data),
    )
    programs = [
        MbcProgram(
            index=int(item.get("index", idx)),
            name=str(item.get("name", "")),
            start=int(item.get("start", 0)),
            end=int(item.get("end", 0)),
            state_raw=int(item.get("state_raw", 0)),
            queue_id=int(item.get("queue_id", 0)),
            unknown_48=int(item.get("unknown_48", 0)),
        )
        for idx, item in enumerate(ir.get("programs") or [])
    ]
    functions = [
        MbcFunction(
            index=int(item.get("index", idx)),
            name=str(item.get("name", "")),
            code_offset=int(item.get("code_offset", 0)),
            program_index_raw=int(item.get("program_index_raw", 0)),
            flags_or_module=int(item.get("flags_or_module", 0)),
        )
        for idx, item in enumerate(ir.get("functions") or [])
    ]
    return MbcScript(
        path=script_path,
        header=header,
        code=code,
        data=data,
        programs=programs,
        functions=functions,
        metadata=metadata,
    )

def load_mbc_as_lossless_ir(path: str | Path) -> JsonDict:
    return script_to_lossless_ir(MbcLoader.load(path))


def dump_lossless_ir(ir: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ir, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_lossless_ir(path: str | Path) -> JsonDict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != IR_FORMAT:
        raise ValueError(f"Unsupported lossless IR format: {data.get('format')!r}")
    return data


def code_bytes_from_ir(ir: Mapping[str, Any]) -> bytes:
    """Rebuild the code section from instruction raw bytes.

    The instruction offsets must still describe a contiguous code stream.  For
    arbitrary length-changing edits the caller should first run a real lowering
    pass that rewrites offsets, branch operands and tables.
    """

    instructions = list(ir.get("instructions") or [])
    if not instructions:
        return _unhex(str(ir.get("code_hex", "")))

    out = bytearray()
    expected = 0
    for row in sorted(instructions, key=lambda item: int(item["offset"])):
        offset = int(row["offset"])
        raw = _unhex(str(row.get("raw_hex", "")))
        if offset != expected:
            raise ValueError(
                f"Non-contiguous instruction stream at 0x{offset:08X}; expected 0x{expected:08X}. "
                "Run a lowering/relocation pass before assembling length-changing edits."
            )
        length = int(row.get("length", len(raw)))
        if length != len(raw):
            raise ValueError(
                f"Instruction 0x{offset:08X} declares length {length}, but raw_hex has {len(raw)} bytes"
            )
        out += raw
        expected += len(raw)
    return bytes(out)


def assemble_lossless_ir(ir: Mapping[str, Any]) -> bytes:
    """Assemble lossless IR back into an MBC byte image."""

    header = ir.get("header") or {}
    return emit_mbc_image(
        checksum_or_tag=int(header.get("checksum_or_tag", 0)),
        module_tag=int(header.get("module_tag", 0)),
        code=code_bytes_from_ir(ir),
        data=_unhex(str(ir.get("data_hex", ""))),
        programs=list(ir.get("programs") or []),
        functions=list(ir.get("functions") or []),
        metadata=_unhex(str(ir.get("metadata_hex", ""))),
    )


def write_mbc_from_lossless_ir(ir: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(assemble_lossless_ir(ir))
    return path


def find_instruction(ir: Mapping[str, Any], offset: int) -> JsonDict:
    for row in ir.get("instructions") or []:
        if int(row.get("offset", -1)) == offset:
            return row
    raise KeyError(f"No instruction at code offset 0x{offset:08X}")


def patch_instruction_raw(ir: JsonDict, offset: int, raw: bytes | bytearray | str) -> None:
    """Replace one instruction's raw bytes without changing its length."""

    row = find_instruction(ir, offset)
    raw_bytes = _unhex(raw)
    old_len = int(row.get("length", 0))
    if old_len != len(raw_bytes):
        raise ValueError(
            f"Length-changing raw patch at 0x{offset:08X}: old {old_len}, new {len(raw_bytes)}. "
            "Use a lowering/relocation pass for this edit."
        )
    row["raw_hex"] = _hex(raw_bytes)
    row["opcode"] = raw_bytes[0] if raw_bytes else -1


def patch_data_bytes(ir: JsonDict, offset: int, data: bytes | bytearray | str) -> None:
    """Patch bytes inside the data section."""

    data_bytes = bytearray(_unhex(str(ir.get("data_hex", ""))))
    patch = _unhex(data)
    end = offset + len(patch)
    if offset < 0 or end > len(data_bytes):
        raise ValueError(f"Data patch 0x{offset:08X}..0x{end:08X} is outside data section")
    data_bytes[offset:end] = patch
    ir["data_hex"] = _hex(data_bytes)
    header = dict(ir.get("header") or {})
    header["data_size"] = len(data_bytes)
    ir["header"] = header


def patch_data_c_string(ir: JsonDict, offset: int, value: str, *, max_length: int, encoding: str = "cp1251") -> None:
    """Patch a fixed-capacity NUL-terminated string in the data section."""

    raw = value.encode(encoding) + b"\x00"
    if len(raw) > max_length:
        raise ValueError(f"String needs {len(raw)} bytes including NUL, capacity is {max_length}")
    patch_data_bytes(ir, offset, raw + b"\x00" * (max_length - len(raw)))


def data_bytes_from_ir(ir: Mapping[str, Any]) -> bytes:
    """Return the current data section bytes from lossless IR."""

    return _unhex(str(ir.get("data_hex", "")))


def read_data_c_string(ir: Mapping[str, Any], offset: int, *, max_length: int | None = None, encoding: str = "cp1251") -> str:
    """Read a NUL-terminated editor string from the data section."""

    data = data_bytes_from_ir(ir)
    if offset < 0 or offset >= len(data):
        raise ValueError(f"Data offset 0x{offset:08X} is outside data section")
    end_limit = len(data) if max_length is None else min(len(data), offset + max(0, max_length))
    end = data.find(b"\x00", offset, end_limit)
    if end < 0:
        end = end_limit
    return data[offset:end].decode(encoding, errors="replace")


def read_data_scalar(ir: Mapping[str, Any], offset: int, *, type_id: int = TYPE_INT) -> int | float:
    """Read a 1/4-byte scalar from the data section for editor prompts."""

    data = data_bytes_from_ir(ir)
    if type_id == TYPE_FLOAT:
        if offset < 0 or offset + 4 > len(data):
            raise ValueError(f"float32 at data[0x{offset:04X}] is outside data section")
        return struct.unpack_from("<f", data, offset)[0]
    if type_id == TYPE_CHAR:
        if offset < 0 or offset + 1 > len(data):
            raise ValueError(f"char at data[0x{offset:04X}] is outside data section")
        return struct.unpack_from("<b", data, offset)[0]
    if offset < 0 or offset + 4 > len(data):
        raise ValueError(f"int32 at data[0x{offset:04X}] is outside data section")
    return struct.unpack_from("<i", data, offset)[0]


def patch_data_scalar(ir: JsonDict, offset: int, value: int | float, *, type_id: int = TYPE_INT) -> None:
    """Patch a fixed-width scalar inside the data section."""

    if type_id == TYPE_FLOAT:
        patch_data_bytes(ir, offset, struct.pack("<f", float(value)))
    elif type_id == TYPE_CHAR:
        ivalue = int(value)
        if not -128 <= ivalue <= 127:
            raise ValueError("char data value must fit in int8")
        patch_data_bytes(ir, offset, struct.pack("<b", ivalue))
    else:
        patch_data_bytes(ir, offset, struct.pack("<i", int(value)))


def patch_inline_span_text(ir: JsonDict, instruction_offset: int, value: str, *, encoding: str = "cp1251") -> None:
    """Patch the data backing a push_inline_span/push_inline_typed_span row."""

    row = find_instruction(ir, instruction_offset)
    operands = row.get("operands") or {}
    data_offset = operands.get("data_offset")
    length = operands.get("length")
    if not isinstance(data_offset, int) or not isinstance(length, int) or length <= 0:
        raise ValueError(f"Instruction 0x{instruction_offset:08X} does not expose a fixed data span")
    patch_data_c_string(ir, data_offset, value, max_length=length, encoding=encoding)


def patch_fixed_integer_operand(ir: JsonDict, offset: int, value: int) -> None:
    """Patch a same-width integer operand on a non-CFG instruction.

    This is intentionally conservative: branches, relative calls and other CFG
    operands are excluded because changing them should go through a relocation /
    lowering pass.  The helper is for source-visible constants such as
    ``program_restart(2)`` or ``add_assign_u16(..., 4)`` where the byte width is
    fixed and the surrounding code layout does not change.
    """

    row = find_instruction(ir, offset)
    raw = bytearray(_unhex(str(row.get("raw_hex", ""))))
    if not raw:
        raise ValueError(f"Instruction at 0x{offset:08X} has no raw bytes")
    opcode = raw[0]
    spec = OPCODES.get(opcode)
    fmt = getattr(spec, "format", "") if spec is not None else ""
    ivalue = int(value)

    operands = dict(row.get("operands") or {})
    if fmt == "u8":
        if not 0 <= ivalue <= 0xFF:
            raise ValueError("u8 operand value must fit in 0..255")
        raw[1] = ivalue & 0xFF
        operands["value"] = ivalue
    elif fmt == "u16":
        if not 0 <= ivalue <= 0xFFFF:
            raise ValueError("u16 operand value must fit in 0..65535")
        raw[1:3] = struct.pack("<H", ivalue)
        operands["value"] = ivalue
    elif fmt == "program_i16":
        if not -0x8000 <= ivalue <= 0x7FFF:
            raise ValueError("program_i16 operand value must fit in -32768..32767")
        raw[1:3] = struct.pack("<h", ivalue)
        operands["program_index"] = ivalue
        operands["program_index_u16"] = ivalue & 0xFFFF
        # A numeric retarget invalidates the old display label, if any.
        operands.pop("program_name", None)
    elif fmt == "data_ref":
        if not 0 <= ivalue <= 0xFFFFFFFF:
            raise ValueError("data_ref operand value must fit in uint32")
        raw[2:6] = struct.pack("<I", ivalue)
        operands["data_offset"] = ivalue
    else:
        raise ValueError(
            f"Instruction at 0x{offset:08X} ({row.get('mnemonic', '?')}) does not expose a supported fixed integer operand"
        )

    patch_instruction_raw(ir, offset, raw)
    row["operands"] = operands


def patch_typed_immediate(ir: JsonDict, offset: int, value: int | float, *, type_id: int | None = None) -> None:
    """Patch a push_imm* instruction while preserving its encoded width.

    Supported rows: ``push_imm32`` (0x39), ``push_imm_u16`` (0x28),
    ``push_imm_i8`` (0x29).  The opcode and existing type byte are preserved
    unless ``type_id`` is provided.
    """

    row = find_instruction(ir, offset)
    raw = bytearray(_unhex(str(row.get("raw_hex", ""))))
    if not raw:
        raise ValueError(f"Instruction at 0x{offset:08X} has no raw bytes")
    opcode = raw[0]
    if opcode not in {0x39, 0x28, 0x29}:
        raise ValueError(f"Instruction at 0x{offset:08X} is not a typed immediate")
    if type_id is not None:
        raw[1] = type_id & 0xFF

    effective_type = raw[1]
    if opcode == 0x39:
        if isinstance(value, float) or effective_type == TYPE_FLOAT:
            raw[2:6] = struct.pack("<f", float(value))
        else:
            raw[2:6] = struct.pack("<i", int(value))
    elif opcode == 0x28:
        if not 0 <= int(value) <= 0xFFFF:
            raise ValueError("push_imm_u16 value must fit in uint16")
        raw[2:4] = struct.pack("<H", int(value))
    elif opcode == 0x29:
        if not -128 <= int(value) <= 127:
            raise ValueError("push_imm_i8 value must fit in int8")
        raw[2] = int(value) & 0xFF

    patch_instruction_raw(ir, offset, raw)
    # Keep the human-readable metadata in sync for editor consumers.
    operands = dict(row.get("operands") or {})
    operands["type"] = effective_type
    operands["type_name"] = type_name(effective_type)
    if opcode == 0x39:
        raw_u32 = struct.unpack_from("<I", raw, 2)[0]
        operands["value_u32"] = raw_u32
        operands["value_i32"] = struct.unpack_from("<i", raw, 2)[0]
        if effective_type == TYPE_FLOAT:
            operands["value_float"] = f32_from_u32(raw_u32)
    elif opcode == 0x28:
        operands["value"] = struct.unpack_from("<H", raw, 2)[0]
    elif opcode == 0x29:
        operands["value"] = struct.unpack_from("<b", raw, 2)[0]
        operands["value_u8"] = raw[2]
    row["operands"] = operands


def render_lossless_instruction(row: Mapping[str, Any]) -> str:
    offset = int(row.get("offset", 0))
    raw = str(row.get("raw_hex", ""))
    mnemonic = str(row.get("mnemonic", "?"))
    operands = row.get("operands") or {}
    parts: list[str] = []
    for key in (
        "type_name",
        "value_i32",
        "value_float",
        "value",
        "data_offset",
        "length",
        "target",
        "fallthrough",
        "program_name",
        "program_index",
        "subopcode_hex",
        "target_name",
    ):
        if key in operands:
            value = operands[key]
            if key in {"data_offset", "target", "fallthrough"} and isinstance(value, int):
                value = f"0x{value:08X}"
            parts.append(f"{key}={value}")
    suffix = " " + ", ".join(parts) if parts else ""
    return f"loc_{offset:08X}: {raw:<29} {mnemonic}{suffix}"


def render_lossless_view(ir: Mapping[str, Any]) -> str:
    lines = [
        f"// Lossless MBC IR: {ir.get('source_name', '<unknown>')}",
        f"// format: {ir.get('format')}",
        f"// code={len(code_bytes_from_ir(ir))} bytes, data={len(_unhex(str(ir.get('data_hex', ''))))} bytes",
        "",
    ]
    lines.extend(render_lossless_instruction(row) for row in ir.get("instructions") or [])
    return "\n".join(lines) + "\n"
