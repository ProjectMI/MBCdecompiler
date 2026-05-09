from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import struct
from typing import List, Optional


MAGIC = b"MBL script v4.0\x00"
CODE_FILE_OFFSET = 0x20


def _u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def _i8(value: int) -> int:
    return value - 0x100 if value >= 0x80 else value


def _read_c_string(buf: bytes, off: int, encoding: str = "cp1251") -> tuple[str, int]:
    end = buf.find(b"\x00", off)
    if end < 0:
        raise ValueError(f"Unterminated string at 0x{off:X}")
    raw = buf[off:end]
    return raw.decode(encoding, errors="replace"), end + 1


@dataclass
class MbcHeader:
    magic: str
    checksum_or_tag: int
    module_tag: int
    code_size: int
    data_size: int

    @property
    def code_file_offset(self) -> int:
        return CODE_FILE_OFFSET


@dataclass
class MbcProgram:
    index: int
    name: str
    start: int
    end: int
    state_raw: int
    queue_id: int
    unknown_48: int

    @property
    def state(self) -> int:
        return _i8(self.state_raw)

    @property
    def file_start(self) -> int:
        return CODE_FILE_OFFSET + self.start

    @property
    def file_end(self) -> int:
        return CODE_FILE_OFFSET + self.end

    def contains(self, code_offset: int) -> bool:
        return self.start <= code_offset <= self.end


@dataclass
class MbcFunction:
    index: int
    name: str
    field_36: int
    field_40: int
    field_44: int


@dataclass
class AdbArrayGuard:
    index: int
    begin_guard_offset: int
    end_guard_offset: int


@dataclass
class MbcScript:
    path: Path
    header: MbcHeader
    code: bytes
    data: bytes
    programs: List[MbcProgram]
    functions: List[MbcFunction]
    metadata: bytes

    def program_by_name(self, name: str) -> Optional[MbcProgram]:
        folded = name.casefold()
        for program in self.programs:
            if program.name.casefold() == folded:
                return program
        return None

    def program_for_offset(self, code_offset: int) -> Optional[MbcProgram]:
        # Program table is not guaranteed to be strictly non-overlapping, but in
        # normal MBC files this works and is enough for CFG labelling.
        for program in self.programs:
            if program.contains(code_offset):
                return program
        return None

    def to_summary_dict(self) -> dict:
        return {
            "path": str(self.path),
            "header": asdict(self.header),
            "program_count": len(self.programs),
            "function_count": len(self.functions),
            "metadata_size": len(self.metadata),
        }


class MbcLoader:
    """Loader for Sphere MBL/MBC bytecode files.

    The parser follows the loader layout reconstructed from the client runtime:
      0x00: 16-byte magic, b"MBL script v4.0\\0"
      0x10: u32 checksum/tag-like value
      0x14: u32 module tag
      0x18: u32 code section size
      0x1C: u32 data section size
      0x20: code bytes
      ... : data bytes
      ... : compact variable-length program table
      ... : compact variable-length function table
      ... : remaining metadata/debug tables
    """

    @staticmethod
    def load(path: str | Path) -> MbcScript:
        path = Path(path)
        buf = path.read_bytes()

        if len(buf) < CODE_FILE_OFFSET:
            raise ValueError(f"{path} is too small to be an MBC file")

        magic = buf[:16]
        if magic != MAGIC:
            raise ValueError(
                f"{path} has bad magic {magic!r}; expected {MAGIC!r}"
            )

        header = MbcHeader(
            magic=magic.rstrip(b"\x00").decode("ascii", errors="replace"),
            checksum_or_tag=_u32(buf, 0x10),
            module_tag=_u32(buf, 0x14),
            code_size=_u32(buf, 0x18),
            data_size=_u32(buf, 0x1C),
        )

        code_start = CODE_FILE_OFFSET
        code_end = code_start + header.code_size
        data_end = code_end + header.data_size

        if data_end > len(buf):
            raise ValueError(
                f"{path} is truncated: sections end at 0x{data_end:X}, "
                f"file size is 0x{len(buf):X}"
            )

        code = buf[code_start:code_end]
        data = buf[code_end:data_end]

        off = data_end
        if off + 4 > len(buf):
            raise ValueError("Missing program table count")

        program_count = _u32(buf, off)
        off += 4

        programs: List[MbcProgram] = []
        for idx in range(program_count):
            name, off = _read_c_string(buf, off)
            if off + 14 > len(buf):
                raise ValueError(f"Truncated program record #{idx}")
            start = _u32(buf, off)
            end = _u32(buf, off + 4)
            state_raw = buf[off + 8]
            queue_id = buf[off + 9]
            unknown_48 = _u32(buf, off + 10)
            off += 14

            programs.append(
                MbcProgram(
                    index=idx,
                    name=name,
                    start=start,
                    end=end,
                    state_raw=state_raw,
                    queue_id=queue_id,
                    unknown_48=unknown_48,
                )
            )

        if off + 4 > len(buf):
            raise ValueError("Missing function table count")

        function_count = _u32(buf, off)
        off += 4

        functions: List[MbcFunction] = []
        for idx in range(function_count):
            name, off = _read_c_string(buf, off)
            if off + 12 > len(buf):
                raise ValueError(f"Truncated function record #{idx}")
            f36, f40, f44 = struct.unpack_from("<III", buf, off)
            off += 12
            functions.append(MbcFunction(idx, name, f36, f40, f44))

        metadata = buf[off:]

        return MbcScript(
            path=path,
            header=header,
            code=code,
            data=data,
            programs=programs,
            functions=functions,
            metadata=metadata,
        )

    @staticmethod
    def load_adb(path: str | Path) -> list[AdbArrayGuard]:
        """Parse optional .adb array-guard file.

        In the client this is used by DebugScriptArrays.cpp-style checks. The
        file is a sequence of 8-byte pairs, not the main CFG table.
        """
        path = Path(path)
        buf = path.read_bytes()
        if len(buf) % 8:
            raise ValueError(f"{path} size must be divisible by 8")
        records: list[AdbArrayGuard] = []
        for index in range(len(buf) // 8):
            begin, end = struct.unpack_from("<II", buf, index * 8)
            records.append(AdbArrayGuard(index, begin, end))
        return records
