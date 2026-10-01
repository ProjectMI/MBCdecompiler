"""Validated v4 metadata. Bytecode operands and data relocation sites stay distinct."""
from __future__ import annotations

from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class ModuleMetadata:
    function_map: tuple[int, ...]
    regions: bytes
    position_offset: int
    reserved: tuple[int, int]
    code_data_relocations: tuple[int, ...]
    data_relocations: tuple[int, ...]
    program_relocations: tuple[int, ...]
    trailing: bytes = b""

    @classmethod
    def parse(cls, data: bytes, *, code_size: int, data_size: int) -> "ModuleMetadata":
        cursor = 0

        def take(size: int) -> bytes:
            nonlocal cursor
            if size < 0 or size > len(data) - cursor:
                raise ValueError(f"Truncated MBC metadata at 0x{cursor:X}: need {size} bytes")
            value = data[cursor:cursor + size]
            cursor += size
            return value

        def word() -> int:
            return struct.unpack("<I", take(4))[0]

        function_map = struct.unpack("<80H", take(160))
        regions = take(word())
        position = word()
        reserved = (word(), word())
        tables = []
        for size, width in ((code_size, 4), (data_size, 4), (code_size, 2)):
            count = word()
            if count > (len(data) - cursor) // 4:
                raise ValueError("MBC relocation count exceeds metadata size")
            table = struct.unpack(f"<{count}I", take(count * 4))
            if any(offset > size or width > size - offset for offset in table):
                raise ValueError("MBC relocation exceeds its destination section")
            if len(set(table)) != len(table):
                raise ValueError("Duplicate relocation would apply an increment twice")
            tables.append(table)
        return cls(function_map, regions, position, reserved, *tables, data[cursor:])
