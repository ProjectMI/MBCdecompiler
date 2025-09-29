"""Utilities for extracting printable strings from raw container segments.

The parser favours ASCII NUL-terminated sequences which mirrors how most
Sphere resources encode UI text.  The resulting tables can be re-used by tests
or surfaced to analysts when reviewing reconstruction output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List

from .lua_formatter import LuaWriter
from .string_utils import (
    Preview,
    chunk_preview,
    decode_latin1,
    printable_ratio,
    trim_ascii_suffix,
)

__all__ = [
    "StringTableEntry",
    "StringTable",
    "parse_string_table",
]


@dataclass(frozen=True)
class StringTableEntry:
    """Single string extracted from a raw byte buffer."""

    offset: int
    data: bytes

    def text(self) -> str:
        return decode_latin1(trim_ascii_suffix(self.data))

    def preview(self, *, limit: int = 60) -> Preview:
        return chunk_preview(self.data, limit=limit)

    def printable_ratio(self) -> float:
        return printable_ratio(self.data)

    def is_printable(self, threshold: float = 0.75) -> bool:
        return self.printable_ratio() >= threshold

    def to_dict(self) -> Dict[str, object]:
        preview = self.preview()
        return {
            "offset": self.offset,
            "length": len(self.data),
            "text": preview.text,
            "truncated": preview.truncated,
            "printable_ratio": self.printable_ratio(),
        }


@dataclass
class StringTable:
    """Collection of :class:`StringTableEntry` objects with helper methods."""

    start_offset: int
    entries: List[StringTableEntry] = field(default_factory=list)

    def __iter__(self) -> Iterator[StringTableEntry]:  # pragma: no cover - trivial
        return iter(self.entries)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.entries)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self.entries)

    @property
    def total_bytes(self) -> int:
        return sum(len(entry.data) for entry in self.entries)

    def add(self, entry: StringTableEntry) -> None:
        self.entries.append(entry)

    def iter_printable(self, *, threshold: float = 0.75) -> Iterator[StringTableEntry]:
        for entry in self.entries:
            if entry.is_printable(threshold=threshold):
                yield entry

    def search(self, needle: str) -> List[StringTableEntry]:
        if not needle:
            return []
        lowered = needle.lower()
        matches: List[StringTableEntry] = []
        for entry in self.entries:
            if lowered in entry.text().lower():
                matches.append(entry)
        return matches

    def statistics(self) -> Dict[str, object]:
        if not self.entries:
            return {"entries": 0, "printable_ratio": 0.0, "bytes": 0}
        printable = sum(1 for entry in self.entries if entry.is_printable())
        return {
            "entries": len(self.entries),
            "bytes": self.total_bytes,
            "printable_ratio": printable / len(self.entries),
        }

    def render(self, *, prefix: str = "string_table") -> str:
        writer = LuaWriter()
        writer.write_comment("decoded string table")
        writer.write_line(f"local {prefix} = {{")
        with writer.indented():
            for entry in self.entries:
                preview = entry.preview()
                comment = preview.text if preview.text else "<empty>"
                writer.write_comment(
                    f"@0x{entry.offset:06X} len={len(entry.data)} {comment}"
                )
                writer.write_line(
                    f"[0x{entry.offset:06X}] = {repr(entry.text())},"
                )
        writer.write_line("}")
        return writer.render()

    def to_dict(self) -> Dict[str, Dict[str, object]]:
        return {f"0x{entry.offset:06X}": entry.to_dict() for entry in self.entries}

    def slice(self, *, start: int, stop: int) -> "StringTable":
        sliced = StringTable(start_offset=self.start_offset + start)
        for entry in self.entries:
            if start <= entry.offset - self.start_offset < stop:
                sliced.add(entry)
        return sliced


def parse_string_table(
    data: bytes,
    *,
    start_offset: int = 0,
    min_length: int = 2,
    terminator: bytes = b"\x00",
) -> StringTable:
    """Scan ``data`` and return a :class:`StringTable` with decoded entries."""

    table = StringTable(start_offset=start_offset)
    if not data:
        return table

    current = bytearray()
    entry_start = start_offset
    index = 0
    terminator_set = set(terminator)
    terminated = False
    while index < len(data):
        byte = data[index]
        if byte in terminator_set:
            if len(current) >= min_length:
                table.add(
                    StringTableEntry(
                        offset=entry_start,
                        data=bytes(current + bytes([byte])),
                    )
                )
            current.clear()
            entry_start = start_offset + index + 1
            terminated = True
        else:
            if not current:
                entry_start = start_offset + index
            current.append(byte)
            terminated = False
        index += 1
    return table
