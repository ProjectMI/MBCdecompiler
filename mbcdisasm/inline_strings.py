"""Utilities that decode inline ASCII payloads embedded in instruction words.

The Sphere bytecode frequently inlines short string fragments directly inside
opcode words.  Those pseudo instructions do not interact with the VM stack; the
interpreter merely copies their payload into auxiliary buffers that eventually
feed structured data writers.  Historically the high level reconstructor treated
them as opaque helper calls which meant that large portions of quest text and
UI strings were effectively invisible in the generated Lua.  The module below
provides a small toolkit that understands the inline encodings and exposes the
captured data in a structured way so that reconstruction layers can surface the
information to operators.

Two important concepts are modelled:

``InlineStringAccumulator``
    Lightweight state machine that consumes :class:`~mbcdisasm.ir.IRInstruction`
    instances and gathers the decoded bytes until a non-inline instruction is
    observed.  Accumulators can be reused across basic blocks which keeps the
    high level translator logic straightforward.

``InlineStringCollector``
    Aggregates the chunks produced by accumulators, grouping them per segment
    and offering convenience accessors that summarise how much data has been
    recovered.  The collector intentionally stays ignorant about rendering so it
    can be reused by command line tools and tests that need direct access to the
    decoded blobs.

The module also exposes :func:`escape_lua_bytes` which mirrors the escaping
rules used elsewhere in the project but accepts arbitrary byte sequences.  This
keeps the output deterministic and shields the rest of the code base from
having to worry about control characters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .ir import IRInstruction
from .lua_formatter import LuaWriter
from .manual_semantics import InstructionSemantics
from .string_utils import chunk_preview, decode_latin1, printable_ratio

# ---------------------------------------------------------------------------
# Inline opcode detection helpers
# ---------------------------------------------------------------------------


_INLINE_PREFIXES: Tuple[str, ...] = (
    "inline_ascii_chunk",
    "inline_mask_chunk",
    "inline_string_token",
)


def is_inline_semantics(semantics: InstructionSemantics) -> bool:
    """Return ``True`` when ``semantics`` encodes inline ASCII payloads.

    Manual annotations use a fairly small vocabulary for these pseudo
    instructions.  We key off the ``manual_name`` field which is stable across
    knowledge base versions and mirrors how analysts refer to the opcodes during
    manual triage sessions.
    """

    name = semantics.manual_name.lower()
    return any(name.startswith(prefix) for prefix in _INLINE_PREFIXES)


def decode_inline_bytes(instruction: IRInstruction) -> bytes:
    """Decode the four ASCII bytes embedded in ``instruction``.

    The encoding mirrors the raw instruction layout: the opcode byte contributes
    the first character, the mode byte contributes the second and the 16-bit
    operand stores the remaining pair.  The helper keeps the logic in one place
    so that callers do not have to duplicate the conversion.
    """

    opcode_hex, mode_hex = instruction.key.split(":", 1)
    opcode = int(opcode_hex, 16) & 0xFF
    mode = int(mode_hex, 16) & 0xFF
    operand = instruction.operand & 0xFFFF
    return bytes([opcode, mode, (operand >> 8) & 0xFF, operand & 0xFF])


# ---------------------------------------------------------------------------
# Inline string accumulation
# ---------------------------------------------------------------------------


@dataclass
class InlineStringChunk:
    """A decoded inline ASCII payload extracted from instruction words."""

    segment_index: int
    block_start: int
    start_offset: int
    end_offset: int
    data: bytes
    instruction_offsets: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def length(self) -> int:
        return len(self.data)

    def text(self) -> str:
        """Decode the chunk using Latin-1 with safe replacements."""

        return decode_latin1(self.data)

    def preview(self, limit: int = 48) -> str:
        """Return a short human-readable description of the bytes."""

        return str(chunk_preview(self.data, limit=limit))

    def printable_ratio(self) -> float:
        return printable_ratio(self.data)

    def is_probably_text(self, threshold: float = 0.75) -> bool:
        """Best effort detection to decide whether the chunk resembles text."""

        return self.printable_ratio() >= threshold


@dataclass(frozen=True)
class InlineStringSequence:
    """Merged view combining consecutive chunks belonging to the same block."""

    segment_index: int
    start_block: int
    chunks: Tuple[InlineStringChunk, ...]
    data: bytes

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("inline string sequence requires at least one chunk")

    @property
    def start_offset(self) -> int:
        return self.chunks[0].start_offset

    @property
    def end_offset(self) -> int:
        return self.chunks[-1].end_offset

    @property
    def total_length(self) -> int:
        return len(self.data)

    def printable_ratio(self) -> float:
        return printable_ratio(self.data)

    def preview(self, limit: int = 80) -> str:
        return str(chunk_preview(self.data, limit=limit))



@dataclass
class InlineStringReport:
    """Aggregated statistics computed from an :class:`InlineStringCollector`."""

    entry_count: int
    segment_count: int
    total_bytes: int
    longest_chunk: Optional[InlineStringChunk]
    average_length: float

    def longest_summary(self) -> Optional[str]:
        if self.longest_chunk is None:
            return None
        return (
            f"{self.longest_chunk.length} bytes at segment "
            f"{self.longest_chunk.segment_index:03d} offset "
            f"0x{self.longest_chunk.start_offset:06X}"
        )

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "entries": self.entry_count,
            "segments": self.segment_count,
            "bytes": self.total_bytes,
            "average_length": self.average_length,
        }
        longest = self.longest_summary()
        if longest is not None:
            payload["largest"] = longest
        return payload


class InlineStringAccumulator:
    """Collect consecutive inline string instructions within a block."""

    def __init__(self) -> None:
        self._bytes = bytearray()
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self._instructions: List[int] = []

    # ------------------------------------------------------------------
    def has_data(self) -> bool:
        return bool(self._bytes)

    def reset(self) -> None:
        self._bytes.clear()
        self._start = None
        self._end = None
        self._instructions.clear()

    def feed(self, instruction: IRInstruction) -> None:
        if self._start is None:
            self._start = instruction.offset
        self._end = instruction.offset
        self._instructions.append(instruction.offset)
        self._bytes.extend(decode_inline_bytes(instruction))

    def finish(self, segment_index: int, block_start: int) -> InlineStringChunk:
        if not self._bytes:
            raise ValueError("inline accumulator is empty")
        start_offset = self._start if self._start is not None else 0
        end_offset = self._end if self._end is not None else start_offset
        chunk = InlineStringChunk(
            segment_index=segment_index,
            block_start=block_start,
            start_offset=start_offset,
            end_offset=end_offset,
            data=bytes(self._bytes),
            instruction_offsets=tuple(self._instructions),
        )
        self.reset()
        return chunk


class InlineStringCollector:
    """Container that stores decoded inline strings grouped by segment."""

    def __init__(self) -> None:
        self._entries: Dict[int, List[InlineStringChunk]] = {}
        self._sequence_cache: Optional[List[InlineStringSequence]] = None

    # ------------------------------------------------------------------
    def add(self, chunk: InlineStringChunk) -> None:
        if not chunk.data:
            return
        self._entries.setdefault(chunk.segment_index, []).append(chunk)
        self._sequence_cache = None

    def clear(self) -> None:
        self._entries.clear()
        self._sequence_cache = None

    def is_empty(self) -> bool:
        return all(not entries for entries in self._entries.values())

    def segments(self) -> Iterator[int]:
        yield from sorted(self._entries)

    def entries_for(self, segment_index: int) -> List[InlineStringChunk]:
        entries = self._entries.get(segment_index, [])
        return sorted(entries, key=lambda chunk: chunk.start_offset)

    def entry_count(self) -> int:
        return sum(len(entries) for entries in self._entries.values())

    def total_bytes(self) -> int:
        return sum(chunk.length for entries in self._entries.values() for chunk in entries)

    def iter_entries(self) -> Iterator[Tuple[int, InlineStringChunk]]:
        for segment in self.segments():
            for chunk in self.entries_for(segment):
                yield segment, chunk

    def _ensure_sequences(self) -> None:
        if self._sequence_cache is not None:
            return
        sequences: List[InlineStringSequence] = []
        for segment in self.segments():
            entries = self.entries_for(segment)
            if not entries:
                continue
            buffer = bytearray()
            current: List[InlineStringChunk] = []
            start_block = entries[0].block_start
            previous_offset: Optional[int] = None
            for chunk in entries:
                expected = None if previous_offset is None else previous_offset + 4
                contiguous = previous_offset is None or chunk.start_offset == expected
                if not contiguous:
                    if current:
                        sequences.append(
                            InlineStringSequence(
                                segment_index=segment,
                                start_block=start_block,
                                chunks=tuple(current),
                                data=bytes(buffer),
                            )
                        )
                    buffer = bytearray()
                    current = []
                    start_block = chunk.block_start
                buffer.extend(chunk.data)
                current.append(chunk)
                previous_offset = chunk.start_offset
            if current:
                sequences.append(
                    InlineStringSequence(
                        segment_index=segment,
                        start_block=start_block,
                        chunks=tuple(current),
                        data=bytes(buffer),
                    )
                )
        self._sequence_cache = sequences

    def iter_sequences(self) -> Iterator[InlineStringSequence]:
        self._ensure_sequences()
        assert self._sequence_cache is not None
        yield from self._sequence_cache

    def segment_count(self) -> int:
        return sum(1 for entries in self._entries.values() if entries)

    def bytes_for_segment(self, segment_index: int) -> int:
        return sum(chunk.length for chunk in self.entries_for(segment_index))

    def longest_chunk(self) -> Optional[InlineStringChunk]:
        longest: Optional[InlineStringChunk] = None
        for _, chunk in self.iter_entries():
            if longest is None or chunk.length > longest.length:
                longest = chunk
        return longest

    def build_report(self) -> InlineStringReport:
        entry_count = self.entry_count()
        segment_count = self.segment_count()
        total_bytes = self.total_bytes()
        longest = self.longest_chunk()
        average = total_bytes / entry_count if entry_count else 0.0
        return InlineStringReport(
            entry_count=entry_count,
            segment_count=segment_count,
            total_bytes=total_bytes,
            longest_chunk=longest,
            average_length=average,
        )

    def to_dict(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        payload: Dict[str, Dict[str, Dict[str, str]]] = {}
        for segment in self.segments():
            entries = self.entries_for(segment)
            if not entries:
                continue
            segment_key = f"{segment:03d}"
            segment_payload: Dict[str, Dict[str, str]] = {}
            for chunk in entries:
                offset = f"0x{chunk.start_offset:06X}"
                segment_payload[offset] = {
                    "hex": chunk.data.hex(),
                    "lua": escape_lua_bytes(chunk.data),
                }
            payload[segment_key] = segment_payload
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def find(self, text: str) -> List[InlineStringChunk]:
        matches: List[InlineStringChunk] = []
        if not text:
            return matches
        needle = text.lower()
        for _, chunk in self.iter_entries():
            haystack = chunk.data.decode("latin-1", "ignore").lower()
            if needle in haystack:
                matches.append(chunk)
        return matches

    def iter_merged(self) -> Iterator[Tuple[int, Tuple[InlineStringChunk, ...], bytes]]:
        for sequence in self.iter_sequences():
            yield sequence.segment_index, sequence.chunks, sequence.data

    def merged_strings(self) -> Dict[int, List[str]]:
        merged: Dict[int, List[str]] = {}
        for segment, chunks, data in self.iter_merged():
            if not chunks:
                continue
            merged.setdefault(segment, []).append(escape_lua_bytes(data))
        return merged


    def filter_segments(self, predicate: Callable[[int], bool]) -> "InlineStringCollector":
        filtered = InlineStringCollector()
        for segment in self.segments():
            if not predicate(segment):
                continue
            for chunk in self.entries_for(segment):
                filtered.add(chunk)
        return filtered


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def escape_lua_bytes(data: bytes) -> str:
    """Return a Lua string literal that reproduces ``data`` byte-for-byte."""

    parts: List[str] = []
    for byte in data:
        if byte == 0x5C:  # backslash
            parts.append("\\\\")
        elif byte == 0x22:  # double quote
            parts.append('\"')
        elif byte == 0x0A:
            parts.append("\\n")
        elif byte == 0x0D:
            parts.append("\\r")
        elif byte == 0x09:
            parts.append("\\t")
        elif 0x20 <= byte <= 0x7E:
            parts.append(chr(byte))
        else:
            parts.append(f"\\x{byte:02X}")
    return '"' + "".join(parts) + '"'


def render_inline_tables(
    collector: InlineStringCollector,
    *,
    prefix: str = "inline_segment",
) -> str:
    """Render Lua tables describing the decoded inline strings.

    Each segment receives its own table which keeps the lookup logic intuitive in
    the generated Lua source.  Callers can freely decide how to reference those
    tables; the renderer merely emits them in a human friendly format.
    """

    writer = LuaWriter()
    writer.write_comment(
        "inline resource strings extracted from inline_* opcode encodings"
    )
    for segment in collector.segments():
        entries = collector.entries_for(segment)
        if not entries:
            continue
        writer.write_line("")
        writer.write_line(f"local {prefix}_{segment:03d} = {{")
        with writer.indented():
            for chunk in entries:
                if chunk.start_offset == chunk.end_offset:
                    comment = f"chunk @0x{chunk.start_offset:06X}"
                else:
                    comment = (
                        f"chunk 0x{chunk.start_offset:06X}..0x{chunk.end_offset:06X}"
                    )
                writer.write_comment(comment)
                writer.write_line(
                    f"[0x{chunk.start_offset:06X}] = {escape_lua_bytes(chunk.data)},"
                )
        writer.write_line("}")
    return writer.render()


__all__ = [
    "InlineStringAccumulator",
    "InlineStringChunk",
    "InlineStringSequence",
    "InlineStringCollector",
    "InlineStringReport",
    "decode_inline_bytes",
    "escape_lua_bytes",
    "is_inline_semantics",
    "render_inline_tables",
]

