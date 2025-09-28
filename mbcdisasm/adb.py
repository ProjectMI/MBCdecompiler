"""Parsers for `.adb` index files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


@dataclass(frozen=True)
class SegmentDescriptor:
    """Description of a segment extracted from an ``.adb`` file."""

    index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


class SegmentIndex(Sequence[SegmentDescriptor]):
    """Load the index table embedded in ``.adb`` files."""

    def __init__(self, offsets: List[int], total_length: int) -> None:
        self._offsets = offsets
        self._total_length = total_length
        self._segments = self._build_segments()

    @classmethod
    def from_file(cls, adb_path: Path, mbc_length: int) -> "SegmentIndex":
        data = adb_path.read_bytes()
        if len(data) % 4 != 0:
            raise ValueError("ADB index file length must be divisible by 4")
        offsets = [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]
        return cls(offsets, mbc_length)

    def _build_segments(self) -> List[SegmentDescriptor]:
        offsets = self._offsets
        total = self._total_length

        if total < 0:
            raise ValueError("Total length must be non-negative")

        if offsets:
            effective_offsets = offsets
        else:
            if total == 0:
                return []
            effective_offsets = [0]

        first_offset = effective_offsets[0]
        if not (0 <= first_offset <= total):
            raise ValueError("ADB index contains offsets outside of container size")

        for prev, curr in zip(effective_offsets, effective_offsets[1:]):
            if curr < prev:
                raise ValueError("ADB offsets must be in non-decreasing order")
            if curr == prev:
                raise ValueError("ADB segments must have a positive length")

        segments: List[SegmentDescriptor] = []
        previous_end = first_offset
        for idx, start in enumerate(effective_offsets):

            end = (
                effective_offsets[idx + 1]
                if idx + 1 < len(effective_offsets)
                else total
            )

            if start < 0 or start > total or end > total:
                raise ValueError("ADB segment bounds fall outside of container size")
            if end <= start:
                raise ValueError("ADB segments must have a positive length")

            segments.append(SegmentDescriptor(idx, start, end))
            previous_end = end

        if previous_end != total:
            raise ValueError("ADB segments must consume the entire container")

        return segments

    def __len__(self) -> int:  # pragma: no cover - trivial container API
        return len(self._segments)

    def __getitem__(self, index: int) -> SegmentDescriptor:  # pragma: no cover
        return self._segments[index]

    def __iter__(self) -> Iterator[SegmentDescriptor]:
        return iter(self._segments)

    def iter_non_empty(self) -> Iterable[SegmentDescriptor]:
        for descriptor in self._segments:
            if descriptor.length > 0:
                yield descriptor

    @property
    def total_length(self) -> int:
        return self._total_length

    def describe(self) -> List[dict]:
        return [
            {
                "index": seg.index,
                "start": seg.start,
                "end": seg.end,
                "length": seg.length,
            }
            for seg in self._segments
        ]
