"""Binary container helpers for ``.mbc`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List

from .adb import SegmentDescriptor, SegmentIndex


@dataclass
class Segment:
    """A contiguous portion of the container."""

    descriptor: SegmentDescriptor
    data: bytes

    @property
    def index(self) -> int:  # pragma: no cover - trivial proxy
        return self.descriptor.index

    @property
    def start(self) -> int:  # pragma: no cover
        return self.descriptor.start

    @property
    def end(self) -> int:  # pragma: no cover
        return self.descriptor.end

    @property
    def length(self) -> int:  # pragma: no cover
        return self.descriptor.length


class MbcContainer:
    """Reader object that exposes container metadata and segments."""

    def __init__(self, path: Path, segments: List[Segment]) -> None:
        self.path = path
        self._segments = segments
        self._header = self._parse_header()

    @classmethod
    def load(cls, mbc_path: Path, adb_path: Path) -> "MbcContainer":
        mbc_data = mbc_path.read_bytes()
        seg_index = SegmentIndex.from_file(adb_path, len(mbc_data))
        segments = [
            cls._slice_segment(mbc_data, descriptor)
            for descriptor in seg_index.iter_non_empty()
        ]
        return cls(mbc_path, segments)

    @staticmethod
    def _slice_segment(mbc_data: bytes, descriptor: SegmentDescriptor) -> Segment:
        total = len(mbc_data)
        start, end = descriptor.start, descriptor.end
        if not (0 <= start < end <= total):
            raise ValueError(
                f"Segment {descriptor.index} boundaries [{start}, {end}) exceed container size {total}"
            )

        segment_bytes = mbc_data[start:end]
        return Segment(descriptor, segment_bytes)

    def segments(self) -> Iterable[Segment]:  # pragma: no cover - simple generator
        return iter(self._segments)

    def iter_segments(self) -> Iterator[Segment]:
        return iter(self._segments)

    def _parse_header(self) -> dict:
        # The format embeds an ASCII banner at the beginning. We expose it to
        # the CLI so operators can sanity-check files quickly.
        banner = self._segments[0].data[:32] if self._segments else b""
        text = banner.split(b"\0", 1)[0].decode("latin-1", "replace")
        return {"banner": text, "segment_count": len(self._segments)}

    @property
    def header(self) -> dict:  # pragma: no cover
        return self._header
