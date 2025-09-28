"""Binary container helpers for ``.mbc`` files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .adb import SegmentDescriptor, SegmentIndex
from .instruction import WORD_SIZE


logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A contiguous portion of the container."""

    descriptor: SegmentDescriptor
    data: bytes
    classification: str

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

    @property
    def is_code(self) -> bool:
        return self.classification == "code"


class MbcContainer:
    """Reader object that exposes container metadata and segments."""

    def __init__(
        self,
        path: Path,
        segments: List[Segment],
        *,
        classifier: Optional["SegmentClassifier"] = None,
    ) -> None:
        self.path = path
        self._segments = segments
        self._header = self._parse_header()
        if classifier is None:
            from .segment_classifier import SegmentClassifier

            classifier = SegmentClassifier()
        self._classifier = classifier

    @classmethod
    def load(
        cls,
        mbc_path: Path,
        adb_path: Path,
        *,
        classifier: Optional["SegmentClassifier"] = None,
    ) -> "MbcContainer":
        mbc_data = mbc_path.read_bytes()
        seg_index = SegmentIndex.from_file(adb_path, len(mbc_data))
        if classifier is None:
            from .segment_classifier import SegmentClassifier

            classifier = SegmentClassifier()
        segments = [
            cls._slice_segment(mbc_data, descriptor, classifier)
            for descriptor in seg_index.iter_non_empty()
        ]
        return cls(mbc_path, segments, classifier=classifier)

    @staticmethod
    def _slice_segment(
        mbc_data: bytes,
        descriptor: SegmentDescriptor,
        classifier: Optional["SegmentClassifier"] = None,
    ) -> Segment:
        total = len(mbc_data)
        start, end = descriptor.start, descriptor.end
        if not (0 <= start < end <= total):
            raise ValueError(
                f"Segment {descriptor.index} boundaries [{start}, {end}) exceed container size {total}"
            )

        segment_bytes = mbc_data[start:end]
        if classifier is None:
            from .segment_classifier import SegmentClassifier

            classifier = SegmentClassifier()
        classification = classifier.classify(descriptor, segment_bytes)
        return Segment(descriptor, segment_bytes, classification)

    def segments(self) -> Iterable[Segment]:  # pragma: no cover - simple generator
        return iter(self._segments)

    def iter_code_segments(self) -> Iterator[Segment]:
        for segment in self._segments:
            if segment.is_code:
                yield segment

    def _parse_header(self) -> dict:
        # The format embeds an ASCII banner at the beginning. We expose it to
        # the CLI so operators can sanity-check files quickly.
        banner = self._segments[0].data[:32] if self._segments else b""
        text = banner.split(b"\0", 1)[0].decode("latin-1", "replace")
        return {"banner": text, "segment_count": len(self._segments)}

    @property
    def header(self) -> dict:  # pragma: no cover
        return self._header


_DEFAULT_CLASSIFIER: Optional["SegmentClassifier"] = None


def classify_segment(
    data: bytes,
    descriptor: Optional[SegmentDescriptor] = None,
    *,
    classifier: Optional["SegmentClassifier"] = None,
) -> str:
    """Compatibility wrapper that proxies to :class:`SegmentClassifier`."""

    global _DEFAULT_CLASSIFIER
    if classifier is None:
        if _DEFAULT_CLASSIFIER is None:
            from .segment_classifier import SegmentClassifier

            _DEFAULT_CLASSIFIER = SegmentClassifier()
        classifier = _DEFAULT_CLASSIFIER

    if descriptor is None:
        descriptor = SegmentDescriptor(index=-1, start=0, end=len(data))

    return classifier.classify(descriptor, data)
