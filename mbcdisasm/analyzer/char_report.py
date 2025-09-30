"""Generate structured summaries for the ``_char`` module.

The `_char` script mixes several layers of content.  It starts with a dense block
of configuration tables, switches to logging messages and finally includes
function entry banners.  The amount of embedded data makes manual inspection
tedious, so the goal of this module is to condense the information into a
machine-friendly report that still carries enough context for human readers.

The builder below walks every segment and collects:

* the recognised ratio exposed by :class:`InstructionProfile`,
* all ASCII strings tagged by :mod:`data_signatures` and classified by
  :mod:`string_classifier`,
* literal markers such as repeated palindromes or explicit sentinel values,
* higher level patterns discovered by :mod:`char_patterns`.

On top of the per-segment summaries the report aggregates global statistics and
extracts a list of function-like names.  This mirrors the workflow used during
manual reversing: start with a long string dump, look for groups of markers that
hint at a configuration structure and then correlate them with runtime log
messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from ..instruction import read_instructions
from ..knowledge import KnowledgeBase
from .instruction_profile import InstructionProfile
from .string_classifier import GLOBAL_STRING_CLASSIFIER
from .char_patterns import find_patterns


@dataclass
class CharString:
    """A single ASCII string extracted from a segment."""

    offset: int
    text: str
    category: str


@dataclass
class CharSegmentSummary:
    """Summary of one container segment."""

    index: int
    recognised_ratio: float
    strings: Tuple[CharString, ...] = field(default_factory=tuple)
    literal_markers: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class FunctionDescriptor:
    """Connect function-like strings with their segments."""

    name: str
    segments: Tuple[int, ...]
    occurrences: int

    def describe(self) -> str:
        segments = ",".join(str(index) for index in self.segments)
        return f"{self.name} (segments: {segments}, occurrences: {self.occurrences})"


@dataclass
class CharReport:
    """Aggregated summary for the entire container."""

    path: Path
    segments: Tuple[CharSegmentSummary, ...]
    functions: Tuple[FunctionDescriptor, ...]
    statistics: Mapping[str, object]

    def average_ratio(self) -> float:
        if not self.segments:
            return 0.0
        total = sum(summary.recognised_ratio for summary in self.segments)
        return total / len(self.segments)

    def function_names(self) -> Tuple[str, ...]:
        return tuple(descriptor.name for descriptor in self.functions)

    def render(self) -> str:
        lines = [f"Report for {self.path}"]
        lines.append(f"Average recognised ratio: {self.average_ratio():.2f}")
        lines.append("Functions:")
        for descriptor in self.functions:
            lines.append("  - " + descriptor.describe())
        lines.append("Segments:")
        for summary in self.segments:
            lines.append(
                f"  * segment {summary.index}: recognised={summary.recognised_ratio:.2f} "
                f"strings={len(summary.strings)} markers={len(summary.literal_markers)}"
            )
            for entry in summary.strings[:5]:
                lines.append(f"      · {entry.category}: {entry.text}")
        return "\n".join(lines)

    def to_dict(self) -> Mapping[str, object]:
        return {
            "path": str(self.path),
            "average_ratio": self.average_ratio(),
            "functions": [descriptor.describe() for descriptor in self.functions],
            "segments": [
                {
                    "index": summary.index,
                    "recognised_ratio": summary.recognised_ratio,
                    "string_count": len(summary.strings),
                    "marker_count": len(summary.literal_markers),
                }
                for summary in self.segments
            ],
        }

    def segments_above_threshold(self, threshold: float) -> Tuple[CharSegmentSummary, ...]:
        return tuple(summary for summary in self.segments if summary.recognised_ratio >= threshold)

    def top_functions(self, limit: int = 5) -> Tuple[FunctionDescriptor, ...]:
        ordered = sorted(self.functions, key=lambda item: item.occurrences, reverse=True)
        return tuple(ordered[:limit])


class CharReportBuilder:
    """Build :class:`CharReport` instances for ``_char`` containers."""

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge

    def build(self, container_path: Path, adb_path: Path) -> CharReport:
        from ..mbc import MbcContainer

        container = MbcContainer.load(container_path, adb_path)
        summaries: List[CharSegmentSummary] = []
        function_map: dict[str, List[int]] = {}
        function_counts: dict[str, int] = {}
        for segment in container.iter_segments():
            words, _ = read_instructions(segment.data, segment.descriptor.start)
            profiles = [InstructionProfile.from_word(word, self.knowledge) for word in words]
            recognised = sum(1 for profile in profiles if profile.kind.name != "UNKNOWN")
            ratio = recognised / len(profiles) if profiles else 0.0
            strings = self._collect_strings(profiles)
            markers = self._collect_markers(profiles)
            for entry in strings:
                if entry.category == "function_name":
                    function_map.setdefault(entry.text, []).append(segment.index)
                    function_counts[entry.text] = function_counts.get(entry.text, 0) + 1
            summaries.append(
                CharSegmentSummary(
                    index=segment.index,
                    recognised_ratio=ratio,
                    strings=strings,
                    literal_markers=markers,
                )
            )
        functions = [
            FunctionDescriptor(
                name=name,
                segments=tuple(sorted(set(indices))),
                occurrences=function_counts.get(name, 0),
            )
            for name, indices in sorted(function_map.items())
        ]
        statistics = self._build_statistics(summaries, functions)
        return CharReport(path=container_path, segments=tuple(summaries), functions=tuple(functions), statistics=statistics)

    def _collect_strings(self, profiles: Sequence[InstructionProfile]) -> Tuple[CharString, ...]:
        entries: List[CharString] = []
        buffer: List[str] = []
        start_offset: Optional[int] = None
        for profile in profiles:
            detector = profile.traits.get("detector")
            if detector == "ascii_chunk":
                if start_offset is None:
                    start_offset = profile.word.offset
                buffer.append(str(profile.traits.get("ascii_text", "")))
                continue
            if buffer and start_offset is not None:
                text = "".join(buffer)
                category = GLOBAL_STRING_CLASSIFIER.classify(text).category
                entries.append(CharString(offset=start_offset, text=text, category=category))
                buffer.clear()
                start_offset = None
        if buffer and start_offset is not None:
            text = "".join(buffer)
            category = GLOBAL_STRING_CLASSIFIER.classify(text).category
            entries.append(CharString(offset=start_offset, text=text, category=category))
        return tuple(entries)

    def _collect_markers(self, profiles: Sequence[InstructionProfile]) -> Tuple[str, ...]:
        markers: List[str] = []
        for profile in profiles:
            detector = profile.traits.get("detector")
            if detector in {"repeat8", "repeat16", "char_marker"}:
                markers.append(detector)
        markers.extend(find_patterns(profiles))
        return tuple(markers)

    def _build_statistics(
        self,
        summaries: Sequence[CharSegmentSummary],
        functions: Sequence[FunctionDescriptor],
    ) -> Mapping[str, object]:
        """Return a mapping with aggregated statistics."""

        marker_histogram: dict[str, int] = {}
        for summary in summaries:
            for marker in summary.literal_markers:
                marker_histogram[marker] = marker_histogram.get(marker, 0) + 1
        return {
            "segment_count": len(summaries),
            "function_count": len(functions),
            "marker_histogram": marker_histogram,
        }

