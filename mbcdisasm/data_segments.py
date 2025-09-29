"""Utilities for extracting and rendering non-code segments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import math
from typing import Dict, Iterable, List, Optional, Sequence

from .lua_formatter import LuaWriter
from .lua_literals import escape_lua_string
from .mbc import Segment

_PRINTABLE_LOW = 32
_PRINTABLE_HIGH = 126


@dataclass(frozen=True)
class ExtractedString:
    offset: int
    text: str

    def render(self) -> str:
        return escape_lua_string(self.text)

    def length(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class HexDumpLine:
    offset: int
    hex_bytes: str
    ascii: str

    def render(self) -> str:
        return f"0x{self.offset:06X}  {self.hex_bytes:<47}  {self.ascii}"


@dataclass(frozen=True)
class ByteFrequency:
    """Frequency entry for the byte histogram of a segment."""

    value: int
    count: int
    ratio: float

    def render(self) -> str:
        display = chr(self.value) if _PRINTABLE_LOW <= self.value <= _PRINTABLE_HIGH else "·"
        return f"0x{self.value:02X} ({display}): {self.count} ({self.ratio:.2f})"


@dataclass(frozen=True)
class ByteRun:
    """Summary of a long run of the same byte inside the segment."""

    segment: int
    offset: int
    value: int
    length: int

    def render(self) -> str:
        glyph = chr(self.value) if _PRINTABLE_LOW <= self.value <= _PRINTABLE_HIGH else "·"
        return (
            f"seg {self.segment} @0x{self.offset:06X}: byte 0x{self.value:02X} ({glyph})"
            f" × {self.length}"
        )

    def is_zero_run(self) -> bool:
        return self.value == 0


@dataclass(frozen=True)
class DataSegmentSummary:
    index: int
    start: int
    length: int
    classification: str
    strings: Sequence[ExtractedString] = field(default_factory=list)
    hex_preview: Sequence[HexDumpLine] = field(default_factory=list)
    printable_ratio: float = 0.0
    entropy: float = 0.0
    byte_histogram: Sequence[ByteFrequency] = field(default_factory=list)
    byte_counts: Dict[int, int] = field(default_factory=dict)
    repeated_runs: Sequence[ByteRun] = field(default_factory=list)
    run_threshold: int = 0
    is_code_segment: bool = False

    def has_strings(self) -> bool:
        return bool(self.strings)

    def string_count(self) -> int:
        return len(self.strings)

    def total_string_length(self) -> int:
        return sum(string.length() for string in self.strings)

    def longest_zero_run(self) -> Optional[ByteRun]:
        for run in self.repeated_runs:
            if run.is_zero_run():
                return run
        return None

    def histogram_summary(self) -> str:
        if not self.byte_histogram:
            return "<no data>"
        parts = [f"{freq.value:02X}:{freq.count}" for freq in self.byte_histogram]
        return ", ".join(parts)

    def stats_summary(self) -> str:
        if not self.strings:
            return "strings=0"
        avg = self.total_string_length() / len(self.strings)
        return f"strings={len(self.strings)} total={self.total_string_length()} avg={avg:.1f}"


class DataSegmentAnalyzer:
    """Analyse non-code segments and extract human readable artefacts."""

    def __init__(
        self,
        *,
        min_string_length: int = 4,
        preview_bytes: int = 128,
        preview_width: int = 16,
        histogram_limit: int = 6,
        run_threshold: int = 8,
        max_runs: int = 3,
        include_code_segments: bool = True,
        code_preview_bytes: int = 48,
        max_code_previews: int = 4,
    ) -> None:
        self.min_string_length = max(1, min_string_length)
        self.preview_bytes = max(0, preview_bytes)
        self.preview_width = max(4, preview_width)
        self.histogram_limit = max(1, histogram_limit)
        self.run_threshold = max(1, run_threshold)
        self.max_runs = max(0, max_runs)
        self.include_code_segments = include_code_segments
        self.code_preview_bytes = max(0, code_preview_bytes)
        self.max_code_previews = max(0, max_code_previews)

    # ------------------------------------------------------------------
    def analyse(self, segment: Segment) -> DataSegmentSummary:
        strings = self._extract_strings(segment.data, segment.start)
        return self._summarise_segment(
            segment,
            strings,
            include_preview=True,
            include_statistics=True,
        )

    def analyse_segments(self, segments: Iterable[Segment]) -> List[DataSegmentSummary]:
        summaries: List[DataSegmentSummary] = []
        for segment in segments:
            strings = self._extract_strings(segment.data, segment.start)
            if segment.is_code:
                if self.include_code_segments:
                    strings = self._filter_code_strings(segment, strings)
                if not self.include_code_segments or not strings:
                    continue
                summary = self._summarise_segment(
                    segment,
                    strings,
                    include_preview=False,
                    include_statistics=True,
                )
                previews = self._code_hex_preview(segment, strings)
                if previews:
                    summary = replace(summary, hex_preview=previews)
                summaries.append(summary)
                continue
            summary = self._summarise_segment(
                segment,
                strings,
                include_preview=True,
                include_statistics=True,
            )
            if summary.strings or summary.hex_preview:
                summaries.append(summary)
        return summaries

    def _filter_code_strings(
        self, segment: Segment, strings: Sequence[ExtractedString]
    ) -> List[ExtractedString]:
        if not strings:
            return []
        filtered: List[ExtractedString] = []
        data = segment.data
        allowed_punctuation = set(" _-:.,()<>/%'\"!?=;[]{}")
        for string in strings:
            text = string.text
            start = string.offset - segment.start
            end = start + len(text)
            prev_byte = data[start - 1] if start > 0 else None
            next_byte = data[end] if end < len(data) else None
            if next_byte != 0 and prev_byte != 0:
                continue
            if text and text[0].isdigit():
                continue
            if len(text) <= 4:
                if text.startswith("_") and all(ch == "_" or ch.isalnum() for ch in text):
                    filtered.append(string)
                    continue
                if text.isalpha() and (text.islower() or text.istitle()):
                    filtered.append(string)
                continue
            letter_count = sum(ch.isalpha() for ch in text)
            if letter_count < 3:
                continue
            if " " not in text and "_" not in text:
                if any(ch.isdigit() or ch in "=+*/" for ch in text):
                    continue
            if not all(ch.isalnum() or ch in allowed_punctuation for ch in text):
                continue
            filtered.append(string)
        return filtered

    def _summarise_segment(
        self,
        segment: Segment,
        strings: Sequence[ExtractedString],
        *,
        include_preview: bool,
        include_statistics: bool,
    ) -> DataSegmentSummary:
        preview: Sequence[HexDumpLine] = []
        printable_ratio = 0.0
        histogram: Sequence[ByteFrequency] = []
        counts: Dict[int, int] = {}
        entropy = 0.0
        runs: Sequence[ByteRun] = []

        if include_preview:
            preview = self._hex_preview(segment.data, segment.start)
        if include_statistics:
            printable_ratio = self._printable_ratio(segment.data)
            histogram, counts = self._byte_frequencies(segment.data)
            entropy = self._entropy(counts, len(segment.data))
            runs = self._repeated_runs(segment, segment.data)

        return DataSegmentSummary(
            index=segment.index,
            start=segment.start,
            length=segment.length,
            classification=segment.classification,
            strings=strings,
            hex_preview=preview,
            printable_ratio=printable_ratio,
            entropy=entropy,
            byte_histogram=histogram,
            byte_counts=counts,
            repeated_runs=runs,
            run_threshold=self.run_threshold,
            is_code_segment=segment.is_code,
        )

    # ------------------------------------------------------------------
    def _extract_strings(self, data: bytes, base_offset: int) -> List[ExtractedString]:
        runs: List[ExtractedString] = []
        start: Optional[int] = None
        buffer: List[str] = []
        for index, byte in enumerate(data):
            if _PRINTABLE_LOW <= byte <= _PRINTABLE_HIGH:
                if start is None:
                    start = base_offset + index
                buffer.append(chr(byte))
                continue
            if start is not None and len(buffer) >= self.min_string_length:
                runs.append(ExtractedString(start, "".join(buffer)))
            start = None
            buffer.clear()
        if start is not None and len(buffer) >= self.min_string_length:
            runs.append(ExtractedString(start, "".join(buffer)))
        return runs

    def _hex_preview(
        self,
        data: bytes,
        base_offset: int,
        *,
        start: int = 0,
        limit: Optional[int] = None,
    ) -> List[HexDumpLine]:
        if not data:
            return []
        if limit is None:
            limit = self.preview_bytes
        if limit <= 0:
            return []
        start = max(0, min(start, len(data)))
        end = min(len(data), start + limit)
        if end <= start:
            return []
        lines: List[HexDumpLine] = []
        for offset in range(start, end, self.preview_width):
            chunk = data[offset : offset + self.preview_width]
            hex_bytes = " ".join(f"{byte:02X}" for byte in chunk)
            ascii = "".join(
                chr(byte) if _PRINTABLE_LOW <= byte <= _PRINTABLE_HIGH else "."
                for byte in chunk
            )
            lines.append(HexDumpLine(base_offset + offset, hex_bytes, ascii))
        return lines

    def _code_hex_preview(
        self,
        segment: Segment,
        strings: Sequence[ExtractedString],
    ) -> List[HexDumpLine]:
        if not self.code_preview_bytes or not self.max_code_previews:
            return []
        previews: List[HexDumpLine] = []
        seen_offsets: set[int] = set()
        limit = min(len(strings), self.max_code_previews)
        for string in strings[:limit]:
            relative_start = string.offset - segment.start
            start = max(0, relative_start - self.code_preview_bytes)
            length = len(string.text)
            preview_length = length + 2 * self.code_preview_bytes
            for line in self._hex_preview(
                segment.data,
                segment.start,
                start=start,
                limit=preview_length,
            ):
                if line.offset in seen_offsets:
                    continue
                seen_offsets.add(line.offset)
                previews.append(line)
        return previews

    def _printable_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        printable = sum(1 for byte in data if _PRINTABLE_LOW <= byte <= _PRINTABLE_HIGH)
        return printable / len(data)

    def _byte_frequencies(self, data: bytes) -> tuple[List[ByteFrequency], Dict[int, int]]:
        if not data:
            return ([], {})
        counts = Counter(data)
        total = len(data)
        entries = [
            ByteFrequency(value=byte, count=count, ratio=count / total)
            for byte, count in counts.items()
        ]
        entries.sort(key=lambda entry: (-entry.count, entry.value))
        return (entries[: self.histogram_limit], dict(counts))

    def _entropy(self, counts: Dict[int, int], total: int) -> float:
        if total <= 0 or not counts:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count <= 0:
                continue
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy

    def _repeated_runs(self, segment: Segment, data: bytes) -> List[ByteRun]:
        if self.max_runs <= 0 or not data:
            return []
        runs: List[ByteRun] = []
        start_index = 0
        current_value: Optional[int] = None
        current_length = 0
        for index, byte in enumerate(data):
            if current_value is None:
                current_value = byte
                start_index = index
                current_length = 1
                continue
            if byte == current_value:
                current_length += 1
                continue
            if current_length >= self.run_threshold:
                runs.append(
                    ByteRun(
                        segment=segment.index,
                        offset=segment.start + start_index,
                        value=current_value,
                        length=current_length,
                    )
                )
            current_value = byte
            start_index = index
            current_length = 1
        if current_value is not None and current_length >= self.run_threshold:
            runs.append(
                ByteRun(
                    segment=segment.index,
                    offset=segment.start + start_index,
                    value=current_value,
                    length=current_length,
                )
            )
        runs.sort(key=lambda run: (-run.length, run.offset))
        return runs[: self.max_runs]


def summarise_data_segments(
    segments: Iterable[Segment],
    *,
    min_length: int = 4,
    preview_bytes: int = 128,
    preview_width: int = 16,
    histogram_limit: int = 6,
    run_threshold: int = 8,
    max_runs: int = 3,
    include_code_segments: bool = True,
    code_preview_bytes: int = 48,
    max_code_previews: int = 4,
) -> List[DataSegmentSummary]:
    analyzer = DataSegmentAnalyzer(
        min_string_length=min_length,
        preview_bytes=preview_bytes,
        preview_width=preview_width,
        histogram_limit=histogram_limit,
        run_threshold=run_threshold,
        max_runs=max_runs,
        include_code_segments=include_code_segments,
        code_preview_bytes=code_preview_bytes,
        max_code_previews=max_code_previews,
    )
    return analyzer.analyse_segments(segments)


def _render_hex_preview(writer: LuaWriter, preview: Sequence[HexDumpLine]) -> None:
    if not preview:
        return
    writer.write_comment("    hex preview:")
    for line in preview:
        writer.write_comment(f"      {line.render()}")


def render_data_summaries(
    summaries: Sequence[DataSegmentSummary],
    *,
    include_hex: bool = True,
) -> str:
    if not summaries:
        return ""
    writer = LuaWriter()
    writer.write_comment("data segments:")
    for summary in summaries:
        header = (
            f"  segment {summary.index} @0x{summary.start:06X} len={summary.length}"
            f" class={summary.classification} printable={summary.printable_ratio:.2f}"
        )
        writer.write_comment(header)
        if summary.is_code_segment and summary.strings:
            writer.write_comment(
                "    embedded printable strings located inside a code segment"
            )
        writer.write_comment(f"    {summary.stats_summary()}")
        writer.write_comment(
            f"    bytes: entropy={summary.entropy:.2f} histogram={summary.histogram_summary()}"
        )
        zero_run = summary.longest_zero_run()
        if zero_run:
            writer.write_comment(
                f"    longest zero run: {zero_run.length} bytes starting @0x{zero_run.offset:06X}"
            )
        if summary.byte_histogram:
            writer.write_comment("    top byte frequencies:")
            for freq in summary.byte_histogram:
                writer.write_comment(f"      {freq.render()}")
        else:
            writer.write_comment("    top byte frequencies: <insufficient data>")
        if summary.repeated_runs:
            writer.write_comment("    repeated runs:")
            for run in summary.repeated_runs:
                writer.write_comment(f"      {run.render()}")
        else:
            writer.write_comment(
                f"    repeated runs: none ≥ {max(1, summary.run_threshold)}"
            )
        if summary.strings:
            for string in summary.strings:
                writer.write_comment(
                    f"    0x{string.offset:06X}: {string.render()}"
                )
        else:
            writer.write_comment("    <no printable strings>")
        if include_hex:
            _render_hex_preview(writer, summary.hex_preview)
        writer.write_line("")
    return writer.render()


@dataclass(frozen=True)
class StringOccurrence:
    segment: int
    offset: int


@dataclass(frozen=True)
class AggregatedString:
    text: str
    occurrences: Sequence[StringOccurrence]

    def render_occurrences(self) -> str:
        parts = [f"seg {occ.segment}@0x{occ.offset:06X}" for occ in self.occurrences]
        return ", ".join(parts)


def aggregate_strings(
    summaries: Sequence[DataSegmentSummary], *, min_occurrences: int = 2
) -> List[AggregatedString]:
    buckets: dict[str, List[StringOccurrence]] = {}
    for summary in summaries:
        for string in summary.strings:
            bucket = buckets.setdefault(string.text, [])
            bucket.append(StringOccurrence(summary.index, string.offset))
    aggregated: List[AggregatedString] = []
    for text, occurrences in buckets.items():
        if len(occurrences) < max(1, min_occurrences):
            continue
        aggregated.append(AggregatedString(text, tuple(sorted(occurrences, key=lambda occ: (occ.segment, occ.offset)))))
    aggregated.sort(key=lambda item: (len(item.occurrences), item.text), reverse=True)
    return aggregated


def render_string_table(strings: Sequence[AggregatedString]) -> str:
    if not strings:
        return ""
    writer = LuaWriter()
    writer.write_comment("string table:")
    for item in strings:
        writer.write_comment(
            f"  {escape_lua_string(item.text)} -> {item.render_occurrences()}"
        )
    return writer.render()


def render_data_table(
    summaries: Sequence[DataSegmentSummary],
    *,
    table_name: str = "__data_segments",
    include_strings: bool = True,
    include_histogram: bool = True,
    include_runs: bool = True,
    return_table: bool = False,
) -> str:
    if not summaries:
        return ""
    writer = LuaWriter()
    writer.write_line(f"local {table_name} = {{")
    with writer.indented():
        for summary in summaries:
            writer.write_line("{")
            with writer.indented():
                writer.write_line(f"index = {summary.index},")
                writer.write_line(f"start = 0x{summary.start:06X},")
                writer.write_line(f"length = {summary.length},")
                writer.write_line(
                    f"classification = {escape_lua_string(summary.classification)},"
                )
                writer.write_line(
                    f"is_code = {'true' if summary.is_code_segment else 'false'},"
                )
                writer.write_line(f"printable = {summary.printable_ratio:.3f},")
                writer.write_line(f"entropy = {summary.entropy:.3f},")
                if include_histogram and summary.byte_histogram:
                    writer.write_line("top_bytes = {")
                    with writer.indented():
                        for freq in summary.byte_histogram:
                            writer.write_line("{")
                            with writer.indented():
                                writer.write_line(f"value = 0x{freq.value:02X},")
                                writer.write_line(f"count = {freq.count},")
                                writer.write_line(f"ratio = {freq.ratio:.4f},")
                            writer.write_line("},")
                    writer.write_line("},")
                if include_runs and summary.repeated_runs:
                    writer.write_line("runs = {")
                    with writer.indented():
                        for run in summary.repeated_runs:
                            writer.write_line("{")
                            with writer.indented():
                                writer.write_line(f"offset = 0x{run.offset:06X},")
                                writer.write_line(f"length = {run.length},")
                                writer.write_line(f"value = 0x{run.value:02X},")
                            writer.write_line("},")
                    writer.write_line("},")
                zero_run = summary.longest_zero_run()
                if zero_run:
                    writer.write_line("zero_run = {")
                    with writer.indented():
                        writer.write_line(f"offset = 0x{zero_run.offset:06X},")
                        writer.write_line(f"length = {zero_run.length},")
                    writer.write_line("},")
                if include_strings and summary.strings:
                    writer.write_line("strings = {")
                    with writer.indented():
                        for item in summary.strings:
                            writer.write_line("{")
                            with writer.indented():
                                writer.write_line(f"offset = 0x{item.offset:06X},")
                                writer.write_line(f"text = {item.render()},")
                            writer.write_line("},")
                    writer.write_line("},")
            writer.write_line("},")
    writer.write_line("}")
    if return_table:
        writer.write_line(f"return {table_name}")
    return writer.render()


@dataclass(frozen=True)
class SegmentStatistics:
    classifications: dict[str, int]
    segment_count: int
    string_segments: int
    string_count: int
    total_string_length: int
    total_bytes: int
    average_printable_ratio: float
    printable_min: float
    printable_max: float
    entropy_min: float
    entropy_max: float
    entropy_average: float
    zero_run_segments: int
    longest_zero_run: Optional[ByteRun]
    common_bytes: Sequence[ByteFrequency]

    def average_strings_per_segment(self) -> float:
        if self.segment_count == 0:
            return 0.0
        return self.string_count / self.segment_count

    def average_string_length(self) -> float:
        if self.string_count == 0:
            return 0.0
        return self.total_string_length / self.string_count

    def printable_ratio_range(self) -> str:
        if self.segment_count == 0:
            return "0.00…0.00"
        return f"{self.printable_min:.2f}…{self.printable_max:.2f}"

    def entropy_range(self) -> str:
        if self.segment_count == 0:
            return "0.00…0.00"
        return f"{self.entropy_min:.2f}…{self.entropy_max:.2f}"

    def has_common_bytes(self) -> bool:
        return bool(self.common_bytes)


def compute_segment_statistics(summaries: Sequence[DataSegmentSummary]) -> SegmentStatistics:
    data_summaries = [summary for summary in summaries if not summary.is_code_segment]
    if not data_summaries:
        return SegmentStatistics(
            classifications={},
            segment_count=0,
            string_segments=0,
            string_count=0,
            total_string_length=0,
            total_bytes=0,
            average_printable_ratio=0.0,
            printable_min=0.0,
            printable_max=0.0,
            entropy_min=0.0,
            entropy_max=0.0,
            entropy_average=0.0,
            zero_run_segments=0,
            longest_zero_run=None,
            common_bytes=tuple(),
        )

    counts: dict[str, int] = {}
    string_segments = 0
    string_count = 0
    total_string_length = 0
    total_bytes = 0
    printable_weighted = 0.0
    printable_min: Optional[float] = None
    printable_max: Optional[float] = None
    entropies: List[float] = []
    zero_run_segments = 0
    zero_runs: List[ByteRun] = []
    aggregate_counts: Counter[int] = Counter()

    for summary in data_summaries:
        counts[summary.classification] = counts.get(summary.classification, 0) + 1
        if summary.strings:
            string_segments += 1
            string_count += len(summary.strings)
            total_string_length += summary.total_string_length()

        total_bytes += summary.length
        printable_weighted += summary.printable_ratio * summary.length
        printable_min = (
            summary.printable_ratio
            if printable_min is None
            else min(printable_min, summary.printable_ratio)
        )
        printable_max = (
            summary.printable_ratio
            if printable_max is None
            else max(printable_max, summary.printable_ratio)
        )
        entropies.append(summary.entropy)
        aggregate_counts.update(summary.byte_counts)

        zero_run = summary.longest_zero_run()
        if zero_run:
            zero_run_segments += 1
            zero_runs.append(zero_run)

    average_printable_ratio = (
        printable_weighted / total_bytes if total_bytes else 0.0
    )
    printable_min = printable_min if printable_min is not None else 0.0
    printable_max = printable_max if printable_max is not None else 0.0
    entropy_min = min(entropies) if entropies else 0.0
    entropy_max = max(entropies) if entropies else 0.0
    entropy_average = (sum(entropies) / len(entropies)) if entropies else 0.0
    longest_zero_run = max(zero_runs, key=lambda run: run.length, default=None)
    common_bytes: List[ByteFrequency] = []
    if total_bytes:
        for value, count in aggregate_counts.most_common(8):
            common_bytes.append(
                ByteFrequency(value=value, count=count, ratio=count / total_bytes)
            )

    return SegmentStatistics(
        classifications=dict(sorted(counts.items())),
        segment_count=len(data_summaries),
        string_segments=string_segments,
        string_count=string_count,
        total_string_length=total_string_length,
        total_bytes=total_bytes,
        average_printable_ratio=average_printable_ratio,
        printable_min=printable_min,
        printable_max=printable_max,
        entropy_min=entropy_min,
        entropy_max=entropy_max,
        entropy_average=entropy_average,
        zero_run_segments=zero_run_segments,
        longest_zero_run=longest_zero_run,
        common_bytes=tuple(common_bytes),
    )


def render_segment_statistics(stats: SegmentStatistics) -> str:
    if stats.segment_count == 0:
        return ""
    writer = LuaWriter()
    writer.write_comment("data segment statistics:")
    writer.write_comment(f"  segments analysed: {stats.segment_count}")
    writer.write_comment(f"  total bytes: {stats.total_bytes}")
    writer.write_comment(f"  segments with strings: {stats.string_segments}")
    writer.write_comment(
        "  total strings: "
        f"{stats.string_count} (avg {stats.average_strings_per_segment():.2f} per segment)"
    )
    writer.write_comment(
        f"  average string length: {stats.average_string_length():.2f}"
    )
    writer.write_comment(
        "  printable ratio: "
        f"avg {stats.average_printable_ratio:.2f} range {stats.printable_ratio_range()}"
    )
    writer.write_comment(
        f"  entropy range: avg {stats.entropy_average:.2f} span {stats.entropy_range()}"
    )
    writer.write_comment(f"  segments with zero runs ≥ threshold: {stats.zero_run_segments}")
    if stats.longest_zero_run is not None:
        writer.write_comment(
            "  longest zero run: "
            f"{stats.longest_zero_run.length} bytes in segment {stats.longest_zero_run.segment}"
            f" starting @0x{stats.longest_zero_run.offset:06X}"
        )
    if stats.has_common_bytes():
        writer.write_comment("  most common bytes:")
        for entry in stats.common_bytes:
            writer.write_comment(f"    {entry.render()}")
    if stats.classifications:
        writer.write_comment("  classifications:")
        for name, count in stats.classifications.items():
            writer.write_comment(f"    {name}: {count}")
    return writer.render()


__all__ = [
    "ExtractedString",
    "HexDumpLine",
    "ByteFrequency",
    "ByteRun",
    "DataSegmentSummary",
    "DataSegmentAnalyzer",
    "summarise_data_segments",
    "render_data_summaries",
    "StringOccurrence",
    "AggregatedString",
    "aggregate_strings",
    "render_string_table",
    "render_data_table",
    "SegmentStatistics",
    "compute_segment_statistics",
    "render_segment_statistics",
]
