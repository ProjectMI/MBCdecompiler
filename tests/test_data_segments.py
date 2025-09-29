from mbcdisasm.adb import SegmentDescriptor
from mbcdisasm.data_segments import (
    DataSegmentAnalyzer,
    aggregate_strings,
    compute_segment_statistics,
    render_data_summaries,
    render_data_table,
    render_segment_statistics,
    render_string_table,
    summarise_data_segments,
)
from mbcdisasm.mbc import Segment


def test_data_segment_string_extraction() -> None:
    descriptor = SegmentDescriptor(index=5, start=0x100, end=0x110)
    data = b"hello\x00world!!\x00"
    segment = Segment(descriptor, data, "strings")
    summaries = summarise_data_segments([segment], min_length=3)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.index == 5
    assert summary.has_strings()
    assert [s.text for s in summary.strings] == ["hello", "world!!"]
    assert summary.hex_preview  # preview should not be empty
    assert summary.stats_summary() == "strings=2 total=12 avg=6.0"
    assert summary.byte_histogram
    assert summary.entropy > 0

    rendered = render_data_summaries(summaries)
    assert "segment 5" in rendered
    assert "hello" in rendered
    assert "world!!" in rendered
    assert "hex preview" in rendered
    assert "histogram" in rendered


def test_code_segments_with_strings_are_reported() -> None:
    data = b"\xAA\xBBHello\x00More\x00\xCC"
    descriptor = SegmentDescriptor(index=11, start=0x500, end=0x500 + len(data))
    segment = Segment(descriptor, data, "code")

    summaries = summarise_data_segments([segment], min_length=4)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.is_code_segment
    assert [s.text for s in summary.strings] == ["Hello", "More"]
    assert summary.hex_preview

    rendered = render_data_summaries(summaries)
    assert "embedded printable strings located inside a code segment" in rendered


def test_data_segment_hex_preview_configuration() -> None:
    descriptor = SegmentDescriptor(index=7, start=0x200, end=0x220)
    data = bytes(range(16))
    segment = Segment(descriptor, data, "blob")
    analyzer = DataSegmentAnalyzer(min_string_length=2, preview_bytes=8, preview_width=8)
    summary = analyzer.analyse(segment)

    assert len(summary.hex_preview) == 1
    line = summary.hex_preview[0]
    assert line.hex_bytes == "00 01 02 03 04 05 06 07"
    assert line.ascii.startswith("....")


def test_data_segment_repeated_runs_and_zero_detection() -> None:
    descriptor = SegmentDescriptor(index=9, start=0x300, end=0x340)
    data = b"\x00" * 12 + b"A" * 9 + b"\xFF" * 5
    segment = Segment(descriptor, data, "blob")
    analyzer = DataSegmentAnalyzer(run_threshold=4, max_runs=3, histogram_limit=4)
    summary = analyzer.analyse(segment)

    assert summary.repeated_runs
    zero_run = summary.longest_zero_run()
    assert zero_run is not None
    assert zero_run.length >= 12
    rendered = render_data_summaries([summary])
    assert "longest zero run" in rendered
    assert "top byte frequencies" in rendered


def test_aggregate_strings_generates_table() -> None:
    descriptor_a = SegmentDescriptor(index=1, start=0x000, end=0x010)
    descriptor_b = SegmentDescriptor(index=2, start=0x100, end=0x110)
    seg_a = Segment(descriptor_a, b"foo\x00bar\x00", "strings")
    seg_b = Segment(descriptor_b, b"foo\x00baz\x00", "strings")
    summaries = summarise_data_segments([seg_a, seg_b], min_length=3)

    aggregated = aggregate_strings(summaries, min_occurrences=2)
    assert aggregated
    assert aggregated[0].text == "foo"
    table = render_string_table(aggregated)
    assert '"foo"' in table
    assert "seg 1" in table and "seg 2" in table


def test_render_data_table_produces_lua_structure() -> None:
    descriptor = SegmentDescriptor(index=6, start=0x400, end=0x420)
    data = b"alpha\x00\x00\x00alpha"
    segment = Segment(descriptor, data, "strings")
    summaries = summarise_data_segments(
        [segment],
        min_length=5,
        histogram_limit=3,
        run_threshold=2,
        max_runs=2,
    )
    lua_table = render_data_table(summaries, table_name="__tbl", return_table=True)
    assert "local __tbl = {" in lua_table
    assert "classification" in lua_table
    assert "is_code = false" in lua_table
    assert "strings" in lua_table
    assert "return __tbl" in lua_table


def test_compute_segment_statistics() -> None:
    descriptor_a = SegmentDescriptor(index=3, start=0x300, end=0x320)
    descriptor_b = SegmentDescriptor(index=4, start=0x400, end=0x420)
    code_data = b"\x01\x02omega\x00"
    descriptor_code = SegmentDescriptor(
        index=12, start=0x540, end=0x540 + len(code_data)
    )
    seg_a = Segment(descriptor_a, b"alpha\x00beta\x00", "strings")
    seg_b = Segment(descriptor_b, b"\x00\x00", "blob")
    seg_code = Segment(descriptor_code, code_data, "code")
    summaries = summarise_data_segments(
        [seg_a, seg_b, seg_code], min_length=5, run_threshold=2
    )
    assert any(summary.is_code_segment for summary in summaries)

    stats = compute_segment_statistics(summaries)
    assert stats.segment_count == 2
    assert stats.string_segments == 1
    assert stats.string_count == 1
    assert stats.total_string_length == len("alpha")
    assert stats.total_bytes == sum(
        summary.length for summary in summaries if not summary.is_code_segment
    )
    assert stats.common_bytes
    assert stats.longest_zero_run is not None

    stats_section = render_segment_statistics(stats)
    assert "data segment statistics" in stats_section
    assert "most common bytes" in stats_section
