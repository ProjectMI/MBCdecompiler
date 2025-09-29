import json

from mbcdisasm.inline_strings import (
    InlineStringAccumulator,
    InlineStringChunk,
    InlineStringCollector,
    escape_lua_bytes,
    render_inline_tables,
)
from mbcdisasm.ir import IRInstruction
from mbcdisasm.manual_semantics import InstructionSemantics, StackEffect


def _make_inline_semantics(name: str) -> InstructionSemantics:
    return InstructionSemantics(
        key="72:65",
        mnemonic=name,
        manual_name=name,
        summary="inline chunk",
        control_flow=None,
        stack_delta=0,
        stack_effect=StackEffect(inputs=0, outputs=0, delta=0, source="test"),
        tags=("literal",),
        comparison_operator=None,
        enum_values={},
        enum_namespace=None,
        struct_context=None,
        stack_inputs=0,
        stack_outputs=0,
        uses_operand=False,
        operand_hint=None,
        vm_method=name,
        vm_call_style="literal",
    )


def _make_instruction(offset: int, key: str, operand: int) -> IRInstruction:
    semantics = _make_inline_semantics("inline_ascii_chunk_7265")
    return IRInstruction(
        offset=offset,
        key=key,
        mnemonic=semantics.mnemonic,
        operand=operand,
        stack_delta=0,
        control_flow=None,
        semantics=semantics,
        stack_inputs=0,
        stack_outputs=0,
    )


def _chunk(
    segment: int,
    offset: int,
    data: bytes,
    *,
    block_start: int = 0,
    end_offset: int | None = None,
) -> InlineStringChunk:
    return InlineStringChunk(
        segment_index=segment,
        block_start=block_start,
        start_offset=offset,
        end_offset=end_offset if end_offset is not None else offset,
        data=data,
        instruction_offsets=(offset,),
    )


def test_accumulator_decodes_bytes() -> None:
    accumulator = InlineStringAccumulator()
    instruction = _make_instruction(0x10, "72:65", 0x6E74)
    accumulator.feed(instruction)
    chunk = accumulator.finish(segment_index=2, block_start=0x10)
    assert chunk.data == b"rent"
    assert chunk.start_offset == 0x10
    assert chunk.end_offset == 0x10


def test_collector_tracks_statistics() -> None:
    collector = InlineStringCollector()
    collector.add(_chunk(1, 0x20, b"foo"))
    collector.add(_chunk(1, 0x40, b"barbaz"))
    assert collector.entry_count() == 2
    assert collector.segment_count() == 1
    assert collector.total_bytes() == 9
    assert collector.bytes_for_segment(1) == 9
    longest = collector.longest_chunk()
    assert longest and longest.data == b"barbaz"
    matches = collector.find("BAR")
    assert matches and matches[0].data == b"barbaz"
    assert collector.find("") == []
    report = collector.build_report()
    assert report.entry_count == 2
    assert report.segment_count == 1
    assert report.total_bytes == 9
    assert report.average_length == 4.5
    assert report.longest_summary() == "6 bytes at segment 001 offset 0x000040"
    report_dict = report.to_dict()
    assert report_dict["largest"] == "6 bytes at segment 001 offset 0x000040"


def test_render_inline_tables_handles_escape_sequences() -> None:
    collector = InlineStringCollector()
    collector.add(_chunk(3, 0x100, b"line1\nline2"))
    rendered = render_inline_tables(collector)
    assert "inline_segment_003" in rendered
    assert "\\n" in rendered


def test_escape_lua_bytes_roundtrip() -> None:
    literal = escape_lua_bytes(b"a\x00b\\")
    assert literal == '"a\\x00b\\\\"'


def test_collector_serialisation() -> None:
    collector = InlineStringCollector()
    collector.add(_chunk(0, 0x10, b"hi"))
    mapping = collector.to_dict()
    assert mapping == {
        "000": {"0x000010": {"hex": "6869", "lua": '"hi"'}}
    }
    payload = json.loads(collector.to_json(indent=0))
    assert payload["000"]["0x000010"]["hex"] == "6869"


def test_merged_strings_combines_contiguous_chunks() -> None:
    collector = InlineStringCollector()
    collector.add(_chunk(2, 0x10, b"ab"))
    collector.add(_chunk(2, 0x14, b"cd"))
    collector.add(_chunk(2, 0x40, b"zz"))
    merged = list(collector.iter_merged())
    assert merged[0][0] == 2
    assert merged[0][2] == b"abcd"
    assert merged[1][2] == b"zz"
    merged_lookup = collector.merged_strings()
    assert merged_lookup[2][0] == '"abcd"'


def test_filter_segments_returns_subset() -> None:
    collector = InlineStringCollector()
    collector.add(_chunk(1, 0x10, b"aa"))
    collector.add(_chunk(2, 0x20, b"bb"))
    filtered = collector.filter_segments(lambda seg: seg == 2)
    assert filtered.entry_count() == 1
    assert filtered.segment_count() == 1
    assert list(filtered.segments()) == [2]


def test_chunk_preview_and_ratio() -> None:
    chunk = _chunk(0, 0x0, b"Dialog\nline")
    assert chunk.is_probably_text()
    preview = chunk.preview(limit=10)
    assert "Dialog" in preview
    noisy = _chunk(0, 0x10, b"\x01\x02\x03\x04")
    assert not noisy.is_probably_text()


def test_collector_sequences_and_summary() -> None:
    collector = InlineStringCollector()
    collector.add(_chunk(7, 0x30, b"He", block_start=0x100))
    collector.add(_chunk(7, 0x34, b"llo", block_start=0x100))
    collector.add(_chunk(8, 0x40, b"\x10\x11", block_start=0x200))
    sequences = list(collector.iter_sequences())
    assert len(sequences) == 2
    assert sequences[0].preview().startswith("Hello")
    summary = [
        (
            f"segment {sequence.segment_index:03d} block 0x{sequence.start_block:06X} "
            f"len={sequence.total_length} bytes"
        )
        for sequence in collector.iter_sequences()
    ]
    assert any("segment 007" in line for line in summary)
