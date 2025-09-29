from mbcdisasm.string_utils import chunk_preview, is_printable_byte, printable_ratio, trim_ascii_suffix


def test_is_printable_byte_and_ratio() -> None:
    assert is_printable_byte(ord("A"))
    assert not is_printable_byte(0x01)
    data = b"ABCD"
    assert printable_ratio(data) == 1.0
    assert printable_ratio(b"\x00AB") < 1.0


def test_chunk_preview_and_trim() -> None:
    preview = chunk_preview(b"Line1\nLine2", limit=5)
    assert preview.text.startswith("Line1")
    assert preview.truncated
    assert trim_ascii_suffix(b"test\x00\x00") == b"test"
