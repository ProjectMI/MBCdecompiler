from mbcdisasm.string_tables import StringTable, StringTableEntry, parse_string_table


def test_parse_string_table_basic() -> None:
    data = b"hello\0world\0garbage"
    table = parse_string_table(data, start_offset=0x200)
    assert len(table.entries) == 2
    assert table.entries[0].text() == "hello"
    assert table.entries[1].offset == 0x206


def test_string_table_statistics_and_search() -> None:
    table = StringTable(start_offset=0)
    table.add(StringTableEntry(offset=0, data=b"Hello\0"))
    table.add(StringTableEntry(offset=6, data=b"\x01\x02\x03"))
    stats = table.statistics()
    assert stats["entries"] == 2
    assert table.search("hello")[0].offset == 0


def test_string_table_rendering() -> None:
    table = StringTable(start_offset=0x300)
    table.add(StringTableEntry(offset=0x300, data=b"Dialog\0"))
    rendered = table.render(prefix="dialogue")
    assert "dialogue" in rendered
    assert "0x000300" in rendered
