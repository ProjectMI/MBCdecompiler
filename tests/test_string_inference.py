from mbcdisasm.string_inference import StringAnalyzer, StringLiteralSequence


def test_analyzer_suggests_chunk_names_and_identifiers() -> None:
    analyzer = StringAnalyzer()
    text = "SaveGame"
    offsets = (0x10, 0x14, 0x18, 0x1C)
    fragments = ["Sa", "ve", "Ga", "me"]
    analysis = analyzer.analyse(text, offsets, fragments, entry_offset=0x10)

    assert analysis.primary_identifier == "SaveGame"
    assert analysis.chunk_name_suggestions[0] == "savegame_str"
    assert analysis.candidates

    sequence = StringLiteralSequence(
        text=analysis.text,
        offsets=analysis.offsets,
        chunk_names=tuple(analysis.chunk_name_suggestions),
        candidates=analysis.candidates,
        primary_identifier=analysis.primary_identifier,
        categories=analysis.categories,
        confidence=analysis.confidence,
        notes=analysis.notes,
    )

    comments = sequence.comment_lines()
    assert comments[0].startswith("string literal sequence: \"SaveGame\"")
    assert any("identifier hint: SaveGame" in line for line in comments)

    selected = analyzer.select_function_name([sequence], entry_offset=0x10)
    assert selected == "SaveGame"


def test_analyzer_classifies_path_strings() -> None:
    analyzer = StringAnalyzer()
    text = "config/path/to/file.lua"
    offsets = (0x200, 0x204, 0x208, 0x20C, 0x210)
    fragments = ["co", "nf", "ig", "/p", "at"]
    analysis = analyzer.analyse(text, offsets, fragments, entry_offset=0x200)

    assert "path" in analysis.categories
    assert analysis.chunk_name_suggestions[0].startswith("config_path")
