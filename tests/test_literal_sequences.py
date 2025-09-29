from mbcdisasm.literal_sequences import (
    LiteralDescriptor,
    LiteralRun,
    LiteralRunCatalogue,
    LiteralRunReport,
    build_literal_run_report,
    compute_literal_statistics,
    literal_report_to_dict,
    literal_statistics_to_dict,
    literal_statistics_to_json,
)
from mbcdisasm.lua_ast import LiteralExpr


def _make_string_run(text: str, *, start: int = 0) -> LiteralRun:
    chunks = [text[i : i + 4] for i in range(0, len(text), 4)]
    descriptors = [
        LiteralDescriptor(kind="string", text=chunk, expression=LiteralExpr(f'"{chunk}"'))
        for chunk in chunks
    ]
    offsets = tuple(start + index * 4 for index in range(len(chunks)))
    return LiteralRun(kind="string", descriptors=tuple(descriptors), offsets=offsets, block_start=start)


def _make_number_run(values: list[int], *, start: int = 0) -> LiteralRun:
    descriptors = [
        LiteralDescriptor(kind="number", text=str(value), expression=LiteralExpr(str(value)))
        for value in values
    ]
    offsets = tuple(start + index * 4 for index in range(len(values)))
    return LiteralRun(kind="number", descriptors=tuple(descriptors), offsets=offsets, block_start=start)


def test_compute_literal_statistics_string_runs() -> None:
    run_a = _make_string_run("hello world", start=0x10)
    run_b = _make_string_run("demo demo", start=0x40)
    stats = compute_literal_statistics([run_a, run_b])

    assert stats.total_runs == 2
    assert stats.kind_counts["string"] == 2
    assert stats.string_stats is not None
    string_stats = stats.string_stats
    assert string_stats.run_count == 2
    assert string_stats.total_length == len("hello world" + "demo demo")
    assert "demo" in string_stats.token_frequency
    assert string_stats.longest_run is run_a
    assert string_stats.run_length_histogram[len('hello world')] == 1
    assert string_stats.token_length_histogram[4] >= 1
    summary = string_stats.summary_lines()
    assert any("top tokens" in line for line in summary)
    stats_dict = literal_statistics_to_dict(stats)
    assert stats_dict['strings']['run_count'] == 2
    json_blob = literal_statistics_to_json(stats)
    assert 'token_frequency' in json_blob


def test_compute_literal_statistics_numeric_runs() -> None:
    run = _make_number_run([1, 2, 3, 4], start=0x20)
    stats = compute_literal_statistics([run])

    assert stats.numeric_stats is not None
    numeric_stats = stats.numeric_stats
    assert numeric_stats.run_count == 1
    assert numeric_stats.total_values == 4
    assert numeric_stats.min_value == 1
    assert numeric_stats.max_value == 4
    assert numeric_stats.average_value == 2.5
    assert numeric_stats.run_length_histogram[4] == 1
    hist_lines = numeric_stats.summary_lines()
    assert any("top values" in line for line in hist_lines)


def test_literal_run_catalogue_filters() -> None:
    run_short = _make_string_run('abc', start=0x10)
    run_long = _make_string_run('longer example string', start=0x20)
    run_number = _make_number_run([1, 2, 3], start=0x30)
    catalogue = LiteralRunCatalogue([run_short, run_long, run_number])
    assert set(catalogue.kinds()) == {'number', 'string'}
    longest = catalogue.longest_runs(kind='string', limit=1)
    assert longest == (run_long,)
    filtered = catalogue.filter_by_min_length(4)
    assert run_short not in filtered and run_long in filtered
    table = catalogue.render_table(limit=2)
    assert 'preview' in table and 'kind' in table
    matches = catalogue.search('example')
    assert matches == (run_long,)
    token_filtered = catalogue.runs_with_min_tokens(2)
    assert run_long in token_filtered and run_short not in token_filtered


def test_literal_run_report_summary() -> None:
    run_a = _make_string_run('hello world', start=0x100)
    run_b = _make_number_run([10, 10, 20], start=0x200)
    report = build_literal_run_report([run_a, run_b])
    assert isinstance(report, LiteralRunReport)
    assert report.total_runs() == 2
    assert report.top_tokens(limit=2)[0][0] == 'hello'
    assert report.top_numbers(limit=1)[0][0] == 10
    previews = report.longest_previews(limit=1)
    assert previews and '0x000100' in previews[0]
    block_lines = report.block_lines(limit=2)
    assert any('block 0x000100' in line for line in block_lines)
    summary_lines = report.summary_lines()
    assert any('top tokens' in line for line in summary_lines)
    payload = literal_report_to_dict(report)
    assert payload['total_runs'] == 2
    assert len(payload['blocks']) == 2
