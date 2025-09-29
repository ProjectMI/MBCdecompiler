#!/usr/bin/env python3
"""Command-line entry point for high-level Lua reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from mbcdisasm import (
    ControlFlowGraphBuilder,
    IRBuilder,
    KnowledgeBase,
    ManualSemanticAnalyzer,
    MbcContainer,
    Segment,
    SegmentClassifier,
)
from mbcdisasm.data_segments import render_data_summaries, summarise_data_segments
from mbcdisasm.highlevel import HighLevelReconstructor, HighLevelFunction
from mbcdisasm.literal_sequences import build_literal_run_report, literal_report_to_dict
from mbcdisasm.lua_formatter import LuaRenderOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adb", type=Path, help="Path to the companion .adb index file")
    parser.add_argument("mbc", type=Path, help="Path to the .mbc container")
    parser.add_argument(
        "--segment",
        type=int,
        action="append",
        dest="segments",
        help="Restrict reconstruction to the selected segment indices",
    )
    parser.add_argument(
        "--max-instr",
        type=int,
        default=None,
        help="Truncate each segment after the specified instruction count",
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/opcode_profiles.json"),
        help="Location of the opcode knowledge base database",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the reconstructed Lua module to the provided path",
    )
    parser.add_argument(
        "--keep-duplicate-comments",
        action="store_true",
        help="Preserve repeated semantic comments instead of collapsing them",
    )
    parser.add_argument(
        "--inline-comment-width",
        type=int,
        default=None,
        help="Maximum width for inline comments before they become standalone",
    )
    parser.add_argument(
        "--no-stub-metadata",
        action="store_true",
        help="Skip emitting helper stub metadata (inputs/outputs annotations)",
    )
    parser.add_argument(
        "--no-enum-metadata",
        action="store_true",
        help="Do not emit enumeration metadata comments above namespace tables",
    )
    parser.add_argument(
        "--no-module-summary",
        action="store_true",
        help="Suppress the module-level summary comment block",
    )
    parser.add_argument(
        "--no-stack-diagnostics",
        action="store_true",
        help="Disable stack diagnostics comments in generated Lua",
    )
    parser.add_argument(
        "--no-control-flow-summary",
        action="store_true",
        help="Skip control flow summary comments",
    )
    parser.add_argument(
        "--no-literal-report",
        action="store_true",
        help="Skip emitting aggregated literal report information",
    )
    parser.add_argument(
        "--min-string-length",
        type=int,
        default=4,
        help="Minimum length for printable strings extracted from data segments",
    )
    parser.add_argument(
        "--data-hex-bytes",
        type=int,
        default=128,
        help="Maximum number of bytes to include in each data segment hex preview",
    )
    parser.add_argument(
        "--data-hex-width",
        type=int,
        default=16,
        help="Number of bytes per row when rendering hex previews",
    )
    parser.add_argument(
        "--no-data-hex",
        action="store_true",
        help="Skip hex previews when rendering data segments",
    )
    parser.add_argument(
        "--data-histogram",
        type=int,
        default=6,
        help="Number of dominant byte values to keep per data segment",
    )
    parser.add_argument(
        "--data-run-threshold",
        type=int,
        default=8,
        help="Minimum length for repeated byte runs to be reported",
    )
    parser.add_argument(
        "--data-max-runs",
        type=int,
        default=3,
        help="Maximum number of repeated byte runs to retain per segment",
    )
    parser.add_argument(
        "--string-table",
        action="store_true",
        help="Aggregate repeated strings across segments and render a lookup table",
    )
    parser.add_argument(
        "--string-table-min-occurrences",
        type=int,
        default=2,
        help="Minimum number of occurrences for strings to appear in the aggregated table",
    )
    parser.add_argument(
        "--data-stats",
        action="store_true",
        help="Include a statistical breakdown of the collected data segments",
    )
    parser.add_argument(
        "--emit-data-table",
        action="store_true",
        help="Emit a Lua table describing data segments alongside comment summaries",
    )
    parser.add_argument(
        "--data-table-name",
        type=str,
        default="__data_segments",
        help="Name of the Lua table emitted when --emit-data-table is set",
    )
    parser.add_argument(
        "--data-table-return",
        action="store_true",
        help="Append a return statement for the generated data table",
    )
    parser.add_argument(
        "--literal-report-json",
        type=Path,
        default=None,
        help="Write literal run statistics to the specified JSON file",
    )
    return parser.parse_args()


def iter_segments(container: MbcContainer, selection: Optional[Sequence[int]]) -> Iterable[Segment]:
    if not selection:
        for segment in container.segments():
            if segment.is_code:
                yield segment
        return
    requested = set(selection)
    for segment in container.segments():
        if segment.index in requested and segment.is_code:
            yield segment


def main() -> None:
    args = parse_args()

    knowledge = KnowledgeBase.load(args.knowledge_base)
    classifier = SegmentClassifier(knowledge)
    container = MbcContainer.load(args.mbc, args.adb, classifier=classifier)

    semantics = ManualSemanticAnalyzer(knowledge)
    cfg_builder = ControlFlowGraphBuilder(knowledge, semantic_analyzer=semantics)
    ir_builder = IRBuilder(knowledge)
    options = LuaRenderOptions()
    if args.inline_comment_width is not None and args.inline_comment_width > 0:
        options.max_inline_comment = args.inline_comment_width
    if args.keep_duplicate_comments:
        options.deduplicate_comments = False
    if args.no_stub_metadata:
        options.emit_stub_metadata = False
    if args.no_enum_metadata:
        options.emit_enum_metadata = False
    if args.no_module_summary:
        options.emit_module_summary = False
    if args.no_stack_diagnostics:
        options.emit_stack_diagnostics = False
    if args.no_control_flow_summary:
        options.emit_control_flow_summary = False
    if args.no_literal_report:
        options.emit_literal_report = False

    reconstructor = HighLevelReconstructor(knowledge, options=options)

    functions: list[HighLevelFunction] = []
    for segment in iter_segments(container, args.segments):
        graph = cfg_builder.build(segment, max_instructions=args.max_instr)
        program = ir_builder.from_cfg(segment, graph)
        functions.append(reconstructor.from_ir(program))

    module_text = reconstructor.render(functions)

    if args.literal_report_json:
        all_runs = [run for func in functions for run in func.metadata.literal_runs]
        report = build_literal_run_report(all_runs)
        payload = literal_report_to_dict(report)
        args.literal_report_json.parent.mkdir(parents=True, exist_ok=True)
        args.literal_report_json.write_text(json.dumps(payload, indent=2), "utf-8")

    data_summaries = summarise_data_segments(
        container.segments(),
        min_length=args.min_string_length,
        preview_bytes=args.data_hex_bytes,
        preview_width=args.data_hex_width,
        histogram_limit=args.data_histogram,
        run_threshold=args.data_run_threshold,
        max_runs=args.data_max_runs,
    )
    data_section = render_data_summaries(
        data_summaries, include_hex=not args.no_data_hex
    )
    if data_section:
        module_text = module_text.rstrip() + "\n\n" + data_section

    if args.data_stats:
        from mbcdisasm.data_segments import compute_segment_statistics, render_segment_statistics

        stats = compute_segment_statistics(data_summaries)
        stats_section = render_segment_statistics(stats)
        if stats_section:
            module_text = module_text.rstrip() + "\n\n" + stats_section

    if args.string_table:
        from mbcdisasm.data_segments import aggregate_strings, render_string_table

        aggregated = aggregate_strings(
            data_summaries, min_occurrences=args.string_table_min_occurrences
        )
        table_section = render_string_table(aggregated)
        if table_section:
            module_text = module_text.rstrip() + "\n\n" + table_section

    if args.emit_data_table:
        from mbcdisasm.data_segments import render_data_table

        table_text = render_data_table(
            data_summaries,
            table_name=args.data_table_name,
            include_strings=True,
            include_histogram=True,
            include_runs=True,
            return_table=args.data_table_return,
        )
        if table_text:
            module_text = module_text.rstrip() + "\n\n" + table_text

    output_path = args.output or args.mbc.with_suffix(".lua")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(module_text, "utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
