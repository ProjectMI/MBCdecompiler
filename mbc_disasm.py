#!/usr/bin/env python3
"""Command-line interface for the Sphere MBC disassembler."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from mbcdisasm import Disassembler, KnowledgeBase, MbcContainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adb", type=Path, help="Path to the companion .adb index file")
    parser.add_argument("mbc", type=Path, help="Path to the .mbc container")
    parser.add_argument(
        "--segment",
        type=int,
        action="append",
        dest="segments",
        help="Restrict disassembly to the selected segment indices",
    )
    parser.add_argument(
        "--max-instr",
        type=int,
        default=None,
        help="Truncate each segment after the specified instruction count",
    )
    parser.add_argument(
        "--disasm-out",
        type=Path,
        default=None,
        help="Override the default <mbc>.disasm.txt output path",
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/manual_annotations.json"),
        help="Location of the manual opcode annotation file",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.adb, args.mbc):
        if not path.exists():
            raise SystemExit(f"missing input file: {path}")


def resolve_segments(args: argparse.Namespace) -> Sequence[int]:
    if args.segments:
        return tuple(args.segments)
    return ()


def main() -> None:
    args = parse_args()
    validate_inputs(args)

    knowledge = KnowledgeBase.load(args.knowledge_base)
    container = MbcContainer.load(args.mbc, args.adb)

    output_path = args.disasm_out or args.mbc.with_suffix(".disasm.txt")
    disassembler = Disassembler(knowledge)
    summary = disassembler.write_listing(
        container,
        output_path,
        segment_indices=resolve_segments(args),
        max_instructions=args.max_instr,
    )
    ir_output = output_path.with_suffix(".ir.txt")
    disassembler.write_ir(
        container,
        ir_output,
        segment_indices=resolve_segments(args),
        max_instructions=args.max_instr,
    )
    print(f"disassembly written to {output_path}")
    print(f"ir written to {ir_output}")
    if summary:
        print(
            "analysis summary: "
            f"unknown kind={summary.unknown_kinds} "
            f"category={summary.unknown_categories} "
            f"pattern={summary.unknown_patterns} "
            f"dominant={summary.unknown_dominant} "
            f"warnings={summary.warning_count}"
        )


if __name__ == "__main__":
    main()
