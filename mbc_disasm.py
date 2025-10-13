#!/usr/bin/env python3
"""Command-line interface for the Sphere MBC disassembler."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from mbcdisasm import (
    ASTBuilder,
    ASTRenderer,
    IRNormalizer,
    IRTextRenderer,
    KnowledgeBase,
    MbcContainer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Either <stem> or <adb> <mbc>. When a single stem is provided,"
        " the input files are resolved relative to --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("mbc"),
        help="Base directory used when resolving inputs from a single stem",
    )
    parser.add_argument(
        "--segment",
        type=int,
        action="append",
        dest="segments",
        help="Restrict disassembly to the selected segment indices",
    )
    parser.add_argument(
        "--ir-out",
        type=Path,
        default=None,
        help="Override the default <mbc>.ir.txt output path",
    )
    parser.add_argument(
        "--ast-out",
        type=Path,
        default=None,
        help="Override the default <mbc>.ast.txt output path",
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/manual_annotations.json"),
        help="Location of the manual opcode annotation file",
    )
    return parser.parse_args()


def validate_inputs(adb_path: Path, mbc_path: Path) -> None:
    for path in (adb_path, mbc_path):
        if not path.exists():
            raise SystemExit(f"missing input file: {path}")


def resolve_segments(args: argparse.Namespace) -> Sequence[int]:
    if args.segments:
        return tuple(args.segments)
    return ()


def resolve_input_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    inputs = args.inputs
    if len(inputs) == 1:
        stem = Path(inputs[0])
        adb_path = (args.root / stem).with_suffix(".adb")
        mbc_path = (args.root / stem).with_suffix(".mbc")
        return adb_path, mbc_path
    if len(inputs) == 2:
        adb_path, mbc_path = map(Path, inputs)
        return adb_path, mbc_path
    raise SystemExit("expected either a single stem or <adb> <mbc> inputs")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    adb_path, mbc_path = resolve_input_paths(args)
    validate_inputs(adb_path, mbc_path)

    knowledge = KnowledgeBase.load(args.knowledge_base)
    container = MbcContainer.load(mbc_path, adb_path)

    selection = resolve_segments(args)

    ir_normalizer = IRNormalizer(knowledge)
    program = ir_normalizer.normalise_container(container, segment_indices=selection)
    ir_output_path = args.ir_out or mbc_path.with_suffix(".ir.txt")
    IRTextRenderer().write(program, ir_output_path)
    print(f"ir written to {ir_output_path}")

    ast_program = ASTBuilder().build(program)
    ast_output_path = args.ast_out or mbc_path.with_suffix(".ast.txt")
    ASTRenderer().write(ast_program, ast_output_path)
    print(f"ast written to {ast_output_path}")

    total_time = time.perf_counter() - start_time
    print(f"total execution time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
