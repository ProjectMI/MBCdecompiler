#!/usr/bin/env python3
"""Command-line entry point for high-level Lua reconstruction."""

from __future__ import annotations

import argparse
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
from mbcdisasm.highlevel import HighLevelReconstructor, HighLevelFunction


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
    reconstructor = HighLevelReconstructor(knowledge)

    functions: list[HighLevelFunction] = []
    for segment in iter_segments(container, args.segments):
        graph = cfg_builder.build(segment, max_instructions=args.max_instr)
        program = ir_builder.from_cfg(segment, graph)
        functions.append(reconstructor.from_ir(program))

    module_text = reconstructor.render(functions)

    output_path = args.output or args.mbc.with_suffix(".lua")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(module_text, "utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
