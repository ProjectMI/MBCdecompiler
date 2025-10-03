"""Command-line interface for the Sphere MBC disassembler."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path
from typing import Iterable, Sequence

from mbcdisasm import Disassembler, KnowledgeBase, MbcContainer, Normalizer, Segment
from mbcdisasm.instruction import read_instructions
from mbcdisasm.ir.normalizer import NormalizerMetrics
from mbcdisasm.ir.serialize import serialize_metrics, serialize_result


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
        "--ir-out",
        type=Path,
        default=None,
        help="Override the default <mbc>.ir.json IR dump path",
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


def iter_selected_segments(
    container: MbcContainer, selection: Sequence[int]
) -> Iterable[Segment]:
    if selection:
        segment_map = {segment.index: segment for segment in container.segments()}
        for index in selection:
            segment = segment_map.get(index)
            if segment is not None:
                yield segment
    else:
        yield from container.segments()


def _empty_metric_totals() -> OrderedDict[str, int]:
    return OrderedDict((field.name, 0) for field in fields(NormalizerMetrics))


def normalise_segments(
    normalizer: Normalizer,
    container: MbcContainer,
    selection: Sequence[int],
    *,
    max_instructions: int | None,
) -> tuple[OrderedDict[str, int], list[dict[str, object]]]:
    totals = _empty_metric_totals()
    segments_payload: list[dict[str, object]] = []

    for segment in iter_selected_segments(container, selection):
        instructions, remainder = read_instructions(segment.data, segment.start)
        if max_instructions is not None:
            instructions = instructions[: max_instructions]
        result = normalizer.normalise(instructions)
        metrics_dict = serialize_metrics(result.metrics)
        for name, value in metrics_dict.items():
            totals[name] += value
        segment_payload: dict[str, object] = {
            "segment_index": segment.index,
            "start_offset": segment.start,
            "length": segment.length,
            "remainder_bytes": remainder,
            "metrics": metrics_dict,
        }
        segment_payload.update(serialize_result(result))
        segments_payload.append(segment_payload)

    return totals, segments_payload


def main() -> None:
    args = parse_args()
    validate_inputs(args)

    knowledge = KnowledgeBase.load(args.knowledge_base)
    container = MbcContainer.load(args.mbc, args.adb)

    output_path = args.disasm_out or args.mbc.with_suffix(".disasm.txt")
    disassembler = Disassembler(knowledge)
    disassembler.write_listing(
        container,
        output_path,
        segment_indices=resolve_segments(args),
        max_instructions=args.max_instr,
    )
    print(f"disassembly written to {output_path}")

    normalizer = Normalizer(knowledge)
    selection = resolve_segments(args)
    totals, segments_payload = normalise_segments(
        normalizer,
        container,
        selection,
        max_instructions=args.max_instr,
    )
    ir_output = args.ir_out or args.mbc.with_suffix(".ir.json")
    payload = {
        "container": str(args.mbc),
        "segments": segments_payload,
        "metrics": dict(totals),
    }
    ir_output.write_text(json.dumps(payload, indent=2), "utf-8")
    print(f"ir written to {ir_output}")

    metrics_summary = " ".join(f"{name}={value}" for name, value in totals.items())
    print(f"normalizer metrics: {metrics_summary}")


if __name__ == "__main__":
    main()
