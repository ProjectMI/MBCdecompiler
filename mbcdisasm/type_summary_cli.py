"""Lightweight CLI that prints segment and opcode statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import Analyzer
from .knowledge import KnowledgeBase
from .mbc import MbcContainer
from .segment_classifier import SegmentClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adb", type=Path)
    parser.add_argument("mbc", type=Path)
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/opcode_profiles.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    knowledge = KnowledgeBase.load(args.knowledge_base)
    classifier = SegmentClassifier(knowledge)
    container = MbcContainer.load(args.mbc, args.adb, classifier=classifier)
    analysis = Analyzer(knowledge).analyze(container)

    print(f"segments: {analysis.total_segments}")
    print(f"instructions: {analysis.total_instructions}")
    for report in analysis.segment_reports[:10]:
        print(
            f"seg {report.index:4d} offset=0x{report.start:06X} len={report.length:6d} "
            f"class={report.classification} instr={report.instruction_count}"
        )

    hot = analysis.most_common(15)
    if hot:
        print("top opcode/mode combinations:")
        for key, count in hot:
            print(f"  {key:7s} -> {count:6d}")

    breakdown = analysis.confidence_breakdown()
    if breakdown:
        print("knowledge confidence:")
        for status, count in breakdown.items():
            print(f"  {status:12s}: {count:6d}")

    conflicts = analysis.conflicting_profiles()
    if conflicts:
        print("potential conflicts:")
        for assessment in conflicts[:5]:
            print(
                f"  {assessment.key:7s} {assessment.notes} "
                f"(existing={assessment.existing_count} new={assessment.new_count})"
            )


if __name__ == "__main__":
    main()
