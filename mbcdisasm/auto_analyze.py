"""Batch analyzer that aggregates opcode statistics across many containers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, Tuple

from .analysis import Analyzer
from .knowledge import KnowledgeBase
from .mbc import MbcContainer
from .segment_classifier import SegmentClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Paths to directories or files containing .mbc/.adb pairs",
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/opcode_profiles.json"),
        help="Knowledge base database to update",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    knowledge = KnowledgeBase.load(args.knowledge_base)
    analyzer = Analyzer(knowledge)
    classifier = SegmentClassifier(knowledge)

    total_files = 0
    for adb_path, mbc_path in find_pairs(args.inputs):
        total_files += 1
        container = MbcContainer.load(mbc_path, adb_path, classifier=classifier)
        result = analyzer.analyze(container)
        merge_report = knowledge.merge_profiles(result.iter_profiles())
        semantic_report = knowledge.apply_semantic_annotations()
        print(f"processed {mbc_path.name}: {result.total_instructions} instructions")
        if merge_report.updates:
            sample = ", ".join(
                f"{u.key}:{u.new_value} ({u.confidence:.2f})"
                for u in merge_report.updates[:3]
            )
            if len(merge_report.updates) > 3:
                sample += f", +{len(merge_report.updates) - 3} more"
            print(f"  learned stack patterns: {sample}")
        if semantic_report.matches:
            preview = ", ".join(
                f"{match.key}->{match.prototype} ({match.score:.2f})"
                for match in semantic_report.matches[:3]
            )
            if len(semantic_report.matches) > 3:
                preview += f", +{len(semantic_report.matches) - 3} more"
            print(f"  semantic assignments: {preview}")
        conflicts = merge_report.conflicts()
        if conflicts:
            names = ", ".join(c.key for c in conflicts[:5])
            if len(conflicts) > 5:
                names += f", +{len(conflicts) - 5} more"
            print(f"  stack conflicts: {names}")
        if merge_report.review_tasks:
            preview = ", ".join(
                f"{task.key}:{task.reason}" for task in merge_report.review_tasks[:5]
            )
            if len(merge_report.review_tasks) > 5:
                preview += f", +{len(merge_report.review_tasks) - 5} more"
            print(f"  review queue: {preview}")

        summary = knowledge.render_merge_report(merge_report)
        if summary:
            print(summary)

    knowledge.save()
    classifier.save_profile()
    print(f"knowledge base saved to {knowledge.path} ({total_files} files)")


def find_pairs(inputs: Iterable[Path]) -> Iterator[Tuple[Path, Path]]:
    for path in inputs:
        if path.is_dir():
            for adb in sorted(path.glob("*.adb")):
                mbc = adb.with_suffix(".mbc")
                if mbc.exists():
                    yield adb, mbc
        else:
            if path.suffix.lower() == ".mbc":
                adb = path.with_suffix(".adb")
                if adb.exists():
                    yield adb, path
            elif path.suffix.lower() == ".adb":
                mbc = path.with_suffix(".mbc")
                if mbc.exists():
                    yield path, mbc


if __name__ == "__main__":
    main()
