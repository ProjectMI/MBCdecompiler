from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from mbcdisasm.analysis import Analyzer
from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.mbc import MbcContainer
from mbcdisasm.segment_classifier import SegmentClassifier


def _mbc_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for adb_path in sorted(root.glob("*.adb")):
        mbc_path = adb_path.with_suffix(".mbc")
        if mbc_path.exists():
            pairs.append((adb_path, mbc_path))
    return pairs


def test_enumerate_stack_gap_inventory() -> None:
    mbc_root = Path("mbc")
    pairs = _mbc_pairs(mbc_root)
    if not pairs:
        pytest.skip("no mbc containers available")

    knowledge = KnowledgeBase.load(Path("knowledge/opcode_profiles.json"))
    analyzer = Analyzer(knowledge)
    classifier = SegmentClassifier(knowledge)

    aggregate_unknown = Counter()
    total = len(pairs)

    for index, (adb_path, mbc_path) in enumerate(pairs, 1):
        container = MbcContainer.load(mbc_path, adb_path, classifier=classifier)
        analysis = analyzer.analyze(container)
        for observation in analysis.stack_observations.values():
            if observation.unknown_samples:
                aggregate_unknown[observation.key] += observation.unknown_samples

        if index % 50 == 0 or index == total:
            remaining = total - index
            print(
                f"processed {index}/{total} containers, remaining: {remaining}",
                flush=True,
            )

    print("stack knowledge gaps:")
    if not aggregate_unknown:
        print("  none")
        return

    for key, count in sorted(aggregate_unknown.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {key}: {count}")
