#!/usr/bin/env python3
"""Utility script to inspect IR normaliser raw_remaining counts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from mbcdisasm import IRNormalizer, KnowledgeBase, MbcContainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the disassembly pipeline across all scripts and list the "
            "containers with the highest number of remaining raw IR nodes."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("mbc"),
        help="Directory containing <stem>.mbc/<stem>.adb pairs.",
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/manual_annotations.json"),
        help="Path to the manual opcode annotation file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of results to display in the final summary.",
    )
    parser.add_argument(
        "--progress-chunk",
        type=int,
        default=50,
        help="Number of processed scripts between progress updates.",
    )
    return parser.parse_args()


def collect_script_pairs(root: Path) -> List[Tuple[str, Path, Path]]:
    """Collect and validate <stem>.mbc/<stem>.adb pairs located under ``root``."""

    if not root.exists():
        raise SystemExit(f"script directory does not exist: {root}")

    stems: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for path in root.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".mbc", ".adb"}:
            continue
        stems[path.stem][suffix] = path

    pairs: List[Tuple[str, Path, Path]] = []
    for stem, files in stems.items():
        mbc_path = files.get(".mbc")
        adb_path = files.get(".adb")
        if not mbc_path or not adb_path:
            missing = ".mbc" if not mbc_path else ".adb"
            raise SystemExit(f"missing {missing} for script '{stem}' in {root}")
        pairs.append((stem, mbc_path, adb_path))

    pairs.sort(key=lambda item: item[0])
    return pairs


def describe_progress(index: int, total: int) -> str:
    processed = index
    remaining = total - processed
    return f"processed {processed}/{total} scripts, remaining {remaining}"


def run_pipeline(
    pairs: Sequence[Tuple[str, Path, Path]],
    knowledge_base: Path,
    *,
    progress_chunk: int,
    limit: int,
) -> None:
    knowledge = KnowledgeBase.load(knowledge_base)
    normalizer = IRNormalizer(knowledge)

    results: List[Tuple[str, int]] = []
    total = len(pairs)

    last_reported = 0

    for index, (stem, mbc_path, adb_path) in enumerate(pairs, start=1):
        container = MbcContainer.load(mbc_path, adb_path)
        program = normalizer.normalise_container(container)
        results.append((stem, program.metrics.raw_remaining))

        if progress_chunk > 0 and index % progress_chunk == 0:
            print(describe_progress(index, total))
            last_reported = index

    if last_reported != total:
        print(describe_progress(total, total))

    results.sort(key=lambda item: item[1], reverse=True)
    top_results = results[: max(limit, 0)] if results else []

    print("\nTop scripts by raw_remaining:")
    for stem, remaining in top_results:
        print(f"{stem}: {remaining}")


def main() -> None:
    args = parse_args()
    pairs = collect_script_pairs(args.root)
    if not pairs:
        raise SystemExit(f"no scripts found under {args.root}")
    run_pipeline(
        pairs,
        args.knowledge_base,
        progress_chunk=args.progress_chunk,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
