#!/usr/bin/env python3
"""Report the scripts with the highest ``raw_remaining`` counts.

The script runs the core disassembly pipeline (instruction loading, IR
normalisation, etc.) over every script available in the local ``mbc`` folder
and prints the top offenders ranked by ``raw_remaining``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbcdisasm import IRNormalizer, KnowledgeBase, MbcContainer


def iter_script_inputs(root: Path) -> Iterable[Tuple[Path, Path]]:
    """Yield ``(mbc, adb)`` pairs for every script in ``root``."""

    for mbc_path in sorted(root.glob("*.mbc")):
        adb_path = mbc_path.with_suffix(".adb")
        if not adb_path.exists():
            continue
        yield mbc_path, adb_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("mbc"),
        help="Directory that stores the <stem>.mbc/<stem>.adb pairs",
    )
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path("knowledge/manual_annotations.json"),
        help="Location of the manual opcode annotation file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Restrict processing to the first N script pairs",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of entries to show in the final summary",
    )
    args = parser.parse_args()

    root = args.root
    knowledge_path = args.knowledge

    if not root.exists():
        raise SystemExit(f"scripts directory not found: {root}")

    knowledge = KnowledgeBase.load(knowledge_path)

    scripts = list(iter_script_inputs(root))
    if args.limit is not None:
        scripts = scripts[: max(0, args.limit)]
    total = len(scripts)

    if total == 0:
        raise SystemExit("no scripts found under the mbc directory")

    results = []
    for index, (mbc_path, adb_path) in enumerate(scripts, start=1):
        container = MbcContainer.load(mbc_path, adb_path)
        normalizer = IRNormalizer(knowledge)
        program = normalizer.normalise_container(container)
        raw_remaining = program.metrics.raw_remaining
        results.append((mbc_path.stem, raw_remaining))

        if index % 50 == 0:
            remaining = total - index
            print(f"Processed {index} of {total} scripts, {remaining} remaining")

    results.sort(key=lambda item: item[1], reverse=True)

    top_n = args.top if args.top > 0 else len(results)
    print(f"Top {min(top_n, len(results))} scripts by raw_remaining:")
    for name, value in results[:top_n]:
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
