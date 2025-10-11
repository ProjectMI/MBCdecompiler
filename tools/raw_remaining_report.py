from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mbcdisasm import IRNormalizer, KnowledgeBase, MbcContainer


def iter_script_pairs(root: Path) -> list[tuple[Path, Path]]:
    adb_stems = {path.stem for path in root.glob("*.adb")}
    mbc_stems = {path.stem for path in root.glob("*.mbc")}
    common = sorted(adb_stems & mbc_stems)
    return [(root / f"{stem}.adb", root / f"{stem}.mbc") for stem in common]


def main() -> None:
    root = Path("mbc")
    pairs = iter_script_pairs(root)
    total = len(pairs)
    if total == 0:
        raise SystemExit(f"no script pairs found in {root}")

    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    normalizer = IRNormalizer(knowledge)

    results: list[tuple[int, str]] = []

    for index, (adb_path, mbc_path) in enumerate(pairs, start=1):
        container = MbcContainer.load(mbc_path, adb_path)
        program = normalizer.normalise_container(container)
        remaining = program.metrics.raw_remaining
        results.append((remaining, mbc_path.stem))
        print(
            f"[{index}/{total}] processed {mbc_path.stem}: raw_remaining={remaining}; "
            f"scripts left {total - index}"
        )

    top5 = sorted(results, reverse=True)[:5]

    print("\nTop 5 scripts by raw_remaining:")
    for rank, (remaining, stem) in enumerate(top5, start=1):
        print(f"{rank}. {stem}: {remaining}")


if __name__ == "__main__":
    main()
