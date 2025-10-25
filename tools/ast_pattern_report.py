from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mbcdisasm import ASTBuilder, ASTRenderer, IRNormalizer, KnowledgeBase, MbcContainer


def iter_script_pairs(root: Path) -> Iterable[tuple[Path, Path]]:
    adb_stems = {path.stem for path in root.glob("*.adb")}
    mbc_stems = {path.stem for path in root.glob("*.mbc")}
    excluded_stems = {"_main"}
    common = sorted(stem for stem in adb_stems & mbc_stems if stem not in excluded_stems)
    for stem in common:
        yield root / f"{stem}.adb", root / f"{stem}.mbc"


PATTERNS: tuple[tuple[str, str], ...] = (
    ("returns=[ret[0]=value0:unknown", "Unknown return value slot"),
    ("value(?(", "Value operand in first argument slot"),
    ("op_", "Operations with op_ prefix"),
)


def main() -> None:
    root = Path("mbc")
    pairs = list(iter_script_pairs(root))
    if not pairs:
        raise SystemExit(f"no script pairs found in {root}")

    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    normalizer = IRNormalizer(knowledge)
    builder = ASTBuilder()
    renderer = ASTRenderer()

    totals: dict[str, list[tuple[int, str]]] = defaultdict(list)

    total_pairs = len(pairs)
    for index, (adb_path, mbc_path) in enumerate(pairs, start=1):
        container = MbcContainer.load(mbc_path, adb_path)
        program = normalizer.normalise_container(container)
        ast_program = builder.build(program)
        ast_text = renderer.render(ast_program)
        stem = mbc_path.stem

        print(f"[{index}/{total_pairs}] processed {stem}")

        for pattern, _ in PATTERNS:
            count = ast_text.count(pattern)
            if count:
                totals[pattern].append((count, stem))

    for pattern, description in PATTERNS:
        entries = sorted(
            totals.get(pattern, ()), key=lambda item: item[0], reverse=True
        )[:5]
        if not entries:
            continue

        print(f"\nTop 5 for '{pattern}' ({description}):")
        for rank, (count, stem) in enumerate(entries, start=1):
            print(f"{rank}. {stem}: {count}")


if __name__ == "__main__":
    main()
