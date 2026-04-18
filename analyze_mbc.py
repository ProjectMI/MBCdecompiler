from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from mbl_vm_tools.report import analyze_module, summarize_many, _annotate_adb_consensus


def _analyze_one(args: tuple[str, dict | None]) -> dict:
    path, overrides = args
    return analyze_module(path, overrides=overrides)


def analyze_many_parallel(paths: list[Path], overrides: dict | None = None, workers: int | None = None) -> dict:
    if not paths:
        return {"summary": {"module_count": 0}, "modules": []}

    if workers is None:
        workers = max(1, os.cpu_count() or 1)
    workers = max(1, min(workers, len(paths)))

    if workers == 1:
        modules = [
            analyze_module(path, overrides=overrides)
            for path in tqdm(paths, desc="Processing scripts", unit="script")
        ]
    else:
        modules: list[dict | None] = [None] * len(paths)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(_analyze_one, (str(path), overrides)): idx
                for idx, path in enumerate(paths)
            }
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing scripts", unit="script"):
                idx = future_to_index[future]
                modules[idx] = future.result()

        modules = [m for m in modules if m is not None]

    adb_summary = _annotate_adb_consensus(modules)
    summary = summarize_many(modules)
    summary["adb_consensus"] = adb_summary
    return {
        "summary": summary,
        "modules": modules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MBL script v4.0 modules")
    parser.add_argument("path", help="Single .mbc file or a directory containing .mbc files")
    parser.add_argument("--out", help="Write JSON report to file")
    parser.add_argument("--overrides", help="Optional known_layouts.json override file")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes for directory mode")
    args = parser.parse_args()

    target = Path(args.path)
    overrides = None
    if args.overrides:
        overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))

    if target.is_dir():
        paths = sorted(target.glob("*.mbc"))
        result = analyze_many_parallel(paths, overrides=overrides, workers=args.workers)
    else:
        result = analyze_module(target, overrides=overrides)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
