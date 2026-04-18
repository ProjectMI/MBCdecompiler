from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from mbl_vm_tools.report import analyze_module, analyze_many


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MBL script v4.0 modules")
    parser.add_argument("path", help="Single .mbc file or a directory containing .mbc files")
    parser.add_argument("--out", help="Write JSON report to file")
    parser.add_argument("--overrides", help="Optional known_layouts.json override file")
    args = parser.parse_args()

    target = Path(args.path)
    overrides = None
    if args.overrides:
        overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))

    if target.is_dir():
        paths = sorted(target.glob("*.mbc"))
        if not paths:
            result = {"summary": {"module_count": 0}, "modules": []}
        else:
            result = analyze_many(tqdm(paths, desc="Processing scripts", unit="script"), overrides=overrides)
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
