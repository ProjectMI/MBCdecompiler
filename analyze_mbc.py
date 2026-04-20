from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from mbl_vm_tools.IR import build_module_ir, summarize_corpus, write_json


PROJECT_ROOT = Path(__file__).resolve().parent
MBC_DIR = PROJECT_ROOT / "mbc"
IR_OUT_DIR = PROJECT_ROOT / "ir"
DEFAULT_WORKERS = 16


def _resolve_script_path(script_arg: str) -> Path:
    candidate = Path(script_arg)
    if candidate.is_file():
        return candidate.resolve()

    if not candidate.suffix:
        by_name = MBC_DIR / f"{candidate.name}.mbc"
        if by_name.is_file():
            return by_name.resolve()

    in_corpus = MBC_DIR / candidate.name
    if in_corpus.is_file():
        return in_corpus.resolve()

    raise FileNotFoundError(f"Script not found: {script_arg}")


def _build_module_ir_one(path: str) -> dict:
    return build_module_ir(path, include_nodes=False)


def build_corpus_report(paths: list[Path], workers: int = DEFAULT_WORKERS) -> dict:
    if not paths:
        return {"summary": {"module_count": 0, "export_count": 0, "total_ir_nodes": 0}, "anomalous_exports": [], "anomalous_modules": []}

    worker_count = min(max(1, workers), len(paths))
    module_irs: list[dict | None] = [None] * len(paths)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_build_module_ir_one, str(path)): idx
            for idx, path in enumerate(paths)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="IR corpus",
            unit="script",
        ):
            idx = future_to_index[future]
            module_irs[idx] = future.result()

    ready_modules = [module for module in module_irs if module is not None]
    return summarize_corpus(ready_modules)


def _default_single_ir_out(script_path: Path) -> Path:
    return IR_OUT_DIR / f"{script_path.stem}.ir.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized IR for one MBC script or emit a corpus-wide anomaly report."
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Optional script path or name. Example: door1.mbc or door1",
    )
    parser.add_argument("--out", help="Output JSON path. In single-script mode this is the IR file; in corpus mode this is the report file.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Max workers for corpus mode (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    if args.script:
        script_path = _resolve_script_path(args.script)
        ir_payload = build_module_ir(script_path, include_nodes=True)
        out_path = Path(args.out) if args.out else _default_single_ir_out(script_path)
        write_json(ir_payload, out_path)
        print(f"Built normalized IR for {script_path.name}")
        print(f"IR nodes: {ir_payload['summary']['total_nodes']}, exports: {ir_payload['summary']['export_count']}")
        print(f"Saved IR to {out_path}")
        return

    paths = sorted(MBC_DIR.glob("*.mbc"))
    report = build_corpus_report(paths, workers=args.workers)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out_path = write_json(report, args.out)
        print(f"Wrote corpus IR report to {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
