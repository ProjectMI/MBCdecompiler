from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mbl_vm_tools.hir import build_module_hir, summarize_corpus, write_json, write_text


PROJECT_ROOT = Path(__file__).resolve().parent
MBC_DIR = PROJECT_ROOT / "mbc"
HIR_OUT_DIR = PROJECT_ROOT / "hir"
DEFAULT_WORKERS = 8



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



def _build_module_hir_one(path: str) -> dict:
    return build_module_hir(path, include_canonical=False, include_text=False)



def build_corpus_report(paths: list[Path], workers: int = DEFAULT_WORKERS) -> dict:
    if not paths:
        return {"summary": {"module_count": 0, "export_count": 0, "total_canonical_instructions": 0}, "heaviest_functions": []}

    worker_count = min(max(1, workers), len(paths))
    module_payloads: list[dict | None] = [None] * len(paths)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_build_module_hir_one, str(path)): idx
            for idx, path in enumerate(paths)
        }
        completed = 0
        total = len(future_to_index)
        failed: list[str] = []
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            path_str = str(paths[idx])
            try:
                module_payloads[idx] = future.result()
            except Exception as exc:  # pragma: no cover - CLI fallback path
                failed.append(f"{Path(path_str).name}: {exc}")
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"HIR corpus: {completed}/{total}")
        if failed:
            print(f"HIR corpus warnings: {len(failed)} module(s) failed")
            for item in failed[:8]:
                print(f"  - {item}")

    ready_modules = [module for module in module_payloads if module is not None]
    return summarize_corpus(ready_modules)



def _default_single_json_out(script_path: Path) -> Path:
    return HIR_OUT_DIR / f"{script_path.stem}.hir.json"



def _default_single_text_out(script_path: Path) -> Path:
    return HIR_OUT_DIR / f"{script_path.stem}.hir.txt"



def _render_module_text(module_payload: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append(f"// {module_payload['script_name']}")
    lines.append(f"// contract: {module_payload['contract']['version']}")
    lines.append("")
    for fn in module_payload.get("functions", []):
        hir_text = fn.get("hir_text", "")
        if hir_text:
            lines.append(hir_text)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build structured HIR for one MBC script or emit a corpus-wide summary."
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Optional script path or name. Example: door1.mbc or door1",
    )
    parser.add_argument("--out", help="Output JSON path. In single-script mode this is the JSON file; in corpus mode this is the report file.")
    parser.add_argument("--text-out", help="Optional text output path for single-script mode. Defaults to hir/<script>.hir.txt")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Max workers for corpus mode (default: {DEFAULT_WORKERS})")
    parser.add_argument("--summary-only", action="store_true", help="In single-script mode omit canonical instructions and text from the JSON payload.")
    args = parser.parse_args()

    if args.script:
        script_path = _resolve_script_path(args.script)
        include_details = not args.summary_only
        payload = build_module_hir(script_path, include_canonical=include_details, include_text=include_details)
        json_out = Path(args.out) if args.out else _default_single_json_out(script_path)
        write_json(payload, json_out)
        if include_details:
            text_out = Path(args.text_out) if args.text_out else _default_single_text_out(script_path)
            write_text(_render_module_text(payload), text_out)
            print(f"Built structured HIR for {script_path.name}")
            print(f"Exports: {payload['summary']['export_count']}, canonical instructions: {payload['summary']['total_canonical_instructions']}")
            print(f"JSON: {json_out}")
            print(f"Text: {text_out}")
        else:
            print(f"Built structured HIR summary for {script_path.name}")
            print(f"JSON: {json_out}")
        return

    paths = sorted(MBC_DIR.glob("*.mbc"))
    report = build_corpus_report(paths, workers=args.workers)
    if args.out:
        out_path = write_json(report, args.out)
        print(f"Wrote HIR corpus report to {out_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
