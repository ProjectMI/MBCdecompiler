from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mbl_vm_tools.ast import build_module_ast, render_module_ast_text_from_payload, summarize_ast_corpus
from mbl_vm_tools.hir import build_module_hir, write_json, write_text


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



def _build_module_ast_report_one(path: str, include_definitions: bool, include_exports: bool, validate: bool) -> dict:
    return build_module_ast(
        path,
        include_canonical=False,
        include_hir=False,
        include_text=False,
        include_ast=False,
        include_diagnostics=False,
        include_definitions=include_definitions,
        include_exports=include_exports,
        validate=validate,
    )


def build_ast_corpus_report(
    paths: list[Path],
    workers: int = DEFAULT_WORKERS,
    *,
    include_definitions: bool = True,
    include_exports: bool = True,
    validate: bool = False,
) -> dict:
    if not paths:
        return summarize_ast_corpus([])

    worker_count = min(max(1, workers), len(paths))
    module_payloads: list[dict | None] = [None] * len(paths)

    if worker_count == 1:
        failed: list[str] = []
        for idx, path in enumerate(paths):
            try:
                module_payloads[idx] = _build_module_ast_report_one(str(path), include_definitions, include_exports, validate)
            except Exception as exc:  # pragma: no cover - CLI fallback path
                failed.append(f"{path.name}: {exc}")
            completed = idx + 1
            if completed % 25 == 0 or completed == len(paths):
                print(f"AST corpus: {completed}/{len(paths)}")
        if failed:
            print(f"AST corpus warnings: {len(failed)} module(s) failed")
            for item in failed[:8]:
                print(f"  - {item}")
        ready_modules = [module for module in module_payloads if module is not None]
        report = summarize_ast_corpus(ready_modules)
        report["summary"]["failed_module_count"] = len(failed)
        report["failed_modules"] = failed
        return report

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_build_module_ast_report_one, str(path), include_definitions, include_exports, validate): idx
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
                print(f"AST corpus: {completed}/{total}")
        if failed:
            print(f"AST corpus warnings: {len(failed)} module(s) failed")
            for item in failed[:8]:
                print(f"  - {item}")

    ready_modules = [module for module in module_payloads if module is not None]
    report = summarize_ast_corpus(ready_modules)
    report["summary"]["failed_module_count"] = len(failed)
    report["failed_modules"] = failed
    return report


def _default_single_json_out(script_path: Path) -> Path:
    return HIR_OUT_DIR / f"{script_path.stem}.hir.json"



def _default_single_text_out(script_path: Path) -> Path:
    return HIR_OUT_DIR / f"{script_path.stem}.ast.txt"



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one MBC script or emit an AST-oriented corpus report. Optional text output is generated from AST."
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Optional script path or name. Example: door1.mbc or door1",
    )
    parser.add_argument("--out", help="Output JSON path. In single-script mode this is the HIR JSON file; in corpus mode this is the AST report file.")
    parser.add_argument("--text-out", help="Optional AST text output path for single-script mode. Defaults to hir/<script>.ast.txt")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Max workers for corpus mode (default: {DEFAULT_WORKERS})")
    parser.add_argument("--summary-only", action="store_true", help="In single-script mode omit canonical instructions from the JSON payload.")
    parser.add_argument("--validate", action="store_true", help="Run HIR validation in single-script mode.")
    parser.add_argument("--validate-corpus", action="store_true", help="Run pre-AST validation while building the AST corpus report.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--exports-only", action="store_true", help="Analyze only export records. Default analyzes definitions plus export-only records.")
    mode_group.add_argument("--definitions-only", action="store_true", help="Analyze only definition records.")
    args = parser.parse_args()

    include_definitions = not args.exports_only
    include_exports = not args.definitions_only

    if args.script:
        script_path = _resolve_script_path(args.script)
        include_details = not args.summary_only
        payload = build_module_hir(
            script_path,
            include_canonical=include_details,
            include_text=False,
            include_definitions=include_definitions,
            include_exports=include_exports,
            validate=args.validate,
        )
        json_out = Path(args.out) if args.out else _default_single_json_out(script_path)
        write_json(payload, json_out)
        if include_details:
            print(f"Built HIR for {script_path.name}")
            print(
                f"Functions: {payload['summary']['function_count']} "
                f"(definitions: {payload['summary']['definition_function_count']}, "
                f"export-only: {payload['summary']['export_only_function_count']}), "
                f"canonical instructions: {payload['summary']['total_canonical_instructions']}"
            )
            print(
                f"Token coverage: {payload['summary']['known_token_ratio']:.4%} known "
                f"({payload['summary']['total_unknown_tokens']} unknown token(s))"
            )
            print(f"JSON: {json_out}")
            text_out = Path(args.text_out) if args.text_out else _default_single_text_out(script_path)
            ast_payload = build_module_ast(
                script_path,
                include_canonical=False,
                include_hir=False,
                include_text=True,
                include_definitions=include_definitions,
                include_exports=include_exports,
                validate=args.validate,
            )
            write_text(render_module_ast_text_from_payload(ast_payload), text_out)
            print(f"AST text: {text_out}")
        else:
            print(f"Built HIR summary for {script_path.name}")
            print(f"JSON: {json_out}")
        return

    paths = sorted(MBC_DIR.glob("*.mbc"))
    report = build_ast_corpus_report(
        paths,
        workers=args.workers,
        include_definitions=include_definitions,
        include_exports=include_exports,
        validate=args.validate_corpus,
    )
    if args.out:
        out_path = write_json(report, args.out)
        print(f"Wrote AST corpus report to {out_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
