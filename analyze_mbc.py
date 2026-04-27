from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mbl_vm_tools.ast import build_module_ast, render_module_ast_text_from_payload, summarize_ast_corpus
from mbl_vm_tools.hir import build_module_hir, write_json, write_text


PROJECT_ROOT = Path(__file__).resolve().parent
MBC_DIR = PROJECT_ROOT / "mbc"
HIR_OUT_DIR = PROJECT_ROOT / "hir"
DEFAULT_WORKERS = 8
CORPUS_RESUME_VERSION = "ast-corpus-resume-v1"



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



def _atomic_write_json(payload: Any, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def _resume_sidecar_path(base_path: Path, suffix: str) -> Path:
    if base_path.suffix:
        return base_path.with_suffix(suffix)
    return base_path.with_name(f"{base_path.name}{suffix}")


def _default_corpus_resume_paths(out_arg: str | None) -> tuple[Path, Path]:
    if out_arg:
        out_path = Path(out_arg)
        return _resume_sidecar_path(out_path, ".pending.json"), _resume_sidecar_path(out_path, ".modules")
    return HIR_OUT_DIR / "ast-report.pending.json", HIR_OUT_DIR / "ast-report.modules"


def _implementation_stamp() -> dict[str, int]:
    candidates = [Path(__file__), *sorted((PROJECT_ROOT / "mbl_vm_tools").glob("*.py"))]
    stamp: dict[str, int] = {}
    for path in candidates:
        try:
            key = str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
            stamp[key] = path.stat().st_mtime_ns
        except OSError:
            continue
    return stamp


def _corpus_run_signature(*, include_definitions: bool, include_exports: bool, validate: bool) -> dict[str, Any]:
    return {
        "include_definitions": include_definitions,
        "include_exports": include_exports,
        "validate": validate,
        "implementation": _implementation_stamp(),
    }


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _ordered_unique_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        clean = Path(str(name)).name
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _module_cache_path(cache_dir: Path, script_name: str) -> Path:
    return cache_dir / f"{Path(script_name).name}.ast-module.json"


def _write_module_cache(payload: dict[str, Any], cache_dir: Path, script_name: str) -> Path:
    return _atomic_write_json(payload, _module_cache_path(cache_dir, script_name))


def _normalize_failed_modules(raw_failed: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(raw_failed, list):
        return normalized

    for item in raw_failed:
        if isinstance(item, dict):
            script_name = Path(str(item.get("script_name") or item.get("name") or "")).name
            error = str(item.get("error") or "")
        else:
            text = str(item)
            head, sep, tail = text.partition(":")
            script_name = Path(head).name
            error = tail.strip() if sep else ""
        if script_name:
            normalized.append({"script_name": script_name, "error": error})

    deduped: dict[str, dict[str, str]] = {}
    for item in normalized:
        deduped[item["script_name"]] = item
    return [deduped[name] for name in sorted(deduped)]


def _failed_module_strings(failed: list[dict[str, str]]) -> list[str]:
    return [
        f"{item['script_name']}: {item.get('error') or 'failed'}"
        for item in sorted(failed, key=lambda row: row["script_name"])
    ]


def _make_corpus_resume_state(paths: list[Path], signature: dict[str, Any]) -> dict[str, Any]:
    script_names = [path.name for path in paths]
    return {
        "version": CORPUS_RESUME_VERSION,
        "signature": signature,
        "planned": script_names,
        "pending": list(script_names),
        "failed": [],
    }


def _reset_corpus_resume_files(state_path: Path, cache_dir: Path) -> None:
    if state_path.exists():
        state_path.unlink()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _prepare_corpus_resume_state(
    paths: list[Path],
    *,
    state_path: Path,
    cache_dir: Path,
    signature: dict[str, Any],
) -> dict[str, Any]:
    script_names = [path.name for path in paths]
    script_name_set = set(script_names)
    raw_state = _read_json_file(state_path)
    state_is_compatible = (
        isinstance(raw_state, dict)
        and raw_state.get("version") == CORPUS_RESUME_VERSION
        and raw_state.get("signature") == signature
    )

    if not state_is_compatible:
        if state_path.exists() or cache_dir.exists():
            print(f"AST corpus resume: state/cache reset because options or implementation changed: {state_path}")
        _reset_corpus_resume_files(state_path, cache_dir)
        state = _make_corpus_resume_state(paths, signature)
        _atomic_write_json(state, state_path)
        return state

    if not _ordered_unique_names(list(raw_state.get("pending") or [])):
        print(f"AST corpus resume: previous corpus report is complete; starting a fresh run: {state_path}")
        _reset_corpus_resume_files(state_path, cache_dir)
        state = _make_corpus_resume_state(paths, signature)
        _atomic_write_json(state, state_path)
        return state

    state = dict(raw_state)
    failed = [
        item
        for item in _normalize_failed_modules(state.get("failed"))
        if item["script_name"] in script_name_set
    ]
    failed_names = {item["script_name"] for item in failed}

    pending = [
        Path(str(name)).name
        for name in state.get("pending") or []
        if Path(str(name)).name in script_name_set and Path(str(name)).name not in failed_names
    ]

    cached_names = {
        path.name
        for path in paths
        if _module_cache_path(cache_dir, path.name).is_file()
    }
    pending = [name for name in pending if name not in cached_names]
    missing_names = [
        name
        for name in script_names
        if name not in cached_names and name not in failed_names and name not in pending
    ]

    state["planned"] = script_names
    state["pending"] = _ordered_unique_names([*pending, *missing_names])
    state["failed"] = failed
    _atomic_write_json(state, state_path)
    return state


def _mark_corpus_module_finished(
    state: dict[str, Any],
    state_path: Path,
    script_name: str,
    *,
    error: str | None = None,
) -> None:
    clean_name = Path(script_name).name
    state["pending"] = [name for name in state.get("pending", []) if Path(str(name)).name != clean_name]

    if error is not None:
        failed = [
            item
            for item in _normalize_failed_modules(state.get("failed"))
            if item["script_name"] != clean_name
        ]
        failed.append({"script_name": clean_name, "error": error})
        state["failed"] = failed

    _atomic_write_json(state, state_path)


def _load_cached_module_payloads(paths: list[Path], cache_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    modules: list[dict[str, Any]] = []
    missing: list[str] = []

    for path in paths:
        cache_path = _module_cache_path(cache_dir, path.name)
        payload = _read_json_file(cache_path)
        if not isinstance(payload, dict) or payload.get("script_name") != path.name:
            missing.append(path.name)
            continue
        modules.append(payload)

    return modules, missing


def _process_corpus_pending_modules(
    paths_by_name: dict[str, Path],
    state: dict[str, Any],
    *,
    state_path: Path,
    cache_dir: Path,
    workers: int,
    max_modules: int | None,
    include_definitions: bool,
    include_exports: bool,
    validate: bool,
) -> None:
    pending_names = [
        name
        for name in _ordered_unique_names(list(state.get("pending") or []))
        if name in paths_by_name
    ]

    if max_modules is not None:
        if max_modules <= 0:
            return
        pending_names = pending_names[:max_modules]

    if not pending_names:
        return

    worker_count = min(max(1, workers), len(pending_names))
    print(
        f"AST corpus resume: processing {len(pending_names)} module(s), "
        f"{len(state.get('pending') or [])} pending before run, workers={worker_count}"
    )

    def handle_success(script_name: str, payload: dict[str, Any]) -> None:
        _write_module_cache(payload, cache_dir, script_name)
        _mark_corpus_module_finished(state, state_path, script_name)

    def handle_failure(script_name: str, exc: Exception) -> None:
        _mark_corpus_module_finished(state, state_path, script_name, error=str(exc))

    completed_this_run = 0
    total_this_run = len(pending_names)

    def print_progress() -> None:
        nonlocal completed_this_run
        completed_this_run += 1
        if completed_this_run % 10 == 0 or completed_this_run == total_this_run:
            print(
                f"AST corpus resume: {completed_this_run}/{total_this_run} this run, "
                f"{len(state.get('pending') or [])} pending"
            )

    if worker_count == 1:
        for script_name in pending_names:
            path = paths_by_name[script_name]
            try:
                payload = _build_module_ast_report_one(str(path), include_definitions, include_exports, validate)
                handle_success(script_name, payload)
            except Exception as exc:  # pragma: no cover - CLI fallback path
                handle_failure(script_name, exc)
            print_progress()
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_script = {
            executor.submit(
                _build_module_ast_report_one,
                str(paths_by_name[script_name]),
                include_definitions,
                include_exports,
                validate,
            ): script_name
            for script_name in pending_names
        }

        for future in as_completed(future_to_script):
            script_name = future_to_script[future]
            try:
                payload = future.result()
                handle_success(script_name, payload)
            except Exception as exc:  # pragma: no cover - CLI fallback path
                handle_failure(script_name, exc)
            print_progress()


def build_ast_corpus_report_resumable(
    paths: list[Path],
    workers: int = DEFAULT_WORKERS,
    *,
    state_path: Path,
    cache_dir: Path,
    include_definitions: bool = True,
    include_exports: bool = True,
    validate: bool = False,
    max_modules: int | None = None,
) -> dict:
    if not paths:
        return summarize_ast_corpus([])

    signature = _corpus_run_signature(
        include_definitions=include_definitions,
        include_exports=include_exports,
        validate=validate,
    )
    state = _prepare_corpus_resume_state(
        paths,
        state_path=state_path,
        cache_dir=cache_dir,
        signature=signature,
    )

    paths_by_name = {path.name: path for path in paths}
    _process_corpus_pending_modules(
        paths_by_name,
        state,
        state_path=state_path,
        cache_dir=cache_dir,
        workers=workers,
        max_modules=max_modules,
        include_definitions=include_definitions,
        include_exports=include_exports,
        validate=validate,
    )

    modules, missing_modules = _load_cached_module_payloads(paths, cache_dir)
    failed = _normalize_failed_modules(state.get("failed"))
    failed_names = {item["script_name"] for item in failed}

    retry_missing = [
        name
        for name in missing_modules
        if name not in failed_names and name in paths_by_name
    ]
    if retry_missing:
        state["pending"] = _ordered_unique_names([*(state.get("pending") or []), *retry_missing])
        _atomic_write_json(state, state_path)

    pending = [
        name
        for name in _ordered_unique_names(list(state.get("pending") or []))
        if name in paths_by_name
    ]
    report = summarize_ast_corpus(modules)
    failed_strings = _failed_module_strings(failed)

    report["summary"]["planned_module_count"] = len(paths)
    report["summary"]["completed_module_count"] = len(modules)
    report["summary"]["pending_module_count"] = len(pending)
    report["summary"]["failed_module_count"] = len(failed_strings)
    report["failed_modules"] = failed_strings
    report["resume"] = {
        "version": CORPUS_RESUME_VERSION,
        "complete": len(pending) == 0,
        "state_path": str(state_path),
        "module_cache_dir": str(cache_dir),
        "pending_module_count": len(pending),
        "pending_modules": pending[:64],
    }
    return report


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
    parser.add_argument("--corpus-state", help="Persistent corpus pending-list JSON path. Defaults next to --out, or hir/ast-report.pending.json.")
    parser.add_argument("--max-corpus-modules", type=int, help="Process at most N pending corpus modules in this run, then write a partial report.")
    parser.add_argument("--no-corpus-resume", action="store_true", help="Disable the persistent corpus queue and use the old one-shot in-memory mode.")
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
    if args.no_corpus_resume:
        report = build_ast_corpus_report(
            paths,
            workers=args.workers,
            include_definitions=include_definitions,
            include_exports=include_exports,
            validate=args.validate_corpus,
        )
    else:
        state_path, cache_dir = _default_corpus_resume_paths(args.out)
        if args.corpus_state:
            state_path = Path(args.corpus_state)
            cache_dir = _resume_sidecar_path(state_path, ".modules")
        report = build_ast_corpus_report_resumable(
            paths,
            workers=args.workers,
            state_path=state_path,
            cache_dir=cache_dir,
            include_definitions=include_definitions,
            include_exports=include_exports,
            validate=args.validate_corpus,
            max_modules=args.max_corpus_modules,
        )

    if args.out:
        out_path = write_json(report, args.out)
        pending = int(report.get("summary", {}).get("pending_module_count", 0) or 0)
        if pending:
            print(f"Wrote partial AST corpus report to {out_path} ({pending} module(s) pending; rerun the same command to continue)")
        else:
            print(f"Wrote AST corpus report to {out_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
