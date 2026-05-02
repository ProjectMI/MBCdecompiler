#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional


def _fmt_hex(value: Any, width: int = 2) -> str:
    if value is None:
        return "?"
    try:
        return f"0x{int(value):0{width}X}"
    except Exception:
        return str(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def analyze_module(job: tuple[str, Optional[int]]) -> dict[str, Any]:
    """Analyze one .mbc module.

    Kept as a top-level function so ProcessPoolExecutor works on Windows too.
    """
    module_path_s, limit_functions = job
    module_path = Path(module_path_s)

    try:
        from mbl_vm_tools.ir import build_module_ir
        from mbl_vm_tools.facts import build_facts

        module_ir = build_module_ir(module_path, limit_functions=limit_functions)
        facts = build_facts(module_ir)
        facts_payload = facts.to_dict()

        unknown_words: list[dict[str, Any]] = []
        unresolved_script_calls: list[dict[str, Any]] = []

        for fn in module_ir.functions:
            fn_start = _safe_int((fn.span or {}).get("start"), 0)

            for word in fn.words:
                if word.terminal_kind != "UNKNOWN":
                    continue
                byte_value = word.operands.get("byte")
                unknown_words.append(
                    {
                        "function": fn.name,
                        "word_index": word.index,
                        "local_offset": word.offset,
                        "absolute_offset": fn_start + word.offset,
                        "byte": byte_value,
                        "byte_hex": _fmt_hex(byte_value),
                        "raw_hex": word.raw.hex(" "),
                        "decoder_rule": word.decoder_rule,
                    }
                )

            for call in fn.calls:
                if call.get("kind") != "script":
                    continue
                target = call.get("target") or {}
                if target.get("resolved"):
                    continue
                unresolved_script_calls.append(
                    {
                        "caller": fn.name,
                        "word_index": call.get("word_index"),
                        "offset": call.get("offset"),
                        "encoded_argc": call.get("encoded_argc"),
                        "absolute_target": target.get("absolute_target"),
                        "formula": target.get("formula"),
                        "alternatives": target.get("alternatives", []),
                    }
                )

        abi_mismatches = []
        for item in facts_payload.get("abi_mismatches", []):
            abi_mismatches.append(dict(item))

        return {
            "status": "ok",
            "module": str(module_path),
            "summary": {
                "function_count": module_ir.summary.get("function_count", 0),
                "total_word_count": module_ir.summary.get("total_word_count", 0),
                "unknown_vm_word_count": len(unknown_words),
                "unresolved_script_call_count": len(unresolved_script_calls),
                "definition_abi_mismatch_count": len(abi_mismatches),
            },
            "unknown_vm_words": unknown_words,
            "unresolved_script_calls": unresolved_script_calls,
            "definition_abi_mismatches": abi_mismatches,
        }
    except Exception as exc:
        return {
            "status": "error",
            "module": str(module_path),
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=12),
        }


def discover_modules(mbc_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(p for p in mbc_dir.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in mbc_dir.glob(pattern) if p.is_file())
    return files


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    return {
        "modules_total": len(results),
        "modules_ok": len(ok),
        "modules_failed": len(failed),
        "unknown_vm_words": sum(_safe_int(r.get("summary", {}).get("unknown_vm_word_count")) for r in ok),
        "unresolved_script_calls": sum(_safe_int(r.get("summary", {}).get("unresolved_script_call_count")) for r in ok),
        "definition_abi_mismatches": sum(_safe_int(r.get("summary", {}).get("definition_abi_mismatch_count")) for r in ok),
        "functions": sum(_safe_int(r.get("summary", {}).get("function_count")) for r in ok),
        "words": sum(_safe_int(r.get("summary", {}).get("total_word_count")) for r in ok),
    }


def _module_rel(module: str, base: Path) -> str:
    path = Path(module)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def _with_module(results: list[dict[str, Any]], key: str, base: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "ok":
            continue
        module = _module_rel(str(result.get("module")), base)
        for item in result.get(key, []):
            payload = dict(item)
            payload["module"] = module
            out.append(payload)
    return out


def _print_limited(items: list[dict[str, Any]], max_items: int, formatter) -> None:
    if not items:
        print("  none")
        return

    for item in items[:max_items]:
        print(formatter(item))

    rest = len(items) - max_items
    if rest > 0:
        print(f"  ... {rest} more; use --max-items {len(items)} or --json report.json")


def print_text_report(results: list[dict[str, Any]], *, base: Path, max_items: int) -> None:
    summary = build_summary(results)

    print("VM corpus report")
    print("=" * 16)
    print(f"modules: {summary['modules_total']} total, {summary['modules_ok']} ok, {summary['modules_failed']} failed")
    print(f"functions: {summary['functions']}")
    print(f"VM words: {summary['words']}")
    print()
    print(f"unknown VM words: {summary['unknown_vm_words']}")
    print(f"unresolved script calls: {summary['unresolved_script_calls']}")
    print(f"definition ABI mismatches: {summary['definition_abi_mismatches']}")

    unknowns = _with_module(results, "unknown_vm_words", base)
    unresolved = _with_module(results, "unresolved_script_calls", base)
    mismatches = _with_module(results, "definition_abi_mismatches", base)
    failed = [r for r in results if r.get("status") != "ok"]

    print()
    print("unknown VM words:")
    _print_limited(
        unknowns,
        max_items,
        lambda x: (
            f"  {x['module']} :: {x.get('function')} "
            f"word#{x.get('word_index')} local={_fmt_hex(x.get('local_offset'), 4)} "
            f"abs={_fmt_hex(x.get('absolute_offset'), 6)} byte={x.get('byte_hex')} raw={x.get('raw_hex')}"
        ),
    )

    print()
    print("unresolved script calls:")
    _print_limited(
        unresolved,
        max_items,
        lambda x: (
            f"  {x['module']} :: {x.get('caller')} "
            f"word#{x.get('word_index')} off={_fmt_hex(x.get('offset'), 4)} "
            f"argc={x.get('encoded_argc')} target={_fmt_hex(x.get('absolute_target'), 6)}"
        ),
    )

    print()
    print("definition ABI mismatches:")
    _print_limited(
        mismatches,
        max_items,
        lambda x: (
            f"  {x['module']} :: {x.get('caller')} "
            f"off={_fmt_hex(x.get('offset'), 4)} -> {x.get('target')} "
            f"call_argc={x.get('encoded_argc')} def_arity={x.get('target_abi_arity')}"
        ),
    )

    if failed:
        print()
        print("failed modules:")
        for result in failed[:max_items]:
            print(f"  {_module_rel(str(result.get('module')), base)} :: {result.get('error')}")
        rest = len(failed) - max_items
        if rest > 0:
            print(f"  ... {rest} more failed modules")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan .mbc modules and report unknown VM words, unresolved script calls, and definition ABI mismatches."
    )
    parser.add_argument("mbc_dir", nargs="?", type=Path, default=Path("mbc"), help="directory with .mbc modules")
    parser.add_argument("--pattern", default="*.mbc", help="file glob, default: *.mbc")
    parser.add_argument("--no-recursive", action="store_true", help="scan only mbc_dir, not nested directories")
    parser.add_argument("-j", "--workers", type=int, default=8, help="worker count, default: 8")
    parser.add_argument("--limit-functions", type=int, default=None, help="debug option: analyze only first N functions per module")
    parser.add_argument("--max-items", type=int, default=100, help="max detailed rows printed per section")
    parser.add_argument("--json", type=Path, default=None, help="write full machine-readable report to JSON")
    args = parser.parse_args(argv)

    try:
        from tqdm import tqdm
    except ImportError:
        print("Missing dependency: tqdm. Install it with: python -m pip install tqdm", file=sys.stderr)
        return 2

    # Preflight imports. The worker imports them again in child processes.
    try:
        import mbl_vm_tools.ir  # noqa: F401
        import mbl_vm_tools.facts  # noqa: F401
    except Exception as exc:
        print(
            "Cannot import mbl_vm_tools. Run this script from the project root "
            "where mbl_vm_tools/ and mbc/ are siblings.",
            file=sys.stderr,
        )
        print(repr(exc), file=sys.stderr)
        return 2

    mbc_dir = args.mbc_dir
    if not mbc_dir.exists() or not mbc_dir.is_dir():
        print(f"Not a directory: {mbc_dir}", file=sys.stderr)
        return 2

    modules = discover_modules(mbc_dir, args.pattern, recursive=not args.no_recursive)
    if not modules:
        print(f"No files matched {args.pattern!r} in {mbc_dir}", file=sys.stderr)
        return 1

    workers = max(1, int(args.workers))
    jobs = [(str(path), args.limit_functions) for path in modules]

    results: list[dict[str, Any]] = []
    if workers == 1:
        for job in tqdm(jobs, total=len(jobs), unit="file", desc="analyzing"):
            results.append(analyze_module(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_module = {pool.submit(analyze_module, job): job[0] for job in jobs}
            for future in tqdm(as_completed(future_to_module), total=len(future_to_module), unit="file", desc="analyzing"):
                module = future_to_module[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "status": "error",
                            "module": module,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=12),
                        }
                    )

    results.sort(key=lambda r: str(r.get("module", "")))
    payload = {
        "summary": build_summary(results),
        "modules": results,
    }

    print_text_report(results, base=Path.cwd(), max_items=max(0, int(args.max_items)))

    if args.json:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"json report: {args.json}")

    return 0 if payload["summary"]["modules_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
