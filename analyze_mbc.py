from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from mbl_vm_tools.parser import MBCModule, TableRecord
from mbl_vm_tools.tokenizer import Token, tokenize_stream


PROJECT_ROOT = Path(__file__).resolve().parent
MBC_DIR = PROJECT_ROOT / "mbc"
DEFAULT_WORKERS = 16


def _record_to_dict(record: TableRecord) -> dict:
    return {
        "offset": record.offset,
        "name": record.name,
        "a": record.a,
        "b": record.b,
        "c": record.c,
    }


def _token_to_dict(token: Token) -> dict:
    return {
        "offset": token.offset,
        "kind": token.kind,
        "size": token.size,
        "payload": token.payload,
    }


def _analyze_export(mod: MBCModule, name: str) -> dict:
    exact_span = mod.get_export_exact_code_span(name)
    public_span = mod.get_export_public_code_span(name)
    using_exact = exact_span is not None
    start, end = exact_span if using_exact else public_span

    raw = mod.get_export_body(name, exact=True)
    if not raw:
        raw = mod.get_export_body(name, exact=False)

    tokens = tokenize_stream(raw)
    token_kinds = [token.kind for token in tokens]
    return_tail = next((token.payload for token in reversed(tokens) if token.kind == "SIG_RETURN_TAIL"), None)

    return {
        "name": name,
        "slice_mode": "definition_exact" if using_exact else "export_public",
        "span": {"start": start, "end": end},
        "public_span": {"start": public_span[0], "end": public_span[1]},
        "byte_size": len(raw),
        "token_count": len(tokens),
        "token_kinds": token_kinds,
        "structure_signature": " -> ".join(token_kinds),
        "return_tail": return_tail,
    }


def analyze_module(path: str | Path) -> dict:
    path = Path(path)
    mod = MBCModule(path, collect_auxiliary=False)
    exports = [_analyze_export(mod, name) for name in mod.export_names()]

    return {
        "path": str(path),
        "has_magic_header": mod.has_magic_header,
        "code_base": mod.code_base,
        "code_size": mod.code_size,
        "data_blob_size": mod.data_blob_size,
        "definition_count": len(mod.definitions),
        "globals_count": len(mod.globals),
        "exports_count": len(mod.exports),
        "definitions": [_record_to_dict(record) for record in mod.definitions],
        "globals": [_record_to_dict(record) for record in mod.globals],
        "exports": exports,
    }


def summarize_many(modules: list[dict]) -> dict:
    export_count = sum(module.get("exports_count", 0) for module in modules)
    exact_exports = sum(
        1
        for module in modules
        for export in module.get("exports", [])
        if export.get("slice_mode") == "definition_exact"
    )
    token_count = sum(export.get("token_count", 0) for module in modules for export in module.get("exports", []))
    structure_counts = Counter(
        export.get("structure_signature")
        for module in modules
        for export in module.get("exports", [])
        if export.get("structure_signature")
    )

    return {
        "module_count": len(modules),
        "export_count": export_count,
        "exact_export_count": exact_exports,
        "token_count": token_count,
        "top_structures": [
            {"structure_signature": signature, "count": count}
            for signature, count in structure_counts.most_common(16)
        ],
    }


def _analyze_one(path: str) -> dict:
    return analyze_module(path)


def analyze_many_parallel(paths: list[Path]) -> dict:
    if not paths:
        return {"summary": {"module_count": 0}, "modules": []}

    worker_count = min(DEFAULT_WORKERS, len(paths))
    modules: list[dict | None] = [None] * len(paths)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_analyze_one, str(path)): idx
            for idx, path in enumerate(paths)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing scripts", unit="script"):
            idx = future_to_index[future]
            modules[idx] = future.result()

    ready_modules = [module for module in modules if module is not None]
    return { "summary": summarize_many(ready_modules) }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bundled MBL scripts")
    parser.add_argument("--out", help="Write JSON report to file")
    args = parser.parse_args()

    paths = sorted(MBC_DIR.glob("*.mbc"))
    result = analyze_many_parallel(paths)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
