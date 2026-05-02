#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MBC_DIR = PROJECT_ROOT / "mbc"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ir"


# Позволяет запускать скрипт из корня проекта без установки пакета.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from mbl_vm_tools.ir import build_callable_index, build_function_ir, build_module_ir, render_function_text
from mbl_vm_tools.parser import MBCModule


def resolve_mbc_path(script_name: str, mbc_dir: Path) -> Path:
    raw = Path(script_name)

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        if raw.suffix.lower() != ".mbc":
            candidates.append(raw.with_suffix(".mbc"))
    elif raw.parent != Path("."):
        candidates.append((PROJECT_ROOT / raw).resolve())
        if raw.suffix.lower() != ".mbc":
            candidates.append((PROJECT_ROOT / raw.with_suffix(".mbc")).resolve())
    else:
        candidates.append((mbc_dir / raw).resolve())
        if raw.suffix.lower() != ".mbc":
            candidates.append((mbc_dir / f"{raw.name}.mbc").resolve())
        candidates.append((PROJECT_ROOT / raw).resolve())
        if raw.suffix.lower() != ".mbc":
            candidates.append((PROJECT_ROOT / f"{raw.name}.mbc").resolve())

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    checked = "\n  - ".join(str(p) for p in unique_candidates)
    raise FileNotFoundError(f"Не нашёл .mbc файл. Проверял:\n  - {checked}")


def build_text(mbc_path: Path, function_name: str | None) -> str:
    if function_name:
        mod = MBCModule(mbc_path)
        callable_index = build_callable_index(mod)
        fn = build_function_ir(mod, function_name, callable_index=callable_index)
        return render_function_text(fn).rstrip() + "\n"

    module_ir = build_module_ir(mbc_path)
    parts = [render_function_text(fn).rstrip() for fn in module_ir.functions]
    return "\n\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print readable VM IR for an .mbc script and save it to a text file."
    )
    parser.add_argument(
        "script",
        help="Имя .mbc скрипта из папки mbc/ без расширения или с расширением, либо путь к .mbc файлу.",
    )
    parser.add_argument(
        "-f",
        "--function",
        default=None,
        help="Опционально: вывести только одну функцию по имени.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Куда сохранить текстовый IR. По умолчанию: ir/<script>.ir.txt",
    )
    parser.add_argument(
        "--mbc-dir",
        type=Path,
        default=DEFAULT_MBC_DIR,
        help="Папка с .mbc скриптами. По умолчанию: <project_root>/mbc",
    )

    args = parser.parse_args(argv)

    try:
        mbc_path = resolve_mbc_path(args.script, args.mbc_dir)
        text = build_text(mbc_path, args.function)

        out_path = args.out
        if out_path is None:
            suffix = f".{args.function}.ir.txt" if args.function else ".ir.txt"
            out_path = DEFAULT_OUT_DIR / f"{mbc_path.stem}{suffix}"
        elif not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        print(text, end="")
        print(f"\n[saved] {out_path}", file=sys.stderr)
        return 0

    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
