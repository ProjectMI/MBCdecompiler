from __future__ import annotations

import argparse
from pathlib import Path
import sys

from mbc_format.loader import MbcLoader
from compile.lossless_ir import (
    assemble_lossless_ir,
    dump_lossless_ir,
    load_lossless_ir,
    render_lossless_view,
    script_to_lossless_ir,
    write_mbc_from_lossless_ir,
)
from compile.writer import write_script


SCRIPT_DIR = Path(__file__).resolve().parent
MBC_DIR = SCRIPT_DIR / "mbc"
IR_DIR = SCRIPT_DIR / "ir_result"
ROUNDTRIP_DIR = SCRIPT_DIR / "roundtrip_result"


def _resolve_mbc_path(script: str | Path) -> Path:
    path = Path(script)
    if path.exists():
        return path
    name = path.name
    if not name.lower().endswith(".mbc"):
        name += ".mbc"
    candidate = MBC_DIR / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"MBC script not found: {script}")


def _iter_targets(script: str | None, all_scripts: bool) -> list[Path]:
    if all_scripts:
        scripts = sorted(MBC_DIR.glob("*.mbc"))
        if not scripts:
            raise FileNotFoundError(f"No .mbc files found in {MBC_DIR}")
        return scripts
    if script is None:
        raise ValueError("script is required unless --all is used")
    return [_resolve_mbc_path(script)]


def cmd_write(args: argparse.Namespace) -> int:
    path = _resolve_mbc_path(args.script)
    script = MbcLoader.load(path)
    out = Path(args.output) if args.output else ROUNDTRIP_DIR / path.name
    write_script(script, out)
    print(f"{path} -> {out}")
    return 0


def cmd_dump_ir(args: argparse.Namespace) -> int:
    path = _resolve_mbc_path(args.script)
    ir = script_to_lossless_ir(MbcLoader.load(path))
    out = Path(args.output) if args.output else IR_DIR / f"{path.stem}.mbcir.json"
    dump_lossless_ir(ir, out)
    if args.view:
        view_path = out.with_suffix(".ir.txt")
        view_path.write_text(render_lossless_view(ir), encoding="utf-8")
        print(f"lossless view -> {view_path}")
    print(f"{path} -> {out}")
    return 0


def cmd_compile_ir(args: argparse.Namespace) -> int:
    ir_path = Path(args.ir)
    ir = load_lossless_ir(ir_path)
    out = Path(args.output) if args.output else ROUNDTRIP_DIR / str(ir.get("source_name", ir_path.stem)).replace(".mbcir.json", ".mbc")
    write_mbc_from_lossless_ir(ir, out)
    print(f"{ir_path} -> {out}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    targets = _iter_targets(args.script, args.all)
    out_dir = Path(args.output_dir) if args.output_dir else None
    ir_dir = Path(args.ir_dir) if args.ir_dir else None
    failures: list[str] = []

    for idx, path in enumerate(targets, start=1):
        script = MbcLoader.load(path)
        ir = script_to_lossless_ir(script, rich=False)
        rebuilt = assemble_lossless_ir(ir)
        original = path.read_bytes()
        ok = rebuilt == original
        status = "OK" if ok else "DIFF"
        print(f"[{idx}/{len(targets)}] {path.name}: {status}", flush=True)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / path.name).write_bytes(rebuilt)
        if ir_dir is not None:
            dump_lossless_ir(ir, ir_dir / f"{path.stem}.mbcir.json")
        if not ok:
            failures.append(path.name)
            if args.stop_on_first_failure:
                break

    if failures:
        print("\nRound-trip failures:", flush=True)
        for name in failures:
            print(f"  {name}", flush=True)
        return 1
    print(f"\nVerified {len(targets)} MBC file(s): byte-identical round-trip.", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lossless MBC compiler/emitter tools: MBC -> lossless IR -> MBC."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write", help="Load and immediately re-emit one MBC file")
    p.add_argument("script")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_write)

    p = sub.add_parser("dump-ir", help="Dump JSON lossless IR for one MBC file")
    p.add_argument("script")
    p.add_argument("-o", "--output")
    p.add_argument("--view", action="store_true", help="Also write a human-readable left-pane IR text file")
    p.set_defaults(func=cmd_dump_ir)

    p = sub.add_parser("compile-ir", help="Assemble JSON lossless IR back into MBC")
    p.add_argument("ir")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_compile_ir)

    p = sub.add_parser("verify", help="Check byte-identical MBC -> IR -> MBC round-trip")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("script", nargs="?", help="Script name/path, unless --all is used")
    group.add_argument("--all", action="store_true", help="Verify every ./mbc/*.mbc")
    p.add_argument("--output-dir", help="Optional directory for rebuilt MBC files")
    p.add_argument("--ir-dir", help="Optional directory for dumped JSON IR files")
    p.add_argument("--stop-on-first-failure", action="store_true")
    p.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
