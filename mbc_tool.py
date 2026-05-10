#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.interpreter import MbcInterpreter, cfg_to_dot
from core.loader import MbcLoader


def _select_programs(cfg: dict[str, Any], selector: str | None) -> dict[str, Any]:
    if not selector:
        return cfg
    wanted = selector.casefold()
    selected = []
    for program in cfg.get("programs", []):
        if str(program.get("index")) == selector or str(program.get("name", "")).casefold() == wanted:
            selected.append(program)
    clone = dict(cfg)
    clone["programs"] = selected
    return clone


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode Sphere MBC bytecode into CFG JSON/DOT and experimental pseudo-AST.")
    parser.add_argument("mbc", type=Path, help="Path to .mbc file")
    parser.add_argument("--json", dest="json_out", type=Path, help="Write full CFG JSON")
    parser.add_argument("--dot", dest="dot_out", type=Path, help="Write Graphviz DOT CFG")
    parser.add_argument("--ast", dest="ast_out", type=Path, help="Write pseudo-AST text")
    parser.add_argument("--tables", dest="tables_out", type=Path, help="Write recovered opcode tables as JSON")
    parser.add_argument("--program", help="Limit output to a program name or numeric index")
    parser.add_argument("--no-ast", action="store_true", help="Skip pseudo-AST construction in JSON")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use legacy nominal start..end decoding instead of reachable CFG decoding",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation; use 0 for compact JSON")
    args = parser.parse_args()

    script = MbcLoader.load(args.mbc)
    interpreter = MbcInterpreter(script, include_ast=not args.no_ast, decode_mode="linear" if args.linear else "reachable")
    cfg = interpreter.build_cfg()
    cfg = _select_programs(cfg, args.program)

    indent = None if args.indent == 0 else args.indent

    if args.tables_out:
        args.tables_out.write_text(json.dumps(interpreter.opcode_tables(), ensure_ascii=False, indent=indent), encoding="utf-8")

    if args.json_out:
        args.json_out.write_text(json.dumps(cfg, ensure_ascii=False, indent=indent), encoding="utf-8")

    if args.dot_out:
        args.dot_out.write_text(cfg_to_dot(cfg), encoding="utf-8")

    if args.ast_out:
        chunks: list[str] = []
        for program in cfg.get("programs", []):
            ast = program.get("ast") or {}
            name = program.get("name") or f"program_{program.get('index')}"
            chunks.append(f"// === {name} @ 0x{program.get('start', 0):08X} ===")
            chunks.append(ast.get("source", ""))
            chunks.append("")
        args.ast_out.write_text("\n".join(chunks), encoding="utf-8")

    if not any([args.json_out, args.dot_out, args.ast_out, args.tables_out]):
        total_programs = len(cfg.get("programs", []))
        total_instructions = sum(len(p.get("instructions", [])) for p in cfg.get("programs", []))
        known = sum(1 for p in cfg.get("programs", []) for ins in p.get("instructions", []) if ins.get("known"))
        print(f"{script.path}: {total_programs} programs, {total_instructions} decoded instructions, {known} known")
        print("Use --json, --dot, --ast, or --tables to write artifacts.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
