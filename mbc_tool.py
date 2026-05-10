#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from core.cfg import MbcControlFlow
from core.decoder import MbcDecoder
from core.linker import MbcStaticLinker
from core.loader import MbcLoader, MbcProgram, MbcScript
from core.stack_ast import build_program_ast


SCRIPT_DIR = Path(__file__).resolve().parent
MBC_DIR = SCRIPT_DIR / "mbc"


def _resolve_mbc_path(script_name: str) -> Path:
    name = Path(script_name).name
    if not name.lower().endswith(".mbc"):
        name = f"{name}.mbc"
    path = MBC_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"MBC script not found: {path}")
    return path


def _output_path(mbc_path: Path) -> Path:
    return SCRIPT_DIR / f"{mbc_path.stem}.txt"


def _program_header(program: MbcProgram) -> str:
    name = program.name or f"program_{program.index}"
    return f"// === {name} @ 0x{program.start:08X} ==="


def decompile_to_text(script: MbcScript) -> str:
    linker = MbcStaticLinker(script)
    decoder = MbcDecoder(script, linker=linker)
    flow = MbcControlFlow(script, decoder=decoder)

    chunks: list[str] = [
        "// Experimental MBC pseudo-source",
        f"// source: {script.path.name}",
        f"// programs: {len(script.programs)}",
        *linker.summary_lines(),
        "",
    ]

    for program in script.programs:
        chunks.append(_program_header(program))

        if not (0 <= program.start < len(script.code)):
            chunks.append("// warning: program start is outside code section")
            chunks.append("")
            continue
        if program.end < program.start:
            chunks.append("// warning: program end is before start")
            chunks.append("")
            continue

        instructions = flow.decode_program(program)
        ast = build_program_ast(script, program, instructions, linker=linker)
        source = ast.get("source", "")
        chunks.append(source if source else "// no decoded statements")
        chunks.append("")

    return "\n".join(chunks).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decompile an MBC script from ./mbc/ into a .txt file next to mbc_tool.py."
    )
    parser.add_argument("script", help="Script name inside ./mbc/; .mbc extension is optional")
    args = parser.parse_args()

    mbc_path = _resolve_mbc_path(args.script)
    script = MbcLoader.load(mbc_path)
    out_path = _output_path(mbc_path)
    out_path.write_text(decompile_to_text(script), encoding="utf-8")

    print(f"{mbc_path.relative_to(SCRIPT_DIR)} -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
