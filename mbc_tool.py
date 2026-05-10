#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from core.cfg import MbcControlFlow
from core.decoder import MbcDecoder
from core.linker import MbcProjectLinker, MbcStaticLinker
from core.loader import MbcLoader, MbcProgram, MbcProject, MbcScript
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


def _program_header(program: MbcProgram, linker: MbcStaticLinker) -> str:
    symbol = linker.internal_at(program.start) or linker.symbol_at(program.start)
    name = program.name or (symbol.name if symbol else f"program_{program.index}")
    signature = symbol.signature.render(include_storage=True) if symbol is not None else "()"
    return f"// === {name}{signature} @ 0x{program.start:08X} ==="


def _load_project_for(mbc_path: Path) -> tuple[MbcProject, MbcScript, MbcProjectLinker]:
    project = MbcProject.load_for_script(mbc_path)
    by_module = project.by_module
    script = by_module.get(mbc_path.stem)
    if script is None:
        # Fallback for unusual paths outside the default mbc/ tree.
        script = MbcLoader.load(mbc_path)
        project = MbcProject(root=mbc_path.parent, scripts=[script])
    project_linker = MbcProjectLinker(project.scripts)
    return project, script, project_linker


def decompile_to_text(script: MbcScript, *, project_linker: MbcProjectLinker | None = None) -> str:
    linker = (project_linker.module(script.path.stem) if project_linker is not None else None) or MbcStaticLinker(script)
    decoder = MbcDecoder(script, linker=linker)
    flow = MbcControlFlow(script, decoder=decoder)

    chunks: list[str] = [
        "// Experimental MBC pseudo-source",
        f"// source: {script.path.name}",
        f"// programs: {len(script.programs)}",
        "// data naming: argN = program_prologue binding; global_XXXX/global_span_XXXX = current process data-section slot",
        "// synthetic call returns are now kept as expressions, not ret_* placeholder variables",
    ]
    if project_linker is not None:
        chunks.extend(project_linker.summary_lines())
    chunks.extend([*linker.summary_lines(), ""])

    for program in script.programs:
        chunks.append(_program_header(program, linker))

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


def decompile_one(script_name: str) -> Path:
    mbc_path = _resolve_mbc_path(script_name)
    _project, script, project_linker = _load_project_for(mbc_path)
    out_path = _output_path(mbc_path)
    out_path.write_text(decompile_to_text(script, project_linker=project_linker), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decompile MBC scripts from ./mbc/. The tool always loads the whole "
            "mbc/ directory first, so import stubs are virtually linked against "
            "neighbouring scripts before IR generation."
        )
    )
    parser.add_argument("script", nargs="?", help="Script name inside ./mbc/; .mbc extension is optional")
    parser.add_argument("--all", action="store_true", help="Decompile every .mbc file in ./mbc/ into .txt files next to mbc_tool.py")
    args = parser.parse_args()

    if args.all:
        project = MbcProject.load_dir(MBC_DIR)
        project_linker = MbcProjectLinker(project.scripts)
        for script in project.scripts:
            out_path = _output_path(script.path)
            out_path.write_text(decompile_to_text(script, project_linker=project_linker), encoding="utf-8")
            print(f"{script.path.relative_to(SCRIPT_DIR)} -> {out_path.name}")
        return 0

    if not args.script:
        parser.error("script is required unless --all is used")

    out_path = decompile_one(args.script)
    print(f"mbc/{Path(args.script).stem}.mbc -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
