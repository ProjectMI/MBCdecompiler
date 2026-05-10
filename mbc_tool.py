from __future__ import annotations
import argparse
from pathlib import Path

from core.decompiler import decompile_to_text, load_project_for
from core.linker import MbcProjectLinker
from core.loader import MbcProject


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


def decompile_one(script_name: str) -> Path:
    mbc_path = _resolve_mbc_path(script_name)
    _project, script, project_linker = load_project_for(mbc_path)
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