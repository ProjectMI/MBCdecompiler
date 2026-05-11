from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from core.decompiler import decompile_to_text, load_project_for
    from core.linker import MbcProjectLinker
    from core.loader import MbcProject
except ModuleNotFoundError:  # packaged layout used by mbc_decompiler_*.zip
    from mbcproj.decompiler import decompile_to_text, load_project_for
    from mbcproj.linker import MbcProjectLinker
    from mbcproj.loader import MbcProject


SCRIPT_DIR = Path(__file__).resolve().parent
MBC_DIR = SCRIPT_DIR / "mbc"
OUT_DIR = SCRIPT_DIR / "decompile_result"

_WORKER_PROJECT: MbcProject | None = None
_WORKER_LINKER: MbcProjectLinker | None = None
_WORKER_SCRIPTS_BY_NAME: dict[str, object] = {}
_WORKER_OUT_DIR: Path | None = None


def _make_project_linker(project: MbcProject) -> MbcProjectLinker:
    """Build the same project linker used by the decompiler package.

    Newer package versions expose ``from_ffprc_plan``; older local ``core``
    layouts may only support the plain constructor.  Keep both paths so this
    script remains drop-in compatible.
    """
    factory = getattr(MbcProjectLinker, "from_ffprc_plan", None)
    if callable(factory):
        return factory(project.scripts)
    return MbcProjectLinker(project.scripts)


def _resolve_mbc_path(script_name: str) -> Path:
    name = Path(script_name).name
    if not name.lower().endswith(".mbc"):
        name = f"{name}.mbc"
    path = MBC_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"MBC script not found: {path}")
    return path


def _output_path(mbc_path: Path) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{mbc_path.stem}.txt"


def decompile_one(script_name: str) -> Path:
    mbc_path = _resolve_mbc_path(script_name)
    _project, script, project_linker = load_project_for(mbc_path)
    out_path = _output_path(mbc_path)
    out_path.write_text(decompile_to_text(script, project_linker=project_linker), encoding="utf-8")
    return out_path


def _init_all_worker(mbc_dir: str, out_dir: str) -> None:
    """Initializer for ProcessPoolExecutor.

    Each process loads the whole project once, then reuses the linker for all
    scripts assigned to that process.  This avoids pickling linker/script
    objects and keeps the implementation safe for Windows spawn mode.
    """
    global _WORKER_PROJECT, _WORKER_LINKER, _WORKER_SCRIPTS_BY_NAME, _WORKER_OUT_DIR

    project = MbcProject.load_dir(Path(mbc_dir))
    _WORKER_PROJECT = project
    _WORKER_LINKER = _make_project_linker(project)
    _WORKER_SCRIPTS_BY_NAME = {script.path.name.lower(): script for script in project.scripts}
    _WORKER_OUT_DIR = Path(out_dir)
    _WORKER_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _decompile_all_worker(script_filename: str) -> tuple[str, str]:
    if _WORKER_LINKER is None or _WORKER_OUT_DIR is None:
        raise RuntimeError("worker was not initialized")

    script = _WORKER_SCRIPTS_BY_NAME.get(script_filename.lower())
    if script is None:
        raise FileNotFoundError(f"MBC script not found in worker project: {script_filename}")

    out_path = _WORKER_OUT_DIR / f"{script.path.stem}.txt"
    out_path.write_text(decompile_to_text(script, project_linker=_WORKER_LINKER), encoding="utf-8")
    return script_filename, out_path.name


def decompile_all(*, jobs: int | None = None) -> int:
    if not MBC_DIR.exists():
        raise FileNotFoundError(f"MBC directory not found: {MBC_DIR}")

    scripts = sorted(path.name for path in MBC_DIR.glob("*.mbc"))
    if not scripts:
        print(f"No .mbc files found in {MBC_DIR}")
        return 0

    max_workers = jobs or (os.cpu_count() or 1)
    max_workers = max(1, min(max_workers, len(scripts)))

    if max_workers == 1:
        # Serial fallback is useful for debugging and avoids multiprocessing
        # startup cost for tiny corpora.
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        project = MbcProject.load_dir(MBC_DIR)
        project_linker = _make_project_linker(project)
        for script in project.scripts:
            out_path = _output_path(script.path)
            out_path.write_text(decompile_to_text(script, project_linker=project_linker), encoding="utf-8")
            print(f"{script.path.relative_to(SCRIPT_DIR)} -> {OUT_DIR.name}/{out_path.name}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Decompiling {len(scripts)} scripts with {max_workers} worker processes into {OUT_DIR.name}/...")

    failed: list[tuple[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_all_worker,
        initargs=(str(MBC_DIR), str(OUT_DIR)),
    ) as executor:
        futures = {executor.submit(_decompile_all_worker, name): name for name in scripts}
        done = 0
        for future in as_completed(futures):
            name = futures[future]
            try:
                script_filename, out_name = future.result()
            except Exception as exc:  # keep processing other scripts
                failed.append((name, str(exc)))
                print(f"mbc/{name} -> ERROR: {exc}")
                continue
            done += 1
            print(f"mbc/{script_filename} -> {OUT_DIR.name}/{out_name} [{done}/{len(scripts)}]")

    if failed:
        print("\nFailed scripts:")
        for name, error in failed:
            print(f"  mbc/{name}: {error}")
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decompile MBC scripts from ./mbc/. Single-script mode loads the "
            "whole mbc/ directory for virtual import linking. --all processes "
            "scripts in parallel by default. Output .txt files are written to ./decompile_result/."
        )
    )
    parser.add_argument("script", nargs="?", help="Script name inside ./mbc/; .mbc extension is optional")
    parser.add_argument("--all", action="store_true", help="Decompile every .mbc file in ./mbc/ into ./decompile_result/")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of worker processes for --all; defaults to all CPU cores. Use 1 for serial mode.",
    )
    args = parser.parse_args()

    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be >= 1")

    if args.all:
        return decompile_all(jobs=args.jobs)

    if not args.script:
        parser.error("script is required unless --all is used")

    out_path = decompile_one(args.script)
    print(f"mbc/{Path(args.script).stem}.mbc -> {OUT_DIR.name}/{out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
