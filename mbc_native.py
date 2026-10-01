"""Generate the native client classes from the complete source script corpus."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

from decompile.native_project import recover_project
from decompile.source_optimize import optimize_project
from decompile.native_normalize import normalize_project
from decompile.native_effects import simplify_project
from decompile.source_compact import compact_project
from decompile.native_families import Families
from decompile.native_emit import NativeGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scripts', type=Path, help='Directory containing the complete .mbc corpus')
    parser.add_argument('--client', required=True, type=Path, help='Client root with core/public/script/MbcCommands.h')
    parser.add_argument('--output', type=Path, help='Output directory; otherwise install into the client')
    parser.add_argument('--statistics', type=Path, help='Optional generation statistics JSON')
    options = parser.parse_args()
    paths = sorted(options.scripts.glob('*.mbc'))
    if not paths:
        parser.error('No .mbc scripts in the supplied directory')
    if not (options.client / 'core/public/script/MbcCommands.h').is_file():
        parser.error('Client root is missing core/public/script/MbcCommands.h')
    start = time.monotonic()

    def progress(message: str) -> None:
        print(f'{time.monotonic() - start:7.1f}s  {message}', flush=True)

    modules, public_types = recover_project(paths, progress, strict=True)
    for module in modules:
        module.linker = None
        for function in module.functions:
            function.instructions = []
    gc.collect()
    removed = optimize_project(modules)
    progress(f'Removed {removed} unused or redundant assignments')
    gc.collect()
    normalize_project(modules)
    gc.collect()
    aliases, snapshots = simplify_project(modules)
    gc.collect()
    locals_removed = compact_project(modules)
    progress(f'Removed {aliases} source-slot arguments, {snapshots} snapshots and {locals_removed} local slots')
    families = Families(modules)
    generator = NativeGenerator(families, public_types, options.client)
    # A failed generation must not overwrite a previously usable pair of files.
    with tempfile.TemporaryDirectory(prefix='mbc-native-') as temporary:
        directory = Path(temporary)
        statistics = generator.write(directory)
        for filename in ('GeneratedScripts.h', 'GeneratedScripts.cpp'):
            destination = (options.output / filename if options.output else options.client / 'core' / ('public' if filename.endswith('.h') else 'private') / 'script' / filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / filename, destination)
    if options.statistics:
        options.statistics.parent.mkdir(parents=True, exist_ok=True)
        options.statistics.write_text(json.dumps(statistics, indent=2) + '\n', encoding='utf-8')
    progress(json.dumps(statistics, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'Generation failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error
