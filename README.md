# Sphere MBC Disassembler

This repository now focuses on analysing Sphere `.mbc` script containers and lifting them into a structured intermediate representation.  The command-line interface reads a container plus its `.adb` index file and emits both the normalised IR and a higher level AST.

The scripts packed into `.mbc` archives are compiled from an early 2000s
Lua-inspired language.  They target a strictly stack based virtual machine that
uses big-endian 32-bit instruction words composed of two 16-bit halves.  Most
opcodes follow rigid stack discipline: literal loaders and inline ASCII chunk
assemblers increase the height by one, reducers collapse multiple values into a
single result, while control flow instructions preserve or deliberately tear
down the frame.  These invariants form the basis for the pipeline analyser
described below.

## Usage

```
python mbc_disasm.py <adb> <mbc> [--segment INDEX ...] [--ir-out PATH] [--ast-out PATH]
```

The tool loads opcode metadata from `knowledge/opcode_profiles.json`.  Manual overrides are merged automatically from `knowledge/manual_annotations.json` and drive the mnemonic and stack effect substitutions in the generated listing.

- `--segment` limits processing to the selected segment indices.
- `--ir-out` overrides the default `<mbc>.ir.txt` output path.
- `--ast-out` overrides the default `<mbc>.ast.txt` output path.
- `--knowledge-base` points to an alternate knowledge database location.

The CLI prints the final output paths and writes the listings to disk.

## Pipeline analyser

The `mbcdisasm.analyzer` package introduces a structural analysis pass on top of
the plain disassembly.  It scans for stable conveyor forms that repeat all over
Lua-style bytecode: literal → push → test sequences, inline ASCII chunk loading
and reduction, call frame preparation, indirect table lookups and the various
return/terminator combinations.  The analyser combines deterministic stack
tracking with small local opcode templates and the surrounding control-flow
context.  Each recognised pipeline block records its stack delta, dominant
instruction kind and confidence score, providing a solid foundation for higher
level decompilation or refactoring tools.  The command-line disassembler embeds
these results directly in the emitted listing: each segment starts with a
pipeline summary and every block is documented with stack ranges, detected
patterns and heuristic notes before the corresponding instructions.

## Tests

Run the test suite with:

```
pytest
```

## Tools

Check raw remaining:

```
python tools/raw_remaining_report.py
```