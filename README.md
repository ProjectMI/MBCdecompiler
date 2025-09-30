# Sphere MBC Disassembler

This repository now focuses on a single task: producing annotated disassembly listings for Sphere `.mbc` script containers.  The command-line interface reads a container plus its `.adb` index file and emits a plain text listing with mnemonics and stack hints sourced from `knowledge/manual_annotations.json`.

## Usage

```
python mbc_disasm.py <adb> <mbc> [--segment INDEX ...] [--max-instr COUNT] [--disasm-out PATH]
```

The tool loads opcode metadata from `knowledge/opcode_profiles.json`.  Manual overrides are merged automatically from `knowledge/manual_annotations.json` and drive the mnemonic and stack effect substitutions in the generated listing.

- `--segment` limits processing to the selected segment indices.
- `--max-instr` truncates each segment after the specified number of instructions.
- `--disasm-out` overrides the default `<mbc>.disasm.txt` output path.
- `--knowledge-base` points to an alternate knowledge database location.

The CLI prints the final output path and writes the listing to disk.

## Tests

Run the test suite with:

```
pytest
```
