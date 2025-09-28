# Sphere MBC Disassembler

This repository provides a research-oriented toolkit for working with Sphere `.mbc` script containers and their companion `.adb` index files. The goal is to offer an extendable pipeline that supports automated opcode discovery, iterative knowledge accumulation, and human-facing disassembly listings.

## Project layout

- `mbcdisasm/` &mdash; Python package containing the core modules.
  - `adb.py` parses the `.adb` segment index.
  - `mbc.py` loads the `.mbc` container and classifies segments.
  - `instruction.py` defines the raw instruction representation.
  - `analysis.py` implements the statistical analyzer used during opcode research.
  - `knowledge.py` maintains the JSON knowledge base for opcode/mode pairs.
  - `disassembler.py` renders annotated disassembly listings using the knowledge base.
  - `auto_analyze.py` runs the analyzer across multiple binaries, merging the findings into the knowledge base.
  - `type_summary_cli.py` exposes a lightweight reporting CLI.
- `mbc_disasm.py` &mdash; main CLI entry point that ties the analyzer and disassembler together.
- `knowledge/` &mdash; persisted knowledge base files (created on demand).
- `tests/` &mdash; sanity tests that exercise the pipeline on bundled fixtures.

All components only depend on the Python standard library.

## Usage

Invoke the main tool by pointing it at the `.adb` index and matching `.mbc` container:

```bash
python mbc_disasm.py <adb> <mbc>
```

Key options:

- `--segment <id ...>` limits the work to selected segment indices.
- `--max-instr <count>` truncates each segment disassembly after the specified number of instructions.
- `--opcode-limit <count>` caps the length of the opcode coverage table shown on stdout.
- `--disasm-out <path>` overrides the default `<mbc name>.disasm.txt` output path.
- `--analysis-out <path>` writes a JSON summary of the current run.
- `--knowledge-base <path>` points to the opcode knowledge base JSON document (default `knowledge/opcode_profiles.json`).
- `--update-knowledge` merges the observed statistics back into the knowledge base and
  automatically refreshes stack delta annotations when the new samples are
  confident.

The tool prints a concise CLI report (segment classifications, detected issues, opcode coverage) and stores the full instruction listing to a file so the console output stays readable.

Runs also emit a "knowledge confidence" table that compares each observed opcode/mode pair with the accumulated statistics from the knowledge base, highlighting stable interpretations versus newly discovered conflicts.

### Quick summaries

To inspect the inferred segment classifications and opcode coverage without generating a full disassembly:

```bash
python -m mbcdisasm.type_summary_cli <adb> <mbc>
```

### Batch analysis

Aggregate opcode statistics from an entire directory and evolve the knowledge base automatically:

```bash
python -m mbcdisasm.auto_analyze mbc/
```

Each processed pair contributes its opcode/mode histograms to the knowledge base, allowing the disassembler to attach better mnemonics and stack hints over time.  Newly
learned stack behaviours are echoed to the console so operators can audit the
automatic deductions.

### Knowledge annotations

The disassembler now renders richer inline comments that summarise the inferred
stack effect, likely control-flow role and dominant operand class for each
instruction.  These hints are derived from the opcode statistics stored in the
knowledge base and can be refined manually by dropping curated entries into
`knowledge/manual_annotations.json`.  When you provide both `operand_hint` and
`operand_confidence` the listing will include the confidence percentage next to
the operand hint so you can gauge how reliable the classification is:

```json
{
  "29:10": {
    "name": "call_indirect",
    "control_flow": "call",
    "summary": "invoke helper routine by table index"
  }
}
```

Manual annotations are merged on load and persisted when the knowledge base is
saved, making it easy to iterate on human-friendly mnemonics.

### JSON summaries

`--analysis-out` reports now include an `opcode_mode_matrix` section that lists
the set of mode values observed for every opcode.  This compact view highlights
how widely an opcode is reused across modes without having to inspect every
opcode/mode histogram individually.

## Tests

Run the tests as

```bash
pytest -s tests/test_reports.py
pytest -s tests/test_stack_gap_inventory.py
pytest -s tests/test_opcode_profiles.py 
```
