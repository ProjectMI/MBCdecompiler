# Sphere MBC Disassembler

This repository provides a research-oriented toolkit for working with Sphere `.mbc` script containers and their companion `.adb` index files. The goal is to offer an extendable pipeline that supports automated opcode discovery, iterative knowledge accumulation, and human-facing disassembly listings.

## Usage

Invoke the main tool by pointing it at the `.adb` index and matching `.mbc` container:

```bash
python mbc_lua_reconstruct.py <adb> <mbc>
```

Key options:

- --segment SEGMENTS 
- --max-instr MAX_INSTR 
- --knowledge-base KNOWLEDGE_BASE
- --output OUTPUT 
- --keep-duplicate-comments
- --inline-comment-width INLINE_COMMENT_WIDTH 
- --no-stub-metadata 
- --no-enum-metadata
- --no-module-summary 
- --no-literal-report 
- --min-string-length MIN_STRING_LENGTH
- --data-hex-bytes DATA_HEX_BYTES 
- --data-hex-width DATA_HEX_WIDTH 
- --no-data-hex
- --data-histogram DATA_HISTOGRAM 
- --data-run-threshold DATA_RUN_THRESHOLD
- --data-max-runs DATA_MAX_RUNS 
- --string-table
- --string-table-min-occurrences STRING_TABLE_MIN_OCCURRENCES 
- --data-stats
- --emit-data-table 
- --data-table-name DATA_TABLE_NAME 
- --data-table-return
- --literal-report-json LITERAL_REPORT_JSON 
- --analysis-text ANALYSIS_TEXT
- --analysis-json ANALYSIS_JSON 
- --analysis-markdown ANALYSIS_MARKDOWN
- --analysis-csv ANALYSIS_CSV 
- --analysis-helper-csv ANALYSIS_HELPER_CSV
- --analysis-summary 
- --analysis-warning-report ANALYSIS_WARNING_REPORT
- --analysis-warning-stdout

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

## Tests

Run the tests as

```bash
pytest -s tests/test_reports.py
pytest -s tests/test_stack_gap_inventory.py
pytest -s tests/test_opcode_profiles.py 
```
