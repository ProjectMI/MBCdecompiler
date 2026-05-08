from __future__ import annotations

import argparse
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Optional

from mbl_vm_tools.branch_model import (
    BASE_MODELS,
    BRANCH_SEMANTICS_CHOICES,
    BRANCH_SEMANTICS_CONSERVATIVE,
    BRANCH_SEMANTICS_STRICT,
    CFGEdge,
    FunctionCFG,
    branch_base_audit_for_module,
    branch_op_name,
    build_cfg_for_function,
    classify_target_landing,
    compare_4a_semantics_for_module,
    normalize_branch_semantics,
    validate_cfg,
)
from mbl_vm_tools.parser import FunctionEntry, MBCModule
from mbl_vm_tools.vm_spec import VMWord, branch_target_offset, decode_words, is_control_branch_word, signed_u16, word_role


STRUCTURAL_TERMINALS = {"MARK", "NOP"}
IMPORTANT_ROLES = {"branch", "call", "return", "unknown", "predicate_no_transfer"}


def resolve_script_path(script: str | Path, *, root: Optional[Path] = None) -> Path:
    """Resolve a script name against the project layout.

    Intended use from project root:
        python -m mbl_vm_tools.dump_branch_flow _main

    Search order keeps the user's requested spelling first, then tries the
    conventional ``mbc/`` directory. Names may be passed with or without the
    ``.mbc`` suffix.
    """

    root = Path.cwd() if root is None else Path(root)
    raw = Path(script)
    names: list[Path] = []
    stem_candidates = [raw]
    if raw.suffix.lower() != ".mbc":
        stem_candidates.append(Path(str(raw) + ".mbc"))
        if not raw.name.startswith("_"):
            stem_candidates.append(raw.with_name("_" + raw.name))
            stem_candidates.append(raw.with_name("_" + raw.name + ".mbc"))

    package_root = Path(__file__).resolve().parents[1]
    if raw.is_absolute() or raw.parent != Path("."):
        search_roots = [Path(""), root, root / "mbc", package_root / "mbc"]
    else:
        # For a bare script name, prefer the project layout described by the
        # corpus: scripts live in mbc/, tools live in mbl_vm_tools/.
        search_roots = [root / "mbc", root, package_root / "mbc", Path("")]
    for name in stem_candidates:
        if name.is_absolute():
            names.append(name)
        else:
            for base in search_roots:
                candidate = base / name
                if candidate not in names:
                    names.append(candidate)

    for candidate in names:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(p) for p in names)
    raise FileNotFoundError(f"MBC script not found: {script}\nTried:\n  {tried}")


def default_output_path(script_path: Path, *, root: Optional[Path] = None) -> Path:
    root = Path.cwd() if root is None else Path(root)
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{script_path.stem}.branch_flow.txt"


def hx(value: Optional[int], width: int = 4) -> str:
    if value is None:
        return "-"
    return f"0x{int(value):0{width}X}"


def signed(value: int) -> str:
    return f"{int(value):+d}"


def _compact_value(value: Any) -> str:
    if isinstance(value, int):
        if abs(value) > 9999:
            return f"{value}"
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return f"[{len(value)}]"
    if isinstance(value, dict):
        return "{...}"
    return str(value)


def summarize_word(word: VMWord, *, include_raw: bool = False) -> str:
    term = word.terminal_kind
    kind = word.kind
    operands = word.operands
    parts = [kind]

    if term == "BR":
        op = int(operands.get("op", -1) or -1) & 0xFF
        off = signed_u16(int(operands.get("off", 0) or 0))
        parts.append(f"{branch_op_name(op)} off={signed(off)}")
    elif term == "CALL_NATIVE":
        parts.append(f"argc={operands.get('argc')} opid={operands.get('opid')}")
    elif term == "CALL_SCRIPT":
        parts.append(f"argc={operands.get('argc')} rel={signed(int(operands.get('rel', 0) or 0))}")
    elif term in {"REF", "REF16"}:
        mode = operands.get("mode")
        mode_s = f"0x{int(mode):02X}" if isinstance(mode, int) else str(mode)
        parts.append(f"mode={mode_s} ref={operands.get('ref')}")
    elif term == "REC41":
        parts.append(f"ref={operands.get('ref')} imm={operands.get('imm')}")
    elif term == "REC61":
        parts.append(
            f"mode={operands.get('mode')} u16={operands.get('u16')} "
            f"a={operands.get('a')} b={operands.get('b')} c={operands.get('c')}"
        )
    elif term == "REC62":
        parts.append(f"mode={operands.get('mode')} u16={operands.get('u16')} c={operands.get('c')}")
    elif term in {"IMM8", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32", "U16"}:
        if "value" in operands:
            parts.append(f"value={operands.get('value')}")
        if "op" in operands:
            parts.append(f"op=0x{int(operands.get('op')):02X}")
    elif term == "F32":
        parts.append(f"value={operands.get('value')}")
    elif term in {"AGG", "AGG0"}:
        parts.append(f"arity={operands.get('arity')} raw_arity={operands.get('raw_arity')}")
    elif term == "CODE_REF":
        parts.append(f"rel={signed(int(operands.get('rel', 0) or 0))}")
    elif term == "BARE_U32":
        parts.append(f"value={operands.get('value')} follower={operands.get('follower_kind')}")
    elif term == "UNKNOWN":
        parts.append(f"byte=0x{int(operands.get('byte', 0) or 0):02X}")
    elif term in {"RETURN_PAIR", "END"}:
        parts.append("return")

    if include_raw:
        parts.append(f"raw={word.raw.hex(' ')}")
    return " ".join(str(p) for p in parts if p)


def _edges_by_source(edges: Iterable[CFGEdge]) -> dict[int, list[CFGEdge]]:
    grouped: dict[int, list[CFGEdge]] = defaultdict(list)
    for edge in edges:
        grouped[int(edge.source)].append(edge)
    return grouped


def _edge_summary(edge: CFGEdge, body_len: int) -> str:
    if edge.target is None:
        return f"{edge.kind}->-"
    target = hx(edge.target, max(4, len(f"{body_len:X}")))
    suffix = "" if edge.valid else f" !{edge.note or 'invalid'}"
    return f"{edge.kind}->{target}{suffix}"


def _target_kind_for_node(body: bytes, word: VMWord, linear_words: list[VMWord]) -> str:
    if word.terminal_kind != "BR" or not is_control_branch_word(word):
        return ""
    landing = classify_target_landing(body, branch_target_offset(word), linear_words)
    if landing.at_function_end:
        return " target=function_end"
    if not landing.in_range:
        return " target=OUT_OF_RANGE"
    if landing.linear_boundary:
        return " target=linear"
    if landing.inside_linear_word:
        return f" target={landing.subentry_kind}@{hx(landing.inside_offset)}"
    return f" target={landing.subentry_kind}"


def should_show_node(word: VMWord, mode: str) -> bool:
    role = word_role(word)
    if mode == "full":
        return True
    if mode == "branches":
        return role in IMPORTANT_ROLES
    if mode == "flow":
        if word.terminal_kind in STRUCTURAL_TERMINALS and role == "structural":
            return False
        return True
    raise ValueError(f"unknown mode: {mode}")


def render_cfg_stream(body: bytes, cfg: FunctionCFG, *, mode: str = "flow") -> list[str]:
    linear_words = decode_words(body)
    linear_offsets = {int(w.offset) for w in linear_words}
    grouped_edges = _edges_by_source(cfg.edges)
    width = max(4, len(f"{max(0, len(body)):X}"))
    lines: list[str] = []
    skipped = 0

    for offset, node in sorted(cfg.nodes.items()):
        word = node.word
        if not should_show_node(word, mode):
            skipped += 1
            continue
        if skipped:
            lines.append(f"    ... {skipped} structural MARK/NOP entries omitted")
            skipped = 0
        sub = "!" if offset not in linear_offsets else " "
        role = word_role(word)
        edge_text = ""
        if offset in grouped_edges:
            interesting = [edge for edge in grouped_edges[offset] if edge.kind != "fallthrough"]
            if interesting:
                edge_text = " ; " + ", ".join(_edge_summary(edge, len(body)) for edge in interesting)
            elif mode == "full":
                edge_text = " ; " + ", ".join(_edge_summary(edge, len(body)) for edge in grouped_edges[offset])
        target_note = _target_kind_for_node(body, word, linear_words)
        raw = word.terminal_kind in {"UNKNOWN", "BR"}
        lines.append(
            f"  {sub}{hx(offset, width)} +{word.size:<2} {role:<21} "
            f"{summarize_word(word, include_raw=raw)}{target_note}{edge_text}"
        )
    if skipped:
        lines.append(f"    ... {skipped} structural MARK/NOP entries omitted")
    return lines


def _fmt_runs(runs: list[tuple[int, int]], *, width: int) -> str:
    if not runs:
        return "-"
    return ", ".join(f"{hx(a, width)}..{hx(b, width)}" for a, b in runs)


def render_function_report(module: MBCModule, entry: FunctionEntry, *, mode: str = "flow", branch_semantics: str = BRANCH_SEMANTICS_CONSERVATIVE, strict_4a_audit: Optional[dict[str, Any]] = None) -> str:
    body = module.get_function_body(entry)
    branch_semantics = normalize_branch_semantics(branch_semantics)
    cfg = build_cfg_for_function(module, entry, branch_semantics=branch_semantics)
    validation = validate_cfg(body, cfg)
    width = max(4, len(f"{max(0, len(body)):X}"))
    span = cfg.span
    span_s = f"{hx(span[0])}..{hx(span[1])}" if span else "-"
    flags = validation["issue_flags"]
    status = "OK" if not flags else "WARN:" + ",".join(flags)
    exported = " exported" if entry.is_exported else ""

    lines = [
        f"FUNC {entry.name} [{entry.source_kind}{exported}] span={span_s} len={len(body)} status={status}",
        (
            "  cfg: "
            f"semantics={branch_semantics} "
            f"nodes={validation['cfg_nodes']} edges={validation['cfg_edges']} "
            f"linear_words={validation['linear_words']} "
            f"branches={validation['cfg_control_branches']} "
            f"predicate_no_transfer={validation['cfg_predicate_no_transfer']} "
            f"unknown={validation['reachable_unknown_words']}"
        ),
        (
            "  branch targets: "
            f"oob={validation['branch_targets_out_of_range']} "
            f"unknown={validation['branch_targets_unknown']} "
            f"linear={validation['branch_targets_linear_boundary']} "
            f"subentry={validation['branch_targets_subentry']} "
            f"terminal={validation['branch_targets_terminal_atom']} "
            f"prefix={validation['branch_targets_prefix_byte']} "
            f"operand={validation['branch_targets_operand_byte']}"
        ),
    ]
    if validation["uncovered_byte_count"]:
        lines.append(
            "  uncovered bytes: "
            f"count={validation['uncovered_byte_count']} "
            f"runs={_fmt_runs(validation['uncovered_runs'], width=width)}"
        )
    if strict_4a_audit is not None:
        strict = strict_4a_audit.get("strict", {})
        conservative = strict_4a_audit.get("conservative", {})
        if strict.get("uncovered_byte_count", 0) and not conservative.get("uncovered_byte_count", 0):
            lines.append(
                "  strict 4A audit: "
                f"strict_uncovered={strict.get('uncovered_byte_count')} "
                f"runs={_fmt_runs(strict.get('uncovered_runs', []), width=width)} "
                "=> covered when 4A fallthrough is followed"
            )
    if cfg.anomalies:
        lines.append(f"  anomalies: {cfg.anomalies}")
    lines.extend(render_cfg_stream(body, cfg, mode=mode))
    return "\n".join(lines)


def render_base_audit(audit: dict[str, Any]) -> list[str]:
    lines = ["BR displacement base audit (CFG-reachable branches):"]
    header = "  model              total  known  oob  unknown  linear  subentry  terminal  prefix  operand"
    lines.append(header)
    for model in BASE_MODELS:
        stats = audit["models"][model]
        mark = "*" if model == audit.get("branch_base_model") else " "
        lines.append(
            f"{mark} {model:<17} "
            f"{stats['total']:>5} "
            f"{stats['known_target']:>6} "
            f"{stats['out_of_range']:>4} "
            f"{stats['unknown_target']:>8} "
            f"{stats['linear_boundary']:>7} "
            f"{stats['subentry']:>9} "
            f"{stats['terminal_atom']:>9} "
            f"{stats['prefix_byte']:>7} "
            f"{stats['operand_byte']:>8}"
        )
    return lines


def render_module_report(
    module: MBCModule,
    *,
    mode: str = "flow",
    function_filter: Optional[str] = None,
    include_base_audit: bool = True,
    branch_semantics: str = BRANCH_SEMANTICS_CONSERVATIVE,
) -> str:
    branch_semantics = normalize_branch_semantics(branch_semantics)
    entries = module.function_entries()
    if function_filter:
        entries = [entry for entry in entries if fnmatch(entry.name, function_filter) or fnmatch(entry.symbol, function_filter)]

    semantics_audit = compare_4a_semantics_for_module(module)
    strict_4a_by_name = {item["function"]: item for item in semantics_audit["functions"]}

    function_reports: list[str] = []
    totals = {
        "functions": 0,
        "body_bytes": 0,
        "linear_words": 0,
        "cfg_nodes": 0,
        "cfg_edges": 0,
        "cfg_control_branches": 0,
        "cfg_predicate_no_transfer": 0,
        "branch_targets_out_of_range": 0,
        "branch_targets_unknown": 0,
        "branch_targets_linear_boundary": 0,
        "branch_targets_subentry": 0,
        "branch_targets_terminal_atom": 0,
        "branch_targets_prefix_byte": 0,
        "branch_targets_operand_byte": 0,
        "reachable_unknown_words": 0,
        "uncovered_byte_count": 0,
        "functions_with_issues": 0,
    }

    for entry in entries:
        body = module.get_function_body(entry)
        cfg = build_cfg_for_function(module, entry, branch_semantics=branch_semantics)
        validation = validate_cfg(body, cfg)
        totals["functions"] += 1
        totals["body_bytes"] += len(body)
        for key in totals:
            if key in validation and isinstance(validation[key], int):
                totals[key] += validation[key]
        if validation["issue_flags"]:
            totals["functions_with_issues"] += 1
        function_reports.append(
            render_function_report(
                module,
                entry,
                mode=mode,
                branch_semantics=branch_semantics,
                strict_4a_audit=strict_4a_by_name.get(entry.name),
            )
        )

    lines = [
        f"SCRIPT {module.path}",
        f"parser_contract={module.parser_contract}",
        (
            f"definitions={len(module.definitions)} exports={len(module.exports)} "
            f"functions_selected={totals['functions']} code_size={module.get_real_code_size()}"
        ),
        (
            "adb="
            f"{module.adb_info.quality} present={module.adb_info.present} "
            f"words={module.adb_info.word_count}"
        ),
        (
            "BRANCH_SEMANTICS "
            f"active={branch_semantics} "
            f"strict_uncovered_bytes={semantics_audit['summary']['strict_uncovered_bytes']} "
            f"strict_functions={semantics_audit['summary']['strict_functions_with_uncovered']} "
            f"conservative_uncovered_bytes={semantics_audit['summary']['conservative_uncovered_bytes']} "
            f"explained_by_4a_fallthrough={semantics_audit['summary']['functions_explained_by_4a_fallthrough']}"
        ),
        (
            "SUMMARY "
            f"body_bytes={totals['body_bytes']} "
            f"linear_words={totals['linear_words']} cfg_nodes={totals['cfg_nodes']} "
            f"edges={totals['cfg_edges']} branches={totals['cfg_control_branches']} "
            f"predicate_no_transfer={totals['cfg_predicate_no_transfer']}"
        ),
        (
            "VALIDATION "
            f"branch_oob={totals['branch_targets_out_of_range']} "
            f"branch_unknown={totals['branch_targets_unknown']} "
            f"reachable_unknown={totals['reachable_unknown_words']} "
            f"subentry_targets={totals['branch_targets_subentry']} "
            f"linear_targets={totals['branch_targets_linear_boundary']} "
            f"uncovered_bytes={totals['uncovered_byte_count']} "
            f"functions_with_warnings={totals['functions_with_issues']}"
        ),
        (
            "SUBENTRY BREAKDOWN "
            f"terminal_atom={totals['branch_targets_terminal_atom']} "
            f"prefix_byte={totals['branch_targets_prefix_byte']} "
            f"operand_byte={totals['branch_targets_operand_byte']}"
        ),
    ]
    if include_base_audit:
        lines.extend(render_base_audit(branch_base_audit_for_module(module, branch_semantics=branch_semantics)))
    lines.append("")
    lines.append("Legend: leading '!' before an offset means CFG entry is not a top-level linear token boundary.")
    lines.append("Branch semantics: conservative_fallthrough follows fallthrough after 0x4A as an audit/over-approximation; strict_4a_jump preserves the old terminating-4A model.")
    lines.append("Mode: " + mode)
    lines.append("")
    lines.extend("\n" + report for report in function_reports)
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump compact function-local token flow with CFG branch annotations for an MBC script."
    )
    parser.add_argument("script", help="Script name or path. Examples: _main, _main.mbc, mbc/_main.mbc")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output text file. Default: out/<script>.branch_flow.txt")
    parser.add_argument(
        "--mode",
        choices=("flow", "branches", "full"),
        default="flow",
        help="flow: skip plain MARK/NOP; branches: calls/branches/returns only; full: every CFG node",
    )
    parser.add_argument("--function", default=None, help="Optional fnmatch filter for function names/symbols, e.g. 'Win*'")
    parser.add_argument(
        "--branch-semantics",
        choices=BRANCH_SEMANTICS_CHOICES,
        default=BRANCH_SEMANTICS_CONSERVATIVE,
        help="CFG semantics for 0x4A. Default follows 0x4A fallthrough to avoid hiding clean bytecode islands.",
    )
    parser.add_argument("--no-base-audit", action="store_true", help="Do not include branch-base model comparison")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    script_path = resolve_script_path(args.script)
    module = MBCModule(script_path)
    out_path = args.out or default_output_path(script_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = render_module_report(
        module,
        mode=args.mode,
        function_filter=args.function,
        include_base_audit=not args.no_base_audit,
        branch_semantics=args.branch_semantics,
    )
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
