from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Optional

from mbl_vm_tools.control_flow import (
    CFGEdge,
    CFGIssue,
    FunctionCFG,
    ModuleCFGReport,
    analyze_module,
    FORMAT_MODES,
)
from mbl_vm_tools.parser import MBCModule
from mbl_vm_tools.vm_spec import (
    VMWord,
    branch_operand_base_offset,
    code_ref_target_offset,
    signed_u16,
    word_role,
)


STRUCTURAL_TERMINALS = {"MARK", "NOP"}
IMPORTANT_ROLES = {"branch", "call", "return", "unknown", "predicate_no_transfer"}
BRANCH_OP_NAMES = {
    0x4A: "BR_4A",
    0x4B: "BR_4B",
    0x4C: "BR_4C",
    0x4D: "BR_4D",
}


ISSUE_CODES_HIDDEN_BY_DEFAULT = {"cfg_uncovered_bytes"}
OVERLAP_RELATIONS = {
    "aggregate_overlap_entry",
    "bare_u32_overlap_entry",
    "literal_payload_overlap_entry",
    "operand_payload_overlap_entry",
    "payload_overlap_entry",
}


def resolve_script_path(script: str | Path, *, root: Optional[Path] = None) -> Path:
    """Resolve an MBC script name against the usual project/corpus layout.

    Intended use from project root:
        python -m mbl_vm_tools.dump_branch_flow _main

    Search order keeps the requested spelling first, then tries the conventional
    ``mbc/`` directory. Names may be passed with or without the ``.mbc`` suffix.
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
        search_roots = [root / "mbc", root, package_root / "mbc", Path("")]

    for name in stem_candidates:
        if name.is_absolute():
            names.append(name)
            continue
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


def branch_op_name(op: int) -> str:
    return BRANCH_OP_NAMES.get(int(op) & 0xFF, f"BR_0x{int(op) & 0xFF:02X}")


def summarize_word(word: VMWord, *, include_raw: bool = False) -> str:
    term = word.terminal_kind
    kind = word.kind
    operands = word.operands
    parts = [kind]

    if term == "BR":
        op = int(operands.get("op", -1) or -1) & 0xFF
        off = signed_u16(int(operands.get("off", 0) or 0))
        parts.append(f"{branch_op_name(op)} off={signed(off)} base={hx(branch_operand_base_offset(word))}")
    elif term == "CALL_NATIVE":
        opid = operands.get("opid")
        opid_s = f"0x{int(opid):02X}" if isinstance(opid, int) else str(opid)
        parts.append(f"frame={operands.get('argc')} opid={opid_s}")
    elif term == "CALL_SCRIPT":
        parts.append(f"frame={operands.get('argc')} rel={signed(int(operands.get('rel', 0) or 0))}")
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
        value = operands.get("value")
        bits = operands.get("bits")
        suffix = f" bits=0x{int(bits):08X}" if isinstance(bits, int) else ""
        parts.append(f"value={value}{suffix}")
    elif term in {"AGG", "AGG0"}:
        parts.append(f"arity={operands.get('arity')} raw_arity={operands.get('raw_arity')}")
    elif term == "CODE_REF":
        rel = int(operands.get("rel", 0) or 0)
        try:
            parts.append(f"rel={signed(rel)} target={hx(code_ref_target_offset(word))}")
        except Exception:
            parts.append(f"rel={signed(rel)}")
    elif term == "BARE_U32":
        parts.append(f"value={operands.get('value')} follower={operands.get('follower_kind')}")
    elif term == "UNKNOWN":
        parts.append(f"byte=0x{int(operands.get('byte', 0) or 0):02X}")
    elif term == "RETURN_PAIR":
        parts.append("return_pair")
    elif term == "END":
        parts.append("return")
    elif term == "MARK":
        op = operands.get("op", operands.get("byte"))
        if isinstance(op, int):
            parts.append(f"op=0x{op:02X}")
    elif term == "NOP":
        pass

    if word.prefixes:
        parts.append("pfx=" + ".".join(f"{int(p):02X}" for p in word.prefixes))
    if include_raw:
        parts.append(f"raw={word.raw.hex(' ')}")
    return " ".join(str(p) for p in parts if p)


def _fmt_runs(runs: list[tuple[int, int]], *, width: int) -> str:
    if not runs:
        return "-"
    return ", ".join(f"{hx(a, width)}..{hx(b, width)}" for a, b in runs)


def _relation_short(relation: Optional[str]) -> str:
    if not relation:
        return "-"
    aliases = {
        "linear_boundary": "linear",
        "prefix_byte_entry": "prefix",
        "terminal_atom_entry": "terminal",
        "function_end": "function_end",
        "out_of_range": "OUT_OF_RANGE",
        "not_covered_by_linear_decode": "linear_gap",
        "aggregate_overlap_entry": "aggregate_overlap",
        "bare_u32_overlap_entry": "bare_u32_overlap",
        "literal_payload_overlap_entry": "literal_overlap",
        "operand_payload_overlap_entry": "operand_overlap",
        "payload_overlap_entry": "payload_overlap",
    }
    return aliases.get(relation, relation)


def _target_note_for_node(cfg: FunctionCFG, word: VMWord, node_edges: Iterable[CFGEdge]) -> str:
    if word.terminal_kind == "BR":
        branch_edges = [edge for edge in node_edges if edge.kind == "branch_conditional"]
        if not branch_edges:
            return " target=fallthrough/predicate"
        edge = branch_edges[0]
        note = _relation_short(edge.target_relation)
        target = hx(edge.dst, max(4, len(f"{cfg.body_size:X}"))) if edge.dst is not None else "-"
        decoded = f" decodes={edge.target_decoded_kind}" if edge.target_decoded_kind else ""
        return f" target={note}@{target}{decoded}"
    if word.terminal_kind == "CODE_REF":
        code_edges = [edge for edge in node_edges if edge.kind == "code_ref_reference"]
        if code_edges:
            edge = code_edges[0]
            note = _relation_short(edge.target_relation)
            target = hx(edge.dst, max(4, len(f"{cfg.body_size:X}"))) if edge.dst is not None else "-"
            decoded = f" decodes={edge.target_decoded_kind}" if edge.target_decoded_kind else ""
            return f" target={note}@{target}{decoded}"
    return ""


def _edges_by_source(edges: Iterable[CFGEdge]) -> dict[int, list[CFGEdge]]:
    grouped: dict[int, list[CFGEdge]] = defaultdict(list)
    for edge in edges:
        grouped[int(edge.src)].append(edge)
    return grouped


def _edge_kind_short(edge: CFGEdge) -> str:
    return {
        "branch_conditional": "branch",
        "predicate_no_transfer_next": "predicate_no_transfer",
        "fallthrough": "fallthrough",
        "next": "next",
        "code_ref_reference": "code_ref_ref",
    }.get(edge.kind, edge.kind)


def _edge_summary(edge: CFGEdge, body_len: int) -> str:
    width = max(4, len(f"{max(0, body_len):X}"))
    target = "-" if edge.dst is None else hx(edge.dst, width)
    relation = _relation_short(edge.target_relation)
    relation_text = "" if relation in {"-", "linear"} else f"/{relation}"
    decoded = "" if not edge.target_decoded_kind else f"/{edge.target_decoded_kind}"
    return f"{_edge_kind_short(edge)}->{target}{relation_text}{decoded}"


def should_show_node(word: VMWord, mode: str, *, relation: str, has_interesting_edges: bool) -> bool:
    role = word_role(word)
    if mode == "full":
        return True
    if mode == "branches":
        return has_interesting_edges or relation != "linear_boundary" or role in IMPORTANT_ROLES
    if mode == "flow":
        if word.terminal_kind in STRUCTURAL_TERMINALS and role == "structural" and relation == "linear_boundary" and not has_interesting_edges:
            return False
        return True
    raise ValueError(f"unknown mode: {mode}")


def render_cfg_stream(cfg: FunctionCFG, *, mode: str = "flow") -> list[str]:
    linear_offsets = {int(w.offset) for w in cfg.linear_words}
    grouped_edges = _edges_by_source(cfg.all_edges)
    width = max(4, len(f"{max(0, cfg.body_size):X}"))
    lines: list[str] = []
    skipped = 0

    for offset, node in sorted(cfg.nodes.items()):
        if offset == node.region_root:
            lines.append(f"    -- region {node.region_kind} root={hx(node.region_root, width)} --")
        word = node.word
        node_edges = grouped_edges.get(offset, [])
        if mode == "full":
            interesting_edges = node_edges
        else:
            interesting_edges = [edge for edge in node_edges if edge.kind not in {"next", "fallthrough"}]
        relation = node.relation.relation
        if not should_show_node(word, mode, relation=relation, has_interesting_edges=bool(interesting_edges)):
            skipped += 1
            continue
        if skipped:
            lines.append(f"    ... {skipped} structural MARK/NOP entries omitted")
            skipped = 0

        sub = "!" if offset not in linear_offsets or relation != "linear_boundary" else " "
        role = word_role(word)
        edge_text = ""
        if interesting_edges:
            edge_text = " ; " + ", ".join(_edge_summary(edge, cfg.body_size) for edge in interesting_edges)
        relation_text = "" if relation == "linear_boundary" else f" relation={_relation_short(relation)}"
        region_text = "" if node.region_kind == "entry" else f" region={node.region_kind}@{hx(node.region_root, width)}"
        target_note = _target_note_for_node(cfg, word, node_edges)
        raw = word.terminal_kind in {"UNKNOWN", "BR"}
        lines.append(
            f"  {sub}{hx(offset, width)} +{word.size:<2} {role:<21} "
            f"{summarize_word(word, include_raw=raw)}{relation_text}{region_text}{target_note}{edge_text}"
        )
    if skipped:
        lines.append(f"    ... {skipped} structural MARK/NOP entries omitted")
    return lines


def _issue_flags(cfg: FunctionCFG) -> list[str]:
    flags: list[str] = []
    if cfg.hard_error_count:
        flags.append(f"errors={cfg.hard_error_count}")
    if cfg.warning_count:
        flags.append(f"warnings={cfg.warning_count}")
    visible_notes = sum(1 for issue in cfg.issues if issue.severity == "note" and issue.code not in ISSUE_CODES_HIDDEN_BY_DEFAULT)
    if visible_notes:
        flags.append(f"notes={visible_notes}")
    uncovered = cfg.body_size - cfg.cfg_covered_bytes
    if uncovered:
        flags.append(f"uncovered={uncovered}")
    if cfg.overlap_target_count:
        flags.append(f"overlap_targets={cfg.overlap_target_count}")
    return flags


def _issue_counts(cfgs: Iterable[FunctionCFG]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for cfg in cfgs:
        for issue in cfg.issues:
            counts[issue.code] += 1
    return counts


def _severity_counts(cfgs: Iterable[FunctionCFG]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for cfg in cfgs:
        for issue in cfg.issues:
            counts[issue.severity] += 1
    return counts


def _target_relation_totals(cfgs: Iterable[FunctionCFG]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for cfg in cfgs:
        totals.update(cfg.branch_target_relations)
    return totals



def render_issue(issue: CFGIssue, *, width: int) -> str:
    loc = ""
    if issue.offset is not None:
        loc += f" @{hx(issue.offset, width)}"
    if issue.target is not None:
        loc += f" target={hx(issue.target, width)}"
    return f"  issue: {issue.severity}:{issue.code}{loc} {issue.message}".rstrip()


def render_function_report(cfg: FunctionCFG, *, mode: str = "flow") -> str:
    width = max(4, len(f"{max(0, cfg.body_size):X}"))
    span_s = f"{hx(cfg.span[0])}..{hx(cfg.span[1])}" if cfg.span else "-"
    flags = _issue_flags(cfg)
    status = "OK" if not flags else "WARN:" + ",".join(flags)
    relation_counts = cfg.branch_target_relations
    linear_targets = relation_counts.get("linear_boundary", 0)
    terminal_targets = relation_counts.get("terminal_atom_entry", 0)
    prefix_targets = relation_counts.get("prefix_byte_entry", 0)
    overlap_targets = sum(relation_counts.get(name, 0) for name in OVERLAP_RELATIONS)
    subentry_targets = sum(
        count
        for relation, count in relation_counts.items()
        if relation not in {"linear_boundary", "out_of_range", "function_end"}
    )
    branch_oob = sum(1 for issue in cfg.issues if issue.code in {"branch_target_out_of_range", "linear_branch_target_out_of_range"})
    branch_unknown = sum(1 for issue in cfg.issues if issue.code in {"branch_target_decodes_unknown", "linear_branch_target_decodes_unknown"})
    reachable_unknown = sum(1 for issue in cfg.issues if issue.code == "reachable_unknown_word")
    uncovered = cfg.body_size - cfg.cfg_covered_bytes

    lines = [
        f"FUNC {cfg.function_name} [{cfg.source_kind}] span={span_s} len={cfg.body_size} status={status}",
        (
            "  cfg: "
            f"nodes={len(cfg.nodes)} transfer_edges={cfg.transfer_edge_count} reference_edges={cfg.reference_edge_count} "
            f"linear_words={len(cfg.linear_words)} "
            f"branches={cfg.control_branch_count_linear} "
            f"predicate_no_transfer={cfg.predicate_no_transfer_count} "
            f"code_refs={cfg.code_ref_count_linear} code_ref_roots={len(cfg.code_ref_region_roots)} "
            f"code_ref_boundaries={cfg.code_ref_region_boundary_count} "
            f"call_scripts_linear={cfg.call_script_count_linear} "
            f"call_scripts_reachable={cfg.reachable_call_script_count} "
            f"unknown={reachable_unknown}"
        ),
        (
            "  branch targets: "
            f"oob={branch_oob} "
            f"unknown={branch_unknown} "
            f"linear={linear_targets} "
            f"subentry={subentry_targets} "
            f"terminal={terminal_targets} "
            f"prefix={prefix_targets} "
            f"overlap={overlap_targets}"
        ),
    ]
    if relation_counts:
        lines.append("  branch target relations(linear): " + ", ".join(f"{k}={v}" for k, v in sorted(relation_counts.items())))
    if cfg.cfg_branch_target_relations:
        lines.append("  branch target relations(reachable_cfg): " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.cfg_branch_target_relations.items())))
    if cfg.code_ref_target_relations:
        lines.append("  code_ref target relations: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.code_ref_target_relations.items())))
    if cfg.call_script_target_relations:
        lines.append("  call_script target relations: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.call_script_target_relations.items())))
    if uncovered:
        lines.append(
            "  uncovered bytes: "
            f"count={uncovered} runs={_fmt_runs(cfg.cfg_uncovered_ranges, width=width)}"
        )

    visible_issues = [issue for issue in cfg.issues if issue.code not in ISSUE_CODES_HIDDEN_BY_DEFAULT]
    for issue in visible_issues[:16]:
        lines.append(render_issue(issue, width=width))
    if len(visible_issues) > 16:
        lines.append(f"  issue: ... {len(visible_issues) - 16} more issues omitted")

    lines.extend(render_cfg_stream(cfg, mode=mode))
    return "\n".join(lines)


def _filter_cfgs(report: ModuleCFGReport, pattern: str | None) -> list[FunctionCFG]:
    cfgs = list(report.function_cfgs)
    if pattern:
        cfgs = [cfg for cfg in cfgs if fnmatch(cfg.function_name, pattern) or fnmatch(cfg.symbol, pattern)]
    return cfgs


def _select_cfgs(cfgs: list[FunctionCFG], *, interesting_only: bool) -> list[FunctionCFG]:
    if not interesting_only:
        return cfgs
    return [cfg for cfg in cfgs if cfg.has_control_interest or _issue_flags(cfg)]


def render_module_report(
    module: MBCModule,
    *,
    mode: str = "flow",
    function_filter: Optional[str] = None,
    interesting_only: bool = False,
) -> str:
    report = analyze_module(module)
    cfgs = _filter_cfgs(report, function_filter)
    selected = _select_cfgs(cfgs, interesting_only=interesting_only)

    totals: Counter[str] = Counter()
    for cfg in selected:
        totals["functions"] += 1
        totals["body_bytes"] += cfg.body_size
        totals["linear_words"] += len(cfg.linear_words)
        totals["cfg_nodes"] += len(cfg.nodes)
        totals["transfer_edges"] += cfg.transfer_edge_count
        totals["reference_edges"] += cfg.reference_edge_count
        totals["linear_branches"] += cfg.branch_count_linear
        totals["control_branches"] += cfg.control_branch_count_linear
        totals["predicate_no_transfer"] += cfg.predicate_no_transfer_count
        totals["code_refs"] += cfg.code_ref_count_linear
        totals["call_scripts"] += cfg.call_script_count_linear
        totals["reachable_call_scripts"] += cfg.reachable_call_script_count
        totals["code_ref_roots"] += len(cfg.code_ref_region_roots)
        totals["code_ref_region_boundaries"] += cfg.code_ref_region_boundary_count
        totals["uncovered_bytes"] += cfg.body_size - cfg.cfg_covered_bytes
        if _issue_flags(cfg):
            totals["functions_with_warnings"] += 1

    relations = _target_relation_totals(selected)
    issue_counts = _issue_counts(selected)
    severity_counts = _severity_counts(selected)
    linear_targets = relations.get("linear_boundary", 0)
    terminal_targets = relations.get("terminal_atom_entry", 0)
    prefix_targets = relations.get("prefix_byte_entry", 0)
    overlap_targets = sum(relations.get(name, 0) for name in OVERLAP_RELATIONS)
    subentry_targets = sum(
        count
        for relation, count in relations.items()
        if relation not in {"linear_boundary", "out_of_range", "function_end"}
    )

    lines = [
        f"SCRIPT {module.path}",
        f"parser_contract={module.parser_contract}",
        (
            f"definitions={len(module.definitions)} exports={len(module.exports)} globals={len(module.globals)} "
            f"functions_selected={totals['functions']} functions_total={report.function_count} code_size={module.get_real_code_size()}"
        ),
        (
            "adb="
            f"{module.adb_info.quality} present={module.adb_info.present} "
            f"words={module.adb_info.word_count}"
        ),
        "ANALYZER control_flow model=region_cfg_v4 code_ref=reference_not_transfer code_ref_roots_are_region_boundaries branch_targets=require_exact_suffix_entry_frame call_script=reachable_cfg_validation",
        (
            "SUMMARY "
            f"body_bytes={totals['body_bytes']} "
            f"linear_words={totals['linear_words']} cfg_nodes={totals['cfg_nodes']} "
            f"transfer_edges={totals['transfer_edges']} reference_edges={totals['reference_edges']} "
            f"branches={totals['control_branches']} predicate_no_transfer={totals['predicate_no_transfer']} "
            f"code_refs={totals['code_refs']} code_ref_roots={totals['code_ref_roots']} "
            f"code_ref_boundaries={totals['code_ref_region_boundaries']} "
            f"call_scripts_linear={totals['call_scripts']} "
            f"call_scripts_reachable={totals['reachable_call_scripts']}"
        ),
        (
            "VALIDATION "
            f"branch_oob={issue_counts.get('branch_target_out_of_range', 0) + issue_counts.get('linear_branch_target_out_of_range', 0)} "
            f"branch_unknown={issue_counts.get('branch_target_decodes_unknown', 0) + issue_counts.get('linear_branch_target_decodes_unknown', 0)} "
            f"branch_frame_bad={issue_counts.get('branch_target_not_suffix_entry_frame', 0) + issue_counts.get('linear_branch_target_not_suffix_entry_frame', 0)} "
            f"reachable_unknown={issue_counts.get('reachable_unknown_word', 0)} "
            f"subentry_targets={subentry_targets} "
            f"linear_targets={linear_targets} "
            f"uncovered_bytes={totals['uncovered_bytes']} "
            f"functions_with_warnings={totals['functions_with_warnings']}"
        ),
        (
            "SUBENTRY BREAKDOWN "
            f"terminal_atom={terminal_targets} "
            f"prefix_byte={prefix_targets} "
            f"overlap={overlap_targets}"
        ),
        f"BRANCH TARGET RELATIONS linear={dict(sorted(relations.items()))}",
        f"BRANCH TARGET RELATIONS reachable_cfg={dict(sorted(sum((cfg.cfg_branch_target_relations for cfg in selected), Counter()).items()))}",
        f"CODE_REF TARGET RELATIONS {dict(sorted(sum((cfg.code_ref_target_relations for cfg in selected), Counter()).items()))}",
        f"CALL_SCRIPT TARGET RELATIONS reachable={dict(sorted(sum((cfg.call_script_target_relations for cfg in selected), Counter()).items()))}",
        f"CALL_SCRIPT SOURCE RELATIONS reachable={dict(sorted(sum((cfg.call_script_source_relations for cfg in selected), Counter()).items()))}",
        f"ISSUES severity={dict(severity_counts)} codes={dict(issue_counts)}",
        "",
        "Legend: leading '!' before an offset means CFG entry is not a top-level linear token boundary.",
        "Edges: branch/fallthrough/next are transfer edges; code_ref_ref is a reference edge that starts a separate region. Branch subentries must decode as exact suffixes of their containing linear word.",
        "Mode: " + mode,
        "",
    ]
    if function_filter:
        lines.insert(4, f"function_filter={function_filter!r}")
    if interesting_only:
        lines.insert(5 if function_filter else 4, "selection=interesting_only")

    lines.extend("\n" + render_function_report(cfg, mode=mode) for cfg in selected)
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump compact function-local token flow with CFG branch annotations for an MBC script."
    )
    parser.add_argument("script", help="Script name or path. Examples: _main, _main.mbc, mbc/_main.mbc")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output text file. Default: out/<script>.branch_flow.txt")
    parser.add_argument(
        "--mode",
        choices=FORMAT_MODES,
        default="flow",
        help="flow: omit plain structural MARK/NOP; branches: calls/branches/returns/labels; full: every CFG node edge",
    )
    parser.add_argument("--function", default=None, help="Optional fnmatch filter for function names/symbols, e.g. 'Win*'")
    parser.add_argument(
        "--interesting-only",
        action="store_true",
        help="Emit only functions with control-flow interest or validation issues. Default emits all selected functions, like the old dump.",
    )
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
        interesting_only=args.interesting_only,
    )
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
