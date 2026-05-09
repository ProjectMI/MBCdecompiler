from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from .parser import MBCModule
from .vm_spec import (
    CONDITIONAL_CONTROL_BRANCH_OPS,
    VMWord,
    branch_target_offset,
    call_script_target_offset,
    code_ref_target_offset,
    decode_word_at,
    decode_words,
    is_control_branch_word,
    terminal_atom_offset,
    word_role,
)

RETURN_TERMINALS = {"RETURN_PAIR", "END"}
BRANCH_TERMINAL = "BR"
NORMAL_TARGET_RELATIONS = {"linear_boundary", "prefix_byte_entry", "terminal_atom_entry"}
OVERLAP_TARGET_RELATIONS = {
    "aggregate_overlap_entry",
    "bare_u32_overlap_entry",
    "literal_payload_overlap_entry",
    "operand_payload_overlap_entry",
    "payload_overlap_entry",
}
FORMAT_MODES = ("flow", "branches", "full")


@dataclass(frozen=True)
class EntryRelation:
    """Relationship between an exact byte entry and the default linear decode."""

    offset: int
    relation: str
    linear_word_offset: Optional[int] = None
    linear_word_size: Optional[int] = None
    linear_word_kind: Optional[str] = None
    linear_word_terminal: Optional[str] = None
    linear_word_raw_hex: Optional[str] = None

    @property
    def is_overlap(self) -> bool:
        return self.relation in OVERLAP_TARGET_RELATIONS

    @property
    def is_normal_entry(self) -> bool:
        return self.relation in NORMAL_TARGET_RELATIONS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CFGEdge:
    src: int
    dst: Optional[int]
    kind: str
    op: Optional[int] = None
    op_hex: Optional[str] = None
    target_relation: Optional[str] = None
    target_decoded_kind: Optional[str] = None
    note: str = ""

    @property
    def is_reference(self) -> bool:
        return self.kind.endswith("_reference")

    @property
    def is_transfer(self) -> bool:
        return not self.is_reference

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CFGNode:
    offset: int
    word: VMWord
    relation: EntryRelation
    edges: list[CFGEdge] = field(default_factory=list)
    region_root: int = 0
    region_kind: str = "entry"

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "word": word_to_dict(self.word),
            "relation": self.relation.to_dict(),
            "edges": [edge.to_dict() for edge in self.edges],
            "region_root": self.region_root,
            "region_kind": self.region_kind,
        }


@dataclass(frozen=True)
class CFGIssue:
    severity: str
    code: str
    offset: Optional[int] = None
    target: Optional[int] = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionCFG:
    function_name: str
    symbol: str
    source_kind: str
    definition_index: Optional[int]
    span: Optional[tuple[int, int]]
    body_size: int
    linear_words: list[VMWord]
    nodes: dict[int, CFGNode]
    edges: list[CFGEdge]
    reference_edges: list[CFGEdge]
    issues: list[CFGIssue]
    branch_target_relations: Counter[str]
    cfg_branch_target_relations: Counter[str]
    code_ref_target_relations: Counter[str]
    call_script_target_relations: Counter[str]
    call_script_source_relations: Counter[str]
    branch_count_linear: int
    control_branch_count_linear: int
    predicate_no_transfer_count: int
    code_ref_count_linear: int
    call_script_count_linear: int
    code_ref_region_roots: list[int]
    cfg_covered_bytes: int
    cfg_uncovered_ranges: list[tuple[int, int]]
    entry_covered_bytes: int = 0
    code_ref_region_boundary_count: int = 0

    @property
    def hard_error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def note_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "note")

    @property
    def overlap_target_count(self) -> int:
        return sum(self.branch_target_relations.get(name, 0) for name in OVERLAP_TARGET_RELATIONS)

    @property
    def reference_edge_count(self) -> int:
        return len(self.reference_edges)

    @property
    def transfer_edge_count(self) -> int:
        return len(self.edges)

    @property
    def reachable_call_script_count(self) -> int:
        return sum(1 for node in self.nodes.values() if node.word.terminal_kind == "CALL_SCRIPT")

    @property
    def all_edges(self) -> list[CFGEdge]:
        return [*self.edges, *self.reference_edges]

    @property
    def has_control_interest(self) -> bool:
        return bool(
            self.control_branch_count_linear
            or self.code_ref_count_linear
            or self.hard_error_count
            or self.warning_count
            or self.overlap_target_count
        )

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "symbol": self.symbol,
            "source_kind": self.source_kind,
            "definition_index": self.definition_index,
            "span": self.span,
            "body_size": self.body_size,
            "linear_word_count": len(self.linear_words),
            "cfg_node_count": len(self.nodes),
            "transfer_edge_count": self.transfer_edge_count,
            "reference_edge_count": self.reference_edge_count,
            "branch_count_linear": self.branch_count_linear,
            "control_branch_count_linear": self.control_branch_count_linear,
            "predicate_no_transfer_count": self.predicate_no_transfer_count,
            "code_ref_count_linear": self.code_ref_count_linear,
            "call_script_count_linear": self.call_script_count_linear,
            "call_script_count_reachable": self.reachable_call_script_count,
            "branch_target_relations_linear": dict(self.branch_target_relations),
            "branch_target_relations_reachable_cfg": dict(self.cfg_branch_target_relations),
            "code_ref_target_relations": dict(self.code_ref_target_relations),
            "call_script_target_relations_reachable": dict(self.call_script_target_relations),
            "call_script_source_relations_reachable": dict(self.call_script_source_relations),
            "code_ref_region_roots": self.code_ref_region_roots[:32],
            "code_ref_region_root_count": len(self.code_ref_region_roots),
            "code_ref_region_boundary_count": self.code_ref_region_boundary_count,
            "overlap_target_count": self.overlap_target_count,
            "hard_error_count": self.hard_error_count,
            "warning_count": self.warning_count,
            "note_count": self.note_count,
            "entry_covered_bytes": self.entry_covered_bytes,
            "cfg_covered_bytes": self.cfg_covered_bytes,
            "cfg_uncovered_byte_count": self.body_size - self.cfg_covered_bytes,
            "cfg_uncovered_ranges": self.cfg_uncovered_ranges[:16],
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class ModuleCFGReport:
    module_path: str
    parser_contract: str
    layout_diagnostics: dict[str, Any]
    definition_count: int
    export_count: int
    global_count: int
    function_count: int
    function_cfgs: list[FunctionCFG]

    @property
    def summary(self) -> dict[str, Any]:
        branch_relations: Counter[str] = Counter()
        cfg_branch_relations: Counter[str] = Counter()
        code_ref_relations: Counter[str] = Counter()
        call_script_relations: Counter[str] = Counter()
        call_script_source_relations: Counter[str] = Counter()
        issue_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        functions_with_branches = 0
        functions_with_errors = 0
        functions_with_overlap_targets = 0
        functions_with_uncovered_bytes = 0
        functions_with_code_refs = 0

        for cfg in self.function_cfgs:
            branch_relations.update(cfg.branch_target_relations)
            cfg_branch_relations.update(cfg.cfg_branch_target_relations)
            code_ref_relations.update(cfg.code_ref_target_relations)
            call_script_relations.update(cfg.call_script_target_relations)
            call_script_source_relations.update(cfg.call_script_source_relations)
            totals["body_bytes"] += cfg.body_size
            totals["linear_words"] += len(cfg.linear_words)
            totals["cfg_nodes"] += len(cfg.nodes)
            totals["transfer_edges"] += cfg.transfer_edge_count
            totals["reference_edges"] += cfg.reference_edge_count
            totals["linear_branches"] += cfg.branch_count_linear
            totals["control_branches"] += cfg.control_branch_count_linear
            totals["predicate_no_transfer_branches"] += cfg.predicate_no_transfer_count
            totals["code_refs"] += cfg.code_ref_count_linear
            totals["call_scripts"] += cfg.call_script_count_linear
            totals["reachable_call_scripts"] += cfg.reachable_call_script_count
            totals["code_ref_region_roots"] += len(cfg.code_ref_region_roots)
            totals["code_ref_region_boundaries"] += cfg.code_ref_region_boundary_count
            totals["entry_covered_bytes"] += cfg.entry_covered_bytes
            totals["cfg_uncovered_bytes"] += cfg.body_size - cfg.cfg_covered_bytes
            if cfg.control_branch_count_linear:
                functions_with_branches += 1
            if cfg.code_ref_count_linear:
                functions_with_code_refs += 1
            if cfg.hard_error_count:
                functions_with_errors += 1
            if cfg.overlap_target_count:
                functions_with_overlap_targets += 1
            if cfg.body_size - cfg.cfg_covered_bytes:
                functions_with_uncovered_bytes += 1
            for issue in cfg.issues:
                issue_counts[issue.code] += 1
                severity_counts[issue.severity] += 1

        return {
            "module_path": self.module_path,
            "parser_contract": self.parser_contract,
            "layout_diagnostics": self.layout_diagnostics,
            "definition_count": self.definition_count,
            "export_count": self.export_count,
            "global_count": self.global_count,
            "function_count": self.function_count,
            "functions_with_control_branches": functions_with_branches,
            "functions_with_code_refs": functions_with_code_refs,
            "functions_with_hard_errors": functions_with_errors,
            "functions_with_overlap_targets": functions_with_overlap_targets,
            "functions_with_uncovered_bytes": functions_with_uncovered_bytes,
            "totals": dict(totals),
            "branch_target_relations_linear": dict(branch_relations),
            "branch_target_relations_reachable_cfg": dict(cfg_branch_relations),
            "code_ref_target_relations": dict(code_ref_relations),
            "call_script_target_relations_reachable": dict(call_script_relations),
            "call_script_source_relations_reachable": dict(call_script_source_relations),
            "issue_counts": dict(issue_counts),
            "severity_counts": dict(severity_counts),
            "notable_functions": [
                cfg.to_summary_dict()
                for cfg in self.function_cfgs
                if cfg.hard_error_count or cfg.warning_count or cfg.overlap_target_count
            ],
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return self.summary


def word_to_dict(word: VMWord) -> dict[str, Any]:
    return {
        "offset": int(word.offset),
        "size": int(word.size),
        "kind": word.kind,
        "terminal_kind": word.terminal_kind,
        "prefixes": [int(p) for p in word.prefixes],
        "operands": dict(word.operands),
        "raw_hex": word.raw.hex(" "),
        "confidence": word.confidence,
        "decoder_rule": word.decoder_rule,
        "role": word_role(word),
    }


def _build_linear_span_map(linear_words: Iterable[VMWord]) -> tuple[dict[int, VMWord], set[int]]:
    span_map: dict[int, VMWord] = {}
    starts: set[int] = set()
    for word in linear_words:
        starts.add(int(word.offset))
        for byte_offset in range(int(word.offset), int(word.offset) + int(word.size)):
            span_map[byte_offset] = word
    return span_map, starts


def classify_entry_offset(offset: int, body_size: int, span_map: dict[int, VMWord], linear_starts: set[int]) -> EntryRelation:
    if offset < 0 or offset > body_size:
        return EntryRelation(offset=offset, relation="out_of_range")
    if offset == body_size:
        return EntryRelation(offset=offset, relation="function_end")

    linear_word = span_map.get(offset)
    if linear_word is None:
        return EntryRelation(offset=offset, relation="not_covered_by_linear_decode")

    if offset in linear_starts and offset == int(linear_word.offset):
        relation = "linear_boundary"
    else:
        terminal_offset = terminal_atom_offset(linear_word)
        if offset == terminal_offset:
            relation = "terminal_atom_entry"
        elif int(linear_word.offset) <= offset < terminal_offset:
            relation = "prefix_byte_entry"
        elif linear_word.terminal_kind in {"AGG", "AGG0"}:
            relation = "aggregate_overlap_entry"
        elif linear_word.terminal_kind == "BARE_U32":
            relation = "bare_u32_overlap_entry"
        elif linear_word.terminal_kind in {"IMM8", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32", "F32", "U16"}:
            relation = "literal_payload_overlap_entry"
        elif linear_word.terminal_kind in {"REF", "REF16", "REC41", "REC61", "REC62", "CODE_REF", "CALL_NATIVE", "CALL_SCRIPT"}:
            relation = "operand_payload_overlap_entry"
        else:
            relation = "payload_overlap_entry"

    return EntryRelation(
        offset=offset,
        relation=relation,
        linear_word_offset=int(linear_word.offset),
        linear_word_size=int(linear_word.size),
        linear_word_kind=linear_word.kind,
        linear_word_terminal=linear_word.terminal_kind,
        linear_word_raw_hex=linear_word.raw.hex(" "),
    )


def _ranges_from_covered(body_size: int, covered: set[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: Optional[int] = None
    prev: Optional[int] = None
    for offset in range(body_size):
        if offset not in covered:
            if start is None:
                start = offset
            prev = offset
        elif start is not None:
            ranges.append((start, int(prev) + 1))
            start = None
            prev = None
    if start is not None:
        ranges.append((start, int(prev) + 1))
    return ranges


def _decode_target_word(body: bytes, target: int) -> VMWord | None:
    if target < 0 or target >= len(body):
        return None
    try:
        return decode_word_at(body, target, limit=len(body))
    except Exception:
        return None


def _decode_target_kind(body: bytes, target: int) -> str | None:
    word = _decode_target_word(body, target)
    return word.kind if word is not None else None


def _target_decodes_unknown(kind: str | None) -> bool:
    return kind is None or kind.endswith("UNKNOWN")


def _branch_target_suffix_frame_is_exact(target: int, relation: EntryRelation, target_word: VMWord | None) -> bool:
    """Return True when an in-word branch target is a real suffix entry.

    Linear entries are accepted directly.  Prefix/terminal entries are accepted
    only when decoding from the branch target consumes exactly the suffix of the
    containing linear word.  This keeps the branch entry frame narrow: branches
    may enter declared linear words or suffixes of fused prefix words, but not
    arbitrary shorter parses inside a word.
    """

    if relation.relation == "linear_boundary":
        return True
    if relation.relation not in {"prefix_byte_entry", "terminal_atom_entry"}:
        return False
    if target_word is None or relation.linear_word_offset is None or relation.linear_word_size is None:
        return False
    return int(target) + int(target_word.size) == int(relation.linear_word_offset) + int(relation.linear_word_size)


def _validate_branch_target(
    *,
    body: bytes,
    body_size: int,
    word: VMWord,
    target: int,
    relation: EntryRelation,
    target_word: VMWord | None,
    target_kind: str | None,
    op_hex: str,
    issues: list[CFGIssue],
    issue_prefix: str,
) -> bool:
    if target < 0 or target >= body_size:
        issues.append(
            CFGIssue(
                severity="error",
                code=f"{issue_prefix}branch_target_out_of_range",
                offset=word.offset,
                target=target,
                message=f"Branch target {target} is outside function body size {body_size}.",
                details={"op": op_hex, "relation": relation.to_dict()},
            )
        )
        return False
    if _target_decodes_unknown(target_kind):
        issues.append(
            CFGIssue(
                severity="error",
                code=f"{issue_prefix}branch_target_decodes_unknown",
                offset=word.offset,
                target=target,
                message="Branch target is in-range, but decoding at the exact target is UNKNOWN.",
                details={"op": op_hex, "relation": relation.to_dict(), "target_decoded_kind": target_kind},
            )
        )
        return False
    if relation.is_overlap:
        issues.append(
            CFGIssue(
                severity="error",
                code=f"{issue_prefix}branch_target_overlap_entry",
                offset=word.offset,
                target=target,
                message="Branch target lands inside a payload/overlap area, not inside a legal branch entry frame.",
                details={"op": op_hex, "relation": relation.to_dict(), "target_decoded_kind": target_kind},
            )
        )
        return False
    if not _branch_target_suffix_frame_is_exact(target, relation, target_word):
        details = {"op": op_hex, "relation": relation.to_dict(), "target_decoded_kind": target_kind}
        if target_word is not None:
            details["target_word"] = word_to_dict(target_word)
            if relation.linear_word_offset is not None and relation.linear_word_size is not None:
                details["target_word_end"] = int(target) + int(target_word.size)
                details["linear_word_end"] = int(relation.linear_word_offset) + int(relation.linear_word_size)
        issues.append(
            CFGIssue(
                severity="error",
                code=f"{issue_prefix}branch_target_not_suffix_entry_frame",
                offset=word.offset,
                target=target,
                message="Branch target decodes, but it is not an exact suffix entry of the containing linear VM word.",
                details=details,
            )
        )
        return False
    return True


def _linear_branch_target_relations(
    body: bytes,
    body_size: int,
    linear_words: list[VMWord],
    span_map: dict[int, VMWord],
    linear_starts: set[int],
    issues: list[CFGIssue],
) -> Counter[str]:
    relations: Counter[str] = Counter()
    for word in linear_words:
        if not is_control_branch_word(word):
            continue
        op = int(word.operands.get("op", -1) or -1) & 0xFF
        op_hex = f"0x{op:02X}"
        try:
            target = branch_target_offset(word)
        except Exception as exc:
            issues.append(CFGIssue("error", "linear_branch_target_resolution_error", word.offset, message=str(exc), details=word_to_dict(word)))
            continue
        relation = classify_entry_offset(target, body_size, span_map, linear_starts)
        target_word = _decode_target_word(body, target)
        target_kind = target_word.kind if target_word is not None else None
        relations[relation.relation] += 1
        _validate_branch_target(
            body=body,
            body_size=body_size,
            word=word,
            target=target,
            relation=relation,
            target_word=target_word,
            target_kind=target_kind,
            op_hex=op_hex,
            issues=issues,
            issue_prefix="linear_",
        )
    return relations


def _linear_code_ref_target_relations(
    body: bytes,
    body_size: int,
    linear_words: list[VMWord],
    span_map: dict[int, VMWord],
    linear_starts: set[int],
    issues: list[CFGIssue],
) -> Counter[str]:
    relations: Counter[str] = Counter()
    for word in linear_words:
        if word.terminal_kind != "CODE_REF":
            continue
        try:
            target = code_ref_target_offset(word)
        except Exception as exc:
            issues.append(CFGIssue("error", "linear_code_ref_target_resolution_error", word.offset, message=str(exc), details=word_to_dict(word)))
            continue
        relation = classify_entry_offset(target, body_size, span_map, linear_starts)
        target_kind = _decode_target_kind(body, target)
        relations[relation.relation] += 1
        if target < 0 or target >= body_size:
            issues.append(
                CFGIssue(
                    "error",
                    "linear_code_ref_target_out_of_range",
                    word.offset,
                    target,
                    f"CODE_REF target {target} is outside function body size {body_size}.",
                    {"relation": relation.to_dict()},
                )
            )
        elif _target_decodes_unknown(target_kind):
            issues.append(
                CFGIssue(
                    "error",
                    "linear_code_ref_target_decodes_unknown",
                    word.offset,
                    target,
                    "CODE_REF target is in-range, but decoding at the exact target is UNKNOWN.",
                    {"relation": relation.to_dict(), "target_decoded_kind": target_kind},
                )
            )
        elif relation.relation != "linear_boundary":
            issues.append(
                CFGIssue(
                    "warning",
                    "linear_code_ref_target_not_linear_boundary",
                    word.offset,
                    target,
                    "CODE_REF target is valid, but it is not a top-level linear word boundary.",
                    {"relation": relation.to_dict(), "target_decoded_kind": target_kind},
                )
            )
    return relations


def build_function_cfg(
    body: bytes,
    *,
    function_name: str = "<anonymous>",
    symbol: str = "<anonymous>",
    source_kind: str = "definition",
    definition_index: Optional[int] = None,
    span: Optional[tuple[int, int]] = None,
) -> FunctionCFG:
    """Build a region-aware byte-entry CFG for one function body.

    Control transfer is produced only by BR and by normal regional fallthrough.
    CODE_REF is a reference to a local code island: it is validated and starts a
    separate region, but it is not a transfer edge from the referencing word.
    """

    body_size = len(body)
    linear_words = decode_words(body)
    span_map, linear_starts = _build_linear_span_map(linear_words)
    branch_count_linear = sum(1 for word in linear_words if word.terminal_kind == BRANCH_TERMINAL)
    control_branch_count_linear = sum(1 for word in linear_words if is_control_branch_word(word))
    predicate_no_transfer_count = branch_count_linear - control_branch_count_linear
    code_ref_count_linear = sum(1 for word in linear_words if word.terminal_kind == "CODE_REF")
    call_script_count_linear = sum(1 for word in linear_words if word.terminal_kind == "CALL_SCRIPT")

    nodes_raw: dict[int, VMWord] = {}
    node_region_root: dict[int, int] = {}
    node_region_kind: dict[int, str] = {}
    edge_buckets: defaultdict[int, list[CFGEdge]] = defaultdict(list)
    transfer_edges: list[CFGEdge] = []
    reference_edges: list[CFGEdge] = []
    issues: list[CFGIssue] = []

    linear_branch_relations = _linear_branch_target_relations(body, body_size, linear_words, span_map, linear_starts, issues)
    code_ref_relations = _linear_code_ref_target_relations(body, body_size, linear_words, span_map, linear_starts, issues)
    declared_code_ref_roots: set[int] = set()
    for ref_word in linear_words:
        if ref_word.terminal_kind != "CODE_REF":
            continue
        try:
            ref_target = code_ref_target_offset(ref_word)
        except Exception:
            continue
        ref_kind = _decode_target_kind(body, ref_target)
        if 0 <= ref_target < body_size and not _target_decodes_unknown(ref_kind):
            declared_code_ref_roots.add(ref_target)
    cfg_branch_relations: Counter[str] = Counter()

    pending_regions: deque[tuple[int, str]] = deque()
    seen_region_roots: set[int] = set()
    code_ref_region_roots: set[int] = set()
    code_ref_region_boundary_count = 0

    def add_transfer_edge(edge: CFGEdge, *, enqueue: bool = False, worklist: deque[int] | None = None) -> None:
        transfer_edges.append(edge)
        edge_buckets[edge.src].append(edge)
        if enqueue and edge.dst is not None and worklist is not None:
            worklist.append(edge.dst)

    def add_reference_edge(edge: CFGEdge) -> None:
        reference_edges.append(edge)
        edge_buckets[edge.src].append(edge)

    def enqueue_region(root: int, kind: str) -> None:
        if 0 <= root < body_size and root not in seen_region_roots:
            seen_region_roots.add(root)
            pending_regions.append((root, kind))

    if body_size:
        enqueue_region(0, "entry")
    else:
        issues.append(CFGIssue(severity="error", code="empty_function_body", message="No bytes available for function body."))

    while pending_regions:
        region_root, region_kind = pending_regions.popleft()
        worklist: deque[int] = deque([region_root])
        while worklist:
            offset = worklist.popleft()
            if offset in nodes_raw:
                continue
            if offset < 0 or offset >= body_size:
                issues.append(
                    CFGIssue(
                        severity="error",
                        code="entry_offset_out_of_range",
                        offset=offset,
                        message=f"CFG entry offset {offset} is outside function body size {body_size}.",
                    )
                )
                continue

            try:
                word = decode_word_at(body, offset, limit=body_size, index=len(nodes_raw))
            except Exception as exc:
                issues.append(CFGIssue("error", "decode_error", offset, message=str(exc)))
                continue

            nodes_raw[offset] = word
            node_region_root[offset] = region_root
            node_region_kind[offset] = region_kind

            if word.terminal_kind == "UNKNOWN":
                issues.append(
                    CFGIssue(
                        severity="error",
                        code="reachable_unknown_word",
                        offset=offset,
                        message=f"Reachable byte 0x{body[offset]:02X} decoded as UNKNOWN.",
                        details={"byte": body[offset], "region_root": region_root, "region_kind": region_kind},
                    )
                )

            next_offset = offset + max(1, int(word.size))
            if next_offset > body_size:
                issues.append(
                    CFGIssue(
                        severity="error",
                        code="word_overruns_body",
                        offset=offset,
                        message=f"Decoded word reaches {next_offset}, beyond function body size {body_size}.",
                        details=word_to_dict(word),
                    )
                )
                continue

            if word.terminal_kind in RETURN_TERMINALS:
                continue

            if word.terminal_kind == BRANCH_TERMINAL:
                op = int(word.operands.get("op", -1) or -1) & 0xFF
                op_hex = f"0x{op:02X}"
                if is_control_branch_word(word):
                    try:
                        target = branch_target_offset(word)
                    except Exception as exc:
                        issues.append(CFGIssue("error", "branch_target_resolution_error", offset, message=str(exc), details=word_to_dict(word)))
                        target = None

                    if target is not None:
                        relation = classify_entry_offset(target, body_size, span_map, linear_starts)
                        target_word = _decode_target_word(body, target)
                        target_kind = target_word.kind if target_word is not None else None
                        cfg_branch_relations[relation.relation] += 1
                        valid_branch_entry = _validate_branch_target(
                            body=body,
                            body_size=body_size,
                            word=word,
                            target=target,
                            relation=relation,
                            target_word=target_word,
                            target_kind=target_kind,
                            op_hex=op_hex,
                            issues=issues,
                            issue_prefix="",
                        )
                        edge = CFGEdge(
                            src=offset,
                            dst=target if 0 <= target < body_size else None,
                            kind="branch_conditional",
                            op=op,
                            op_hex=op_hex,
                            target_relation=relation.relation,
                            target_decoded_kind=target_kind,
                        )
                        add_transfer_edge(edge, enqueue=valid_branch_entry, worklist=worklist)

                    if op in CONDITIONAL_CONTROL_BRANCH_OPS and next_offset < body_size:
                        relation = classify_entry_offset(next_offset, body_size, span_map, linear_starts)
                        edge = CFGEdge(
                            src=offset,
                            dst=next_offset,
                            kind="fallthrough",
                            op=op,
                            op_hex=op_hex,
                            target_relation=relation.relation,
                            target_decoded_kind=_decode_target_kind(body, next_offset),
                        )
                        add_transfer_edge(edge, enqueue=True, worklist=worklist)
                else:
                    if next_offset < body_size:
                        relation = classify_entry_offset(next_offset, body_size, span_map, linear_starts)
                        edge = CFGEdge(
                            src=offset,
                            dst=next_offset,
                            kind="predicate_no_transfer_next",
                            op=op,
                            op_hex=op_hex,
                            target_relation=relation.relation,
                            target_decoded_kind=_decode_target_kind(body, next_offset),
                            note="BR-shaped predicate; encoded target is fallthrough, so it is not a CFG branch.",
                        )
                        add_transfer_edge(edge, enqueue=True, worklist=worklist)
                continue

            if word.terminal_kind == "CODE_REF":
                try:
                    target = code_ref_target_offset(word)
                except Exception as exc:
                    issues.append(CFGIssue("error", "code_ref_target_resolution_error", offset, message=str(exc), details=word_to_dict(word)))
                    target = None
                if target is not None:
                    relation = classify_entry_offset(target, body_size, span_map, linear_starts)
                    target_kind = _decode_target_kind(body, target)
                    if target < 0 or target >= body_size:
                        issues.append(
                            CFGIssue(
                                "error",
                                "code_ref_target_out_of_range",
                                offset,
                                target,
                                f"CODE_REF target {target} is outside function body size {body_size}.",
                                {"relation": relation.to_dict(), "region_root": region_root, "region_kind": region_kind},
                            )
                        )
                    elif _target_decodes_unknown(target_kind):
                        issues.append(
                            CFGIssue(
                                "error",
                                "code_ref_target_decodes_unknown",
                                offset,
                                target,
                                "CODE_REF target is in-range, but decoding at the exact target is UNKNOWN.",
                                {"relation": relation.to_dict(), "target_decoded_kind": target_kind},
                            )
                        )
                    elif relation.relation != "linear_boundary":
                        issues.append(
                            CFGIssue(
                                "warning",
                                "code_ref_target_not_linear_boundary",
                                offset,
                                target,
                                "CODE_REF target is valid, but it is not a top-level linear word boundary.",
                                {"relation": relation.to_dict(), "target_decoded_kind": target_kind},
                            )
                        )
                    edge = CFGEdge(
                        src=offset,
                        dst=target if 0 <= target < body_size else None,
                        kind="code_ref_reference",
                        target_relation=relation.relation,
                        target_decoded_kind=target_kind,
                        note="CODE_REF is a local code-island reference, not a control-transfer edge.",
                    )
                    add_reference_edge(edge)
                    if 0 <= target < body_size and not _target_decodes_unknown(target_kind):
                        code_ref_region_roots.add(target)
                        enqueue_region(target, "code_ref")

            if next_offset < body_size:
                crosses_declared_code_ref_root = (
                    region_kind == "code_ref"
                    and next_offset in declared_code_ref_roots
                    and next_offset != region_root
                )
                if crosses_declared_code_ref_root:
                    code_ref_region_boundary_count += 1
                else:
                    relation = classify_entry_offset(next_offset, body_size, span_map, linear_starts)
                    edge = CFGEdge(
                        src=offset,
                        dst=next_offset,
                        kind="next",
                        target_relation=relation.relation,
                        target_decoded_kind=_decode_target_kind(body, next_offset),
                    )
                    add_transfer_edge(edge, enqueue=True, worklist=worklist)

    nodes: dict[int, CFGNode] = {}
    for offset, word in nodes_raw.items():
        nodes[offset] = CFGNode(
            offset=offset,
            word=word,
            relation=classify_entry_offset(offset, body_size, span_map, linear_starts),
            edges=list(edge_buckets.get(offset, [])),
            region_root=node_region_root.get(offset, 0),
            region_kind=node_region_kind.get(offset, "entry"),
        )

    covered: set[int] = set()
    entry_covered: set[int] = set()
    for offset, word in nodes_raw.items():
        byte_range = range(offset, min(body_size, offset + max(1, int(word.size))))
        covered.update(byte_range)
        if node_region_kind.get(offset) == "entry":
            entry_covered.update(byte_range)
    uncovered_ranges = _ranges_from_covered(body_size, covered)
    if uncovered_ranges:
        issues.append(
            CFGIssue(
                severity="note",
                code="cfg_uncovered_bytes",
                message="Some bytes are present in the function span but are not covered by entry or CODE_REF regions.",
                details={"ranges": uncovered_ranges[:16], "uncovered_byte_count": body_size - len(covered)},
            )
        )

    return FunctionCFG(
        function_name=function_name,
        symbol=symbol,
        source_kind=source_kind,
        definition_index=definition_index,
        span=span,
        body_size=body_size,
        linear_words=linear_words,
        nodes=nodes,
        edges=transfer_edges,
        reference_edges=reference_edges,
        issues=issues,
        branch_target_relations=linear_branch_relations,
        cfg_branch_target_relations=cfg_branch_relations,
        code_ref_target_relations=code_ref_relations,
        call_script_target_relations=Counter(),
        call_script_source_relations=Counter(),
        branch_count_linear=branch_count_linear,
        control_branch_count_linear=control_branch_count_linear,
        predicate_no_transfer_count=predicate_no_transfer_count,
        code_ref_count_linear=code_ref_count_linear,
        call_script_count_linear=call_script_count_linear,
        code_ref_region_roots=sorted(code_ref_region_roots),
        cfg_covered_bytes=len(covered),
        cfg_uncovered_ranges=uncovered_ranges,
        entry_covered_bytes=len(entry_covered),
        code_ref_region_boundary_count=code_ref_region_boundary_count,
    )


def _classify_call_script_absolute_target(module: MBCModule, target: int) -> tuple[str, dict[str, Any]]:
    code_size = module.get_real_code_size()
    if target < 0 or target >= code_size:
        return "out_of_range", {"code_size": code_size}

    slot = module.initial_slot_for_offset(target)
    if slot is not None:
        return "initial_slot" if slot.valid else "invalid_initial_slot", {"slot": slot.to_dict()}

    definition_names = [rec.name for rec in module.definitions if int(rec.a) == target]
    if definition_names:
        return "definition_entry", {"definitions": definition_names}

    return "code_offset", {"code_size": code_size}


def _validate_call_script_targets(module: MBCModule, cfg: FunctionCFG) -> None:
    """Validate every reachable CALL_SCRIPT entry in the region CFG.

    Linear CALL_SCRIPT words are still counted separately by
    ``call_script_count_linear``.  Validation must use CFG nodes, not only the
    linear stream, because legal branch suffix entries can expose prefixed
    CALL_SCRIPT atoms at prefix-byte coordinates.
    """

    if cfg.span is None:
        return
    function_start = int(cfg.span[0])
    cfg.call_script_target_relations.clear()
    cfg.call_script_source_relations.clear()

    for node in sorted(cfg.nodes.values(), key=lambda n: n.offset):
        word = node.word
        if word.terminal_kind != "CALL_SCRIPT":
            continue
        local_target = call_script_target_offset(word)
        absolute_target = function_start + local_target
        relation, details = _classify_call_script_absolute_target(module, absolute_target)
        source_relation = node.relation.relation
        source_kind = "linear" if source_relation == "linear_boundary" else "subentry"
        cfg.call_script_target_relations[relation] += 1
        cfg.call_script_source_relations[f"{source_kind}:{source_relation}"] += 1
        details = dict(details)
        details.update({
            "local_target": local_target,
            "absolute_target": absolute_target,
            "function_start": function_start,
            "source_kind": source_kind,
            "source_relation": source_relation,
            "region_root": node.region_root,
            "region_kind": node.region_kind,
            "word": word_to_dict(word),
        })
        if relation == "out_of_range":
            cfg.issues.append(
                CFGIssue(
                    "error",
                    "call_script_target_out_of_range",
                    word.offset,
                    absolute_target,
                    "CALL_SCRIPT absolute target is outside module code.",
                    details,
                )
            )
        elif relation == "invalid_initial_slot":
            cfg.issues.append(
                CFGIssue(
                    "error",
                    "call_script_target_invalid_initial_slot",
                    word.offset,
                    absolute_target,
                    "CALL_SCRIPT target points at an initial slot whose bytes/table record are invalid.",
                    details,
                )
            )
        elif relation == "code_offset":
            cfg.issues.append(
                CFGIssue(
                    "warning",
                    "call_script_target_not_definition_or_initial_slot",
                    word.offset,
                    absolute_target,
                    "CALL_SCRIPT target is inside module code but not at a definition entry or valid initial slot.",
                    details,
                )
            )

def _empty_cfg_for_unavailable_span(entry_name: str, symbol: str, source_kind: str, definition_index: Optional[int], issues: list[CFGIssue]) -> FunctionCFG:
    return FunctionCFG(
        function_name=entry_name,
        symbol=symbol,
        source_kind=source_kind,
        definition_index=definition_index,
        span=None,
        body_size=0,
        linear_words=[],
        nodes={},
        edges=[],
        reference_edges=[],
        issues=issues,
        branch_target_relations=Counter(),
        cfg_branch_target_relations=Counter(),
        code_ref_target_relations=Counter(),
        call_script_target_relations=Counter(),
        call_script_source_relations=Counter(),
        branch_count_linear=0,
        control_branch_count_linear=0,
        predicate_no_transfer_count=0,
        code_ref_count_linear=0,
        call_script_count_linear=0,
        code_ref_region_roots=[],
        cfg_covered_bytes=0,
        cfg_uncovered_ranges=[],
        entry_covered_bytes=0,
        code_ref_region_boundary_count=0,
    )


def analyze_module(module: MBCModule) -> ModuleCFGReport:
    function_cfgs: list[FunctionCFG] = []
    for entry in module.function_entries(include_definitions=True, include_exports=True, dedupe=True):
        span, reason = module.get_function_exact_code_span_with_reason(entry)
        body = module.get_function_body(entry)
        if span is None:
            issues = [
                CFGIssue(
                    severity="error",
                    code="function_span_unavailable",
                    message=f"Function body span unavailable: {reason}",
                    details=entry.to_dict(),
                )
            ]
            cfg = _empty_cfg_for_unavailable_span(entry.name, entry.symbol, entry.source_kind, entry.definition_index, issues)
        else:
            cfg = build_function_cfg(
                body,
                function_name=entry.name,
                symbol=entry.symbol,
                source_kind=entry.source_kind,
                definition_index=entry.definition_index,
                span=span,
            )
            _validate_call_script_targets(module, cfg)
        function_cfgs.append(cfg)

    return ModuleCFGReport(
        module_path=str(module.path),
        parser_contract=module.parser_contract,
        layout_diagnostics=module.layout_diagnostics,
        definition_count=len(module.definitions),
        export_count=len(module.exports),
        global_count=len(module.globals),
        function_count=len(function_cfgs),
        function_cfgs=function_cfgs,
    )
