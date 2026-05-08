from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .parser import FunctionEntry, MBCModule
from .vm_spec import (
    CONDITIONAL_CONTROL_BRANCH_OPS,
    VMWord,
    branch_operand_base_offset,
    branch_target_offset,
    code_ref_target_offset,
    decode_word_at,
    decode_words,
    is_control_branch_word,
    signed_u16,
    terminal_atom_offset,
    word_role,
)

RETURN_TERMINALS = {"RETURN_PAIR", "END"}
BRANCH_TERMINAL = "BR"
UNCONDITIONAL_BRANCH_OP = None  # all BR opcodes are modeled with fallthrough at CFG level
NORMAL_TARGET_RELATIONS = {"linear_boundary", "prefix_byte_entry", "terminal_atom_entry"}
OVERLAP_TARGET_RELATIONS = {
    "aggregate_overlap_entry",
    "bare_u32_overlap_entry",
    "literal_payload_overlap_entry",
    "operand_payload_overlap_entry",
    "payload_overlap_entry",
}

STRUCTURAL_TERMINALS = {"MARK", "NOP"}
IMPORTANT_ROLES = {"branch", "call", "return", "unknown", "predicate_no_transfer"}
FORMAT_MODES = ("flow", "branches", "full")


@dataclass(frozen=True)
class EntryRelation:
    """How a byte offset relates to the default linear tokenization."""

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CFGNode:
    offset: int
    word: VMWord
    relation: EntryRelation
    edges: list[CFGEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "word": word_to_dict(self.word),
            "relation": self.relation.to_dict(),
            "edges": [edge.to_dict() for edge in self.edges],
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
    issues: list[CFGIssue]
    branch_target_relations: Counter[str]
    branch_count_linear: int
    control_branch_count_linear: int
    predicate_no_transfer_count: int
    cfg_covered_bytes: int
    cfg_uncovered_ranges: list[tuple[int, int]]

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
    def has_control_interest(self) -> bool:
        return bool(self.control_branch_count_linear or self.hard_error_count or self.warning_count or self.overlap_target_count)

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
            "edge_count": len(self.edges),
            "branch_count_linear": self.branch_count_linear,
            "control_branch_count_linear": self.control_branch_count_linear,
            "predicate_no_transfer_count": self.predicate_no_transfer_count,
            "branch_target_relations": dict(self.branch_target_relations),
            "overlap_target_count": self.overlap_target_count,
            "hard_error_count": self.hard_error_count,
            "warning_count": self.warning_count,
            "note_count": self.note_count,
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
        target_relations: Counter[str] = Counter()
        issue_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        functions_with_branches = 0
        functions_with_errors = 0
        functions_with_overlap_targets = 0
        functions_with_uncovered_bytes = 0

        for cfg in self.function_cfgs:
            target_relations.update(cfg.branch_target_relations)
            totals["body_bytes"] += cfg.body_size
            totals["linear_words"] += len(cfg.linear_words)
            totals["cfg_nodes"] += len(cfg.nodes)
            totals["cfg_edges"] += len(cfg.edges)
            totals["linear_branches"] += cfg.branch_count_linear
            totals["control_branches"] += cfg.control_branch_count_linear
            totals["predicate_no_transfer_branches"] += cfg.predicate_no_transfer_count
            totals["cfg_uncovered_bytes"] += cfg.body_size - cfg.cfg_covered_bytes
            if cfg.control_branch_count_linear:
                functions_with_branches += 1
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
            "functions_with_hard_errors": functions_with_errors,
            "functions_with_overlap_targets": functions_with_overlap_targets,
            "functions_with_uncovered_bytes": functions_with_uncovered_bytes,
            "totals": dict(totals),
            "branch_target_relations": dict(target_relations),
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

    relation: str
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


def _decode_target_kind(body: bytes, target: int) -> str | None:
    if target < 0 or target >= len(body):
        return None
    try:
        return decode_word_at(body, target, limit=len(body)).kind
    except Exception:
        return None


def _signed_branch_off(word: VMWord) -> int:
    return signed_u16(int(word.operands.get("off", 0) or 0))


def build_function_cfg(
    body: bytes,
    *,
    function_name: str = "<anonymous>",
    symbol: str = "<anonymous>",
    source_kind: str = "definition",
    definition_index: Optional[int] = None,
    span: Optional[tuple[int, int]] = None,
) -> FunctionCFG:
    """Build a byte-entry CFG for one function body.

    The important bit is that branch targets are decoded from their exact byte
    offset.  This intentionally permits branch targets into prefix bytes, terminal
    atoms, and overlap entries inside a linear fused word.
    """

    body_size = len(body)
    linear_words = decode_words(body)
    span_map, linear_starts = _build_linear_span_map(linear_words)
    branch_count_linear = sum(1 for word in linear_words if word.terminal_kind == BRANCH_TERMINAL)
    control_branch_count_linear = sum(1 for word in linear_words if is_control_branch_word(word))
    predicate_no_transfer_count = branch_count_linear - control_branch_count_linear

    nodes_raw: dict[int, VMWord] = {}
    edge_buckets: defaultdict[int, list[CFGEdge]] = defaultdict(list)
    edges: list[CFGEdge] = []
    issues: list[CFGIssue] = []
    branch_target_relations: Counter[str] = Counter()

    if body_size:
        worklist: deque[int] = deque([0])
    else:
        worklist = deque()
        issues.append(CFGIssue(severity="error", code="empty_function_body", message="No bytes available for function body."))

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
        except Exception as exc:  # pragma: no cover - defensive decoder boundary
            issues.append(
                CFGIssue(
                    severity="error",
                    code="decode_error",
                    offset=offset,
                    message=str(exc),
                )
            )
            continue

        nodes_raw[offset] = word
        if word.terminal_kind == "UNKNOWN":
            issues.append(
                CFGIssue(
                    severity="error",
                    code="reachable_unknown_word",
                    offset=offset,
                    message=f"Reachable byte 0x{body[offset]:02X} decoded as UNKNOWN.",
                    details={"byte": body[offset]},
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
                except Exception as exc:  # pragma: no cover - defensive decoder boundary
                    issues.append(
                        CFGIssue(
                            severity="error",
                            code="branch_target_resolution_error",
                            offset=offset,
                            message=str(exc),
                            details=word_to_dict(word),
                        )
                    )
                    target = None

                if target is not None:
                    relation = classify_entry_offset(target, body_size, span_map, linear_starts)
                    branch_target_relations[relation.relation] += 1
                    target_kind = _decode_target_kind(body, target)
                    if target < 0 or target >= body_size:
                        issues.append(
                            CFGIssue(
                                severity="error",
                                code="branch_target_out_of_range",
                                offset=offset,
                                target=target,
                                message=f"Branch target {target} is outside function body size {body_size}.",
                                details={"op": op_hex, "relation": relation.to_dict()},
                            )
                        )
                    elif target_kind is None or target_kind.endswith("UNKNOWN"):
                        issues.append(
                            CFGIssue(
                                severity="error",
                                code="branch_target_decodes_unknown",
                                offset=offset,
                                target=target,
                                message="Branch target is in-range, but decoding at the exact target is UNKNOWN.",
                                details={"op": op_hex, "relation": relation.to_dict(), "target_decoded_kind": target_kind},
                            )
                        )
                    elif relation.is_overlap:
                        issues.append(
                            CFGIssue(
                                severity="note",
                                code="branch_target_overlap_entry",
                                offset=offset,
                                target=target,
                                message="Branch target lands inside a linear fused word but decodes as a valid byte/sub-entry word.",
                                details={"op": op_hex, "relation": relation.to_dict(), "target_decoded_kind": target_kind},
                            )
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
                    edges.append(edge)
                    edge_buckets[offset].append(edge)
                    if 0 <= target < body_size:
                        worklist.append(target)

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
                    edges.append(edge)
                    edge_buckets[offset].append(edge)
                    worklist.append(next_offset)
            else:
                relation = classify_entry_offset(next_offset, body_size, span_map, linear_starts) if next_offset <= body_size else EntryRelation(next_offset, "out_of_range")
                if next_offset < body_size:
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
                    edges.append(edge)
                    edge_buckets[offset].append(edge)
                    worklist.append(next_offset)
            continue

        if word.terminal_kind == "CODE_REF":
            try:
                target = code_ref_target_offset(word)
            except Exception as exc:  # pragma: no cover - defensive decoder boundary
                issues.append(
                    CFGIssue(
                        severity="error",
                        code="code_ref_target_resolution_error",
                        offset=offset,
                        message=str(exc),
                        details=word_to_dict(word),
                    )
                )
                target = None
            if target is not None:
                relation = classify_entry_offset(target, body_size, span_map, linear_starts)
                target_kind = _decode_target_kind(body, target)
                if target < 0 or target >= body_size:
                    issues.append(
                        CFGIssue(
                            severity="error",
                            code="code_ref_target_out_of_range",
                            offset=offset,
                            target=target,
                            message=f"CODE_REF target {target} is outside function body size {body_size}.",
                            details={"relation": relation.to_dict()},
                        )
                    )
                elif target_kind is None or target_kind.endswith("UNKNOWN"):
                    issues.append(
                        CFGIssue(
                            severity="error",
                            code="code_ref_target_decodes_unknown",
                            offset=offset,
                            target=target,
                            message="CODE_REF target is in-range, but decoding at the exact target is UNKNOWN.",
                            details={"relation": relation.to_dict(), "target_decoded_kind": target_kind},
                        )
                    )
                edge = CFGEdge(
                    src=offset,
                    dst=target if 0 <= target < body_size else None,
                    kind="code_ref",
                    target_relation=relation.relation,
                    target_decoded_kind=target_kind,
                )
                edges.append(edge)
                edge_buckets[offset].append(edge)
                if 0 <= target < body_size:
                    worklist.append(target)

        if next_offset < body_size:
            relation = classify_entry_offset(next_offset, body_size, span_map, linear_starts)
            edge = CFGEdge(
                src=offset,
                dst=next_offset,
                kind="next",
                target_relation=relation.relation,
                target_decoded_kind=_decode_target_kind(body, next_offset),
            )
            edges.append(edge)
            edge_buckets[offset].append(edge)
            worklist.append(next_offset)

    nodes: dict[int, CFGNode] = {}
    for offset, word in nodes_raw.items():
        nodes[offset] = CFGNode(
            offset=offset,
            word=word,
            relation=classify_entry_offset(offset, body_size, span_map, linear_starts),
            edges=list(edge_buckets.get(offset, [])),
        )

    covered: set[int] = set()
    for offset, word in nodes_raw.items():
        covered.update(range(offset, min(body_size, offset + max(1, int(word.size)))))
    uncovered_ranges = _ranges_from_covered(body_size, covered)
    if uncovered_ranges:
        issues.append(
            CFGIssue(
                severity="note",
                code="cfg_uncovered_bytes",
                message="Some bytes are present in the function span but are not reached from CFG entry 0.",
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
        edges=edges,
        issues=issues,
        branch_target_relations=branch_target_relations,
        branch_count_linear=branch_count_linear,
        control_branch_count_linear=control_branch_count_linear,
        predicate_no_transfer_count=predicate_no_transfer_count,
        cfg_covered_bytes=len(covered),
        cfg_uncovered_ranges=uncovered_ranges,
    )


def analyze_module(module: MBCModule) -> ModuleCFGReport:
    function_cfgs: list[FunctionCFG] = []
    for entry in module.function_entries(include_definitions=True, include_exports=True, dedupe=True):
        span, reason = module.get_function_exact_code_span_with_reason(entry)
        body = module.get_function_body(entry)
        issues: list[CFGIssue] = []
        if span is None:
            issues.append(
                CFGIssue(
                    severity="error",
                    code="function_span_unavailable",
                    message=f"Function body span unavailable: {reason}",
                    details=entry.to_dict(),
                )
            )
            cfg = FunctionCFG(
                function_name=entry.name,
                symbol=entry.symbol,
                source_kind=entry.source_kind,
                definition_index=entry.definition_index,
                span=None,
                body_size=0,
                linear_words=[],
                nodes={},
                edges=[],
                issues=issues,
                branch_target_relations=Counter(),
                branch_count_linear=0,
                control_branch_count_linear=0,
                predicate_no_transfer_count=0,
                cfg_covered_bytes=0,
                cfg_uncovered_ranges=[],
            )
        else:
            cfg = build_function_cfg(
                body,
                function_name=entry.name,
                symbol=entry.symbol,
                source_kind=entry.source_kind,
                definition_index=entry.definition_index,
                span=span,
            )
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


def resolve_mbc_path(target: str | Path, *, mbc_dir: str | Path = "mbc") -> Path:
    target_path = Path(target)
    candidates: list[Path] = []
    if target_path.suffix.lower() == ".mbc":
        candidates.append(target_path)
        candidates.append(Path(mbc_dir) / target_path.name)
    else:
        candidates.append(target_path.with_suffix(".mbc"))
        candidates.append(Path(mbc_dir) / f"{target_path.name}.mbc")
        candidates.append(Path(mbc_dir) / target_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot find target MBC script {target!r}. Tried: {joined}")


def _hex_raw(raw: bytes, *, max_bytes: int = 18) -> str:
    text = raw[:max_bytes].hex(" ")
    if len(raw) > max_bytes:
        text += " ..."
    return text


def _format_value(value: Any) -> str:
    if isinstance(value, int):
        if value < 0:
            return str(value)
        return f"{value}"
    return str(value)


def format_word_concise(word: VMWord) -> str:
    k = word.terminal_kind
    op = word.operands.get("op")
    prefix = ""
    if word.prefixes:
        prefix = " pfx=" + ".".join(f"{int(p):02X}" for p in word.prefixes)

    if k == "BR":
        op_int = int(word.operands.get("op", -1) or -1) & 0xFF
        raw_off = int(word.operands.get("off", 0) or 0)
        return f"{word.kind}{prefix} op=0x{op_int:02X} off={signed_u16(raw_off)} base={branch_operand_base_offset(word)}"
    if k == "CALL_NATIVE":
        return f"{word.kind}{prefix} argc={word.operands.get('argc')} opid=0x{int(word.operands.get('opid', 0) or 0):02X}"
    if k == "CALL_SCRIPT":
        return f"{word.kind}{prefix} argc={word.operands.get('argc')} rel={word.operands.get('rel')}"
    if k in {"AGG", "AGG0"}:
        children = word.operands.get("children") or []
        child_text = ",".join(f"{ch.get('tag')}:{ch.get('ref')}" for ch in children[:6])
        if len(children) > 6:
            child_text += ",..."
        return f"{word.kind} arity={word.operands.get('arity')} children=[{child_text}]"
    if k in {"REF", "REF16"}:
        return f"{word.kind}{prefix} op=0x{int(op or 0):02X} mode=0x{int(word.operands.get('mode', 0) or 0):02X} ref={word.operands.get('ref')}"
    if k in {"REC41", "REC61", "REC62"}:
        keys = ["mode", "u16", "ref", "imm", "a", "b", "c"]
        body = " ".join(f"{key}={_format_value(word.operands[key])}" for key in keys if key in word.operands)
        return f"{word.kind}{prefix} {body}".rstrip()
    if k == "CODE_REF":
        return f"{word.kind}{prefix} rel={word.operands.get('rel')}"
    if k == "BARE_U32":
        return f"{word.kind} value={word.operands.get('value')} follower={word.operands.get('follower_kind')}"
    if k in {"IMM8", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32", "F32", "U16"}:
        if k == "F32":
            return f"{word.kind}{prefix} value={word.operands.get('value')} bits=0x{int(word.operands.get('bits', 0) or 0):08X}"
        return f"{word.kind}{prefix} op={f'0x{int(op):02X}' if op is not None else '-'} value={word.operands.get('value')}"
    if k == "RETURN_PAIR":
        return f"{word.kind} return_pair"
    if k == "END":
        return f"{word.kind} end"
    if k == "MARK":
        return f"{word.kind}{prefix} op=0x{int(op or word.operands.get('byte', 0) or 0):02X}"
    if k == "NOP":
        return f"{word.kind}"
    if k == "UNKNOWN":
        return f"{word.kind} byte=0x{int(word.operands.get('byte', 0) or 0):02X}"
    return f"{word.kind}{prefix}"


def _format_edge(edge: CFGEdge) -> str:
    dst = "EXIT" if edge.dst is None else f"L{edge.dst:04X}"
    if edge.kind == "next":
        return f"next {dst}"
    if edge.kind == "fallthrough":
        return f"fallthrough {dst}"
    if edge.kind == "predicate_no_transfer_next":
        return f"predicate-no-transfer {dst}"
    relation = f" [{edge.target_relation}]" if edge.target_relation else ""
    target_kind = f" decodes={edge.target_decoded_kind}" if edge.target_decoded_kind else ""
    return f"{edge.kind} -> {dst}{relation}{target_kind}"


def _module_summary_lines(report: ModuleCFGReport) -> list[str]:
    summary = report.summary
    totals = summary["totals"]
    lines = [
        f"# module: {report.module_path}",
        f"# parser_contract: {report.parser_contract}",
        f"# defs={report.definition_count} exports={report.export_count} globals={report.global_count} functions={report.function_count}",
        (
            "# cfg: "
            f"branch_functions={summary['functions_with_control_branches']} "
            f"linear_branches={totals.get('linear_branches', 0)} "
            f"control_branches={totals.get('control_branches', 0)} "
            f"nodes={totals.get('cfg_nodes', 0)} edges={totals.get('cfg_edges', 0)}"
        ),
        f"# target_relations: {dict(summary['branch_target_relations'])}",
        f"# issues: severity={dict(summary['severity_counts'])} codes={dict(summary['issue_counts'])}",
        "# note: overlap target = branch lands inside a linear fused word, but exact byte target decodes as a valid CFG entry.",
        "",
    ]
    return lines


def _should_show_formatted_word(
    word: VMWord,
    *,
    mode: str,
    has_label: bool,
    has_non_next_edges: bool,
    is_alt: bool,
) -> bool:
    if mode not in FORMAT_MODES:
        raise ValueError(f"unknown format mode: {mode!r}")
    if mode == "full":
        return True

    role = word_role(word)
    if mode == "branches":
        return has_label or has_non_next_edges or role in IMPORTANT_ROLES

    # flow mode keeps the readable control-flow stream while suppressing plain
    # structural padding.  Structural entries are still shown when they are CFG
    # labels, branch targets, or non-linear alternate entries.
    if word.terminal_kind in STRUCTURAL_TERMINALS and not (has_label or has_non_next_edges or is_alt):
        return False
    return True


def format_function_cfg(cfg: FunctionCFG, *, mode: str = "flow") -> str:
    label_sources: defaultdict[int, list[str]] = defaultdict(list)
    for edge in cfg.edges:
        if edge.dst is not None and edge.kind != "next":
            label_sources[edge.dst].append(f"from {edge.src:04X}:{edge.kind}:{edge.target_relation}")
    label_sources[0].append("entry")

    display_offsets = {int(word.offset) for word in cfg.linear_words}
    display_offsets.update(cfg.nodes.keys())
    sorted_offsets = sorted(display_offsets)

    relation_counts = dict(cfg.branch_target_relations)
    uncovered_count = cfg.body_size - cfg.cfg_covered_bytes
    span_text = "-" if cfg.span is None else f"{cfg.span[0]}..{cfg.span[1]}"
    lines = [
        f"== {cfg.function_name}  symbol={cfg.symbol} source={cfg.source_kind} def_index={cfg.definition_index} span={span_text} bytes={cfg.body_size}",
        (
            f"   words={len(cfg.linear_words)} cfg_nodes={len(cfg.nodes)} edges={len(cfg.edges)} "
            f"branches={cfg.control_branch_count_linear}/{cfg.branch_count_linear} "
            f"target_relations={relation_counts} errors={cfg.hard_error_count} notes={cfg.note_count} uncovered={uncovered_count}"
        ),
    ]

    interesting_issues = [issue for issue in cfg.issues if issue.code != "cfg_uncovered_bytes"]
    for issue in interesting_issues[:12]:
        target = "" if issue.target is None else f" target={issue.target:04X}"
        offset = "" if issue.offset is None else f" @{issue.offset:04X}"
        lines.append(f"   ! {issue.severity}:{issue.code}{offset}{target} {issue.message}")
    if len(interesting_issues) > 12:
        lines.append(f"   ! ... {len(interesting_issues) - 12} more issues omitted")

    linear_by_offset = {int(word.offset): word for word in cfg.linear_words}
    span_map, linear_starts = _build_linear_span_map(cfg.linear_words)
    skipped = 0
    for offset in sorted_offsets:
        word = cfg.nodes[offset].word if offset in cfg.nodes and offset not in linear_by_offset else linear_by_offset[offset]
        is_alt = offset in cfg.nodes and offset not in linear_by_offset
        node = cfg.nodes.get(offset)
        labels = label_sources.get(offset, [])
        non_next_edges = [edge for edge in node.edges if edge.kind != "next"] if node and node.edges else []
        if not _should_show_formatted_word(
            word,
            mode=mode,
            has_label=bool(labels),
            has_non_next_edges=bool(non_next_edges),
            is_alt=is_alt,
        ):
            skipped += 1
            continue
        if skipped:
            lines.append(f"      ... {skipped} structural entries omitted")
            skipped = 0
        label_text = f"L{offset:04X}:" if labels else "      "
        alt_text = "@ALT " if is_alt else "     "
        raw = _hex_raw(word.raw)
        raw_pad = raw.ljust(54)
        relation = node.relation.relation if node is not None else classify_entry_offset(offset, cfg.body_size, span_map, linear_starts).relation
        relation_text = f" [{relation}]" if is_alt or relation != "linear_boundary" else ""
        edge_text = ""
        if non_next_edges:
            edge_text = "  ; " + " | ".join(_format_edge(edge) for edge in non_next_edges)
        source_text = ""
        if labels:
            source_text = "  ; labels=" + ", ".join(labels[:4])
            if len(labels) > 4:
                source_text += ", ..."
        lines.append(
            f"{label_text} {alt_text}{offset:04X}-{offset + word.size:04X} {raw_pad} {format_word_concise(word)}{relation_text}{edge_text}{source_text}"
        )
    if skipped:
        lines.append(f"      ... {skipped} structural entries omitted")
    lines.append("")
    return "\n".join(lines)


def format_module_report(
    report: ModuleCFGReport,
    *,
    include_all_functions: bool = False,
    include_functions_without_branches: bool = False,
    mode: str = "flow",
) -> str:
    lines = _module_summary_lines(report)
    selected: list[FunctionCFG] = []
    for cfg in report.function_cfgs:
        if include_all_functions:
            selected.append(cfg)
        elif cfg.has_control_interest:
            selected.append(cfg)
        elif include_functions_without_branches and cfg.branch_count_linear == 0:
            selected.append(cfg)

    lines.append(f"# emitted_functions={len(selected)} mode={mode} (use include_all_functions=True for a full token stream)\n")
    for cfg in selected:
        lines.append(format_function_cfg(cfg, mode=mode))
    return "\n".join(lines)
