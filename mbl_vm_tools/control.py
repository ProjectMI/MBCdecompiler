from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .vm_spec import (
    VMWord,
    branch_operand_base_offset,
    branch_target_offset,
    decode_word_at,
    signed_u16,
    terminal_atom_offset,
    word_role,
)
from .semantic import classify_branch


CONTROL_CONTRACT_VERSION = "vm-control-v3"
BRANCH_TARGET_FORMULA = "terminal_op_offset + 1 + signed_u16(off)"


@dataclass(frozen=True)
class BranchTargetCandidate:
    formula: str
    local_target: int
    absolute_target: int
    status: str
    confidence: float
    alignment_delta: int = 0
    base_formula: Optional[str] = None
    base_local_target: Optional[int] = None
    entry_kind: Optional[str] = None
    decoded_kind: Optional[str] = None
    decoder_rule: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BranchResolution:
    word_index: Optional[int]
    offset: int
    size: int
    op: int
    encoded_offset: int
    signed_offset: int
    prefixes_hex: list[str]
    status: str
    exact_candidates: list[BranchTargetCandidate] = field(default_factory=list)
    aligned_candidates: list[BranchTargetCandidate] = field(default_factory=list)
    selected_local_target: Optional[int] = None
    selected_absolute_target: Optional[int] = None
    selected_formula: Optional[str] = None
    terminal_op_offset: Optional[int] = None
    operand_base_offset: Optional[int] = None
    target_entry_kind: Optional[str] = None
    target_decoder_rule: Optional[str] = None
    semantic: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exact_candidates"] = [c.to_dict() for c in self.exact_candidates]
        payload["aligned_candidates"] = [c.to_dict() for c in self.aligned_candidates]
        return payload


@dataclass(frozen=True)
class VMControlEdge:
    source: str
    target: str
    kind: str
    status: str
    word_index: Optional[int] = None
    formula: Optional[str] = None
    local_target: Optional[int] = None
    alignment_delta: int = 0
    instruction_offset: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMControlBlock:
    id: str
    start_offset: int
    end_offset: int
    word_indices: list[int]
    instruction_offsets: list[int]
    successors: list[str]
    candidate_successors: list[str]
    terminator: Optional[dict[str, Any]] = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMControlGraph:
    contract: str
    blocks: list[VMControlBlock]
    edges: list[VMControlEdge]
    branch_resolutions: list[BranchResolution]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "blocks": [b.to_dict() for b in self.blocks],
            "edges": [e.to_dict() for e in self.edges],
            "branch_resolutions": [r.to_dict() for r in self.branch_resolutions],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Entry-coordinate model


def _stable_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _stable_unique_int(items: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _reconstruct_raw(words: list[VMWord]) -> bytes:
    if not words:
        return b""
    size = max(int(w.offset) + int(w.size) for w in words)
    data = bytearray(size)
    filled = bytearray(size)
    for word in words:
        start = int(word.offset)
        end = start + int(word.size)
        raw = bytes(word.raw)
        data[start:end] = raw[: max(0, end - start)]
        filled[start:end] = b"\x01" * max(0, end - start)
    # Linear decode should cover the whole selected function body.  If a caller
    # passed a sparse custom word list, leave missing bytes as zero instead of
    # inventing layout repairs here.
    return bytes(data)


def _add_entry_kind(kinds: dict[int, set[str]], offset: int, kind: str) -> None:
    if offset >= 0:
        kinds[offset].add(kind)


def entry_offset_kinds(words: list[VMWord]) -> dict[int, set[str]]:
    """Return byte offsets that are valid VM entry coordinates.

    Top-level linear word starts remain entries, but branch targets may also land
    on prefix-chain sub-entries or aggregate structural tail bytes.  This helper
    enumerates only VM-level entry coordinates; it does not infer source-level
    control constructs.
    """

    kinds: dict[int, set[str]] = defaultdict(set)
    for word in words:
        start = int(word.offset)
        _add_entry_kind(kinds, start, "word_start")

        # A prefixed word can be entered at any nested prefix, producing the
        # suffix of the original prefix chain, or at the terminal atom itself.
        for i in range(1, len(word.prefixes)):
            _add_entry_kind(kinds, start + i, "prefix_subentry")
        if word.prefixes:
            _add_entry_kind(kinds, terminal_atom_offset(word), "terminal_atom")

        # Aggregate prologues sometimes expose structural tail bytes as VM entry
        # points.  The byte-level decoder already knows what these bytes mean;
        # the control layer only records them as possible branch destinations.
        if word.terminal_kind in {"AGG", "AGG0"}:
            if "term2" in word.operands and word.size >= 2:
                _add_entry_kind(kinds, start + word.size - 2, "aggregate_tail")
                _add_entry_kind(kinds, start + word.size - 1, "aggregate_tail")
            elif "term" in word.operands and word.size >= 1:
                _add_entry_kind(kinds, start + word.size - 1, "aggregate_tail")
    return dict(kinds)


def _entry_kind_text(kinds: dict[int, set[str]], offset: int) -> Optional[str]:
    values = kinds.get(offset)
    if not values:
        return None
    return "|".join(sorted(values))


def _decode_valid_entry(raw: bytes, offset: int) -> VMWord | None:
    if offset < 0 or offset >= len(raw):
        return None
    try:
        word = decode_word_at(raw, offset)
    except Exception:
        return None
    if word.terminal_kind == "UNKNOWN":
        return None
    return word


# ---------------------------------------------------------------------------
# Branch resolution


def resolve_branch_target(
    word: VMWord,
    *,
    function_start: int,
    entry_kinds: Optional[dict[int, set[str]]] = None,
    raw: Optional[bytes] = None,
    valid_offsets: Optional[set[int]] = None,
    alignment_radius: int = 4,
    source_word_index: Optional[int] = None,
) -> BranchResolution:
    """Resolve a branch using the VM operand-base invariant.

    ``BR.off`` is a signed u16 displacement from the first operand byte of the
    terminal branch atom: ``terminal_op_offset + 1 + signed_u16(off)``.  The
    resolver does not snap to nearby word starts and does not preserve the old
    start/end based formulas as candidates.
    """

    raw_off = int(word.operands.get("off", 0) or 0) & 0xFFFF
    signed = signed_u16(raw_off)
    term_off = terminal_atom_offset(word)
    operand_base = branch_operand_base_offset(word)
    target = branch_target_offset(word)
    absolute = int(function_start) + int(target)
    entry_kinds = entry_kinds or ({off: {"word_start"} for off in (valid_offsets or set())})

    candidate: Optional[BranchTargetCandidate] = None
    status = "unresolved"
    target_entry_kind: Optional[str] = None
    target_decoder_rule: Optional[str] = None
    semantic: Optional[dict[str, Any]] = None

    known_kind = _entry_kind_text(entry_kinds, target)
    if known_kind is not None:
        status = "exact_entry"
        target_entry_kind = known_kind
        decoded_kind = None
        if raw is not None and 0 <= target < len(raw):
            decoded = _decode_valid_entry(raw, target)
            if decoded is not None:
                decoded_kind = decoded.terminal_kind
                target_decoder_rule = decoded.decoder_rule
        candidate = BranchTargetCandidate(
            formula=BRANCH_TARGET_FORMULA,
            local_target=target,
            absolute_target=absolute,
            status="exact_vm_entry",
            confidence=1.0,
            entry_kind=known_kind,
            decoded_kind=decoded_kind,
            decoder_rule=target_decoder_rule,
        )
    elif raw is not None and 0 <= target < len(raw):
        decoded = _decode_valid_entry(raw, target)
        if decoded is not None:
            status = "decoded_entry"
            target_entry_kind = "decoded_hidden_entry"
            target_decoder_rule = decoded.decoder_rule
            candidate = BranchTargetCandidate(
                formula=BRANCH_TARGET_FORMULA,
                local_target=target,
                absolute_target=absolute,
                status="decoded_vm_entry",
                confidence=1.0,
                entry_kind="decoded_hidden_entry",
                decoded_kind=decoded.terminal_kind,
                decoder_rule=decoded.decoder_rule,
            )
        else:
            status = "unresolved"
    elif raw is not None:
        status = "out_of_range"

    selected_local_target = candidate.local_target if candidate is not None else None
    selected_absolute_target = candidate.absolute_target if candidate is not None else None
    selected_formula = candidate.formula if candidate is not None else None

    word_index = source_word_index
    if word_index is None:
        word_index = int(word.index) if int(word.index) >= 0 else None

    semantic = classify_branch(word, source_word_index=word_index).to_dict()

    return BranchResolution(
        word_index=word_index,
        offset=word.offset,
        size=word.size,
        op=int(word.operands.get("op", -1) or -1),
        encoded_offset=raw_off,
        signed_offset=signed,
        prefixes_hex=[f"0x{p:02X}" for p in word.prefixes],
        status=status,
        exact_candidates=[candidate] if candidate is not None else [],
        aligned_candidates=[],
        selected_local_target=selected_local_target,
        selected_absolute_target=selected_absolute_target,
        selected_formula=selected_formula,
        terminal_op_offset=term_off,
        operand_base_offset=operand_base,
        target_entry_kind=target_entry_kind,
        target_decoder_rule=target_decoder_rule,
        semantic=semantic,
    )


# ---------------------------------------------------------------------------
# CFG construction


def _reachable_blocks(blocks: list[VMControlBlock], edges: list[VMControlEdge]) -> set[str]:
    if not blocks:
        return set()
    succs: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.status == "proven":
            succs[edge.source].append(edge.target)
    start = blocks[0].id
    seen = {start}
    queue: deque[str] = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in succs.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _top_level_index_by_offset(words: list[VMWord]) -> dict[int, int]:
    return {int(w.offset): int(w.index) for w in words}


def _decode_control_word(raw: bytes, offset: int, top_level_by_offset: dict[int, int]) -> VMWord | None:
    decoded = _decode_valid_entry(raw, offset)
    if decoded is None:
        return None
    top_index = top_level_by_offset.get(offset)
    if top_index is not None:
        return VMWord(
            index=top_index,
            offset=decoded.offset,
            size=decoded.size,
            kind=decoded.kind,
            terminal_kind=decoded.terminal_kind,
            prefixes=decoded.prefixes,
            operands=decoded.operands,
            raw=decoded.raw,
            confidence=decoded.confidence,
            decoder_rule=decoded.decoder_rule,
        )
    return VMWord(
        index=-1,
        offset=decoded.offset,
        size=decoded.size,
        kind=decoded.kind,
        terminal_kind=decoded.terminal_kind,
        prefixes=decoded.prefixes,
        operands=decoded.operands,
        raw=decoded.raw,
        confidence=decoded.confidence,
        decoder_rule=decoded.decoder_rule,
    )


def _add_split_offset(
    split_offsets: set[int],
    queue: deque[int],
    raw: bytes,
    offset: Optional[int],
) -> None:
    if offset is None or offset < 0 or offset >= len(raw):
        return
    if _decode_valid_entry(raw, offset) is None:
        return
    if offset not in split_offsets:
        split_offsets.add(offset)
        queue.append(offset)


def _discover_block_starts(
    raw: bytes,
    words: list[VMWord],
    *,
    function_start: int,
    entry_kinds_map: dict[int, set[str]],
) -> tuple[set[int], dict[int, BranchResolution]]:
    if not raw:
        return set(), {}

    top_level_by_offset = _top_level_index_by_offset(words)
    split_offsets: set[int] = {0}
    queue: deque[int] = deque([0])
    processed_starts: set[int] = set()
    branch_by_offset: dict[int, BranchResolution] = {}

    while queue:
        start = queue.popleft()
        if start in processed_starts:
            continue
        processed_starts.add(start)
        if _decode_valid_entry(raw, start) is None:
            continue

        pos = start
        safety = 0
        while 0 <= pos < len(raw):
            if pos != start and pos in split_offsets:
                break
            word = _decode_control_word(raw, pos, top_level_by_offset)
            if word is None:
                break
            role = word_role(word)
            next_pos = pos + max(1, int(word.size))

            if role == "branch":
                res = resolve_branch_target(
                    word,
                    function_start=function_start,
                    entry_kinds=entry_kinds_map,
                    raw=raw,
                    source_word_index=top_level_by_offset.get(pos),
                )
                branch_by_offset[pos] = res
                sem = classify_branch(word, source_word_index=top_level_by_offset.get(pos))
                _add_split_offset(split_offsets, queue, raw, res.selected_local_target)
                if sem.has_fallthrough_edge:
                    _add_split_offset(split_offsets, queue, raw, next_pos)
                break

            if role == "return":
                # Keep the old VMIR coverage behavior: bytes after a return are
                # allowed to start another disconnected bytecode island.  This is
                # a coverage split, not a fallthrough edge.
                _add_split_offset(split_offsets, queue, raw, next_pos)
                break

            if next_pos <= pos:
                break
            pos = next_pos
            safety += 1
            if safety > len(raw) + 1:
                break

    return split_offsets, branch_by_offset


def build_control_graph(
    words: list[VMWord],
    *,
    function_start: int = 0,
    alignment_radius: int = 4,
    split_on_aligned_candidates: bool = True,
    raw: Optional[bytes] = None,
) -> VMControlGraph:
    """Build a byte/sub-entry VM control graph without source-level lowering.

    Branch targets are resolved by the VM-spec invariant
    ``terminal_op_offset + 1 + signed_u16(off)``.  The graph does not snap branch
    targets to nearby top-level word starts.  If a target lands inside a prefixed
    word, the target is represented as a byte/sub-entry block and decoded from
    that byte.
    """

    raw = bytes(raw) if raw is not None else _reconstruct_raw(words)
    if not words or not raw:
        return VMControlGraph(
            contract=CONTROL_CONTRACT_VERSION,
            blocks=[],
            edges=[],
            branch_resolutions=[],
            summary={
                "block_count": 0,
                "branch_count": 0,
                "branch_status_histogram": {},
                "edge_status_histogram": {},
                "edge_kind_histogram": {},
                "proven_edge_count": 0,
                "candidate_edge_count": 0,
                "policy": "BR targets use terminal-op operand-base byte/sub-entry coordinates.",
            },
        )

    entry_kinds_map = entry_offset_kinds(words)
    top_level_by_offset = _top_level_index_by_offset(words)
    split_offsets, branch_by_offset = _discover_block_starts(
        raw,
        words,
        function_start=function_start,
        entry_kinds_map=entry_kinds_map,
    )

    block_starts = sorted(o for o in split_offsets if 0 <= o < len(raw) and _decode_valid_entry(raw, o) is not None)
    start_to_id = {offset: f"bb{i}" for i, offset in enumerate(block_starts)}

    blocks: list[VMControlBlock] = []
    edges: list[VMControlEdge] = []

    for start in block_starts:
        block_id = start_to_id[start]
        instruction_offsets: list[int] = []
        word_indices: list[int] = []
        decoded_words: list[VMWord] = []
        pos = start
        safety = 0

        while 0 <= pos < len(raw):
            if pos != start and pos in start_to_id:
                break
            word = _decode_control_word(raw, pos, top_level_by_offset)
            if word is None:
                break
            instruction_offsets.append(pos)
            decoded_words.append(word)
            if pos in top_level_by_offset:
                word_indices.append(top_level_by_offset[pos])
            role = word_role(word)
            if role in {"branch", "return"}:
                break
            next_pos = pos + max(1, int(word.size))
            if next_pos <= pos:
                break
            pos = next_pos
            safety += 1
            if safety > len(raw) + 1:
                break

        if not decoded_words:
            continue

        last = decoded_words[-1]
        last_offset = instruction_offsets[-1]
        end_offset = last_offset + max(1, int(last.size))
        successors: list[str] = []
        candidate_successors: list[str] = []
        terminator: Optional[dict[str, Any]] = None
        flags: list[str] = []
        if start not in top_level_by_offset:
            flags.append("subentry_block")
        if start in entry_kinds_map:
            flags.extend(f"entry:{kind}" for kind in sorted(entry_kinds_map[start]) if kind != "word_start")
        elif start != 0:
            flags.append("entry:decoded_hidden_entry")

        if word_role(last) == "branch":
            resolution = branch_by_offset.get(last_offset)
            if resolution is None:
                resolution = resolve_branch_target(
                    last,
                    function_start=function_start,
                    entry_kinds=entry_kinds_map,
                    raw=raw,
                    source_word_index=top_level_by_offset.get(last_offset),
                )
                branch_by_offset[last_offset] = resolution
            semantic = classify_branch(last, source_word_index=resolution.word_index)
            terminator = {"kind": "branch", "resolution": resolution.to_dict(), "semantic": semantic.to_dict()}
            if resolution.selected_local_target is not None:
                target_id = start_to_id.get(resolution.selected_local_target)
                if target_id:
                    successors.append(target_id)
                    edges.append(
                        VMControlEdge(
                            source=block_id,
                            target=target_id,
                            kind=semantic.taken_edge_kind,
                            status="proven",
                            word_index=resolution.word_index,
                            formula=resolution.selected_formula,
                            local_target=resolution.selected_local_target,
                            instruction_offset=last_offset,
                        )
                    )
            if semantic.has_fallthrough_edge:
                ft_id = start_to_id.get(end_offset)
                if ft_id:
                    successors.append(ft_id)
                    edges.append(
                        VMControlEdge(
                            source=block_id,
                            target=ft_id,
                            kind=semantic.fallthrough_edge_kind or "fallthrough",
                            status="proven",
                            word_index=resolution.word_index,
                            instruction_offset=last_offset,
                        )
                    )
        elif word_role(last) == "return":
            terminator = {"kind": "return"}
        else:
            ft_id = start_to_id.get(end_offset)
            if ft_id:
                successors.append(ft_id)
                edges.append(
                    VMControlEdge(
                        source=block_id,
                        target=ft_id,
                        kind="fallthrough",
                        status="proven",
                        instruction_offset=last_offset,
                    )
                )

        blocks.append(
            VMControlBlock(
                id=block_id,
                start_offset=start,
                end_offset=end_offset,
                word_indices=_stable_unique_int(word_indices),
                instruction_offsets=_stable_unique_int(instruction_offsets),
                successors=_stable_unique(successors),
                candidate_successors=_stable_unique(candidate_successors),
                terminator=terminator,
                flags=_stable_unique(flags),
            )
        )

    branch_resolutions = [branch_by_offset[offset] for offset in sorted(branch_by_offset)]
    status_hist = Counter(r.status for r in branch_resolutions)
    branch_semantic_hist = Counter((r.semantic or {}).get("branch_kind", "unknown") for r in branch_resolutions)
    branch_terminator_semantic_hist = Counter(
        ((b.terminator or {}).get("semantic") or {}).get("branch_kind", "unknown")
        for b in blocks
        if (b.terminator or {}).get("kind") == "branch"
    )
    edge_status_hist = Counter(e.status for e in edges)
    edge_kind_hist = Counter(e.kind for e in edges)
    reachable = _reachable_blocks(blocks, edges)
    subentry_blocks = sum(1 for b in blocks if "subentry_block" in b.flags)
    summary = {
        "block_count": len(blocks),
        "subentry_block_count": subentry_blocks,
        "branch_count": len(branch_resolutions),
        "branch_status_histogram": dict(sorted(status_hist.items())),
        "branch_semantic_kind_histogram": dict(sorted(branch_semantic_hist.items())),
        "branch_terminator_count": sum(branch_terminator_semantic_hist.values()),
        "branch_terminator_semantic_kind_histogram": dict(sorted(branch_terminator_semantic_hist.items())),
        "exact_entry_branch_count": status_hist.get("exact_entry", 0),
        "decoded_entry_branch_count": status_hist.get("decoded_entry", 0),
        "unresolved_branch_count": status_hist.get("unresolved", 0),
        "out_of_range_branch_count": status_hist.get("out_of_range", 0),
        "edge_status_histogram": dict(sorted(edge_status_hist.items())),
        "edge_kind_histogram": dict(sorted(edge_kind_hist.items())),
        "proven_edge_count": edge_status_hist.get("proven", 0),
        "candidate_edge_count": edge_status_hist.get("candidate", 0),
        "reachable_proven_block_count": len(reachable),
        "unreachable_under_proven_edges_count": max(0, len(blocks) - len(reachable)),
        "entry_coordinate_policy": "word starts, prefix subentries, terminal atoms, aggregate tails, and decoded branch targets are legal VM entry coordinates.",
        "branch_target_formula": BRANCH_TARGET_FORMULA,
        "policy": "BR targets use terminal-op operand-base byte/sub-entry coordinates; op 0x4A is modelled as jump without fallthrough, op 0x4B/0x4C/0x4D as conditional taken/fallthrough branches.",
    }
    return VMControlGraph(
        contract=CONTROL_CONTRACT_VERSION,
        blocks=blocks,
        edges=edges,
        branch_resolutions=branch_resolutions,
        summary=summary,
    )
