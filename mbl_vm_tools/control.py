from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .vm_spec import VMWord, word_role


CONTROL_CONTRACT_VERSION = "vm-control-v1"


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BranchResolution:
    word_index: int
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMControlBlock:
    id: str
    start_offset: int
    end_offset: int
    word_indices: list[int]
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


def _branch_base_targets(word: VMWord) -> list[tuple[str, int]]:
    raw = int(word.operands.get("off", 0) or 0)
    signed = raw - 0x10000 if raw & 0x8000 else raw
    # These are VM-coordinate observations only.  Bias/alignment repairs are not
    # treated as proven edges by this layer.
    bases = [
        ("absolute_local", raw),
        ("absolute_signed", signed),
        ("start_plus_u16", word.offset + raw),
        ("end_plus_u16", word.offset + word.size + raw),
        ("start_plus_s16", word.offset + signed),
        ("end_plus_s16", word.offset + word.size + signed),
    ]
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for item in bases:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _nearest_word_start(local_target: int, valid_offsets: set[int], *, radius: int = 4) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for off in valid_offsets:
        delta = off - local_target
        dist = abs(delta)
        if dist > radius:
            continue
        if best is None or dist < abs(best[1]) or (dist == abs(best[1]) and off < best[0]):
            best = (off, delta)
    return best


def resolve_branch_target(
    word: VMWord,
    *,
    function_start: int,
    valid_offsets: set[int],
    alignment_radius: int = 4,
) -> BranchResolution:
    raw = int(word.operands.get("off", 0) or 0)
    signed = raw - 0x10000 if raw & 0x8000 else raw
    exact_by_target: dict[int, list[str]] = defaultdict(list)
    aligned_by_target: dict[int, tuple[list[str], int, int]] = {}

    for formula, local_target in _branch_base_targets(word):
        if local_target in valid_offsets:
            exact_by_target[local_target].append(formula)
        else:
            nearest = _nearest_word_start(local_target, valid_offsets, radius=alignment_radius)
            if nearest is None:
                continue
            aligned_target, delta = nearest
            bucket = aligned_by_target.setdefault(aligned_target, ([], delta, local_target))
            bucket[0].append(formula)

    exact_candidates = [
        BranchTargetCandidate(
            formula="|".join(formulas),
            local_target=target,
            absolute_target=function_start + target,
            status="exact_word_start",
            confidence=1.0,
        )
        for target, formulas in sorted(exact_by_target.items())
    ]
    aligned_candidates = [
        BranchTargetCandidate(
            formula="|".join(formulas),
            local_target=target,
            absolute_target=function_start + target,
            status="aligned_word_start",
            confidence=0.5,
            alignment_delta=delta,
            base_formula="|".join(formulas),
            base_local_target=base_target,
        )
        for target, (formulas, delta, base_target) in sorted(aligned_by_target.items())
        if target not in exact_by_target
    ]

    selected_local_target: Optional[int] = None
    selected_formula: Optional[str] = None
    selected_absolute_target: Optional[int] = None
    if len(exact_candidates) == 1:
        status = "exact"
        selected_local_target = exact_candidates[0].local_target
        selected_absolute_target = exact_candidates[0].absolute_target
        selected_formula = exact_candidates[0].formula
    elif len(exact_candidates) > 1:
        status = "ambiguous_exact"
    elif aligned_candidates:
        status = "aligned_only"
    else:
        status = "unresolved"

    return BranchResolution(
        word_index=word.index,
        offset=word.offset,
        size=word.size,
        op=int(word.operands.get("op", -1) or -1),
        encoded_offset=raw,
        signed_offset=signed,
        prefixes_hex=[f"0x{p:02X}" for p in word.prefixes],
        status=status,
        exact_candidates=exact_candidates,
        aligned_candidates=aligned_candidates,
        selected_local_target=selected_local_target,
        selected_absolute_target=selected_absolute_target,
        selected_formula=selected_formula,
    )


def _stable_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


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


def build_control_graph(
    words: list[VMWord],
    *,
    function_start: int = 0,
    alignment_radius: int = 4,
    split_on_aligned_candidates: bool = True,
) -> VMControlGraph:
    """Build a bytecode control graph without source-level lowering.

    Proven branch edges require an exact VM-word-start target.  If a branch
    displacement only lands near a word boundary, the near target is reported as
    an aligned candidate but is not promoted to a proven edge.  This keeps the
    VMIR honest while still exposing enough evidence for later VM-spec work.
    """

    valid_offsets = {w.offset for w in words}
    offset_to_word = {w.offset: w for w in words}
    split_offsets: set[int] = {words[0].offset} if words else set()
    terminator_indices: set[int] = set()
    branch_by_word: dict[int, BranchResolution] = {}

    for word in words:
        role = word_role(word)
        if role == "branch":
            terminator_indices.add(word.index)
            resolution = resolve_branch_target(
                word,
                function_start=function_start,
                valid_offsets=valid_offsets,
                alignment_radius=alignment_radius,
            )
            branch_by_word[word.index] = resolution
            for cand in resolution.exact_candidates:
                split_offsets.add(cand.local_target)
            if split_on_aligned_candidates:
                for cand in resolution.aligned_candidates:
                    split_offsets.add(cand.local_target)
            next_offset = word.offset + word.size
            if next_offset in valid_offsets:
                split_offsets.add(next_offset)
        elif role == "return":
            terminator_indices.add(word.index)
            next_offset = word.offset + word.size
            if next_offset in valid_offsets:
                split_offsets.add(next_offset)

    block_starts = sorted(o for o in split_offsets if o in offset_to_word)
    if not block_starts and words:
        block_starts = [words[0].offset]
    start_to_id = {offset: f"bb{i}" for i, offset in enumerate(block_starts)}
    word_index_by_offset = {w.offset: i for i, w in enumerate(words)}

    blocks: list[VMControlBlock] = []
    edges: list[VMControlEdge] = []

    for bi, start in enumerate(block_starts):
        start_idx = word_index_by_offset[start]
        end_offset_exclusive = block_starts[bi + 1] if bi + 1 < len(block_starts) else None
        word_indices: list[int] = []
        idx = start_idx
        while idx < len(words):
            word = words[idx]
            if end_offset_exclusive is not None and word.offset >= end_offset_exclusive:
                break
            word_indices.append(word.index)
            if word.index in terminator_indices:
                break
            idx += 1
        if not word_indices:
            continue

        block_id = start_to_id[start]
        last = words[word_indices[-1]]
        successors: list[str] = []
        candidate_successors: list[str] = []
        terminator: Optional[dict[str, Any]] = None

        if word_role(last) == "branch":
            resolution = branch_by_word[last.index]
            terminator = {"kind": "branch", "resolution": resolution.to_dict()}
            if resolution.selected_local_target is not None:
                target_id = start_to_id.get(resolution.selected_local_target)
                if target_id:
                    successors.append(target_id)
                    edges.append(
                        VMControlEdge(
                            source=block_id,
                            target=target_id,
                            kind="branch",
                            status="proven",
                            word_index=last.index,
                            formula=resolution.selected_formula,
                            local_target=resolution.selected_local_target,
                        )
                    )
            for cand in resolution.aligned_candidates:
                target_id = start_to_id.get(cand.local_target)
                if target_id:
                    candidate_successors.append(target_id)
                    edges.append(
                        VMControlEdge(
                            source=block_id,
                            target=target_id,
                            kind="branch_candidate",
                            status="candidate",
                            word_index=last.index,
                            formula=cand.formula,
                            local_target=cand.local_target,
                            alignment_delta=cand.alignment_delta,
                        )
                    )
            fallthrough = last.offset + last.size
            ft_id = start_to_id.get(fallthrough)
            if ft_id:
                successors.append(ft_id)
                edges.append(
                    VMControlEdge(
                        source=block_id,
                        target=ft_id,
                        kind="fallthrough",
                        status="proven",
                        word_index=last.index,
                    )
                )
        elif word_role(last) == "return":
            terminator = {"kind": "return"}
        else:
            fallthrough = last.offset + last.size
            ft_id = start_to_id.get(fallthrough)
            if ft_id:
                successors.append(ft_id)
                edges.append(
                    VMControlEdge(
                        source=block_id,
                        target=ft_id,
                        kind="fallthrough",
                        status="proven",
                    )
                )

        blocks.append(
            VMControlBlock(
                id=block_id,
                start_offset=start,
                end_offset=last.offset + last.size,
                word_indices=word_indices,
                successors=_stable_unique(successors),
                candidate_successors=_stable_unique(candidate_successors),
                terminator=terminator,
            )
        )

    branch_resolutions = [branch_by_word[idx] for idx in sorted(branch_by_word)]
    status_hist = Counter(r.status for r in branch_resolutions)
    edge_status_hist = Counter(e.status for e in edges)
    edge_kind_hist = Counter(e.kind for e in edges)
    reachable = _reachable_blocks(blocks, edges)
    summary = {
        "block_count": len(blocks),
        "branch_count": len(branch_resolutions),
        "branch_status_histogram": dict(sorted(status_hist.items())),
        "exact_branch_count": status_hist.get("exact", 0),
        "ambiguous_exact_branch_count": status_hist.get("ambiguous_exact", 0),
        "aligned_only_branch_count": status_hist.get("aligned_only", 0),
        "unresolved_branch_count": status_hist.get("unresolved", 0),
        "edge_status_histogram": dict(sorted(edge_status_hist.items())),
        "edge_kind_histogram": dict(sorted(edge_kind_hist.items())),
        "proven_edge_count": edge_status_hist.get("proven", 0),
        "candidate_edge_count": edge_status_hist.get("candidate", 0),
        "reachable_proven_block_count": len(reachable),
        "unreachable_under_proven_edges_count": max(0, len(blocks) - len(reachable)),
        "policy": "Only exact VM-word-start branch targets become proven CFG edges; aligned near-boundary targets remain candidates.",
    }
    return VMControlGraph(
        contract=CONTROL_CONTRACT_VERSION,
        blocks=blocks,
        edges=edges,
        branch_resolutions=branch_resolutions,
        summary=summary,
    )
