from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from .vm_spec import (
    VMWord,
    branch_operand_base_offset,
    branch_target_offset,
    is_control_branch_word,
    signed_u16,
    terminal_atom_offset,
)

if TYPE_CHECKING:  # pragma: no cover - avoids runtime import cycles with control.py
    from .control import VMControlGraph


SEMANTIC_CONTRACT_VERSION = "vm-semantic-v5"
CFG_SEMANTIC_CONTRACT_VERSION = "vm-cfg-semantic-v7"
BRANCH_SEMANTIC_POLICY = (
    "Opcode 0x4A is a VM control jump; opcodes 0x4B/0x4C/0x4D are "
    "conditional VM branches. If a conditional branch target equals its "
    "fallthrough coordinate, it is a VM conditional no-transfer atom, not a "
    "control-flow split. Predicate polarity and source-level condition meaning "
    "are intentionally not inferred here."
)
CFG_SEMANTIC_POLICY = (
    "CFG semantic lifting consumes proven VM control graphs plus hierarchical "
    "region/branch facts. It names VM-level control regions, preserves taken/"
    "fallthrough edge identity, and keeps predicate polarity unresolved. It does "
    "not rebuild CFG, mutate byte-level decoding, or emit source-level if/while/"
    "break/return constructs. Cross-SCC and cyclic-SCC branch envelopes are lifted "
    "from branch facts; no-transfer conditional atoms remain byte/branch facts and "
    "are not CFG branch regions."
)


@dataclass(frozen=True)
class VMBranchSemantics:
    contract: str
    word_index: Optional[int]
    offset: int
    size: int
    terminal_op_offset: int
    operand_base_offset: int
    op: int
    prefixes_hex: list[str]
    encoded_offset: int
    signed_offset: int
    target_offset: int
    fallthrough_offset: Optional[int]
    branch_kind: str
    taken_edge_kind: str
    fallthrough_edge_kind: Optional[str]
    has_fallthrough_edge: bool
    predicate_source: Optional[str]
    predicate_polarity: Optional[str]
    stack_pop: Optional[int]
    confidence: float
    policy: str = BRANCH_SEMANTIC_POLICY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMCFGPredicateSemantics:
    source: Optional[str]
    polarity: Optional[str]
    branch_kind: Optional[str]
    op: Optional[int]
    word_index: Optional[int]
    offset: Optional[int]
    terminal_op_offset: Optional[int]
    target_offset: Optional[int]
    fallthrough_offset: Optional[int]
    stack_pop: Optional[int]
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMSemanticArm:
    id: str
    role: str
    edge_kind: str
    successor: str
    successor_scc: str
    sccs: list[str]
    nodes: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMSemanticRegion:
    id: str
    kind: str
    source_kind: str
    header: str
    nodes: list[str]
    header_scc: Optional[str] = None
    exit: Optional[str] = None
    exit_kind: Optional[str] = None
    predicate: Optional[VMCFGPredicateSemantics] = None
    arms: list[VMSemanticArm] = field(default_factory=list)
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["predicate"] = self.predicate.to_dict() if self.predicate is not None else None
        payload["arms"] = [arm.to_dict() for arm in self.arms]
        return payload


@dataclass(frozen=True)
class VMSemanticDeferred:
    id: str
    kind: str
    header: str
    reason: str
    nodes: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMCFGSemanticsReport:
    contract: str
    summary: dict[str, Any]
    regions: list[VMSemanticRegion]
    deferred: list[VMSemanticDeferred] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "summary": self.summary,
            "regions": [region.to_dict() for region in self.regions],
            "deferred": [item.to_dict() for item in self.deferred],
        }


UNCONDITIONAL_BRANCH_OPS = {0x4A}
CONDITIONAL_BRANCH_OPS = {0x4B, 0x4C, 0x4D}


# ---------------------------------------------------------------------------
# Branch atom semantics used by control.py


def classify_branch(word: VMWord, *, source_word_index: Optional[int] = None) -> VMBranchSemantics:
    """Classify VM control-branch semantics without source-level lowering.

    This pass only labels control-edge behavior. BR-shaped words that cannot
    change control flow, such as target==fallthrough predicate atoms, are handled
    by ``vm_spec.word_role`` and are not valid inputs here.
    """

    if word.terminal_kind != "BR" or not is_control_branch_word(word):
        raise ValueError("classify_branch requires a control BR VMWord")

    op = int(word.operands.get("op", -1) or -1) & 0xFF
    encoded = int(word.operands.get("off", 0) or 0) & 0xFFFF
    signed = signed_u16(encoded)
    target = branch_target_offset(word)
    fallthrough = int(word.offset) + int(word.size)
    word_index = source_word_index
    if word_index is None:
        word_index = int(word.index) if int(word.index) >= 0 else None

    if op in UNCONDITIONAL_BRANCH_OPS:
        branch_kind = "unconditional_jump"
        taken_edge_kind = "jump"
        fallthrough_edge_kind = None
        has_fallthrough_edge = False
        predicate_source = None
        predicate_polarity = None
        stack_pop: Optional[int] = 0
        confidence = 1.0
    elif op in CONDITIONAL_BRANCH_OPS:
        branch_kind = "conditional_branch"
        taken_edge_kind = "conditional_taken"
        fallthrough_edge_kind = "conditional_fallthrough"
        has_fallthrough_edge = True
        predicate_source = "prefix_chain_or_stack" if word.prefixes else "stack_top_or_flag"
        # Polarity is deliberately unresolved. The VM can tell us which edge is
        # taken by the branch atom, but not yet whether that edge represents a
        # source-level true or false condition.
        predicate_polarity = "unresolved"
        stack_pop = None
        confidence = 0.95
    else:
        branch_kind = "unknown_branch"
        taken_edge_kind = "branch"
        fallthrough_edge_kind = "fallthrough"
        has_fallthrough_edge = True
        predicate_source = "unknown"
        predicate_polarity = "unknown"
        stack_pop = None
        confidence = 0.0

    return VMBranchSemantics(
        contract=SEMANTIC_CONTRACT_VERSION,
        word_index=word_index,
        offset=int(word.offset),
        size=int(word.size),
        terminal_op_offset=terminal_atom_offset(word),
        operand_base_offset=branch_operand_base_offset(word),
        op=op,
        prefixes_hex=[f"0x{p:02X}" for p in word.prefixes],
        encoded_offset=encoded,
        signed_offset=signed,
        target_offset=target,
        fallthrough_offset=fallthrough if has_fallthrough_edge else None,
        branch_kind=branch_kind,
        taken_edge_kind=taken_edge_kind,
        fallthrough_edge_kind=fallthrough_edge_kind,
        has_fallthrough_edge=has_fallthrough_edge,
        predicate_source=predicate_source,
        predicate_polarity=predicate_polarity,
        stack_pop=stack_pop,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# CFG semantic lifting


def _as_dicts(cfg: "VMControlGraph | dict[str, Any]") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if hasattr(cfg, "blocks") and hasattr(cfg, "edges"):
        return [b.to_dict() for b in cfg.blocks], [e.to_dict() for e in cfg.edges]
    return list(cfg.get("blocks") or []), list(cfg.get("edges") or [])


def _stable_sorted(values: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    return sorted(values, key=lambda x: (len(x), x))


def _predicate_from_header(block_by_id: dict[str, dict[str, Any]], header: str) -> Optional[VMCFGPredicateSemantics]:
    block = block_by_id.get(header) or {}
    terminator = block.get("terminator") or {}
    semantic = (terminator.get("semantic") or {}) if isinstance(terminator, dict) else {}
    if not semantic:
        return None
    return VMCFGPredicateSemantics(
        source=semantic.get("predicate_source"),
        polarity=semantic.get("predicate_polarity"),
        branch_kind=semantic.get("branch_kind"),
        op=semantic.get("op"),
        word_index=semantic.get("word_index"),
        offset=semantic.get("offset"),
        terminal_op_offset=semantic.get("terminal_op_offset"),
        target_offset=semantic.get("target_offset"),
        fallthrough_offset=semantic.get("fallthrough_offset"),
        stack_pop=semantic.get("stack_pop"),
        confidence=float(semantic.get("confidence", 1.0) or 0.0),
    )


def _arm_role(edge_kind: str) -> str:
    kinds = set(str(edge_kind).split("|"))
    if "conditional_taken" in kinds:
        return "vm_taken"
    if "conditional_fallthrough" in kinds:
        return "vm_fallthrough"
    if "jump" in kinds:
        return "vm_jump"
    return "vm_edge"


def _semantic_region_kind(branch: dict[str, Any]) -> str:
    region_kind = branch.get("region_kind")
    if region_kind:
        return str(region_kind)
    if branch.get("kind") == "conditional_scc_branch" and branch.get("exit_kind") == "virtual_function_exit":
        return "conditional_multi_exit_region"
    if branch.get("kind") == "conditional_scc_branch":
        return "conditional_region"
    return str(branch.get("kind") or "unknown_region")


def _semantic_arm_from_branch_arm(arm: dict[str, Any]) -> VMSemanticArm:
    edge_kind = str(arm.get("edge_kind") or "unknown")
    evidence = dict(arm.get("diagnostics") or {})
    if arm.get("entry_edges"):
        evidence["entry_edges"] = arm.get("entry_edges")
    return VMSemanticArm(
        id=str(arm.get("id") or "arm"),
        role=_arm_role(edge_kind),
        edge_kind=edge_kind,
        successor=str(arm.get("successor")),
        successor_scc=str(arm.get("successor_scc")),
        sccs=list(arm.get("sccs") or []),
        nodes=list(arm.get("nodes") or []),
        evidence=evidence,
    )


def analyze_cfg_semantics(cfg: "VMControlGraph | dict[str, Any]", *, include_details: bool = True) -> VMCFGSemanticsReport:
    """Lift proven CFG facts into VM-level semantic regions.

    The semantic layer consumes ``control``, ``regions`` and ``branches`` output.
    It does not recompute byte decoding or patch region coverage. Branches are
    emitted only from canonical branch facts already proven by the branch lifting
    pass, including cyclic-SCC branch envelopes.
    """

    # Lazy imports keep semantic.py usable by control.py without creating an
    # import cycle: control -> semantic -> branches -> control.
    from .branches import analyze_branch_lifts
    from .regions import REGION_CONTRACT_VERSION, analyze_regions

    blocks, edges = _as_dicts(cfg)
    block_by_id = {str(block.get("id")): block for block in blocks}
    branch_report = analyze_branch_lifts(cfg, include_branches=include_details).to_dict()
    branch_summary = branch_report.get("summary") or {}

    semantic_regions: list[VMSemanticRegion] = []
    deferred: list[VMSemanticDeferred] = []
    region_report: dict[str, Any] = {"contract": REGION_CONTRACT_VERSION, "summary": {}}

    if not include_details:
        real_conditional_count = int(branch_summary.get("real_join_branch_count", 0) or 0)
        multi_exit_count = int(branch_summary.get("function_exit_branch_count", 0) or 0)
        cyclic_branch_count = int(branch_summary.get("lifted_cyclic_scc_branch_count", 0) or 0)
        cyclic_region_count = int(branch_summary.get("cyclic_scc_count", 0) or 0)
        kind_hist = Counter()
        if real_conditional_count:
            kind_hist["conditional_region"] = real_conditional_count
        if multi_exit_count:
            kind_hist["conditional_multi_exit_region"] = multi_exit_count
        if cyclic_branch_count:
            kind_hist["cyclic_branch_region"] = cyclic_branch_count
        if cyclic_region_count:
            kind_hist["cyclic_scc_region"] = cyclic_region_count
        deferred_hist = Counter()
        summary = {
            "block_count": len(blocks),
            "edge_count": len(edges),
            "proven_edge_count": int(branch_summary.get("proven_edge_count", 0) or 0),
            "reachable_block_count": int(branch_summary.get("reachable_block_count", 0) or 0),
            "scc_count": int(branch_summary.get("scc_count", 0) or 0),
            "cyclic_scc_count": cyclic_region_count,
            "conditional_branch_count": int(branch_summary.get("conditional_branch_count", 0) or 0),
            "conditional_two_successor_count": int(branch_summary.get("conditional_two_successor_count", 0) or 0),
            "conditional_two_edge_count": int(branch_summary.get("conditional_two_edge_count", 0) or 0),
            "conditional_same_scc_successors_count": int(branch_summary.get("conditional_same_scc_successors_count", 0) or 0),
            "conditional_cross_scc_count": int(branch_summary.get("conditional_cross_scc_count", 0) or 0),
            "lifted_branch_count": int(branch_summary.get("lifted_branch_count", 0) or 0),
            "semantic_region_count": int(branch_summary.get("lifted_branch_count", 0) or 0) + cyclic_region_count,
            "semantic_conditional_region_count": real_conditional_count,
            "semantic_multi_exit_region_count": multi_exit_count,
            "semantic_cyclic_branch_region_count": cyclic_branch_count,
            "semantic_cyclic_region_count": cyclic_region_count,
            "semantic_region_kind_histogram": dict(sorted(kind_hist.items())),
            "semantic_deferred_count": 0,
            "semantic_deferred_kind_histogram": dict(sorted(deferred_hist.items())),
            "branch_contract": branch_report.get("contract"),
            "region_contract": REGION_CONTRACT_VERSION,
            "policy": CFG_SEMANTIC_POLICY,
        }
        return VMCFGSemanticsReport(CFG_SEMANTIC_CONTRACT_VERSION, summary, [], [])

    region_report = analyze_regions(cfg).to_dict()

    for branch in branch_report.get("branches") or []:
        header = str(branch.get("header"))
        predicate = _predicate_from_header(block_by_id, header)
        arms = [_semantic_arm_from_branch_arm(arm) for arm in branch.get("arms") or []]
        semantic_regions.append(
            VMSemanticRegion(
                id=f"sem_{branch.get('id')}",
                kind=_semantic_region_kind(branch),
                source_kind=str(branch.get("kind") or "unknown"),
                header=header,
                header_scc=branch.get("header_scc"),
                exit=branch.get("exit"),
                exit_kind=branch.get("exit_kind"),
                nodes=list(branch.get("nodes") or []),
                predicate=predicate,
                arms=arms,
                confidence=float(branch.get("confidence", 1.0) or 0.0),
                evidence={
                    **dict(branch.get("evidence") or {}),
                    "source_contract": branch_report.get("contract"),
                    "semantic_rule": "lift proven VM branch envelope into semantic CFG region",
                },
            )
        )

    cyclic_region_count = 0
    for scc in region_report.get("scc_regions") or []:
        cyclic_region_count += 1
        nodes = list(scc.get("nodes") or [])
        semantic_regions.append(
            VMSemanticRegion(
                id=f"sem_{scc.get('id')}",
                kind="cyclic_scc_region",
                source_kind="scc_region",
                header=str((scc.get("entry_blocks") or nodes or [scc.get("id")])[0]),
                header_scc=scc.get("id"),
                exit=None,
                exit_kind="cyclic_deferred",
                nodes=nodes,
                predicate=None,
                arms=[],
                confidence=1.0,
                evidence={
                    "source_contract": region_report.get("contract"),
                    "entry_blocks": scc.get("entry_blocks") or [],
                    "exit_blocks": scc.get("exit_blocks") or [],
                    "semantic_rule": "strongly connected VM blocks form a cyclic semantic region; detailed loop lifting is a later pass",
                },
            )
        )

    region_summary = region_report.get("summary") or {}
    kind_hist = Counter(region.kind for region in semantic_regions)
    deferred_hist = Counter(item.kind for item in deferred)
    real_conditional_count = int(kind_hist.get("conditional_region", 0))
    multi_exit_count = int(kind_hist.get("conditional_multi_exit_region", 0))
    summary = {
        "block_count": len(blocks),
        "edge_count": len(edges),
        "proven_edge_count": int(branch_summary.get("proven_edge_count", region_summary.get("proven_edge_count", 0)) or 0),
        "reachable_block_count": int(branch_summary.get("reachable_block_count", region_summary.get("reachable_block_count", 0)) or 0),
        "scc_count": int(branch_summary.get("scc_count", region_summary.get("scc_count", 0)) or 0),
        "cyclic_scc_count": int(branch_summary.get("cyclic_scc_count", region_summary.get("cyclic_scc_count", 0)) or 0),
        "conditional_branch_count": int(branch_summary.get("conditional_branch_count", 0) or 0),
        "conditional_two_successor_count": int(branch_summary.get("conditional_two_successor_count", 0) or 0),
        "conditional_two_edge_count": int(branch_summary.get("conditional_two_edge_count", 0) or 0),
        "conditional_same_scc_successors_count": int(branch_summary.get("conditional_same_scc_successors_count", 0) or 0),
        "conditional_cross_scc_count": int(branch_summary.get("conditional_cross_scc_count", 0) or 0),
        "lifted_branch_count": int(branch_summary.get("lifted_branch_count", 0) or 0),
        "semantic_region_count": len(semantic_regions),
        "semantic_conditional_region_count": real_conditional_count,
        "semantic_multi_exit_region_count": multi_exit_count,
        "semantic_cyclic_branch_region_count": int(kind_hist.get("cyclic_branch_region", 0)),
        "semantic_cyclic_region_count": cyclic_region_count,
        "semantic_region_kind_histogram": dict(sorted(kind_hist.items())),
        "semantic_deferred_count": len(deferred),
        "semantic_deferred_kind_histogram": dict(sorted(deferred_hist.items())),
        "branch_contract": branch_report.get("contract"),
        "region_contract": region_report.get("contract"),
        "policy": CFG_SEMANTIC_POLICY,
    }
    return VMCFGSemanticsReport(CFG_SEMANTIC_CONTRACT_VERSION, summary, semantic_regions, deferred)


# Backward-friendly alias: the module-level task is semantic lifting, but
# callers may prefer either verb.
lift_cfg_semantics = analyze_cfg_semantics


# ---------------------------------------------------------------------------
# Module CLI helpers


def analyze_module_cfg_semantics(
    module: Any,
    *,
    function: Optional[str] = None,
    limit_functions: Optional[int] = None,
) -> dict[str, Any]:
    from .control import build_control_graph
    from .ir import select_function_body_vmir
    from .parser import MBCModule
    from .vm_spec import decode_words

    mod = module if isinstance(module, MBCModule) else MBCModule(module)
    entries = [mod.get_function_entry(function)] if function else mod.function_entries(include_definitions=True, include_exports=True, dedupe=True)
    if limit_functions is not None:
        entries = entries[: max(0, int(limit_functions))]

    functions: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    region_kind_totals: Counter[str] = Counter()
    deferred_kind_totals: Counter[str] = Counter()
    for entry in entries:
        raw, selection = select_function_body_vmir(mod, entry)
        words = decode_words(raw)
        span = selection.get("span") or {"start": 0, "end": 0}
        cfg = build_control_graph(words, function_start=int(span.get("start", 0) or 0), raw=raw)
        report = analyze_cfg_semantics(cfg, include_details=bool(function)).to_dict()
        summary = report["summary"]
        for key in [
            "block_count",
            "edge_count",
            "proven_edge_count",
            "reachable_block_count",
            "scc_count",
            "cyclic_scc_count",
            "conditional_branch_count",
            "conditional_two_successor_count",
            "conditional_two_edge_count",
            "conditional_same_scc_successors_count",
            "conditional_cross_scc_count",
            "lifted_branch_count",
            "semantic_region_count",
            "semantic_conditional_region_count",
            "semantic_multi_exit_region_count",
            "semantic_cyclic_branch_region_count",
            "semantic_cyclic_region_count",
            "semantic_deferred_count",
        ]:
            totals[key] += int(summary.get(key, 0) or 0)
        region_kind_totals.update(summary.get("semantic_region_kind_histogram", {}))
        deferred_kind_totals.update(summary.get("semantic_deferred_kind_histogram", {}))
        functions.append({
            "name": entry.name,
            "symbol": entry.symbol,
            "span": span,
            "summary": summary,
            "regions": report["regions"] if function else [],
            "deferred": report["deferred"] if function else [],
        })

    return {
        "contract": CFG_SEMANTIC_CONTRACT_VERSION,
        "module": str(mod.path),
        "summary": {
            **dict(totals),
            "function_count": len(functions),
            "semantic_region_kind_histogram": dict(sorted(region_kind_totals.items())),
            "semantic_deferred_kind_histogram": dict(sorted(deferred_kind_totals.items())),
            "policy": CFG_SEMANTIC_POLICY,
        },
        "functions": functions,
    }


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Lift proven VM CFG facts into semantic regions")
    parser.add_argument("module", type=Path)
    parser.add_argument("--function", default=None)
    parser.add_argument("--limit-functions", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = analyze_module_cfg_semantics(args.module, function=args.function, limit_functions=args.limit_functions)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
