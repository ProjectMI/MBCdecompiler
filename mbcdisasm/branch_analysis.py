"""Advanced branch analysis helpers.

This module introduces a higher level understanding of branches that goes
far beyond the light-weight hints exposed by :mod:`knowledge`.  The original
control-flow graph builder only differentiated between a handful of control
flow categories (``jump``, ``branch``, ``call`` and ``return``).  In practice
the ``branch`` group spans a diverse set of opcodes that cover conditional
tests, multi-way dispatch, iterator control and virtual method lookups.  The
lack of structure meant that the rest of the pipeline could not reliably
identify compound statements, nested functions or even basic ``if``/``else``
sections.  This file offers a dedicated analysis pass that inspects annotated
instructions, consults statistical knowledge and performs flow-sensitive
reasoning to describe each branch precisely.

The analysis is organised into three layers:

``BranchDescriptor``
    Captures instruction level information such as the kind of branch,
    whether the fallthrough path is conditional and the conditions encoded in
    the operand or stack inputs.

``BranchStructure``
    Describes how branch descriptors connect basic blocks.  Structures are
    aware of dominance and are able to determine whether a branch creates a
    fork, loop or multi-way decision tree.

``BranchRegistry``
    A convenience wrapper that exposes lookup helpers allowing downstream
    passes to quickly answer questions such as "does block X close a loop" or
    "what are the mutually exclusive outcomes from block Y".

The helpers in this module favour correctness and visibility over brevity.
The algorithms are intentionally verbose and heavily documented to aid
ongoing research into the MBC VM semantics.  Many opcodes still lack formal
documentation and having an explicit reasoning trail greatly simplifies
manual auditing when heuristics misfire.
"""

from __future__ import annotations

import enum
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    DefaultDict,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .cfg import BasicBlock, ControlFlowGraph
from .instruction import WORD_SIZE
from .manual_semantics import AnnotatedInstruction, InstructionSemantics
from .knowledge import KnowledgeBase
from .vm_analysis import estimate_stack_io


# ---------------------------------------------------------------------------
# Branch kinds
# ---------------------------------------------------------------------------


class BranchKind(enum.Enum):
    """Taxonomy for branch instructions.

    The enum is purposefully rich to enable downstream translation passes to
    make fine grained decisions when reconstructing high level constructs.  A
    traditional VM might only need to differentiate between unconditional and
    conditional jumps, however the MBC bytecode exposes a large collection of
    helpers that multiplex stack manipulation, iterator control, virtual
    method dispatch and exception handling.  The ``BranchResolver`` expands the
    original hints into the following categories:

    ``DIRECT``
        Unconditional transfer of execution without side effects.

    ``CONDITIONAL``
        Two-way conditional branch that relies on a boolean (or truthy) value
        present on the stack.  The fallthrough path is treated as the ``false``
        branch unless heuristics conclude otherwise.

    ``PREDICATED``
        A specialisation of ``CONDITIONAL`` where the test is derived from a
        comparison operation encoded within the instruction itself.  These
        branches typically compare the operand against a register or literal.

    ``DISPATCH``
        Multi-way jump driven by a table or operand range.  All known switch
        constructs fall within this category.

    ``ITERATOR``
        Branch that controls VM level iterators (``for``/``foreach``).  The
        pattern is important because iterator branches often spawn implicit
        nested functions.

    ``CALL``
        Tail calls or dynamic dispatch operations that may behave as function
        calls from a high level point of view.  They still manipulate control
        flow so the analysis tracks them alongside standard branches.

    ``EXIT``
        Hard terminators (return, stop, throw).  They contribute to the control
        flow lattice even though they do not have successors.

    ``UNKNOWN``
        Fallback category used when heuristics cannot infer a sensible role.
    """

    DIRECT = "direct"
    CONDITIONAL = "conditional"
    PREDICATED = "predicated"
    DISPATCH = "dispatch"
    ITERATOR = "iterator"
    CALL = "call"
    EXIT = "exit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BranchOutcome:
    """Represents a single outgoing edge from a branch."""

    target: Optional[int]
    """Target offset. ``None`` is used for fall-through from exit blocks."""

    probability: Optional[float] = None
    """Probability hint gathered from opcode profiles when available."""

    condition: Optional[str] = None
    """Human readable condition guarding the edge."""

    description: str = ""
    """Free form text describing why the edge was produced."""


@dataclass(frozen=True)
class BranchDescriptor:
    """Detailed description of a branch instruction."""

    offset: int
    block_start: int
    kind: BranchKind
    semantics: InstructionSemantics
    outcomes: Tuple[BranchOutcome, ...]
    fallthrough: Optional[int]
    operand_hint: Optional[str]
    stack_inputs: int
    stack_outputs: int
    consumed_symbols: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    @property
    def is_conditional(self) -> bool:
        return self.kind in {BranchKind.CONDITIONAL, BranchKind.PREDICATED, BranchKind.ITERATOR}

    @property
    def is_multiway(self) -> bool:
        return len(self.outcomes) > 2 or self.kind in {BranchKind.DISPATCH, BranchKind.ITERATOR}

    def describe(self) -> str:
        outcomes = []
        for outcome in self.outcomes:
            target = "<exit>" if outcome.target is None else f"0x{outcome.target:06X}"
            if outcome.condition:
                label = f"{target} when {outcome.condition}"
            else:
                label = target
            outcomes.append(label)
        summary = ", ".join(outcomes)
        return f"{self.kind.value} -> {summary}"


@dataclass
class BranchStructure:
    """Aggregates the branch descriptors within a CFG.

    Each basic block may produce at most one descriptor since only the last
    instruction is allowed to alter control-flow.  ``BranchStructure`` stores
    the descriptors, predecessor relationships and a dominance forest derived
    from the CFG.  The relationships are expensive to compute so they are
    cached on demand.
    """

    graph: ControlFlowGraph
    descriptors: Dict[int, BranchDescriptor]
    predecessors: Dict[int, Set[int]] = field(default_factory=dict)
    _dominators: Optional[Dict[int, Set[int]]] = None
    _immediate_dominators: Optional[Dict[int, Optional[int]]] = None
    _post_dominators: Optional[Dict[int, Set[int]]] = None
    _immediate_post_dominators: Optional[Dict[int, Optional[int]]] = None

    def block_descriptor(self, block_start: int) -> Optional[BranchDescriptor]:
        return self.descriptors.get(block_start)

    # ------------------------------------------------------------------
    # Dominance information
    # ------------------------------------------------------------------
    def dominators(self) -> Dict[int, Set[int]]:
        if self._dominators is None:
            self._dominators = _compute_dominators(self.graph)
        return self._dominators

    def immediate_dominators(self) -> Dict[int, Optional[int]]:
        if self._immediate_dominators is None:
            dominators = self.dominators()
            self._immediate_dominators = _compute_immediate_dominators(dominators)
        return self._immediate_dominators

    def post_dominators(self) -> Dict[int, Set[int]]:
        if self._post_dominators is None:
            self._post_dominators = _compute_post_dominators(self.graph)
        return self._post_dominators

    def immediate_post_dominators(self) -> Dict[int, Optional[int]]:
        if self._immediate_post_dominators is None:
            post = self.post_dominators()
            self._immediate_post_dominators = _compute_immediate_post_dominators(post)
        return self._immediate_post_dominators

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------
    def loop_headers(self) -> Set[int]:
        headers: Set[int] = set()
        idoms = self.immediate_dominators()
        for start, descriptor in self.descriptors.items():
            if descriptor.kind in {BranchKind.DIRECT, BranchKind.EXIT}:
                continue
            for outcome in descriptor.outcomes:
                target = outcome.target
                if target is None:
                    continue
                # A block is a loop header if it dominates one of its targets
                # (i.e. there is a back-edge).
                if start in self.dominators().get(target, set()):
                    headers.add(start)
                    break
            else:
                continue
        # Some loops rely on straight-line fallthrough edges.  Detect them by
        # walking the immediate dominator tree.
        for block in self.graph.blocks.values():
            fallthrough = _fallthrough_successor(block)
            if fallthrough is None:
                continue
            if idoms.get(fallthrough) == block.start and block.start in self.dominators().get(fallthrough, set()):
                headers.add(block.start)
        return headers

    def enclosing_header(self, block_start: int) -> Optional[int]:
        for header in sorted(self.loop_headers()):
            doms = self.dominators().get(block_start, set())
            if header in doms:
                return header
        return None

    def is_post_dominated_by(self, block_start: int, candidate: int) -> bool:
        return candidate in self.post_dominators().get(block_start, set())

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------
    def ordered_blocks(self) -> List[BasicBlock]:
        return self.graph.block_order()

    def __iter__(self) -> Iterator[BranchDescriptor]:
        for block in self.graph.block_order():
            descriptor = self.descriptors.get(block.start)
            if descriptor is not None:
                yield descriptor


class BranchRegistry:
    """User friendly facade on top of :class:`BranchStructure`."""

    def __init__(self, structure: BranchStructure) -> None:
        self.structure = structure
        self._block_to_descriptor: Dict[int, BranchDescriptor] = structure.descriptors
        self._target_map: DefaultDict[int, List[BranchDescriptor]] = defaultdict(list)
        for descriptor in structure:
            for outcome in descriptor.outcomes:
                if outcome.target is not None:
                    self._target_map[outcome.target].append(descriptor)

    def branch_at(self, block_start: int) -> Optional[BranchDescriptor]:
        return self._block_to_descriptor.get(block_start)

    def incoming_branches(self, target: int) -> List[BranchDescriptor]:
        return list(self._target_map.get(target, ()))

    def enclosing_loop(self, block_start: int) -> Optional[int]:
        return self.structure.enclosing_header(block_start)

    def loop_headers(self) -> Set[int]:
        return self.structure.loop_headers()

    def render_tree(self, entry: Optional[int] = None) -> str:
        return render_branch_tree(self.structure, entry=entry)


# ---------------------------------------------------------------------------
# Branch resolver
# ---------------------------------------------------------------------------


class BranchResolver:
    """Infer branch descriptors for all blocks in a CFG."""

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge

    # Public API -------------------------------------------------------
    def analyse(self, graph: ControlFlowGraph) -> BranchRegistry:
        descriptors: Dict[int, BranchDescriptor] = {}
        predecessors: Dict[int, Set[int]] = defaultdict(set)

        for block in graph.block_order():
            for successor in block.successors:
                predecessors[successor].add(block.start)

        for block in graph.block_order():
            descriptor = self._analyse_block(block, predecessors)
            if descriptor is not None:
                descriptors[block.start] = descriptor

        structure = BranchStructure(graph=graph, descriptors=descriptors, predecessors=predecessors)
        return BranchRegistry(structure)

    # Private helpers --------------------------------------------------
    def _analyse_block(
        self, block: BasicBlock, predecessors: Mapping[int, Set[int]]
    ) -> Optional[BranchDescriptor]:
        if not block.instructions:
            return None

        instr = block.instructions[-1]
        semantics = instr.semantics
        kind = self._classify_kind(block, instr)

        fallthrough = _fallthrough_successor(block)
        outcomes = list(self._build_outcomes(block, instr, kind, fallthrough))
        operand_hint = semantics.operand_hint or self.knowledge.flow_target_hint(instr.label())
        inputs, outputs = estimate_stack_io(semantics)

        consumed = self._describe_consumed_values(block, instr, inputs)
        warnings = []
        if inputs and len(consumed) < inputs:
            warnings.append("insufficient stack inputs detected")

        # Promote single edge branches to DIRECT if no condition exists.
        if len(outcomes) == 1 and not outcomes[0].condition and kind == BranchKind.CONDITIONAL:
            kind = BranchKind.DIRECT

        descriptor = BranchDescriptor(
            offset=instr.word.offset,
            block_start=block.start,
            kind=kind,
            semantics=semantics,
            outcomes=tuple(outcomes),
            fallthrough=fallthrough,
            operand_hint=operand_hint,
            stack_inputs=int(inputs),
            stack_outputs=int(outputs),
            consumed_symbols=tuple(consumed),
            warnings=tuple(warnings),
        )

        return descriptor

    def _classify_kind(
        self, block: BasicBlock, instr: AnnotatedInstruction
    ) -> BranchKind:
        semantics = instr.semantics
        control_flow = semantics.control_flow or self.knowledge.control_flow_hint(instr.label())
        manual = semantics.manual_name.lower()
        summary = (semantics.summary or "").lower()

        # Start with explicit hints from the annotations.
        if control_flow == "return" or "return" in manual:
            return BranchKind.EXIT
        if control_flow == "stop" or "halt" in manual:
            return BranchKind.EXIT
        if control_flow == "call" or "call" in manual:
            if "tail" in manual or "jump" in manual:
                return BranchKind.DIRECT
            return BranchKind.CALL
        if control_flow == "branch":
            if "iterator" in manual or "loop" in summary:
                return BranchKind.ITERATOR
            if any(keyword in manual for keyword in ("switch", "case", "jump table")):
                return BranchKind.DISPATCH
            return BranchKind.CONDITIONAL
        if control_flow == "jump":
            if "switch" in manual or "table" in manual:
                return BranchKind.DISPATCH
            return BranchKind.DIRECT

        # Heuristic fallbacks ------------------------------------------------
        label = instr.word.label()
        mnemonic = semantics.mnemonic.lower()
        operand_hint = semantics.operand_hint or self.knowledge.flow_target_hint(label)

        # 1) Multi way dispatch typically consumes more than one stack input
        #    or makes use of operand tables.
        if self._looks_like_dispatch(semantics, operand_hint):
            return BranchKind.DISPATCH

        # 2) Stack consuming jumps tend to be conditional.
        stack_inputs = semantics.stack_inputs
        if stack_inputs is None:
            stack_inputs = 1 if "test" in manual or "compare" in manual else 0

        if stack_inputs > 0:
            return BranchKind.CONDITIONAL

        # 3) Keywords in manual names or summaries hint at more specific roles.
        keywords = {
            BranchKind.ITERATOR: ("iterator", "foreach", "for loop"),
            BranchKind.PREDICATED: ("less", "greater", "equal", "cmp", "test"),
        }
        for kind, tokens in keywords.items():
            if any(token in manual or token in summary for token in tokens):
                return kind

        if "call" in summary:
            return BranchKind.CALL

        if "branch" in summary or "jump" in summary or "branch" in manual:
            return BranchKind.CONDITIONAL

        if mnemonic.startswith("jmp") or mnemonic.startswith("bra"):
            return BranchKind.DIRECT

        return BranchKind.UNKNOWN

    def _looks_like_dispatch(
        self, semantics: InstructionSemantics, operand_hint: Optional[str]
    ) -> bool:
        if semantics.stack_outputs and semantics.stack_outputs > 1:
            return True
        text = f"{semantics.manual_name} {semantics.summary or ''}".lower()
        if any(token in text for token in ("switch", "case", "lookup", "table", "dispatch")):
            return True
        if operand_hint in {"table", "switch", "vector"}:
            return True
        return False

    def _build_outcomes(
        self,
        block: BasicBlock,
        instr: AnnotatedInstruction,
        kind: BranchKind,
        fallthrough: Optional[int],
    ) -> Iterator[BranchOutcome]:
        semantics = instr.semantics
        label = instr.word.label()
        hint = self.knowledge.control_flow_hint(label)
        target_hint = self.knowledge.flow_target_hint(label)

        successors = sorted(block.successors)
        if not successors:
            if kind == BranchKind.EXIT:
                yield BranchOutcome(target=None, probability=1.0, description="exit")
            return

        stack_inputs = semantics.stack_inputs or 0
        condition = self._infer_condition(instr, kind, stack_inputs)

        # When the CFG already reports explicit successors simply wrap them in
        # BranchOutcome objects.  For multi-way dispatch we try to preserve all
        # edges to avoid losing information.
        for successor in successors:
            probability = self._edge_probability(label, successor)
            if fallthrough is not None and successor == fallthrough and condition:
                outcome_condition = f"not ({condition})"
                description = "fallthrough"
            elif successor == fallthrough:
                outcome_condition = None
                description = "fallthrough"
            else:
                outcome_condition = condition
                description = "branch"
            yield BranchOutcome(
                target=successor,
                probability=probability,
                condition=outcome_condition,
                description=description,
            )

        # Some iterators push closure functions and therefore exhibit hidden
        # control-flow edges.  When the manual summary hints at nested function
        # creation annotate an implicit branch so the high level passes can
        # synthesize a function literal.
        if kind == BranchKind.ITERATOR and semantics.stack_outputs:
            description = "iterator-body"
            yield BranchOutcome(target=fallthrough, probability=None, condition=condition, description=description)

    def _edge_probability(self, key: str, successor: int) -> Optional[float]:
        profile = self.knowledge.opcode_profile(key)
        if profile is None:
            return None
        following = profile.following
        total = sum(following.values())
        if not total:
            return None
        label = f"0x{successor:06X}"
        count = following.get(label)
        if not count:
            return None
        probability = count / total
        if probability < 1e-3:
            return None
        return probability

    def _infer_condition(
        self, instr: AnnotatedInstruction, kind: BranchKind, stack_inputs: int
    ) -> Optional[str]:
        semantics = instr.semantics
        manual = semantics.manual_name.lower()
        summary = (semantics.summary or "").lower()

        if kind == BranchKind.CONDITIONAL and stack_inputs > 0:
            return "top_of_stack"
        if kind == BranchKind.ITERATOR:
            return "iterator-advance"
        if kind == BranchKind.PREDICATED:
            if "equal" in manual or "==" in summary:
                return "compare == operand"
            if "less" in manual or "<" in summary:
                return "compare < operand"
            if "greater" in manual or ">" in summary:
                return "compare > operand"
        if "zero" in manual:
            return "test == 0"
        if "nonzero" in manual or "nz" in manual:
            return "test != 0"
        if "truth" in summary:
            return "truthy"
        return None

    def _describe_consumed_values(
        self, block: BasicBlock, instr: AnnotatedInstruction, count: int
    ) -> List[str]:
        if count <= 0:
            return []

        symbols: List[str] = []
        # Walk backwards within the block gathering temporary names assigned by
        # previous instructions.  This provides context for branch conditions
        # which helps the Lua reconstructor emit expressive conditions.
        for previous in reversed(block.instructions[:-1]):
            semantics = previous.semantics
            outputs = semantics.stack_outputs or 0
            if outputs <= 0:
                continue
            label = semantics.manual_name
            base = label.lower().replace(" ", "_")
            for index in range(outputs):
                name = f"{base}_{index}"
                symbols.append(name)
                if len(symbols) >= count:
                    return list(reversed(symbols[:count]))
        return list(reversed(symbols))


# ---------------------------------------------------------------------------
# Dominator utilities
# ---------------------------------------------------------------------------


def _compute_dominators(graph: ControlFlowGraph) -> Dict[int, Set[int]]:
    """Return dominator sets for each block using the classical algorithm."""

    blocks = graph.block_order()
    if not blocks:
        return {}

    start = blocks[0].start
    doms: Dict[int, Set[int]] = {block.start: set(graph.blocks) for block in blocks}
    doms[start] = {start}

    changed = True
    while changed:
        changed = False
        for block in blocks[1:]:
            preds = block.predecessors
            if not preds:
                new_set = {block.start}
            else:
                pred_sets = [doms[pred] for pred in preds if pred in doms]
                if pred_sets:
                    intersection = set.intersection(*pred_sets)
                else:
                    intersection = set()
                new_set = {block.start} | intersection
            if new_set != doms[block.start]:
                doms[block.start] = new_set
                changed = True
    return doms


def _compute_immediate_dominators(dominators: Mapping[int, Set[int]]) -> Dict[int, Optional[int]]:
    idoms: Dict[int, Optional[int]] = {}
    for node, doms in dominators.items():
        candidates = doms - {node}
        idom = None
        for candidate in candidates:
            if all(candidate == other or candidate not in dominators[other] for other in candidates):
                idom = candidate
                break
        idoms[node] = idom
    return idoms


def _compute_post_dominators(graph: ControlFlowGraph) -> Dict[int, Set[int]]:
    blocks = graph.block_order()
    if not blocks:
        return {}

    exits = [block.start for block in blocks if not block.successors]
    if not exits:
        # The graph might not have explicit exits when the segment falls through
        # into trailing data.  Treat the last block as the exit in that case.
        exits = [blocks[-1].start]

    post_doms: Dict[int, Set[int]] = {block.start: set(graph.blocks) for block in blocks}
    for exit_start in exits:
        post_doms[exit_start] = {exit_start}

    changed = True
    while changed:
        changed = False
        for block in blocks:
            succs = block.successors
            if not succs:
                continue
            succ_sets = [post_doms[succ] for succ in succs if succ in post_doms]
            if succ_sets:
                intersection = set.intersection(*succ_sets)
            else:
                intersection = set()
            new_set = {block.start} | intersection
            if new_set != post_doms[block.start]:
                post_doms[block.start] = new_set
                changed = True
    return post_doms


def _compute_immediate_post_dominators(
    post_dominators: Mapping[int, Set[int]]
) -> Dict[int, Optional[int]]:
    immediate: Dict[int, Optional[int]] = {}
    for node, pdoms in post_dominators.items():
        candidates = sorted(pdoms - {node})
        ipdom = None
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if other == candidate:
                    continue
                if candidate in post_dominators.get(other, set()):
                    dominated = True
                    break
            if not dominated:
                ipdom = candidate
                break
        immediate[node] = ipdom
    return immediate


def _fallthrough_successor(block: BasicBlock) -> Optional[int]:
    if not block.instructions:
        return None
    next_offset = block.instructions[-1].word.offset + WORD_SIZE
    if next_offset in block.successors:
        return next_offset
    return None


# ---------------------------------------------------------------------------
# Public convenience helpers
# ---------------------------------------------------------------------------


def analyse_branches(knowledge: KnowledgeBase, graph: ControlFlowGraph) -> BranchRegistry:
    """Return a :class:`BranchRegistry` populated from ``graph``.

    The helper offers a concise entry point for code that only needs the final
    registry.  Internally it instantiates :class:`BranchResolver` and performs
    the analysis in a single call.  The function is intentionally small but its
    presence keeps the call sites tidy and emphasises the separation between
    branch discovery and representation.
    """

    resolver = BranchResolver(knowledge)
    return resolver.analyse(graph)


# ---------------------------------------------------------------------------
# Metrics and reporting helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchMetrics:
    """Aggregated statistics about the branches present in a graph."""

    total: int
    conditionals: int
    dispatch: int
    iterators: int
    exits: int
    calls: int
    unknown: int
    loops: int
    multiway: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "conditionals": self.conditionals,
            "dispatch": self.dispatch,
            "iterators": self.iterators,
            "exits": self.exits,
            "calls": self.calls,
            "unknown": self.unknown,
            "loops": self.loops,
            "multiway": self.multiway,
        }


@dataclass(frozen=True)
class LoopDescriptor:
    """Describes a natural loop discovered within the branch structure."""

    header: int
    back_edges: Tuple[int, ...]
    nesting_level: int


@dataclass
class BranchNarrative:
    """Human-friendly explanation of the reconstructed branch structure."""

    lines: List[str]

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"


@dataclass
class BranchTreeNode:
    """Hierarchical representation of branch decisions."""

    block_start: int
    descriptor: Optional[BranchDescriptor]
    children: List["BranchTreeNode"] = field(default_factory=list)

    def add_child(self, child: "BranchTreeNode") -> None:
        self.children.append(child)

    def render(self, indent: str = "  ") -> List[str]:
        label = f"0x{self.block_start:06X}"
        if self.descriptor is None:
            heading = f"block {label}: entry"
        else:
            heading = f"block {label}: {self.descriptor.describe()}"
        lines = [heading]
        for child in self.children:
            for line in child.render(indent):
                lines.append(indent + line)
        return lines


def compute_branch_metrics(structure: BranchStructure) -> BranchMetrics:
    """Summarise the distribution of branch kinds present in ``structure``."""

    totals = Counter()
    loops = structure.loop_headers()
    multiway = 0
    for descriptor in structure:
        totals[descriptor.kind] += 1
        if descriptor.is_multiway:
            multiway += 1
    return BranchMetrics(
        total=sum(totals.values()),
        conditionals=int(totals[BranchKind.CONDITIONAL] + totals[BranchKind.PREDICATED]),
        dispatch=int(totals[BranchKind.DISPATCH]),
        iterators=int(totals[BranchKind.ITERATOR]),
        exits=int(totals[BranchKind.EXIT]),
        calls=int(totals[BranchKind.CALL]),
        unknown=int(totals[BranchKind.UNKNOWN]),
        loops=len(loops),
        multiway=multiway,
    )


def enumerate_loops(structure: BranchStructure) -> List[LoopDescriptor]:
    """Return loop descriptors ordered by their nesting depth."""

    headers = sorted(structure.loop_headers())
    idoms = structure.immediate_dominators()
    descriptors: List[LoopDescriptor] = []
    for header in headers:
        back_edges = []
        for descriptor in structure:
            for outcome in descriptor.outcomes:
                if outcome.target == header and descriptor.block_start != header:
                    back_edges.append(descriptor.block_start)
        level = 0
        current = idoms.get(header)
        while current is not None:
            if current in headers:
                level += 1
            current = idoms.get(current)
        descriptors.append(LoopDescriptor(header=header, back_edges=tuple(sorted(back_edges)), nesting_level=level))
    return sorted(descriptors, key=lambda item: (item.nesting_level, item.header))


def narrate_branches(structure: BranchStructure) -> BranchNarrative:
    """Return a textual description of the branch layout."""

    metrics = compute_branch_metrics(structure)
    lines = [
        "branch summary:",
        f"  total branches: {metrics.total}",
        f"  conditional: {metrics.conditionals}",
        f"  dispatch: {metrics.dispatch}",
        f"  iterator: {metrics.iterators}",
        f"  exit: {metrics.exits}",
        f"  call-like: {metrics.calls}",
        f"  unknown: {metrics.unknown}",
        f"  loop headers: {metrics.loops}",
        f"  multi-way branches: {metrics.multiway}",
    ]
    if not metrics.total:
        return BranchNarrative(lines)

    lines.append("")
    lines.append("branch catalogue:")
    for descriptor in structure:
        label = f"0x{descriptor.block_start:06X}"
        summary = descriptor.describe()
        lines.append(f"  {label}: {summary}")
        if descriptor.warnings:
            for warning in descriptor.warnings:
                lines.append(f"    warning: {warning}")
        if descriptor.consumed_symbols:
            joined = ", ".join(descriptor.consumed_symbols)
            lines.append(f"    consumes: {joined}")

    loops = enumerate_loops(structure)
    if loops:
        lines.append("")
        lines.append("loops:")
        for loop in loops:
            header_label = f"0x{loop.header:06X}"
            back = ", ".join(f"0x{edge:06X}" for edge in loop.back_edges) or "<implicit>"
            lines.append(
                f"  header {header_label} (nesting={loop.nesting_level}) back-edges: {back}"
            )

    return BranchNarrative(lines)


def branch_metrics_to_dict(structure: BranchStructure) -> Dict[str, object]:
    """Return a JSON-serialisable dictionary summarising the branch registry."""

    metrics = compute_branch_metrics(structure)
    payload: Dict[str, object] = {"metrics": metrics.to_dict(), "branches": []}
    for descriptor in structure:
        payload["branches"].append(
            {
                "offset": descriptor.offset,
                "block_start": descriptor.block_start,
                "kind": descriptor.kind.value,
                "outcomes": [
                    {
                        "target": outcome.target,
                        "probability": outcome.probability,
                        "condition": outcome.condition,
                        "description": outcome.description,
                    }
                    for outcome in descriptor.outcomes
                ],
                "fallthrough": descriptor.fallthrough,
                "operand_hint": descriptor.operand_hint,
                "stack_inputs": descriptor.stack_inputs,
                "stack_outputs": descriptor.stack_outputs,
                "warnings": list(descriptor.warnings),
                "consumed": list(descriptor.consumed_symbols),
            }
        )
    payload["loops"] = [
        {
            "header": loop.header,
            "back_edges": list(loop.back_edges),
            "nesting_level": loop.nesting_level,
        }
        for loop in enumerate_loops(structure)
    ]
    return payload


def render_branch_report(structure: BranchStructure) -> str:
    """Return a multiline textual report summarising the branch registry."""

    narrative = narrate_branches(structure)
    return narrative.render()


def build_branch_tree(structure: BranchStructure, entry: Optional[int] = None) -> BranchTreeNode:
    """Construct a depth-first tree representation of the branch registry."""

    blocks = structure.graph.block_order()
    if not blocks:
        return BranchTreeNode(block_start=0, descriptor=None)
    entry_start = entry if entry is not None else blocks[0].start
    visited: Set[int] = set()

    def _build(start: int) -> BranchTreeNode:
        descriptor = structure.block_descriptor(start)
        node = BranchTreeNode(block_start=start, descriptor=descriptor)
        if start in visited:
            return node
        visited.add(start)
        successors = structure.graph.blocks[start].successors
        for successor in sorted(successors):
            node.add_child(_build(successor))
        return node

    return _build(entry_start)


def render_branch_tree(structure: BranchStructure, entry: Optional[int] = None) -> str:
    """Render the branch structure as an indented tree."""

    tree = build_branch_tree(structure, entry=entry)
    lines = tree.render()
    return "\n".join(lines) + "\n"


@dataclass
class BranchReportBuilder:
    """Composable report generator for branch registries."""

    structure: BranchStructure
    include_tree: bool = True
    include_metrics: bool = True
    include_loops: bool = True
    include_catalogue: bool = True

    def build(self) -> str:
        sections: List[str] = []
        if self.include_metrics:
            sections.append(render_branch_report(self.structure).rstrip())
        if self.include_tree:
            sections.append(render_branch_tree(self.structure).rstrip())
        if self.include_loops and self.structure.loop_headers():
            lines = ["loop summary:"]
            for loop in enumerate_loops(self.structure):
                header = f"0x{loop.header:06X}"
                edges = ", ".join(f"0x{edge:06X}" for edge in loop.back_edges) or "<implicit>"
                lines.append(f"  header {header} nesting={loop.nesting_level} edges=[{edges}]")
            sections.append("\n".join(lines))
        if self.include_catalogue:
            catalogue_lines = ["branch catalogue (compact):"]
            for descriptor in self.structure:
                label = f"0x{descriptor.block_start:06X}"
                catalogue_lines.append(f"  {label}: {descriptor.kind.value}")
            sections.append("\n".join(catalogue_lines))
        return "\n\n".join(section for section in sections if section) + "\n"


def build_branch_report(structure: BranchStructure, *, include_tree: bool = True) -> str:
    """Convenience helper returning a combined textual report."""

    builder = BranchReportBuilder(
        structure=structure,
        include_tree=include_tree,
        include_metrics=True,
        include_loops=True,
        include_catalogue=True,
    )
    return builder.build()


def branch_registry_to_json(registry: BranchRegistry, *, indent: int = 2) -> str:
    """Serialise ``registry`` to JSON."""

    import json

    payload = branch_metrics_to_dict(registry.structure)
    return json.dumps(payload, indent=indent, sort_keys=True)


def branch_registry_summary(registry: BranchRegistry) -> str:
    """Return a short one-line summary describing the branch registry."""

    metrics = compute_branch_metrics(registry.structure)
    return (
        f"branches={metrics.total} conditionals={metrics.conditionals} "
        f"loops={metrics.loops} dispatch={metrics.dispatch}"
    )

