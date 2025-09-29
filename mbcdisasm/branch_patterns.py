"""Higher level branch pattern analysis built on top of :mod:`branch_analysis`.

The :mod:`branch_analysis` module focuses on the low level mechanics of control
flow – identifying which instructions terminate blocks, classifying their
behaviour and exposing structural queries such as dominator sets or loop
headers.  While that information is sufficient for expert tooling it is rather
verbose for the higher level reconstruction layers.  This module offers a
friendlier abstraction that groups related blocks into patterns resembling
familiar programming constructs (``if`` statements, ``while`` loops, dispatch
tables and so on).

The pattern analyser purposely trades completeness for clarity: it attempts to
describe what can be recognised with high confidence and records rich metadata
for consumers to fall back to raw descriptors when heuristics fail.  Each
pattern includes a ``confidence`` score and a ``notes`` list that document why a
particular interpretation was chosen.  Downstream passes – especially the Lua
structurer – can use this information to decide whether it is safe to produce a
structured construct or whether a more conservative translation should be
emitted instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from .branch_analysis import BranchDescriptor, BranchKind, BranchRegistry, BranchStructure


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def _sorted_tuple(items: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(set(items)))


def _outcome_targets(descriptor: BranchDescriptor) -> Tuple[Optional[int], ...]:
    return tuple(outcome.target for outcome in descriptor.outcomes)


# ---------------------------------------------------------------------------
# Pattern dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternNote:
    """Simple annotation describing additional insights about a pattern."""

    message: str


@dataclass(frozen=True)
class BranchPattern:
    """Base class for the higher level branch patterns."""

    block_start: int
    descriptor: BranchDescriptor
    confidence: float
    notes: Tuple[PatternNote, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        fragments = [f"confidence={self.confidence:.2f}"]
        if self.notes:
            fragments.append("notes=" + ", ".join(note.message for note in self.notes))
        return f"block 0x{self.block_start:06X}: {self.descriptor.kind.value} ({'; '.join(fragments)})"


@dataclass(frozen=True)
class ConditionalPattern(BranchPattern):
    """Pattern describing two-way (or fallthrough) branches."""

    true_target: Optional[int] = None
    false_target: Optional[int] = None
    join_block: Optional[int] = None
    true_region: Tuple[int, ...] = field(default_factory=tuple)
    false_region: Tuple[int, ...] = field(default_factory=tuple)
    style: str = "if"
    loop_body_target: Optional[int] = None
    loop_exit_target: Optional[int] = None

    def is_guard(self) -> bool:
        return self.style in {"loop_guard", "tail_guard"}


@dataclass(frozen=True)
class DispatchPattern(BranchPattern):
    """Pattern describing multi-way dispatch statements."""

    targets: Tuple[Optional[int], ...] = field(default_factory=tuple)
    join_block: Optional[int] = None
    ladder_blocks: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LoopPattern(BranchPattern):
    """Natural loop discovered in the control-flow graph."""

    header: int = 0
    latch_blocks: Tuple[int, ...] = field(default_factory=tuple)
    body: Tuple[int, ...] = field(default_factory=tuple)
    exits: Tuple[int, ...] = field(default_factory=tuple)
    nesting_level: int = 0
    style: str = "while"

    def contains(self, block: int) -> bool:
        return block == self.header or block in self.body


@dataclass(frozen=True)
class TailCallPattern(BranchPattern):
    """Pattern that flags tail-call like blocks."""

    callee: Optional[int] = None
    returns: bool = False


@dataclass
class BranchPatternRegistry:
    """Container exposing lookup helpers for the derived patterns."""

    structure: BranchStructure
    conditionals: Dict[int, ConditionalPattern] = field(default_factory=dict)
    dispatches: Dict[int, DispatchPattern] = field(default_factory=dict)
    loops: Dict[int, LoopPattern] = field(default_factory=dict)
    tail_calls: Dict[int, TailCallPattern] = field(default_factory=dict)

    def conditional(self, block_start: int) -> Optional[ConditionalPattern]:
        return self.conditionals.get(block_start)

    def dispatch(self, block_start: int) -> Optional[DispatchPattern]:
        return self.dispatches.get(block_start)

    def loop(self, header: int) -> Optional[LoopPattern]:
        return self.loops.get(header)

    def enclosing_loop(self, block_start: int) -> Optional[LoopPattern]:
        for loop in sorted(self.loops.values(), key=lambda item: item.nesting_level):
            if loop.contains(block_start):
                return loop
        return None

    def tail_call(self, block_start: int) -> Optional[TailCallPattern]:
        return self.tail_calls.get(block_start)

    def summary_lines(self) -> List[str]:
        lines = ["branch pattern summary:"]
        lines.append(f"  conditionals: {len(self.conditionals)}")
        lines.append(f"  dispatches: {len(self.dispatches)}")
        lines.append(f"  loops: {len(self.loops)}")
        lines.append(f"  tail-calls: {len(self.tail_calls)}")
        return lines

    def __iter__(self) -> Iterator[BranchPattern]:
        yield from sorted(self.conditionals.values(), key=lambda pattern: pattern.block_start)
        yield from sorted(self.dispatches.values(), key=lambda pattern: pattern.block_start)
        yield from sorted(self.loops.values(), key=lambda pattern: pattern.block_start)
        yield from sorted(self.tail_calls.values(), key=lambda pattern: pattern.block_start)


# ---------------------------------------------------------------------------
# Pattern analyser
# ---------------------------------------------------------------------------


class BranchPatternAnalyzer:
    """Analyse a :class:`BranchStructure` and derive higher level patterns."""

    def __init__(self, structure: BranchStructure) -> None:
        self.structure = structure
        self.graph = structure.graph
        self._dominators = structure.dominators()
        self._post_dominators = structure.post_dominators()
        self._immediate_post_dominators = structure.immediate_post_dominators()
        self._predecessors = structure.predecessors

    # ------------------------------------------------------------------
    def analyse(self) -> BranchPatternRegistry:
        registry = BranchPatternRegistry(structure=self.structure)
        loop_patterns = self._discover_loops()
        registry.loops.update({loop.header: loop for loop in loop_patterns})

        for descriptor in self.structure:
            if descriptor.kind in {BranchKind.CONDITIONAL, BranchKind.PREDICATED, BranchKind.ITERATOR}:
                pattern = self._build_conditional_pattern(descriptor, registry)
                if pattern is not None:
                    registry.conditionals[descriptor.block_start] = pattern
            elif descriptor.kind == BranchKind.DISPATCH or descriptor.is_multiway:
                pattern = self._build_dispatch_pattern(descriptor)
                registry.dispatches[descriptor.block_start] = pattern
            elif descriptor.kind == BranchKind.CALL:
                pattern = self._build_tail_call_pattern(descriptor)
                if pattern is not None:
                    registry.tail_calls[descriptor.block_start] = pattern

        return registry

    # ------------------------------------------------------------------
    def _discover_loops(self) -> List[LoopPattern]:
        patterns: List[LoopPattern] = []
        headers = self.structure.loop_headers()
        if not headers:
            return patterns

        for header in sorted(headers):
            descriptor = self.structure.block_descriptor(header)
            if descriptor is None:
                continue
            latch_blocks = self._loop_latches(header, descriptor)
            body = self._loop_body(header, latch_blocks)
            exits = self._loop_exits(header, body)
            nesting = self._loop_nesting_level(header, headers)
            style = "while"
            notes: List[PatternNote] = []
            if descriptor.kind == BranchKind.ITERATOR:
                style = "iterator"
                notes.append(PatternNote("loop header flagged as iterator"))
            elif descriptor.kind == BranchKind.DISPATCH:
                style = "dispatch"
                notes.append(PatternNote("multi-way dispatch inside loop"))
            pattern = LoopPattern(
                block_start=header,
                descriptor=descriptor,
                confidence=0.9,
                notes=tuple(notes),
                header=header,
                latch_blocks=_sorted_tuple(latch_blocks),
                body=_sorted_tuple(body - {header}),
                exits=_sorted_tuple(exits),
                nesting_level=nesting,
                style=style,
            )
            patterns.append(pattern)
        return patterns

    def _loop_latches(self, header: int, descriptor: BranchDescriptor) -> Set[int]:
        latches: Set[int] = set()
        for outcome in descriptor.outcomes:
            if outcome.target == header:
                latches.add(descriptor.block_start)
            elif outcome.target is not None and self._forward_reaches_header(outcome.target, header):
                latches.add(outcome.target)
        for predecessor in self._predecessors.get(header, ()):  # type: ignore[arg-type]
            if self._forward_reaches_header(predecessor, header):
                latches.add(predecessor)
        return latches

    def _loop_body(self, header: int, latch_blocks: Set[int]) -> Set[int]:
        body: Set[int] = {header}
        worklist: List[int] = list(latch_blocks)
        while worklist:
            current = worklist.pop()
            if current in body:
                continue
            body.add(current)
            for predecessor in self._predecessors.get(current, ()):  # type: ignore[arg-type]
                if predecessor not in body:
                    worklist.append(predecessor)
        return body

    def _forward_reaches_header(self, start: int, header: int) -> bool:
        if start == header:
            return True
        visited: Set[int] = set()
        stack: List[int] = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current == header:
                return True
            block = self.graph.blocks.get(current)
            if block is None:
                continue
            for successor in block.successors:
                if successor not in visited:
                    stack.append(successor)
        return False

    def _loop_exits(self, header: int, body: Set[int]) -> Set[int]:
        exits: Set[int] = set()
        for block_start in body:
            block = self.graph.blocks.get(block_start)
            if block is None:
                continue
            for successor in block.successors:
                if successor not in body:
                    exits.add(successor)
        return exits

    def _loop_nesting_level(self, header: int, headers: Set[int]) -> int:
        level = 0
        for candidate in headers:
            if candidate == header:
                continue
            doms = self._dominators.get(header, set())
            if candidate in doms:
                level += 1
        return level

    # ------------------------------------------------------------------
    def _build_conditional_pattern(
        self, descriptor: BranchDescriptor, registry: BranchPatternRegistry
    ) -> Optional[ConditionalPattern]:
        true_target, false_target = self._deduce_binary_targets(descriptor)
        join_block = self._immediate_post_dominators.get(descriptor.block_start)
        style = "if"
        notes: List[PatternNote] = []

        loop = registry.enclosing_loop(descriptor.block_start)
        loop_body_target = None
        loop_exit_target = None
        if loop and loop.header == descriptor.block_start:
            style = "loop_guard"
            notes.append(PatternNote("branch is loop header"))
            if true_target is not None and loop.contains(true_target):
                loop_body_target = true_target
                loop_exit_target = false_target
            elif false_target is not None and loop.contains(false_target):
                loop_body_target = false_target
                loop_exit_target = true_target
            else:
                style = "if"
                notes.append(PatternNote("no loop body target resolved"))
        elif loop and loop.contains(descriptor.block_start):
            notes.append(PatternNote(f"branch inside loop headed at 0x{loop.header:06X}"))

        if descriptor.kind == BranchKind.ITERATOR:
            style = "loop_guard"
            notes.append(PatternNote("iterator control flow"))
        elif descriptor.kind == BranchKind.PREDICATED:
            notes.append(PatternNote("predicated branch"))

        true_region = self._collect_region(true_target, join_block, guard=descriptor.block_start)
        false_region = self._collect_region(false_target, join_block, guard=descriptor.block_start)

        confidence = 0.75
        if join_block is None:
            confidence = 0.5
            notes.append(PatternNote("no join block detected"))
        elif join_block == descriptor.block_start:
            confidence = 0.45
            notes.append(PatternNote("self post-dominating branch"))
        elif style == "loop_guard":
            confidence = 0.85

        return ConditionalPattern(
            block_start=descriptor.block_start,
            descriptor=descriptor,
            confidence=confidence,
            notes=tuple(notes),
            true_target=true_target,
            false_target=false_target,
            join_block=join_block,
            true_region=_sorted_tuple(true_region),
            false_region=_sorted_tuple(false_region),
            style=style,
            loop_body_target=loop_body_target,
            loop_exit_target=loop_exit_target,
        )

    def _deduce_binary_targets(
        self, descriptor: BranchDescriptor
    ) -> Tuple[Optional[int], Optional[int]]:
        fallthrough = descriptor.fallthrough
        true_target: Optional[int] = None
        false_target: Optional[int] = fallthrough
        for outcome in descriptor.outcomes:
            target = outcome.target
            if target == fallthrough or (fallthrough is None and target is None):
                false_target = target
            elif true_target is None:
                true_target = target
        if true_target is None and fallthrough is not None:
            true_target = fallthrough
        return true_target, false_target

    def _collect_region(
        self,
        start: Optional[int],
        stop: Optional[int],
        *,
        guard: Optional[int] = None,
    ) -> Set[int]:
        if start is None or start == stop:
            return set()
        visited: Set[int] = set()
        stack: List[int] = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            if current == stop or current == guard:
                continue
            visited.add(current)
            block = self.graph.blocks.get(current)
            if block is None:
                continue
            for successor in block.successors:
                if successor not in visited and successor != guard:
                    stack.append(successor)
        return visited

    # ------------------------------------------------------------------
    def _build_dispatch_pattern(self, descriptor: BranchDescriptor) -> DispatchPattern:
        join_block = self._immediate_post_dominators.get(descriptor.block_start)
        targets = _outcome_targets(descriptor)
        ladder: Set[int] = set()
        for outcome in descriptor.outcomes:
            if outcome.target is None:
                continue
            branch = self.structure.block_descriptor(outcome.target)
            if branch and branch.kind in {BranchKind.CONDITIONAL, BranchKind.PREDICATED}:
                if branch.fallthrough == join_block:
                    ladder.add(outcome.target)
        notes: List[PatternNote] = []
        if join_block is None:
            notes.append(PatternNote("dispatch without join block"))
        if ladder:
            notes.append(PatternNote("partial ladder detected"))
        return DispatchPattern(
            block_start=descriptor.block_start,
            descriptor=descriptor,
            confidence=0.7 if join_block is None else 0.85,
            notes=tuple(notes),
            targets=targets,
            join_block=join_block,
            ladder_blocks=_sorted_tuple(ladder),
        )

    # ------------------------------------------------------------------
    def _build_tail_call_pattern(self, descriptor: BranchDescriptor) -> Optional[TailCallPattern]:
        # Treat blocks with no fallthrough and a single outgoing edge as tail-calls.
        if descriptor.fallthrough is not None:
            return None
        targets = [outcome.target for outcome in descriptor.outcomes if outcome.target is not None]
        if len(targets) != 1:
            return None
        notes: List[PatternNote] = []
        if descriptor.warnings:
            notes.extend(PatternNote(message) for message in descriptor.warnings)
        return TailCallPattern(
            block_start=descriptor.block_start,
            descriptor=descriptor,
            confidence=0.6,
            notes=tuple(notes),
            callee=targets[0],
            returns=descriptor.kind == BranchKind.CALL,
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def analyse_branch_patterns(registry: BranchRegistry) -> BranchPatternRegistry:
    """Return a :class:`BranchPatternRegistry` derived from ``registry``."""

    analyzer = BranchPatternAnalyzer(registry.structure)
    return analyzer.analyse()


def render_branch_patterns(registry: BranchPatternRegistry) -> str:
    """Return a multi-line textual summary for the provided registry."""

    lines = registry.summary_lines()
    lines.append("")
    if registry.loops:
        lines.append("loops:")
        for loop in sorted(registry.loops.values(), key=lambda item: item.nesting_level):
            latch = ", ".join(f"0x{edge:06X}" for edge in loop.latch_blocks) or "<implicit>"
            exits = ", ".join(f"0x{exit:06X}" for exit in loop.exits) or "<implicit>"
            lines.append(
                f"  header 0x{loop.header:06X} style={loop.style} body={list(loop.body)} latch=[{latch}] exits=[{exits}]"
            )
    if registry.conditionals:
        lines.append("conditionals:")
        for pattern in sorted(registry.conditionals.values(), key=lambda item: item.block_start):
            lines.append(
                "  0x{block:06X}: style={style} true={true} false={false} join={join} confidence={conf:.2f}".format(
                    block=pattern.block_start,
                    style=pattern.style,
                    true=_format_optional(pattern.true_target),
                    false=_format_optional(pattern.false_target),
                    join=_format_optional(pattern.join_block),
                    conf=pattern.confidence,
                )
            )
    if registry.dispatches:
        lines.append("dispatches:")
        for pattern in sorted(registry.dispatches.values(), key=lambda item: item.block_start):
            targets = ", ".join(_format_optional(target) for target in pattern.targets)
            join = _format_optional(pattern.join_block)
            lines.append(
                f"  0x{pattern.block_start:06X}: targets=[{targets}] join={join} confidence={pattern.confidence:.2f}"
            )
    if registry.tail_calls:
        lines.append("tail-calls:")
        for pattern in sorted(registry.tail_calls.values(), key=lambda item: item.block_start):
            callee = _format_optional(pattern.callee)
            lines.append(
                f"  0x{pattern.block_start:06X}: callee={callee} returns={pattern.returns} confidence={pattern.confidence:.2f}"
            )
    return "\n".join(lines) + "\n"


def _format_optional(value: Optional[int]) -> str:
    if value is None:
        return "<none>"
    return f"0x{value:06X}"

