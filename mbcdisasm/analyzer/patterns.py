"""Pipeline pattern recognition helpers.

The module defines a declarative representation for instruction pipelines.  Each
:class:`PipelinePattern` consists of :class:`PatternToken` objects that describe
the expected instruction kinds, stack deltas and auxiliary constraints.  The
pipeline analyser compiles these patterns into deterministic finite automata that
can quickly scan long segments.  The actual matching logic is intentionally kept
human readable to make it easier to tweak heuristics without having to rewrite
large chunks of code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .instruction_profile import InstructionKind
from .stack import StackEvent


@dataclass(frozen=True)
class PatternToken:
    """Describe a single position in a pipeline pattern."""

    kinds: Tuple[InstructionKind, ...]
    min_delta: int = -999
    max_delta: int = 999
    allow_unknown: bool = False
    description: str = ""

    def matches(self, event: StackEvent) -> bool:
        """Return ``True`` if ``event`` satisfies this token."""

        if not self.allow_unknown and event.kind is InstructionKind.UNKNOWN:
            return False
        if self.kinds and event.kind not in self.kinds:
            return False
        if event.delta < self.min_delta or event.delta > self.max_delta:
            return False
        return True

    def describe(self) -> str:
        names = ",".join(kind.name for kind in self.kinds) if self.kinds else "*"
        return f"token({names} Δ∈[{self.min_delta},{self.max_delta}])"


@dataclass(frozen=True)
class PipelinePattern:
    """Declarative description of a known pipeline form."""

    name: str
    category: str
    tokens: Tuple[PatternToken, ...]
    allow_extra: bool = False
    stack_change: Optional[int] = None
    description: str = ""

    def match(self, events: Sequence[StackEvent]) -> Optional["PatternMatch"]:
        """Return a :class:`PatternMatch` if ``events`` fit the pattern."""

        if len(events) < len(self.tokens):
            return None

        slices = events[: len(self.tokens)]
        for token, event in zip(self.tokens, slices):
            if not token.matches(event):
                return None

        if not self.allow_extra and len(events) != len(self.tokens):
            return None

        if self.stack_change is not None:
            delta = sum(event.delta for event in events)
            if delta != self.stack_change:
                return None

        score = 1.0
        if any(event.uncertain for event in events):
            score *= 0.85
        if any(event.kind is InstructionKind.UNKNOWN for event in events):
            score *= 0.5

        return PatternMatch(pattern=self, events=tuple(events), score=score)

    def describe(self) -> str:
        return f"pattern {self.name} ({self.category})"


@dataclass(frozen=True)
class PatternMatch:
    """Result of a pattern match."""

    pattern: PipelinePattern
    events: Tuple[StackEvent, ...]
    score: float

    def span(self) -> Tuple[int, int]:
        start = self.events[0].profile.word.offset
        end = self.events[-1].profile.word.offset
        return start, end

    def describe(self) -> str:
        return f"{self.pattern.name} score={self.score:.2f}"


class PatternRegistry:
    """Container for all available patterns."""

    def __init__(self, patterns: Optional[Iterable[PipelinePattern]] = None) -> None:
        self._patterns: List[PipelinePattern] = list(patterns or [])

    def add(self, pattern: PipelinePattern) -> None:
        self._patterns.append(pattern)

    def extend(self, patterns: Iterable[PipelinePattern]) -> None:
        for pattern in patterns:
            self.add(pattern)

    def __iter__(self) -> Iterable[PipelinePattern]:
        return iter(self._patterns)

    def best_match(self, events: Sequence[StackEvent]) -> Optional[PatternMatch]:
        """Return the best scoring pattern for ``events``."""

        best: Optional[PatternMatch] = None
        for pattern in self._patterns:
            match = pattern.match(events)
            if match is None:
                continue
            if best is None or match.score > best.score:
                best = match
        return best


def default_patterns() -> PatternRegistry:
    """Return a :class:`PatternRegistry` populated with built-in patterns."""

    registry = PatternRegistry()
    registry.extend(
        [
            literal_run_pattern(),
            ascii_run_pattern(),
            marker_run_pattern(),
            literal_pipeline(),
            ascii_pipeline(),
            *guarded_return_patterns(),
            reduce_pipeline(),
            call_preparation_pipeline(),
            return_pipeline(),
            indirect_load_pipeline(),
            table_builder_pipeline(),
        ]
    )
    return registry


def literal_run_pattern() -> PipelinePattern:
    """Return a pattern that matches dense literal pushes."""

    tokens = tuple(
        PatternToken(
            kinds=(
                InstructionKind.LITERAL,
                InstructionKind.ASCII_CHUNK,
                InstructionKind.PUSH,
            ),
            min_delta=0,
            max_delta=1,
            description="literal burst",
        )
        for _ in range(4)
    )
    return PipelinePattern(
        name="literal_bulk",
        category="literal",
        tokens=tokens,
        allow_extra=True,
        description="Sequence of literal pushes (≥4 instructions)",
    )


def ascii_run_pattern() -> PipelinePattern:
    """Return a pattern that matches clusters of ASCII chunks."""

    tokens = tuple(
        PatternToken(
            kinds=(InstructionKind.ASCII_CHUNK,),
            min_delta=0,
            max_delta=0,
            description="ascii chunk",
        )
        for _ in range(3)
    )
    return PipelinePattern(
        name="ascii_stream",
        category="literal",
        tokens=tokens,
        allow_extra=True,
        stack_change=0,
        description="Run of inline ASCII words",
    )


def marker_run_pattern() -> PipelinePattern:
    """Return a pattern matching stack-neutral literal markers."""

    tokens = tuple(
        PatternToken(
            kinds=(InstructionKind.LITERAL,),
            min_delta=0,
            max_delta=0,
            description="literal marker",
        )
        for _ in range(3)
    )
    return PipelinePattern(
        name="marker_cluster",
        category="literal",
        tokens=tokens,
        allow_extra=True,
        stack_change=0,
        description="Cluster of literal markers",
    )


def literal_pipeline() -> PipelinePattern:
    """Return the pattern for ``literal -> push -> test`` style blocks."""

    tokens = (
        PatternToken(
            kinds=(InstructionKind.LITERAL, InstructionKind.ASCII_CHUNK),
            min_delta=1,
            description="literal load",
        ),
        PatternToken(
            kinds=(InstructionKind.PUSH,),
            min_delta=1,
            description="stack push",
        ),
        PatternToken(
            kinds=(InstructionKind.TEST, InstructionKind.BRANCH),
            min_delta=-1,
            max_delta=0,
            description="test/branch",
        ),
    )
    return PipelinePattern(
        name="literal_push_test",
        category="literal",
        tokens=tokens,
        stack_change=1,
        description="Load literal value, push to stack, perform test",
    )


def ascii_pipeline() -> PipelinePattern:
    """Return the pattern for inline ASCII chunk loaders."""

    tokens = (
        PatternToken(
            kinds=(InstructionKind.ASCII_CHUNK,),
            min_delta=1,
            description="ascii chunk load",
        ),
        PatternToken(
            kinds=(InstructionKind.REDUCE,),
            min_delta=-3,
            max_delta=-1,
            description="chunk reducer",
        ),
        PatternToken(
            kinds=(InstructionKind.PUSH,),
            min_delta=1,
            description="push reduced value",
        ),
    )
    return PipelinePattern(
        name="ascii_reduce_push",
        category="literal",
        tokens=tokens,
        stack_change=-1,
        description="Load ASCII chunk and collapse into a single value",
    )


def guarded_return_patterns() -> Tuple[PipelinePattern, ...]:
    """Return patterns describing conditional guards with immediate returns."""

    branch_token = PatternToken(
        kinds=(InstructionKind.BRANCH, InstructionKind.TEST),
        min_delta=-2,
        max_delta=1,
        description="guard branch",
    )
    guard_token = PatternToken(
        kinds=(InstructionKind.PUSH, InstructionKind.LITERAL, InstructionKind.STACK_COPY),
        min_delta=-1,
        max_delta=2,
        description="guard payload",
    )
    return_token = PatternToken(
        kinds=(InstructionKind.RETURN, InstructionKind.TERMINATOR),
        min_delta=-2,
        max_delta=0,
        description="guarded return",
    )

    prep_token = PatternToken(
        kinds=(InstructionKind.STACK_TEARDOWN,),
        min_delta=-4,
        max_delta=0,
        description="guard prep",
    )
    literal_token = PatternToken(
        kinds=(InstructionKind.LITERAL, InstructionKind.PUSH),
        min_delta=-1,
        max_delta=2,
        description="guard literal",
    )

    variants: Tuple[Tuple[PatternToken, ...], ...] = (
        (),
        (prep_token,),
        (literal_token,),
        (prep_token, literal_token),
    )

    patterns: List[PipelinePattern] = []
    for prefix in variants:
        tokens = prefix + (branch_token, guard_token, return_token)
        patterns.append(
            PipelinePattern(
                name="guarded_return",
                category="return",
                tokens=tokens,
                description="Conditional branch with guarded return",
            )
        )
    return tuple(patterns)


def reduce_pipeline() -> PipelinePattern:
    """Return a pattern for multi-argument reducers."""

    tokens = (
        PatternToken(
            kinds=(InstructionKind.PUSH, InstructionKind.LITERAL),
            min_delta=1,
            description="operand feed",
        ),
        PatternToken(
            kinds=(InstructionKind.PUSH, InstructionKind.LITERAL),
            min_delta=1,
            description="operand feed",
        ),
        PatternToken(
            kinds=(InstructionKind.REDUCE,),
            min_delta=-2,
            max_delta=-1,
            description="reduce",
        ),
    )
    return PipelinePattern(
        name="reduce_chain",
        category="compute",
        tokens=tokens,
        allow_extra=True,
        description="Feed operands then reduce",
    )


def call_preparation_pipeline() -> PipelinePattern:
    """Return the pattern for call frame initialisation."""

    tokens = (
        PatternToken(
            kinds=(InstructionKind.PUSH, InstructionKind.LITERAL, InstructionKind.TABLE_LOOKUP),
            min_delta=1,
            description="operand push",
        ),
        PatternToken(
            kinds=(InstructionKind.PUSH, InstructionKind.LITERAL, InstructionKind.TABLE_LOOKUP),
            min_delta=0,
            max_delta=2,
            description="additional operands",
        ),
        PatternToken(
            kinds=(InstructionKind.CALL, InstructionKind.TAILCALL, InstructionKind.META),
            min_delta=-1,
            max_delta=1,
            description="call helper",
        ),
    )
    return PipelinePattern(
        name="call_preparation",
        category="call",
        tokens=tokens,
        allow_extra=True,
        description="Prepare call frame and invoke helper",
    )


def table_builder_pipeline() -> PipelinePattern:
    """Return a pattern describing table builder prologues."""

    tokens = (
        PatternToken(
            kinds=(InstructionKind.UNKNOWN,),
            min_delta=-1,
            max_delta=1,
            description="table prologue",
        ),
        PatternToken(
            kinds=(
                InstructionKind.LITERAL,
                InstructionKind.ASCII_CHUNK,
                InstructionKind.PUSH,
            ),
            min_delta=-1,
            max_delta=2,
            description="table parameters",
            allow_unknown=True,
        ),
        PatternToken(
            kinds=(InstructionKind.UNKNOWN,),
            min_delta=-1,
            max_delta=1,
            description="table entries",
        ),
    )

    return PipelinePattern(
        name="table_builder",
        category="table",
        tokens=tokens,
        allow_extra=True,
        description="Prologue with literals followed by dense table construction",
    )


def return_pipeline() -> PipelinePattern:
    """Return the pattern describing teardown/return sequences."""

    tokens = (
        PatternToken(
            kinds=(InstructionKind.STACK_TEARDOWN,),
            min_delta=-4,
            max_delta=-1,
            description="frame teardown",
        ),
        PatternToken(
            kinds=(InstructionKind.RETURN, InstructionKind.TERMINATOR),
            min_delta=-2,
            max_delta=0,
            description="return/terminator",
        ),
    )
    return PipelinePattern(
        name="return_teardown",
        category="return",
        tokens=tokens,
        allow_extra=True,
        description="Drop frame values and return",
    )


def indirect_load_pipeline() -> PipelinePattern:
    """Return the pattern for table/indirect loads."""

    tokens = (
        PatternToken(
            kinds=(InstructionKind.PUSH, InstructionKind.LITERAL),
            min_delta=1,
            description="base push",
        ),
        PatternToken(
            kinds=(InstructionKind.PUSH, InstructionKind.LITERAL),
            min_delta=1,
            description="key push",
        ),
        PatternToken(
            kinds=(
                InstructionKind.INDIRECT,
                InstructionKind.INDIRECT_LOAD,
                InstructionKind.INDIRECT_STORE,
                InstructionKind.TABLE_LOOKUP,
            ),
            min_delta=-1,
            max_delta=0,
            description="indirect read",
        ),
    )
    return PipelinePattern(
        name="indirect_lookup",
        category="indirect",
        tokens=tokens,
        description="Push base/key then resolve table entry",
    )
