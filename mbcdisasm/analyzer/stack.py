"""Stack tracking primitives used by the pipeline analyser.

The interpreter targeted by this project is stack based.  Every instruction
manipulates the evaluation stack in highly constrained ways which makes the
stack delta a reliable feature for classification.  The :class:`StackTracker`
class included in this module keeps a running tally of the stack height while it
visits the instruction stream.  The tracker is intentionally conservative: when
 faced with ambiguous data it records a range instead of guessing.  Downstream
consumers can then decide whether a block is reliable enough to be labelled as a
literal loader or whether it should be flagged for manual review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .instruction_profile import InstructionKind, InstructionProfile, StackEffectHint


@dataclass
class StackEvent:
    """Represents the effect of a single instruction on the stack."""

    profile: InstructionProfile
    delta: int
    minimum: int
    maximum: int
    confidence: float
    depth_before: int
    depth_after: int
    uncertain: bool = False
    marker_run: bool = False
    span: int = 1

    def describe(self) -> str:
        return (
            f"{self.profile.word.offset:08X} {self.profile.label:<7} "
            f"Δ={self.delta:+d} range=({self.minimum:+d},{self.maximum:+d}) "
            f"depth={self.depth_before}->{self.depth_after}"
        )


@dataclass
class StackState:
    """Running stack statistics."""

    depth: int = 0
    minimum_depth: int = 0
    maximum_depth: int = 0
    uncertain: bool = False

    def apply(self, hint: StackEffectHint) -> Tuple[int, int, int, bool]:
        """Apply ``hint`` and return the resulting statistics."""

        minimum = self.depth + hint.minimum
        maximum = self.depth + hint.maximum

        after = self.depth + hint.nominal
        self.depth = after
        self.minimum_depth = min(self.minimum_depth, minimum)
        self.maximum_depth = max(self.maximum_depth, maximum)
        uncertainty = self.uncertain or hint.confidence < 0.75
        self.uncertain = uncertainty
        return minimum, maximum, after, uncertainty

    def fork(self) -> "StackState":
        """Return a copy of the state.

        The method is handy for speculative flows where a caller wants to compute
        the effect of several instructions without mutating the primary state.
        """

        return StackState(
            depth=self.depth,
            minimum_depth=self.minimum_depth,
            maximum_depth=self.maximum_depth,
            uncertain=self.uncertain,
        )


@dataclass
class StackSummary:
    """Aggregate statistics for a block of instructions."""

    change: int
    minimum: int
    maximum: int
    uncertain: bool
    events: Tuple[StackEvent, ...]

    def describe(self) -> str:
        flag = "±" if self.uncertain else "="
        return f"stack{flag}{self.change:+d} range=({self.minimum:+d},{self.maximum:+d})"


class StackTracker:
    """Track the stack height as instructions are processed."""

    def __init__(self, initial_depth: int = 0) -> None:
        self._state = StackState(depth=initial_depth)
        self._events: List[StackEvent] = []

    def process(self, profile: InstructionProfile) -> StackEvent:
        """Process ``profile`` and record the resulting stack event."""

        hint = profile.estimated_stack_delta()
        minimum, maximum, after, uncertain = self._state.apply(hint)
        event = StackEvent(
            profile=profile,
            delta=hint.nominal,
            minimum=minimum,
            maximum=maximum,
            confidence=hint.confidence,
            depth_before=after - hint.nominal,
            depth_after=after,
            uncertain=uncertain,
        )
        self._events.append(event)
        return event

    def run(self, profiles: Sequence[InstructionProfile]) -> StackSummary:
        """Process ``profiles`` sequentially and return the summary."""

        for profile in profiles:
            self.process(profile)
        return self.summarise()

    def summarise(self) -> StackSummary:
        """Return a :class:`StackSummary` covering all processed instructions."""

        change = sum(event.delta for event in self._events)
        if not self._events:
            return StackSummary(change=0, minimum=0, maximum=0, uncertain=False, events=tuple())
        minimum = min(event.minimum for event in self._events)
        maximum = max(event.maximum for event in self._events)
        uncertain = any(event.uncertain for event in self._events)
        return StackSummary(
            change=change,
            minimum=minimum,
            maximum=maximum,
            uncertain=uncertain,
            events=tuple(self._events),
        )

    def reset(self, depth: int = 0) -> None:
        """Clear the recorded events and reset the depth."""

        self._state = StackState(depth=depth)
        self._events.clear()

    def depth(self) -> int:
        """Return the current stack depth estimate."""

        return self._state.depth

    def copy(self) -> "StackTracker":
        """Return a duplicate tracker with the same state."""

        clone = StackTracker()
        clone._state = self._state.fork()
        clone._events = list(self._events)
        return clone

    def process_block(self, profiles: Sequence[InstructionProfile]) -> StackSummary:
        """Process ``profiles`` using a dedicated tracker copy."""

        clone = self.copy()
        summary = clone.run(profiles)
        return StackSummary(
            change=summary.change,
            minimum=summary.minimum,
            maximum=summary.maximum,
            uncertain=summary.uncertain,
            events=summary.events,
        )


def stack_change(profiles: Sequence[InstructionProfile]) -> int:
    """Return the combined stack delta for ``profiles``."""

    tracker = StackTracker()
    summary = tracker.run(profiles)
    return summary.change


def contains_stack_teardown(profiles: Iterable[InstructionProfile]) -> bool:
    """Return ``True`` if any profile represents a stack teardown operation."""

    for profile in profiles:
        if profile.kind is InstructionKind.STACK_TEARDOWN:
            return True
    return False


def contains_literal_push(profiles: Iterable[InstructionProfile]) -> bool:
    """Return ``True`` if the block looks like a literal/push sequence."""

    literal_like = {
        InstructionKind.LITERAL,
        InstructionKind.ASCII_CHUNK,
        InstructionKind.PUSH,
        InstructionKind.TABLE_LOOKUP,
    }
    return any(profile.kind in literal_like for profile in profiles)
