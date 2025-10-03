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
from enum import Enum, auto
from typing import Iterable, List, Optional, Sequence, Tuple

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
    popped: Tuple["StackValue", ...] = tuple()
    pushed: Tuple["StackValue", ...] = tuple()
    tag: str | None = None
    ignore_for_tokens: bool = False

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


class StackValueKind(Enum):
    """Lightweight classification of values tracked on the stack."""

    UNKNOWN = auto()
    NUMBER = auto()
    SLOT = auto()
    IDENTIFIER = auto()
    MARKER = auto()


@dataclass(frozen=True)
class StackValue:
    """Typed value stored on the analysis stack."""

    kind: StackValueKind
    weight: int = 1
    source: Optional[str] = None


@dataclass
class StackEffectDescriptor:
    """Describes the typed effect of an instruction on the stack."""

    hint: StackEffectHint
    pop_weight: int = 0
    pop_markers: int = 0
    push_values: Tuple[StackValue, ...] = tuple()
    tag: Optional[str] = None
    ignore_for_tokens: bool = False


class StackTracker:
    """Track the stack height as instructions are processed."""

    def __init__(self, initial_depth: int = 0) -> None:
        self._state = StackState(depth=initial_depth)
        self._events: List[StackEvent] = []
        self._values: List[StackValue] = [
            StackValue(StackValueKind.UNKNOWN, source="initial") for _ in range(initial_depth)
        ]

    def process(
        self,
        profile: InstructionProfile,
        *,
        following: Sequence[InstructionProfile] | None = None,
    ) -> StackEvent:
        """Process ``profile`` and record the resulting stack event."""

        descriptor = self._describe_effect(profile, tuple(following or ()))
        popped_markers = self._pop_markers(descriptor.pop_markers)
        popped_weight = self._pop_weight(descriptor.pop_weight)
        popped = tuple(popped_markers + popped_weight)
        for value in descriptor.push_values:
            self._values.append(value)
        hint = descriptor.hint
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
            popped=popped,
            pushed=descriptor.push_values,
            tag=descriptor.tag,
            ignore_for_tokens=descriptor.ignore_for_tokens,
        )
        self._events.append(event)
        return event

    def run(self, profiles: Sequence[InstructionProfile]) -> StackSummary:
        """Process ``profiles`` sequentially and return the summary."""

        self._events.clear()
        total = len(profiles)
        for idx, profile in enumerate(profiles):
            following = profiles[idx + 1 : idx + 4]
            self.process(profile, following=following)
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
        self._values = [StackValue(StackValueKind.UNKNOWN, source="initial") for _ in range(depth)]

    def depth(self) -> int:
        """Return the current stack depth estimate."""

        return self._state.depth

    def copy(self) -> "StackTracker":
        """Return a duplicate tracker with the same state."""

        clone = StackTracker()
        clone._state = self._state.fork()
        clone._events = list(self._events)
        clone._values = list(self._values)
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

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _describe_effect(
        self, profile: InstructionProfile, following: Sequence[InstructionProfile]
    ) -> StackEffectDescriptor:
        if self._is_marker(profile):
            marker = StackValue(StackValueKind.MARKER, weight=0, source=profile.label)
            return StackEffectDescriptor(
                hint=StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=0.95),
                push_values=(marker,),
                ignore_for_tokens=True,
            )

        if profile.kind in {InstructionKind.LITERAL, InstructionKind.PUSH}:
            value = StackValue(StackValueKind.NUMBER, source=profile.label)
            hint = profile.estimated_stack_delta()
            return StackEffectDescriptor(hint=hint, push_values=(value,))

        if profile.kind is InstructionKind.ASCII_CHUNK:
            value = StackValue(StackValueKind.IDENTIFIER, source=profile.label)
            hint = profile.estimated_stack_delta()
            return StackEffectDescriptor(hint=hint, push_values=(value,))

        if profile.kind is InstructionKind.INDIRECT or self._looks_like_indirect(profile):
            return self._describe_indirect(profile, following)

        hint = profile.estimated_stack_delta()
        pop_weight = max(0, -hint.nominal)
        push_count = max(0, hint.nominal)
        push_values: Tuple[StackValue, ...] = tuple(
            StackValue(StackValueKind.UNKNOWN, source=profile.label) for _ in range(push_count)
        )
        return StackEffectDescriptor(hint=hint, pop_weight=pop_weight, push_values=push_values)

    def _describe_indirect(
        self, profile: InstructionProfile, following: Sequence[InstructionProfile]
    ) -> StackEffectDescriptor:
        marker_count = self._count_trailing_markers()
        has_value = self._has_value_before_markers(marker_count)
        store_candidate = has_value

        next_profile = self._next_non_marker(following)
        if next_profile is not None:
            if next_profile.kind in {InstructionKind.REDUCE, InstructionKind.STACK_TEARDOWN}:
                store_candidate = True
            elif next_profile.kind in {InstructionKind.LITERAL, InstructionKind.PUSH}:
                store_candidate = False

        if store_candidate:
            return StackEffectDescriptor(
                hint=StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=0.8),
                pop_markers=marker_count,
                tag="indirect_store",
            )

        value = StackValue(StackValueKind.NUMBER, source=profile.label)
        return StackEffectDescriptor(
            hint=StackEffectHint(nominal=1, minimum=0, maximum=1, confidence=0.8),
            pop_markers=marker_count,
            push_values=(value,),
            tag="indirect_load",
        )

    def _count_trailing_markers(self) -> int:
        count = 0
        for value in reversed(self._values):
            if value.kind is StackValueKind.MARKER:
                count += 1
                continue
            break
        return count

    def _has_value_before_markers(self, marker_count: int) -> bool:
        index = len(self._values) - marker_count - 1
        if index < 0:
            return False
        value = self._values[index]
        return value.weight > 0

    def _next_non_marker(
        self, following: Sequence[InstructionProfile]
    ) -> Optional[InstructionProfile]:
        for profile in following:
            if not self._is_marker(profile):
                return profile
        return None

    def _pop_markers(self, count: int) -> List[StackValue]:
        popped: List[StackValue] = []
        while count > 0 and self._values:
            value = self._values[-1]
            if value.kind is not StackValueKind.MARKER:
                break
            popped.append(self._values.pop())
            count -= 1
        return popped

    def _pop_weight(self, amount: int) -> List[StackValue]:
        if amount <= 0:
            return []
        popped: List[StackValue] = []
        remaining = amount
        while remaining > 0 and self._values:
            value = self._values.pop()
            popped.append(value)
            remaining -= max(0, value.weight)
        return popped

    @staticmethod
    def _is_marker(profile: InstructionProfile) -> bool:
        if profile.kind is InstructionKind.MARKER:
            return True
        if profile.mnemonic == "literal_marker":
            return True
        if profile.label == "69:00":
            return True
        return False

    @staticmethod
    def _looks_like_indirect(profile: InstructionProfile) -> bool:
        label = profile.label
        if label.startswith("69:") and label != "69:00":
            return True
        return False


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
