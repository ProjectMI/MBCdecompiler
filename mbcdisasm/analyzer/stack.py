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


class StackValueType(Enum):
    """Classify values tracked on the analysis stack."""

    UNKNOWN = auto()
    NUMBER = auto()
    SLOT = auto()
    IDENTIFIER = auto()
    MARKER = auto()


@dataclass
class TypeEffect:
    """Typed stack effect produced by an instruction."""

    hint: StackEffectHint
    pop_count: int
    push_types: Tuple[StackValueType, ...]
    token: Optional[str] = None
    invalidate_top: bool = False


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
    popped_types: Tuple[StackValueType, ...] = tuple()
    pushed_types: Tuple[StackValueType, ...] = tuple()
    token: Optional[str] = None

    def describe(self) -> str:
        token = f" token={self.token}" if self.token else ""
        return (
            f"{self.profile.word.offset:08X} {self.profile.label:<7} "
            f"Δ={self.delta:+d} range=({self.minimum:+d},{self.maximum:+d}) "
            f"depth={self.depth_before}->{self.depth_after}{token}"
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
        self._type_stack: List[StackValueType] = []

    def process(self, profile: InstructionProfile) -> StackEvent:
        """Process ``profile`` and record the resulting stack event."""

        hint = profile.estimated_stack_delta()
        effect = self._typed_effect(profile, hint)
        minimum, maximum, after, uncertain = self._state.apply(effect.hint)
        popped = self._pop_types(effect.pop_count)
        if effect.invalidate_top and self._type_stack:
            self._type_stack[-1] = StackValueType.UNKNOWN
        self._type_stack.extend(effect.push_types)
        event = StackEvent(
            profile=profile,
            delta=effect.hint.nominal,
            minimum=minimum,
            maximum=maximum,
            confidence=effect.hint.confidence,
            depth_before=after - effect.hint.nominal,
            depth_after=after,
            uncertain=uncertain,
            popped_types=popped,
            pushed_types=effect.push_types,
            token=effect.token,
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
        self._type_stack.clear()

    def depth(self) -> int:
        """Return the current stack depth estimate."""

        return self._state.depth

    def copy(self) -> "StackTracker":
        """Return a duplicate tracker with the same state."""

        clone = StackTracker()
        clone._state = self._state.fork()
        clone._events = list(self._events)
        clone._type_stack = list(self._type_stack)
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
    def _typed_effect(self, profile: InstructionProfile, hint: StackEffectHint) -> TypeEffect:
        if self._is_marker(profile):
            return TypeEffect(
                hint=StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=max(0.8, hint.confidence)),
                pop_count=0,
                push_types=tuple(),
                token="marker",
            )

        if profile.word.opcode == 0x69:
            return self._classify_indirect(profile, hint)

        push_types = self._infer_push_types(profile)
        if not push_types and hint.nominal > 0:
            push_types = tuple(StackValueType.UNKNOWN for _ in range(hint.nominal))

        pop_count = self._pop_count_from_hint(hint)
        return TypeEffect(hint=hint, pop_count=pop_count, push_types=push_types)

    def _pop_types(self, count: int) -> Tuple[StackValueType, ...]:
        if count <= 0:
            return tuple()
        popped: List[StackValueType] = []
        for _ in range(count):
            if not self._type_stack:
                popped.append(StackValueType.UNKNOWN)
                continue
            popped.append(self._type_stack.pop())
        popped.reverse()
        return tuple(popped)

    def _infer_push_types(self, profile: InstructionProfile) -> Tuple[StackValueType, ...]:
        kind = profile.kind

        if kind is InstructionKind.ASCII_CHUNK:
            return (StackValueType.IDENTIFIER,)

        if kind in {InstructionKind.LITERAL, InstructionKind.PUSH}:
            if self._looks_like_slot_literal(profile):
                return (StackValueType.SLOT,)
            if self._looks_like_identifier_literal(profile):
                return (StackValueType.IDENTIFIER,)
            return (StackValueType.NUMBER,)

        return tuple()

    def _classify_indirect(self, profile: InstructionProfile, hint: StackEffectHint) -> TypeEffect:
        stack = self._type_stack

        slot_on_top = bool(stack) and stack[-1] is StackValueType.SLOT
        value_on_top = bool(stack) and stack[-1] in {StackValueType.NUMBER, StackValueType.IDENTIFIER}
        slot_below = len(stack) >= 2 and stack[-2] is StackValueType.SLOT

        if slot_on_top and not (len(stack) >= 2 and stack[-2] in {StackValueType.NUMBER, StackValueType.IDENTIFIER}):
            adjusted = StackEffectHint(nominal=1, minimum=0, maximum=1, confidence=max(0.65, hint.confidence))
            return TypeEffect(
                hint=adjusted,
                pop_count=0,
                push_types=(StackValueType.NUMBER,),
                token="indirect_load",
            )

        if value_on_top and slot_below:
            adjusted = StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=max(0.6, hint.confidence * 0.95))
            return TypeEffect(
                hint=adjusted,
                pop_count=0,
                push_types=tuple(),
                token="indirect_store",
                invalidate_top=True,
            )

        widened = hint.widen(1)
        return TypeEffect(
            hint=widened,
            pop_count=self._pop_count_from_hint(widened),
            push_types=tuple(),
            token="indirect_access",
        )

    def _looks_like_slot_literal(self, profile: InstructionProfile) -> bool:
        operand = profile.operand & 0xFFFF
        if operand == 0:
            return False
        if operand & 0xFF:
            return False
        high = (operand >> 8) & 0xFF
        return high <= 0x20

    def _looks_like_identifier_literal(self, profile: InstructionProfile) -> bool:
        operand = profile.operand & 0xFFFF
        high = (operand >> 8) & 0xFF
        low = operand & 0xFF
        return 0x20 <= high <= 0x7E and 0x20 <= low <= 0x7E

    def _is_marker(self, profile: InstructionProfile) -> bool:
        mnemonic = profile.mnemonic.lower()
        return "marker" in mnemonic

    def _pop_count_from_hint(self, hint: StackEffectHint) -> int:
        if hint.nominal < 0:
            return abs(hint.nominal)
        if hint.minimum < 0:
            return abs(hint.minimum)
        return 0


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
