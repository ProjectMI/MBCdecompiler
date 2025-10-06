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

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, List, Optional, Sequence, Tuple

from .instruction_profile import InstructionKind, InstructionProfile, StackEffectHint


CALL_HELPER_LABELS = {
    "10:E8",
    "16:00",
    "16:01",
    "16:02",
    "16:04",
    "16:AC",
}


TAIL_MASK_LABELS = {
    "29:10",
    "32:29",
    "52:05",
    "5E:29",
    "70:29",
    "0B:29",
    "06:66",
    "F0:4B",
    "6C:01",
    "4A:05",
    "05:00",
}


ZERO_STACK_MNEMONICS = {"fanout", "stack_shuffle", "call_helpers"}


class StackValueType(Enum):
    """Lightweight classification for stack values."""

    UNKNOWN = auto()
    NUMBER = auto()
    SLOT = auto()
    IDENTIFIER = auto()
    MARKER = auto()


@dataclass(frozen=True)
class StackEffectDetails:
    """Combined stack effect information for an instruction."""

    hint: StackEffectHint
    popped: Tuple[StackValueType, ...] = tuple()
    pushed: Tuple[StackValueType, ...] = tuple()
    kind: InstructionKind | None = None


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
    kind: InstructionKind
    popped_types: Tuple[StackValueType, ...] = tuple()
    pushed_types: Tuple[StackValueType, ...] = tuple()
    uncertain: bool = False

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
    types: List[StackValueType] = field(default_factory=list)

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
            types=list(self.types),
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

    def process(
        self, profile: InstructionProfile, *, effect: StackEffectDetails | None = None
    ) -> StackEvent:
        """Process ``profile`` and record the resulting stack event."""

        if effect is None:
            effect = infer_stack_effect((profile,), 0, prior_types=tuple(self._state.types))

        hint = effect.hint
        before = self._state.depth
        minimum, maximum, after, uncertain = self._state.apply(hint)
        self._apply_types(effect.popped, effect.pushed)
        event = StackEvent(
            profile=profile,
            delta=hint.nominal,
            minimum=minimum,
            maximum=maximum,
            confidence=hint.confidence,
            depth_before=before,
            depth_after=after,
            kind=effect.kind or profile.kind,
            popped_types=effect.popped,
            pushed_types=effect.pushed,
            uncertain=uncertain,
        )
        self._events.append(event)
        return event

    def _apply_types(
        self, popped: Sequence[StackValueType], pushed: Sequence[StackValueType]
    ) -> None:
        if not popped and not pushed:
            return

        # Trim the current stack representation using best-effort semantics.
        for value in popped:
            if value is StackValueType.MARKER:
                continue
            if self._state.types:
                self._state.types.pop()
        for value in pushed:
            if value is StackValueType.MARKER:
                continue
            self._state.types.append(value)

    def run(self, profiles: Sequence[InstructionProfile]) -> StackSummary:
        """Process ``profiles`` sequentially and return the summary."""

        self.process_sequence(profiles)
        return self.summarise()

    def process_sequence(self, profiles: Sequence[InstructionProfile]) -> Tuple[StackEvent, ...]:
        """Process ``profiles`` while taking local context into account."""

        events: List[StackEvent] = []
        total = len(profiles)
        for index, profile in enumerate(profiles):
            effect = infer_stack_effect(
                profiles,
                index,
                prior_types=tuple(self._state.types),
            )
            events.append(self.process(profile, effect=effect))
        return tuple(events)

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


class IndirectVariant(Enum):
    """Flavours of ``indirect_access`` opcodes recognised by the tracker."""

    LOAD = auto()
    STORE = auto()


def infer_stack_effect(
    profiles: Sequence[InstructionProfile],
    index: int,
    *,
    prior_types: Sequence[StackValueType] = (),
) -> StackEffectDetails:
    """Return a :class:`StackEffectDetails` for ``profiles[index]``."""

    profile = profiles[index]
    hint = profile.estimated_stack_delta()
    pushed = list(_default_push_types(profile))
    popped: List[StackValueType] = []
    kind_override: InstructionKind | None = None

    if profile.is_literal_marker():
        hint = StackEffectHint(
            nominal=0,
            minimum=0,
            maximum=0,
            confidence=max(0.85, hint.confidence),
        )
        # Markers carry metadata but should not occupy stack slots.
        pushed = [StackValueType.MARKER]

    if _is_indirect_candidate(profile):
        variant = classify_indirect_variant(profiles, index, prior_types)
        if variant is IndirectVariant.STORE:
            hint = StackEffectHint(
                nominal=0,
                minimum=0,
                maximum=0,
                confidence=max(0.65, hint.confidence),
            )
            kind_override = InstructionKind.INDIRECT_STORE
            popped = [StackValueType.SLOT, StackValueType.NUMBER]
            pushed = []
        else:
            hint = StackEffectHint(
                nominal=1,
                minimum=1,
                maximum=1,
                confidence=max(0.65, hint.confidence),
            )
            kind_override = InstructionKind.INDIRECT_LOAD
            popped = [StackValueType.SLOT]
            pushed = [StackValueType.NUMBER]

    override = _stack_convention_override(profile)
    if override is not None:
        hint = override.hint
        popped = list(override.popped)
        pushed = list(override.pushed)
        if override.kind is not None:
            kind_override = override.kind

    return StackEffectDetails(
        hint=hint,
        popped=tuple(popped),
        pushed=tuple(pushed),
        kind=kind_override,
    )


def _default_push_types(profile: InstructionProfile) -> Tuple[StackValueType, ...]:
    if profile.is_literal_marker():
        return (StackValueType.MARKER,)
    if profile.kind is InstructionKind.LITERAL:
        return (StackValueType.NUMBER,)
    if profile.kind is InstructionKind.ASCII_CHUNK:
        return (StackValueType.IDENTIFIER,)
    if profile.kind is InstructionKind.PUSH:
        return (StackValueType.SLOT,)
    if profile.kind in {
        InstructionKind.TABLE_LOOKUP,
        InstructionKind.INDIRECT,
        InstructionKind.INDIRECT_LOAD,
    }:
        return (StackValueType.NUMBER,)
    return tuple()


def _stack_convention_override(profile: InstructionProfile) -> Optional[StackEffectDetails]:
    mnemonic = profile.mnemonic
    label = profile.label.upper()

    if mnemonic in ZERO_STACK_MNEMONICS or label in CALL_HELPER_LABELS or label in TAIL_MASK_LABELS:
        hint = StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=0.9)
        return StackEffectDetails(hint=hint)

    count = _stack_teardown_count(mnemonic, profile.operand)
    if count is not None and count >= 0:
        hint = StackEffectHint(nominal=-count, minimum=-count, maximum=-count, confidence=0.95)
        popped = tuple(StackValueType.UNKNOWN for _ in range(count))
        return StackEffectDetails(hint=hint, popped=popped)

    return None


def _stack_teardown_count(mnemonic: str, operand: int) -> Optional[int]:
    if mnemonic.startswith("stack_teardown_"):
        suffix = mnemonic.rsplit("_", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return None
    if mnemonic == "stack_teardown" and operand:
        return operand & 0xFF
    return None


def _is_indirect_candidate(profile: InstructionProfile) -> bool:
    if profile.kind in {
        InstructionKind.INDIRECT,
        InstructionKind.INDIRECT_LOAD,
        InstructionKind.INDIRECT_STORE,
    }:
        return True
    mnemonic = profile.mnemonic.lower()
    if "indirect" in mnemonic:
        return True
    return profile.label.startswith("69:")


def classify_indirect_variant(
    profiles: Sequence[InstructionProfile],
    index: int,
    prior_types: Sequence[StackValueType],
) -> IndirectVariant:
    """Return whether the current instruction behaves like a load or a store."""

    # Look ahead for teardown helpers that typically follow store sequences.
    for offset in range(1, 4):
        next_index = index + offset
        if next_index >= len(profiles):
            break
        follower = profiles[next_index]
        label = follower.label.upper()
        if label in {"F1:3D", "01:F0"}:
            return IndirectVariant.STORE
        if follower.kind in {InstructionKind.REDUCE, InstructionKind.STACK_TEARDOWN}:
            return IndirectVariant.STORE
        if follower.kind not in {
            InstructionKind.LITERAL,
            InstructionKind.ASCII_CHUNK,
            InstructionKind.PUSH,
            InstructionKind.UNKNOWN,
            InstructionKind.META,
        }:
            break

    if prior_types:
        if prior_types[-1] is StackValueType.SLOT:
            return IndirectVariant.LOAD

    return IndirectVariant.LOAD


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

