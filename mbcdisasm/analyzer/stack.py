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
from typing import Iterable, List, Sequence, Tuple

from ..constants import CALL_SHUFFLE_STANDARD, LITERAL_MARKER_HINTS, RET_MASK
from .instruction_profile import InstructionKind, InstructionProfile, StackEffectHint


_LITERAL_MARKER_VALUES = frozenset(LITERAL_MARKER_HINTS)


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

    hint, popped, pushed, kind_override = _apply_literal_chain_overrides(
        profiles,
        index,
        profile,
        hint,
        popped,
        pushed,
        kind_override,
    )

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

    hint, popped, pushed, kind_override = _apply_wrapper_overrides(
        profiles,
        index,
        profile,
        hint,
        popped,
        pushed,
        kind_override,
    )

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


# Known stack shuffle operands associated with call wrappers.  The constants are
# derived from the helper call metadata and the normaliser's cleanup rules.
_CALL_SHUFFLE_VARIANTS = {CALL_SHUFFLE_STANDARD, 0x3032, 0x7223}

# Opcodes that participate in the helper/tailcall wrapper but do not modify the
# stack height.  They typically handle book-keeping tasks such as masking or
# pointer updates and therefore can be modelled as stack-neutral operations with
# high confidence once recognised.
_WRAPPER_META_OPS = {
    (0x4A, 0x05),  # op_4A_05 helper dispatch
    (0x32, 0x29),  # op_32_29 cleanup helper
    (0x52, 0x05),  # op_52_05 ascii helpers
    (0x70, 0x29),  # mask propagation helpers
    (0x0B, 0x29),  # alternate mask helpers
    (0x06, 0x66),  # shuffle/mask adapters
}

# Modes frequently observed for the ``call_helpers`` opcodes.  The manual
# annotation database historically used decimal labels for these instructions
# which made the automated lookup brittle.  The tracker therefore recognises
# them heuristically based on the opcode/mode pair.
_CALL_HELPER_MODES = {0x00, 0x01, 0x02, 0x04, 0x64, 0x84, 0xAC, 0xD0, 0xE8}


def _apply_wrapper_overrides(
    profiles: Sequence[InstructionProfile],
    index: int,
    profile: InstructionProfile,
    hint: StackEffectHint,
    popped: List[StackValueType],
    pushed: List[StackValueType],
    kind_override: InstructionKind | None,
) -> Tuple[StackEffectHint, List[StackValueType], List[StackValueType], InstructionKind | None]:
    """Inject deterministic stack behaviour for helper wrappers."""

    opcode = profile.opcode
    mode = profile.mode
    operand = profile.operand
    original_kind = profile.kind

    # Tail dispatchers that merely shuttle the return mask are stack neutral.
    if opcode == 0x29 and mode == 0x10 and operand == RET_MASK:
        hint = StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=0.9)
        kind_override = InstructionKind.TAILCALL
        return hint, popped, pushed, kind_override

    # Dedicated helper entrypoints (opcode 0x10/0x16) behave like a ``call``
    # instruction in the IR but do not change the evaluation stack.  Boost the
    # confidence so that downstream passes can treat them as reliable anchors.
    if (
        opcode in {0x10, 0x16}
        and mode in _CALL_HELPER_MODES
        and original_kind in {InstructionKind.UNKNOWN, InstructionKind.CALL, InstructionKind.META}
    ):
        hint = StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=0.85)
        kind_override = InstructionKind.CALL
        return hint, popped, pushed, kind_override

    # The primary call dispatch instruction (0x28) similarly leaves the stack
    # unchanged – the helpers clean up afterwards.  Mark it as a high-confidence
    # call so that the pipeline classifier can rely on it when segmenting
    # blocks.
    if opcode == 0x28:
        hint = StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=0.85)
        kind_override = InstructionKind.CALL
        return hint, popped, pushed, kind_override

    if opcode == 0x30 and hint.confidence < 0.75:
        hint = StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=0.85)
        kind_override = InstructionKind.RETURN
        return hint, popped, pushed, kind_override

    # Recognise the canonical stack shuffles used before helper calls.  These
    # are pure permutations of the current frame and therefore stack neutral.
    if opcode == 0x66 and operand in _CALL_SHUFFLE_VARIANTS:
        hint = StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=0.9)
        kind_override = InstructionKind.STACK_COPY
        return hint, popped, pushed, kind_override

    # Stack teardown helpers encode the arity in their mnemonic (e.g.
    # ``stack_teardown_4``).  Parse the suffix to obtain a deterministic stack
    # delta and emit the matching number of ``UNKNOWN`` pops so that the tracker
    # can keep the virtual stack depth accurate.
    if opcode == 0x01:
        count = _teardown_count(profile)
        if count:
            hint = StackEffectHint(nominal=-count, minimum=-count, maximum=-count, confidence=0.9)
            popped = [StackValueType.UNKNOWN] * count
            pushed = []
            kind_override = InstructionKind.STACK_TEARDOWN
            return hint, popped, pushed, kind_override

    # Miscellaneous helper wrappers that only update metadata.
    if (opcode, mode) in _WRAPPER_META_OPS and _has_wrapper_context(profiles, index):
        hint = StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=0.85)
        if kind_override is None or kind_override is InstructionKind.UNKNOWN:
            kind_override = InstructionKind.META
        return hint, popped, pushed, kind_override

    return hint, popped, pushed, kind_override


def _has_wrapper_context(profiles: Sequence[InstructionProfile], index: int) -> bool:
    """Return ``True`` if nearby instructions resemble call/tail wrappers."""

    total = len(profiles)
    for offset in range(1, 4):
        prev = index - offset
        if prev >= 0 and profiles[prev].opcode in {0x10, 0x16, 0x28, 0x29, 0x66}:
            return True
        nxt = index + offset
        if nxt < total and profiles[nxt].opcode in {0x10, 0x16, 0x28, 0x29, 0x66, 0x01}:
            return True
    return False


def _teardown_count(profile: InstructionProfile) -> int:
    """Extract the number of slots cleared by a stack teardown helper."""

    name = profile.mnemonic.lower()
    if "stack_teardown" not in name:
        return 0
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return 0
    suffix = parts[1]
    if not suffix.isdigit():
        return 0
    return int(suffix)


def _apply_literal_chain_overrides(
    profiles: Sequence[InstructionProfile],
    index: int,
    profile: InstructionProfile,
    hint: StackEffectHint,
    popped: Sequence[StackValueType],
    pushed: Sequence[StackValueType],
    kind_override: InstructionKind | None,
) -> tuple[StackEffectHint, List[StackValueType], List[StackValueType], InstructionKind | None]:
    """Boost confidence for literal/reduce chains.

    The bytecode frequently emits the ``push_literal``/``push_literal``/
    ``reduce_pair`` trio when materialising constants.  The manual knowledge base
    does not yet annotate their stack effects which forces the tracker to rely on
    low-confidence heuristics.  Recognising the pattern allows us to attribute a
    deterministic stack delta to every participant which removes the "uncertain"
    noise from pipeline reports.
    """

    opcode = profile.opcode
    role = _literal_chain_role(profiles, index)

    if role in {"literal_head", "literal_tail"}:
        stable_hint = StackEffectHint(nominal=1, minimum=1, maximum=1, confidence=0.9)
        return stable_hint, list(popped), [StackValueType.NUMBER], kind_override

    if role == "reduce_pair":
        stable_hint = StackEffectHint(nominal=-1, minimum=-1, maximum=-1, confidence=0.9)
        pops = [StackValueType.NUMBER, StackValueType.NUMBER]
        pushes = [StackValueType.NUMBER]
        return stable_hint, pops, pushes, InstructionKind.REDUCE

    if opcode == 0x00 and profile.operand in _LITERAL_MARKER_VALUES:
        stable_hint = StackEffectHint(nominal=1, minimum=1, maximum=1, confidence=0.9)
        pushes = [StackValueType.NUMBER]
        if kind_override is None:
            kind_override = InstructionKind.LITERAL
        return stable_hint, list(popped), pushes, kind_override

    return hint, list(popped), list(pushed), kind_override


def _literal_chain_role(
    profiles: Sequence[InstructionProfile], index: int
) -> str | None:
    """Return the role ``profiles[index]`` plays in a literal reduction chain."""

    current = profiles[index]
    opcode = current.opcode
    mode = current.mode

    if opcode == 0x04 and mode == 0x00:
        if index >= 2 and all(profiles[pos].opcode == 0x00 for pos in (index - 1, index - 2)):
            return "reduce_pair"
        return None

    if opcode != 0x00:
        return None

    total = len(profiles)

    if index + 2 < total:
        second = profiles[index + 1]
        third = profiles[index + 2]
        if second.opcode == 0x00 and third.opcode == 0x04 and third.mode == 0x00:
            return "literal_head"

    if index >= 1 and index + 1 < total:
        prev = profiles[index - 1]
        nxt = profiles[index + 1]
        if prev.opcode == 0x00 and nxt.opcode == 0x04 and nxt.mode == 0x00:
            return "literal_tail"

    return None


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

