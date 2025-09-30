"""Instruction profiling helpers.

The :class:`InstructionProfile` class wraps :class:`~mbcdisasm.instruction.InstructionWord`
instances with metadata derived from the knowledge base and with additional
heuristic classification.  Pipeline recognition operates on these enriched
profiles rather than the bare instruction words because the metadata exposes
events such as *"this instruction consumes the top stack value"* or *"this is a
terminator"*.  Each profile attempts to normalise the low level signals into a
small set of categories that can be consumed by finite state matchers or by the
stack tracker.

The module is intentionally verbose – every helper method is documented and the
logic is broken into many small units.  This makes the heuristics easy to unit
 test and provides numerous hooks for future experimentation.  The heavy
commentary also serves as a form of living documentation for the reversing work
which has traditionally been scattered across private notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from ..instruction import InstructionWord
from ..knowledge import KnowledgeBase, OpcodeInfo

# ---------------------------------------------------------------------------
# Heuristic opcode helpers
# ---------------------------------------------------------------------------

ASCII_ALLOWED = set(range(0x20, 0x7F))
ASCII_ALLOWED.update({0x09, 0x0A, 0x0D})  # tab/newline characters often occur

ASCII_HEURISTIC_SUMMARY = (
    "Эвристически восстановленный ASCII-блок (четыре печатаемых байта)."
)

# Families of opcodes that can be classified purely from the opcode value.  The
# manual annotation database does not yet cover the hundreds of helper opcodes
# present in ``_char`` so the fallback classifier relies on these sets to assign
# sensible :class:`InstructionKind` values.
CALL_OPCODE_FAMILY = {
    0x10,
    0x11,
    0x12,
    0x13,
    0x16,
    0x17,
    0x18,
    0x19,
    0x28,
    0x4B,
    0x84,
    0x88,
    0x8C,
    0xA4,
    0xA8,
    0xAC,
    0xB4,
    0xB8,
    0xBC,
    0xF0,
    0xF1,
}

INDIRECT_OPCODE_FAMILY = {0x69}

TABLE_LOOKUP_OPCODE_FAMILY = {0x75}


class InstructionKind(Enum):
    """High level classification of an opcode.

    The enumeration is intentionally exhaustive and contains several flavours of
    stack manipulation instructions.  Lua-style VMs tend to encode literal
    loaders, stack adjustments and indirect loads via distinct opcodes which are
    extremely regular across the binary.  Having discrete entries for these
    groups simplifies the later pattern matching passes.
    """

    CONTROL = auto()
    TERMINATOR = auto()
    BRANCH = auto()
    CALL = auto()
    TAILCALL = auto()
    RETURN = auto()
    LITERAL = auto()
    ASCII_CHUNK = auto()
    PUSH = auto()
    REDUCE = auto()
    STACK_TEARDOWN = auto()
    STACK_COPY = auto()
    TEST = auto()
    INDIRECT = auto()
    TABLE_LOOKUP = auto()
    ARITHMETIC = auto()
    LOGICAL = auto()
    BITWISE = auto()
    META = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class StackEffectHint:
    """Represents a stack delta hint used during analysis.

    The hint stores both a nominal delta and a range.  When the manual
    annotations specify an exact ``stack_delta`` the ``minimum`` and ``maximum``
    values are equal to that number.  When only pushes or pops are defined the
    class normalises them into a signed delta.  The pipeline analyser prefers to
    work with :class:`StackEffectHint` instances instead of raw integers because
    several opcodes expose mode-dependent behaviour.  In those cases a wide
    ``minimum``/``maximum`` range is recorded which tells the stack tracker to be
    conservative.
    """

    nominal: int
    minimum: int
    maximum: int
    confidence: float = 1.0

    @classmethod
    def from_info(cls, info: Optional[OpcodeInfo]) -> "StackEffectHint":
        """Build a hint from :class:`OpcodeInfo` metadata.

        When the manual annotations do not expose any stack information a neutral
        hint (``nominal == minimum == maximum == 0``) is returned with a low
        confidence score.  The analyser will treat such instructions as
        stack-neutral unless additional heuristics override the estimate.
        """

        if info is None:
            return cls(nominal=0, minimum=0, maximum=0, confidence=0.25)

        if info.stack_delta is not None:
            delta = int(info.stack_delta)
            return cls(nominal=delta, minimum=delta, maximum=delta, confidence=0.95)

        pushed = info.stack_push if info.stack_push is not None else 0
        popped = info.stack_pop if info.stack_pop is not None else 0

        delta = pushed - popped
        spread = max(abs(pushed), abs(popped))
        confidence = 0.5 if spread else 0.25

        minimum = -popped
        maximum = pushed

        return cls(nominal=delta, minimum=minimum, maximum=maximum, confidence=confidence)

    def widen(self, amount: int) -> "StackEffectHint":
        """Return a new hint with an expanded range.

        Some instructions (for example polymorphic reducers) have a mode-dependent
        stack delta.  The knowledge database can mark such cases by creating a
        neutral hint and letting the classifier widen the range when a specific
        mode is encountered.
        """

        return StackEffectHint(
            nominal=self.nominal,
            minimum=self.minimum - amount,
            maximum=self.maximum + amount,
            confidence=self.confidence * 0.9,
        )


@dataclass
class InstructionProfile:
    """Bundle low level instruction details with analysis helpers."""

    word: InstructionWord
    info: Optional[OpcodeInfo]
    mnemonic: str
    summary: Optional[str]
    category: Optional[str]
    control_flow: Optional[str]
    stack_hint: StackEffectHint
    kind: InstructionKind
    traits: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_word(cls, word: InstructionWord, knowledge: KnowledgeBase) -> "InstructionProfile":
        """Create a profile for ``word`` using ``knowledge`` annotations."""

        info, heuristic = resolve_opcode_info(word, knowledge)
        mnemonic = info.mnemonic if info else f"op_{word.opcode:02X}_{word.mode:02X}"
        summary = info.summary if info else None
        category = info.category if info else None
        control_flow = info.control_flow if info else None
        stack_hint = StackEffectHint.from_info(info)
        kind = classify_kind(word, info)
        if info and info.attributes:
            traits: Mapping[str, object] = dict(info.attributes)
        else:
            traits = {}
        if heuristic:
            traits = dict(traits)
            traits.setdefault("heuristic", True)
        return cls(
            word=word,
            info=info,
            mnemonic=mnemonic,
            summary=summary,
            category=category,
            control_flow=control_flow,
            stack_hint=stack_hint,
            kind=kind,
            traits=traits,
        )

    @property
    def label(self) -> str:
        return self.word.label()

    @property
    def opcode(self) -> int:
        return self.word.opcode

    @property
    def mode(self) -> int:
        return self.word.mode

    @property
    def operand(self) -> int:
        return self.word.operand

    def estimated_stack_delta(self) -> StackEffectHint:
        """Return the stack hint adjusted by heuristics."""

        hint = self.stack_hint
        modifier = heuristic_stack_adjustment(self)
        if modifier is None:
            return hint
        nominal = hint.nominal + modifier
        return StackEffectHint(
            nominal=nominal,
            minimum=hint.minimum + modifier,
            maximum=hint.maximum + modifier,
            confidence=min(1.0, hint.confidence + 0.1),
        )

    def is_control(self) -> bool:
        return self.kind in {InstructionKind.CONTROL, InstructionKind.BRANCH, InstructionKind.TERMINATOR}

    def is_terminator(self) -> bool:
        return self.kind is InstructionKind.TERMINATOR

    def describe(self) -> str:
        """Return a debugging string summarising the profile."""

        return (
            f"{self.word.offset:08X} {self.label:<7} {self.kind.name:<14} "
            f"stack≈{self.stack_hint.nominal:+d}"
        )


def classify_kind(word: InstructionWord, info: Optional[OpcodeInfo]) -> InstructionKind:
    """Classify ``word`` using ``info`` heuristics."""

    if info is None:
        if looks_like_ascii_chunk(word):
            return InstructionKind.ASCII_CHUNK
        return guess_kind_from_opcode(word)

    if info.control_flow:
        control = info.control_flow.lower()
        if "return" in control:
            return InstructionKind.RETURN
        if "terminator" in control or "halt" in control:
            return InstructionKind.TERMINATOR
        if "branch" in control or "jump" in control:
            return InstructionKind.BRANCH
        if "call" in control:
            if "tail" in (info.category or ""):
                return InstructionKind.TAILCALL
            return InstructionKind.CALL
        if "control" in control:
            return InstructionKind.CONTROL

    if info.category:
        category = info.category.lower()
        if "ascii" in category:
            return InstructionKind.ASCII_CHUNK
        if "literal" in category:
            return InstructionKind.LITERAL
        if "push" in category:
            return InstructionKind.PUSH
        if "reduce" in category or "fold" in category:
            return InstructionKind.REDUCE
        if "teardown" in category or "pop" in category:
            return InstructionKind.STACK_TEARDOWN
        if "copy" in category or "dup" in category:
            return InstructionKind.STACK_COPY
        if "test" in category:
            return InstructionKind.TEST
        if "indirect" in category or "table" in category:
            return InstructionKind.INDIRECT
        if "tailcall" in category:
            return InstructionKind.TAILCALL
        if "return" in category:
            return InstructionKind.RETURN
        if "terminator" in category:
            return InstructionKind.TERMINATOR

    mnemonic = (info.mnemonic or "").lower()
    summary = (info.summary or "").lower()

    for source in (mnemonic, summary):
        if not source:
            continue
        if "literal" in source or "const" in source:
            return InstructionKind.LITERAL
        if "push" in source and "stack" in source:
            return InstructionKind.PUSH
        if "reduce" in source or "fold" in source:
            return InstructionKind.REDUCE
        if "test" in source and "branch" in source:
            return InstructionKind.TEST
        if "stack" in source and ("clear" in source or "teardown" in source or "drop" in source):
            return InstructionKind.STACK_TEARDOWN
        if "duplicate" in source or "copy" in source:
            return InstructionKind.STACK_COPY
        if "table" in source or "index" in source:
            return InstructionKind.TABLE_LOOKUP
        if "arith" in source or "math" in source:
            return InstructionKind.ARITHMETIC
        if "logic" in source or "boolean" in source:
            return InstructionKind.LOGICAL
        if "bit" in source:
            return InstructionKind.BITWISE
        if "meta" in source or "helper" in source:
            return InstructionKind.META

    if looks_like_ascii_chunk(word):
        return InstructionKind.ASCII_CHUNK
    return guess_kind_from_opcode(word)


def guess_kind_from_opcode(word: InstructionWord) -> InstructionKind:
    """Fallback heuristic when no manual metadata is available."""

    opcode = word.opcode
    mode = word.mode

    if opcode == 0x00:
        return InstructionKind.LITERAL

    if opcode in {0x29, 0x30}:
        if mode in {0x00, 0x10, 0x30, 0x69}:
            return InstructionKind.TERMINATOR
        return InstructionKind.RETURN

    if opcode in {0x22, 0x23, 0x24, 0x25, 0x26, 0x27}:
        return InstructionKind.BRANCH

    if opcode in CALL_OPCODE_FAMILY:
        return InstructionKind.CALL

    if opcode in INDIRECT_OPCODE_FAMILY:
        return InstructionKind.INDIRECT

    if opcode in TABLE_LOOKUP_OPCODE_FAMILY:
        return InstructionKind.TABLE_LOOKUP

    if opcode == 0xFF:
        return InstructionKind.TERMINATOR

    if opcode in {0x41, 0x47, 0x90, 0xDC, 0xF4}:
        return InstructionKind.ASCII_CHUNK

    if opcode in {0x01}:
        return InstructionKind.STACK_TEARDOWN

    if opcode in {0x02, 0x03}:
        return InstructionKind.PUSH

    if opcode in {0x04}:
        return InstructionKind.REDUCE

    if opcode in {0x05, 0x06, 0x07, 0x08}:
        return InstructionKind.ARITHMETIC

    return InstructionKind.META


def resolve_opcode_info(
    word: InstructionWord, knowledge: KnowledgeBase
) -> Tuple[Optional[OpcodeInfo], bool]:
    """Return the most suitable :class:`OpcodeInfo` for ``word``.

    The helper mirrors :meth:`KnowledgeBase.lookup` but augments the results
    with lightweight heuristics that recognise frequently occurring instruction
    families (for example inline ASCII data blocks).  The boolean flag reports
    whether heuristics were involved which allows callers to highlight the
    source in diagnostic output.
    """

    info = knowledge.lookup(word.label())
    if info is not None:
        return info, False

    heuristic = heuristic_opcode_info(word)
    return heuristic, heuristic is not None


def heuristic_opcode_info(word: InstructionWord) -> Optional[OpcodeInfo]:
    """Return a synthetic :class:`OpcodeInfo` inferred from ``word``."""

    if looks_like_ascii_chunk(word):
        return OpcodeInfo(
            mnemonic="inline_ascii_chunk",
            summary=ASCII_HEURISTIC_SUMMARY,
            control_flow="fallthrough",
            category="ascii_literal",
            stack_delta=0,
            attributes={"source": "heuristic", "kind": "ascii_chunk"},
        )
    return None


def looks_like_ascii_chunk(word: InstructionWord) -> bool:
    """Return ``True`` if ``word`` resembles a packed ASCII chunk."""

    raw = word.raw.to_bytes(4, "big")
    if all(byte == 0 for byte in raw):
        return False
    printable = 0
    zero_seen = False
    for index, byte in enumerate(raw):
        if byte == 0:
            if not zero_seen:
                # Allow trailing NUL terminators but reject embedded zeros.
                if any(value != 0 for value in raw[index + 1 :]):
                    return False
            zero_seen = True
            continue
        if zero_seen:
            return False
        if byte not in ASCII_ALLOWED:
            return False
        if 0x20 <= byte <= 0x7E:
            printable += 1
    return printable > 0


def heuristic_stack_adjustment(profile: InstructionProfile) -> Optional[int]:
    """Return additional stack delta adjustments derived from heuristics."""

    kind = profile.kind
    word = profile.word

    if kind is InstructionKind.ASCII_CHUNK and profile.stack_hint.nominal == 0:
        return 1

    if kind is InstructionKind.LITERAL and profile.stack_hint.nominal == 0:
        return 1

    if kind is InstructionKind.PUSH and profile.stack_hint.nominal == 0:
        return 1

    if kind in {InstructionKind.STACK_TEARDOWN, InstructionKind.REDUCE} and profile.stack_hint.nominal == 0:
        return -1

    if kind is InstructionKind.TABLE_LOOKUP and profile.stack_hint.nominal == 0:
        return 0

    if kind is InstructionKind.TEST and profile.stack_hint.nominal == 0:
        return -1 if word.mode & 0x01 else 0

    if kind in {InstructionKind.CALL, InstructionKind.TAILCALL} and profile.stack_hint.nominal == 0:
        return 0

    return None


def summarise_profiles(profiles: Sequence[InstructionProfile]) -> Mapping[InstructionKind, int]:
    """Return a histogram of instruction kinds in ``profiles``."""

    histogram: dict[InstructionKind, int] = {}
    for profile in profiles:
        histogram[profile.kind] = histogram.get(profile.kind, 0) + 1
    return histogram


# When multiple kinds tie for dominance we prefer the ones earlier in this
# ordering.  The priority list favours call/return/branch classifications before
# literal and compute categories which mirrors how the pipeline is typically
# consumed: recovering the control structure takes precedence over the literal
# payload.
DOMINANT_PRIORITY = [
    InstructionKind.CALL,
    InstructionKind.TAILCALL,
    InstructionKind.RETURN,
    InstructionKind.TERMINATOR,
    InstructionKind.BRANCH,
    InstructionKind.TEST,
    InstructionKind.INDIRECT,
    InstructionKind.TABLE_LOOKUP,
    InstructionKind.LITERAL,
    InstructionKind.ASCII_CHUNK,
    InstructionKind.PUSH,
    InstructionKind.REDUCE,
    InstructionKind.ARITHMETIC,
    InstructionKind.LOGICAL,
    InstructionKind.BITWISE,
    InstructionKind.STACK_TEARDOWN,
    InstructionKind.STACK_COPY,
    InstructionKind.META,
]


def dominant_kind(profiles: Sequence[InstructionProfile]) -> InstructionKind:
    """Return the most common instruction kind in ``profiles``."""

    histogram = summarise_profiles(profiles)
    if not histogram:
        return InstructionKind.UNKNOWN

    if InstructionKind.UNKNOWN in histogram and len(histogram) > 1:
        histogram = {
            kind: count
            for kind, count in histogram.items()
            if kind is not InstructionKind.UNKNOWN
        }

    if not histogram:
        return InstructionKind.UNKNOWN

    priority = {kind: idx for idx, kind in enumerate(DOMINANT_PRIORITY)}

    def score(item: Tuple[InstructionKind, int]) -> Tuple[int, int]:
        kind, count = item
        return (count, -priority.get(kind, len(priority)))

    return max(histogram.items(), key=score)[0]


def filter_profiles(
    profiles: Sequence[InstructionProfile], kinds: Iterable[InstructionKind]
) -> Tuple[InstructionProfile, ...]:
    """Return a tuple with the profiles matching ``kinds``."""

    allowed = set(kinds)
    return tuple(profile for profile in profiles if profile.kind in allowed)
