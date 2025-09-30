"""Heuristic pipeline signatures.

The pattern matcher in :mod:`mbcdisasm.analyzer.patterns` focuses on strict
instruction templates that can be expressed as deterministic finite automata.
While extremely fast, these templates struggle with the sprawling literal
sections present in scripts such as ``_char`` where hundreds of consecutive
instructions merely shuttle data around.  The :class:`SignatureDetector`
defined in this module complements the automaton-based approach with a set of
loosely defined *signatures*.  Each signature encodes a higher level behaviour
observed during manual reverse engineering sessions – for example "a run of
ASCII words" or "table slot initialisation" – and can match even when the
exact instruction mix varies slightly between occurrences.

The detector is intentionally opinionated and biased towards literal-heavy
pipelines because those are the hardest to classify without manual hints.  The
implementation is verbose; rich docstrings and descriptive variable names make
the heuristics easier to audit and tweak during future reversing sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from .instruction_profile import InstructionKind, InstructionProfile
from .stack import StackSummary


LiteralLike = {
    InstructionKind.LITERAL,
    InstructionKind.ASCII_CHUNK,
    InstructionKind.PUSH,
}


@dataclass(frozen=True)
class SignatureMatch:
    """Result of a successful signature detection."""

    name: str
    category: str
    confidence: float
    notes: Tuple[str, ...] = tuple()


class SignatureRule:
    """Base class for heuristic signature rules."""

    name: str = "signature"
    category: str = "unknown"
    base_confidence: float = 0.55

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        raise NotImplementedError


class AsciiRunSignature(SignatureRule):
    """Match blocks composed purely of ASCII chunk instructions."""

    name = "ascii_run"
    category = "literal"
    base_confidence = 0.68

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 3:
            return None
        if not all(profile.kind is InstructionKind.ASCII_CHUNK for profile in profiles):
            return None
        notes = (
            f"ascii_run length={len(profiles)}",
            f"stackΔ={stack.change:+d}",
        )
        return SignatureMatch(self.name, self.category, self.base_confidence, notes)


class LiteralRunSignature(SignatureRule):
    """Match blocks that contain a dense sequence of literal pushes."""

    name = "literal_run"
    category = "literal"
    base_confidence = 0.62

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None
        literal_like = sum(1 for profile in profiles if profile.kind in LiteralLike)
        density = literal_like / len(profiles)
        if density < 0.7:
            return None
        notes = (
            f"literal_density={density:.2f}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = self.base_confidence + 0.05 * (density - 0.7)
        return SignatureMatch(self.name, self.category, confidence, notes)


class MarkerRunSignature(SignatureRule):
    """Detect clusters of literal marker instructions."""

    name = "marker_run"
    category = "literal"
    base_confidence = 0.57

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 3:
            return None
        if not all(profile.mnemonic == "literal_marker" for profile in profiles):
            return None
        notes = (
            f"marker_cluster={len(profiles)}",
            f"stackΔ={stack.change:+d}",
        )
        return SignatureMatch(self.name, self.category, self.base_confidence, notes)


class TableStoreSignature(SignatureRule):
    """Recognise the table slot initialisation pattern used in ``_char``."""

    name = "table_store"
    category = "literal"
    base_confidence = 0.65

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 5:
            return None
        labels = [profile.label for profile in profiles]
        if labels[0] != "66:75":
            return None
        if "10:0E" not in labels[:3]:
            return None
        if not any(profile.kind is InstructionKind.PUSH for profile in profiles):
            return None
        notes = (
            "detected table slot flush",
            f"operands={[profile.operand for profile in profiles[:4]]}",
        )
        confidence = self.base_confidence
        if stack.change >= 0:
            confidence += 0.05
        return SignatureMatch(self.name, self.category, confidence, notes)


class IndirectFetchSignature(SignatureRule):
    """Match the indirect access setup commonly found around character tables."""

    name = "indirect_fetch"
    category = "indirect"
    base_confidence = 0.66

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None
        if profiles[-1].kind is not InstructionKind.INDIRECT:
            return None
        if not any(profile.label == "75:30" for profile in profiles):
            return None
        if not any(profile.kind in LiteralLike for profile in profiles):
            return None
        notes = (
            "indirect_fetch detected",
            f"stackΔ={stack.change:+d}",
        )
        return SignatureMatch(self.name, self.category, self.base_confidence, notes)


class SignatureDetector:
    """Run a collection of :class:`SignatureRule` objects on a block."""

    def __init__(self, rules: Optional[Iterable[SignatureRule]] = None) -> None:
        self.rules: Tuple[SignatureRule, ...] = tuple(rules or self._default_rules())

    @staticmethod
    def _default_rules() -> Tuple[SignatureRule, ...]:
        return (
            AsciiRunSignature(),
            TableStoreSignature(),
            IndirectFetchSignature(),
            LiteralRunSignature(),
            MarkerRunSignature(),
        )

    def detect(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        for rule in self.rules:
            match = rule.match(profiles, stack)
            if match is not None:
                return match
        return None
