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


INDIRECT_RETURN_TERMINALS = {"2C:01", "66:3E", "F1:3D", "10:48"}


def is_literal_marker(profile: InstructionProfile) -> bool:
    """Return ``True`` when ``profile`` represents a literal marker opcode."""

    if profile.mnemonic == "literal_marker":
        return True

    opcode = profile.label.split(":", 1)[0]
    if opcode in {"40", "67", "69"}:
        return True
    return False


def is_literal_like(profile: InstructionProfile) -> bool:
    """Return ``True`` when the instruction behaves like a literal loader."""

    return profile.kind in LiteralLike or is_literal_marker(profile)


def is_call_helper(profile: InstructionProfile) -> bool:
    """Return ``True`` for helper opcodes involved in call setup/teardown."""

    mnemonic = profile.mnemonic.lower()
    summary = (profile.summary or "").lower()
    label = profile.label

    if "helper" in mnemonic or "call_helper" in mnemonic:
        return True
    if "helper" in summary:
        return True

    if label.startswith("10:"):
        return True

    if label.startswith("16:") and profile.kind in {InstructionKind.CALL, InstructionKind.META}:
        return True
    return False


def is_tailcall(profile: InstructionProfile) -> bool:
    """Return ``True`` when ``profile`` behaves like a tailcall dispatch."""

    if profile.kind is InstructionKind.TAILCALL:
        return True

    label = profile.label
    if label.startswith("29:"):
        return True

    mnemonic = profile.mnemonic.lower()
    summary = (profile.summary or "").lower()
    if "tail" in mnemonic or "tail" in summary:
        return True

    return False


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


class HeaderAsciiCtrlSeqSignature(SignatureRule):
    """Match ASCII headers that transition into control sequences."""

    name = "header_ascii_ctrl_seq"
    category = "literal"
    base_confidence = 0.6
    _ctrl_labels = {"34:2E", "33:FF", "EB:0B", "C9:29"}

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        ascii_prefix = 0
        for profile in profiles:
            if profile.kind is InstructionKind.ASCII_CHUNK:
                ascii_prefix += 1
                continue
            break

        if ascii_prefix < 2:
            return None

        trailing = profiles[ascii_prefix:]
        if len(trailing) < 2:
            return None

        ctrl_hits = sum(1 for profile in trailing if profile.label in self._ctrl_labels)
        if ctrl_hits < 2:
            return None

        notes = (
            f"ascii_prefix={ascii_prefix}",
            f"ctrl_hits={ctrl_hits}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.85, self.base_confidence + 0.05 * (ctrl_hits - 1))
        return SignatureMatch(self.name, self.category, confidence, notes)


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


class LiteralRunWithMarkersSignature(SignatureRule):
    """Detect literal bursts that interleave explicit marker opcodes."""

    name = "literal_run_with_markers"
    category = "literal"
    base_confidence = 0.61

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        marker_positions = [idx for idx, profile in enumerate(profiles) if is_literal_marker(profile)]
        if len(marker_positions) < 2:
            return None

        if not any(b - a == 1 for a, b in zip(marker_positions, marker_positions[1:])):
            return None

        literal_like = sum(1 for profile in profiles if is_literal_like(profile))
        density = literal_like / len(profiles)
        if density < 0.6:
            return None

        notes = (
            f"marker_pairs={len(marker_positions)}",
            f"literal_density={density:.2f}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.85, self.base_confidence + 0.04 * (density - 0.6))
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


class LiteralReduceChainExSignature(SignatureRule):
    """Recognise literal chains punctuated by reduction helpers."""

    name = "literal_reduce_chain_ex"
    category = "literal"
    base_confidence = 0.63

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 5:
            return None

        reduce_count = sum(1 for profile in profiles if profile.kind is InstructionKind.REDUCE)
        if reduce_count == 0:
            return None

        literal_like = sum(1 for profile in profiles if is_literal_like(profile))
        if literal_like < 3:
            return None

        density = literal_like / len(profiles)
        if density < 0.55:
            return None

        notes = (
            f"reduces={reduce_count}",
            f"literal_density={density:.2f}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(
            0.88,
            self.base_confidence + 0.05 * min(reduce_count, 3) + 0.03 * (density - 0.55),
        )
        return SignatureMatch(self.name, self.category, confidence, notes)


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


class AsciiTailcallPatternSignature(SignatureRule):
    """Match tailcalls that dispatch using ASCII identifiers."""

    name = "ascii_tailcall_pattern"
    category = "call"
    base_confidence = 0.59
    _anchors = {"00:52", "4A:05", "03:00", "30:32"}

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 5:
            return None

        tail_idx = next((idx for idx, profile in enumerate(profiles) if is_tailcall(profile)), None)
        if tail_idx is None or tail_idx >= len(profiles) - 1:
            return None

        ascii_after = any(profile.kind is InstructionKind.ASCII_CHUNK for profile in profiles[tail_idx + 1 :])
        if not ascii_after:
            return None

        literal_prefix = sum(1 for profile in profiles[:tail_idx] if is_literal_like(profile))
        if literal_prefix < 2:
            return None

        anchor_hits = sum(1 for profile in profiles if profile.label in self._anchors)

        notes = (
            f"tail_idx={tail_idx}",
            f"anchor_hits={anchor_hits}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.87, self.base_confidence + 0.04 * anchor_hits)
        return SignatureMatch(self.name, self.category, confidence, notes)


class TailcallReturnComboSignature(SignatureRule):
    """Detect tailcalls that immediately collapse into a return."""

    name = "tailcall_return_combo"
    category = "call"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        tail_idx = next((idx for idx, profile in enumerate(profiles) if is_tailcall(profile)), None)
        if tail_idx is None:
            return None

        return_idx = next(
            (
                idx
                for idx, profile in enumerate(profiles)
                if idx > tail_idx and profile.kind in {InstructionKind.RETURN, InstructionKind.TERMINATOR}
            ),
            None,
        )
        if return_idx is None:
            return None

        prefix_literals = sum(1 for profile in profiles[:tail_idx] if is_literal_like(profile))
        if prefix_literals == 0:
            return None

        notes = (
            f"tail_idx={tail_idx}",
            f"return_idx={return_idx}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = self.base_confidence
        if stack.change <= 0:
            confidence += 0.05
        return SignatureMatch(self.name, self.category, confidence, notes)


class CallprepAsciiDispatchSignature(SignatureRule):
    """Recognise call helpers that dispatch via ASCII payloads."""

    name = "callprep_ascii_dispatch"
    category = "call"
    base_confidence = 0.6
    _anchors = {"4B:3C", "41:A4", "00:05"}

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        if not is_call_helper(profiles[0]):
            return None

        ascii_count = sum(1 for profile in profiles[1:] if profile.kind is InstructionKind.ASCII_CHUNK)
        literal_count = sum(1 for profile in profiles[1:] if is_literal_like(profile))
        if ascii_count == 0 or literal_count == 0:
            return None

        anchor_hits = sum(1 for profile in profiles if profile.label in self._anchors)
        if anchor_hits == 0:
            return None

        notes = (
            f"ascii_count={ascii_count}",
            f"literal_count={literal_count}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.86, self.base_confidence + 0.05 * anchor_hits)
        return SignatureMatch(self.name, self.category, confidence, notes)


class FanoutTeardownSignature(SignatureRule):
    """Match helper blocks that duplicate arguments and then tear the stack down."""

    name = "fanout_teardown_seq"
    category = "call"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 3:
            return None

        has_call_helper = any(is_call_helper(profile) for profile in profiles)
        if not has_call_helper:
            return None

        has_fanout = any(
            profile.kind is InstructionKind.STACK_COPY
            or profile.mnemonic.lower().startswith("fanout")
            or profile.label.startswith("66:")
            for profile in profiles
        )
        has_teardown = any(profile.kind is InstructionKind.STACK_TEARDOWN for profile in profiles)
        if not (has_fanout and has_teardown):
            return None

        notes = (
            "fanout_teardown detected",
            f"stackΔ={stack.change:+d}",
        )
        confidence = self.base_confidence
        if stack.change < 0:
            confidence += 0.05
        return SignatureMatch(self.name, self.category, confidence, notes)


class IndirectCallExSignature(SignatureRule):
    """Recognise extended indirect call setup blocks."""

    name = "indirect_call_ex"
    category = "call"
    base_confidence = 0.58

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 3:
            return None

        if not is_call_helper(profiles[0]):
            return None

        marker_idx = next(
            (
                idx
                for idx, profile in enumerate(profiles[1:], start=1)
                if is_literal_marker(profile) and profile.label.startswith("69:")
            ),
            None,
        )
        if marker_idx is None:
            return None

        if profiles[-1].label in INDIRECT_RETURN_TERMINALS:
            return None

        trailing_literal = any(
            is_literal_like(profile) or profile.kind is InstructionKind.ASCII_CHUNK
            for profile in profiles[marker_idx + 1 :]
        )
        if not trailing_literal:
            return None

        notes = (
            f"marker_idx={marker_idx}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.85, self.base_confidence + 0.04 * (len(profiles) - marker_idx))
        return SignatureMatch(self.name, self.category, confidence, notes)


class IndirectReturnExSignature(SignatureRule):
    """Detect the tail of indirect call sequences with unusual terminators."""

    name = "indirect_return_ex"
    category = "call"
    base_confidence = 0.56
    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 3:
            return None

        if profiles[-1].label not in INDIRECT_RETURN_TERMINALS:
            return None

        if not any(is_call_helper(profile) for profile in profiles[:-1]):
            return None

        literal_tail = sum(
            1 for profile in profiles[:-1] if is_literal_like(profile) or profile.kind is InstructionKind.ASCII_CHUNK
        )
        if literal_tail == 0:
            return None

        notes = (
            f"terminal={profiles[-1].label}",
            f"literal_tail={literal_tail}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = self.base_confidence
        if stack.change <= 0:
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
            HeaderAsciiCtrlSeqSignature(),
            TableStoreSignature(),
            IndirectFetchSignature(),
            LiteralRunWithMarkersSignature(),
            LiteralReduceChainExSignature(),
            AsciiTailcallPatternSignature(),
            TailcallReturnComboSignature(),
            CallprepAsciiDispatchSignature(),
            FanoutTeardownSignature(),
            IndirectCallExSignature(),
            IndirectReturnExSignature(),
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
