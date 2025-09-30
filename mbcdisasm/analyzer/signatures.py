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


class MarkerPairWithHeaderSignature(SignatureRule):
    """Detect paired literal markers that introduce a fixed header sequence."""

    name = "marker_pair_with_header"
    category = "literal"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        pair_idx = next(
            (
                idx
                for idx in range(len(profiles) - 1)
                if is_literal_marker(profiles[idx]) and is_literal_marker(profiles[idx + 1])
            ),
            None,
        )
        if pair_idx is None:
            return None

        header_idx = pair_idx + 2
        if header_idx >= len(profiles):
            return None
        if not profiles[header_idx].label.startswith("41:B4"):
            return None

        if header_idx + 1 >= len(profiles):
            return None
        if not profiles[header_idx + 1].label.startswith("08:"):
            return None

        notes = (
            f"pair_idx={pair_idx}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = self.base_confidence
        if stack.change >= 0:
            confidence += 0.04
        return SignatureMatch(self.name, self.category, confidence, notes)


class AsciiPrologMarkerComboSignature(SignatureRule):
    """Recognise ASCII literals wrapped by a ``05:00`` prolog and marker pair."""

    name = "ascii_prolog_marker_combo"
    category = "literal"
    base_confidence = 0.59

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 5:
            return None

        if not profiles[0].label.startswith("05:"):
            return None

        ascii_idx = next(
            (
                idx
                for idx, profile in enumerate(profiles[1:], start=1)
                if profile.kind is InstructionKind.ASCII_CHUNK
            ),
            None,
        )
        if ascii_idx is None:
            return None

        helper_idx = next(
            (
                idx
                for idx in range(ascii_idx + 1, len(profiles))
                if profiles[idx].label.startswith("10:0F")
            ),
            None,
        )
        if helper_idx is None:
            return None

        marker_idx = next(
            (
                idx
                for idx in range(helper_idx + 1, len(profiles) - 1)
                if is_literal_marker(profiles[idx]) and is_literal_marker(profiles[idx + 1])
            ),
            None,
        )
        if marker_idx is None:
            return None

        notes = (
            f"ascii_idx={ascii_idx}",
            f"marker_idx={marker_idx}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.83, self.base_confidence + 0.04)
        return SignatureMatch(self.name, self.category, confidence, notes)


class AsciiWrapperEf48Signature(SignatureRule):
    """Match ASCII payloads wrapped by ``EF:28`` and ``48:00`` sentinels."""

    name = "ascii_wrapper_ef48"
    category = "literal"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        labels = [profile.label for profile in profiles]
        try:
            start_idx = next(idx for idx, label in enumerate(labels) if label.startswith("EF:28"))
        except StopIteration:
            return None

        try:
            end_idx = next(
                idx for idx in range(start_idx + 1, len(labels)) if labels[idx].startswith("48:00")
            )
        except StopIteration:
            return None

        if end_idx <= start_idx + 1:
            return None

        ascii_count = sum(
            1 for profile in profiles[start_idx + 1 : end_idx] if profile.kind is InstructionKind.ASCII_CHUNK
        )
        if ascii_count == 0:
            return None

        notes = (
            f"start_idx={start_idx}",
            f"end_idx={end_idx}",
            f"ascii_count={ascii_count}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.84, self.base_confidence + 0.05 * ascii_count)
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


class AsciiIndirectTailcallSignature(SignatureRule):
    """Match tailcalls that resolve through indirect markers and ASCII IDs."""

    name = "ascii_indirect_tailcall"
    category = "call"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 6:
            return None

        first_label = profiles[0].label
        if not (is_literal_marker(profiles[0]) or first_label.startswith(("00:38", "10:38"))):
            return None

        if not is_tailcall(profiles[1]):
            return None

        labels = [profile.label for profile in profiles]

        marker_4b_idx = next(
            (idx for idx in range(2, len(profiles)) if labels[idx].startswith("4B:")),
            None,
        )
        if marker_4b_idx is None:
            return None

        marker_69_idx = next(
            (idx for idx in range(marker_4b_idx + 1, len(profiles)) if labels[idx].startswith("69:")),
            None,
        )
        if marker_69_idx is None:
            return None

        literal_idx = next(
            (
                idx
                for idx in range(marker_69_idx + 1, len(profiles))
                if is_literal_like(profiles[idx])
            ),
            None,
        )
        if literal_idx is None:
            return None

        ascii_idx = next(
            (
                idx
                for idx in range(literal_idx + 1, len(profiles))
                if profiles[idx].kind is InstructionKind.ASCII_CHUNK
            ),
            None,
        )
        if ascii_idx is None:
            return None

        notes = (
            f"tail_idx=1",
            f"ascii_idx={ascii_idx}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.86, self.base_confidence + 0.05)
        return SignatureMatch(self.name, self.category, confidence, notes)


class TailcallPostJumpSignature(SignatureRule):
    """Recognise tailcalls that are immediately preceded by a jump."""

    name = "tailcall_post_jump"
    category = "call"
    base_confidence = 0.58

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        first = profiles[0]
        if first.kind not in {InstructionKind.BRANCH, InstructionKind.CONTROL} and not first.label.startswith("22:"):
            return None

        tail_idx = next((idx for idx, profile in enumerate(profiles[1:], start=1) if is_tailcall(profile)), None)
        if tail_idx is None:
            return None

        marker_idx = next(
            (idx for idx in range(tail_idx + 1, len(profiles)) if profiles[idx].label.startswith("69:")),
            None,
        )
        if marker_idx is None:
            return None

        notes = (
            f"jump_idx=0",
            f"tail_idx={tail_idx}",
            f"marker_idx={marker_idx}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = self.base_confidence
        if stack.change <= 0:
            confidence += 0.04
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


class TailcallReturnIndirectSignature(SignatureRule):
    """Detect tailcalls that return through helper + indirect cleanup."""

    name = "tailcall_return_indirect"
    category = "call"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 6:
            return None

        if not profiles[0].label.startswith("4C:"):
            return None

        tail_idx = next((idx for idx, profile in enumerate(profiles) if is_tailcall(profile)), None)
        if tail_idx is None:
            return None

        return_idx = next(
            (
                idx
                for idx in range(tail_idx + 1, len(profiles))
                if profiles[idx].kind in {InstructionKind.RETURN, InstructionKind.TERMINATOR}
            ),
            None,
        )
        if return_idx is None:
            return None

        literal_idx = next(
            (
                idx
                for idx in range(return_idx + 1, len(profiles))
                if is_literal_like(profiles[idx])
            ),
            None,
        )
        if literal_idx is None:
            return None

        helper_idx = next(
            (
                idx
                for idx in range(literal_idx + 1, len(profiles))
                if is_call_helper(profiles[idx]) and profiles[idx].label.startswith("10:")
            ),
            None,
        )
        if helper_idx is None:
            return None

        indirect_idx = next(
            (idx for idx in range(helper_idx + 1, len(profiles)) if profiles[idx].label.startswith("69:")),
            None,
        )
        if indirect_idx is None:
            return None

        notes = (
            f"tail_idx={tail_idx}",
            f"return_idx={return_idx}",
            f"indirect_idx={indirect_idx}",
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


class IndirectCallDualLiteralSignature(SignatureRule):
    """Detect indirect call setups that use two literal pushes with F1/10 markers."""

    name = "indirect_call_dual_literal"
    category = "call"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 6:
            return None

        labels = [profile.label for profile in profiles]
        if "10:38" not in labels:
            return None

        literal_indices = [idx for idx, profile in enumerate(profiles) if is_literal_like(profile)]
        if len(literal_indices) < 2:
            return None

        first_push = literal_indices[0]
        second_push = next((idx for idx in literal_indices[1:] if idx > first_push), None)
        if second_push is None:
            return None

        f1_idx = next(
            (idx for idx in range(first_push + 1, second_push) if labels[idx] == "F1:3D"),
            None,
        )
        if f1_idx is None:
            return None

        helper_idx = next(
            (idx for idx in range(f1_idx + 1, second_push) if labels[idx] == "10:48"),
            None,
        )
        if helper_idx is None:
            return None

        if not any(label.startswith("69:") for label in labels[second_push + 1 :]):
            return None

        notes = (
            f"first_push={first_push}",
            f"second_push={second_push}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.88, self.base_confidence + 0.04 * (second_push - first_push))
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
            MarkerPairWithHeaderSignature(),
            AsciiPrologMarkerComboSignature(),
            AsciiWrapperEf48Signature(),
            LiteralRunWithMarkersSignature(),
            LiteralReduceChainExSignature(),
            AsciiTailcallPatternSignature(),
            AsciiIndirectTailcallSignature(),
            TailcallPostJumpSignature(),
            TailcallReturnComboSignature(),
            TailcallReturnIndirectSignature(),
            CallprepAsciiDispatchSignature(),
            FanoutTeardownSignature(),
            IndirectCallDualLiteralSignature(),
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
