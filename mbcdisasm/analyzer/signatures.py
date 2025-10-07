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

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from ..constants import RET_MASK
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
    if opcode in {"40", "67"}:
        return True
    return False


def is_indirect_access(profile: InstructionProfile) -> bool:
    """Return ``True`` for helper opcodes involved in indirect access."""

    if profile.kind in {
        InstructionKind.INDIRECT,
        InstructionKind.INDIRECT_LOAD,
        InstructionKind.INDIRECT_STORE,
    }:
        return True

    return profile.label.startswith("69:")


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

    name = "ascii_header"
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


class ScriptHeaderPrologSignature(SignatureRule):
    """Recognise the ``script v`` style ASCII prologs with control tail."""

    name = "script_header_prolog"
    category = "literal"
    base_confidence = 0.58

    _required_suffix = ("34:2E", "58:26", "06:00", "63:00")

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 5:
            return None

        ascii_prefix = 0
        for profile in profiles:
            if profile.kind is InstructionKind.ASCII_CHUNK:
                ascii_prefix += 1
                continue
            break

        if ascii_prefix < 2:
            return None

        labels = [profile.label for profile in profiles]
        search_from = ascii_prefix
        for label in self._required_suffix:
            try:
                idx = labels.index(label, search_from)
            except ValueError:
                return None
            if idx < search_from:
                return None
            search_from = idx + 1

        notes = (
            f"ascii_prefix={ascii_prefix}",
            f"tail_labels={self._required_suffix}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.86, self.base_confidence + 0.05 * (ascii_prefix - 1))
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


class ReduceAsciiPrologSignature(SignatureRule):
    """Recognise the ``reduce`` + ``00:4F`` marker that precedes ASCII payloads."""

    name = "ascii_inline"
    category = "literal"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 3:
            return None

        first, second = profiles[0], profiles[1]
        if first.kind is not InstructionKind.REDUCE:
            return None
        if second.label != "00:4F":
            return None

        ascii_after = sum(
            1 for profile in profiles[2:] if profile.kind is InstructionKind.ASCII_CHUNK
        )
        if ascii_after == 0:
            return None

        notes = (
            f"ascii_after={ascii_after}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.88, self.base_confidence + 0.05 * min(ascii_after, 3))
        return SignatureMatch(self.name, self.category, confidence, notes)


class MarkerPairWithHeaderSignature(SignatureRule):
    """Detect paired literal markers that introduce a fixed header sequence."""

    name = "ascii_marker"
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

    name = "ascii_marker_combo"
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

    name = "ascii_wrapper"
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


class AsciiReduceMarkerSignature(SignatureRule):
    """Detect ASCII or reduction prologs that end with ``66:1B`` → markers."""

    name = "ascii_block"
    category = "literal"
    base_confidence = 0.58

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        labels = [profile.label for profile in profiles]
        try:
            idx_reduce = labels.index("66:1B")
        except ValueError:
            return None

        if idx_reduce + 2 >= len(profiles):
            return None
        if labels[idx_reduce + 1] != "00:52" or labels[idx_reduce + 2] != "4A:05":
            return None

        ascii_hits = sum(
            1 for profile in profiles[:idx_reduce] if profile.kind is InstructionKind.ASCII_CHUNK
        )
        reduce_prefix = any(
            profile.kind is InstructionKind.REDUCE for profile in profiles[:idx_reduce]
        )

        if ascii_hits == 0 and not reduce_prefix:
            return None

        trailing_label = labels[idx_reduce + 3] if idx_reduce + 3 < len(labels) else "<none>"

        notes = (
            f"ascii_hits={ascii_hits}",
            f"reduce_prefix={reduce_prefix}",
            f"trailing={trailing_label}",
            f"stackΔ={stack.change:+d}",
        )

        confidence = self.base_confidence
        if ascii_hits:
            confidence += 0.05
        if reduce_prefix:
            confidence += 0.03
        return SignatureMatch(self.name, self.category, min(0.87, confidence), notes)


class AsciiControlClusterSignature(SignatureRule):
    """Recognise ASCII clusters followed by control marker combos."""

    name = "ascii_control_cluster"
    category = "literal"
    base_confidence = 0.57
    _control_labels = {"34:2E", "33:FF", "EB:0B", "34:29"}

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        ascii_prefix = 0
        for profile in profiles:
            if profile.kind is InstructionKind.ASCII_CHUNK:
                ascii_prefix += 1
            else:
                break

        if ascii_prefix < 2:
            return None

        control_slice = profiles[ascii_prefix : ascii_prefix + 3]
        control_hits = sum(1 for profile in control_slice if profile.label in self._control_labels)
        if control_hits == 0:
            return None

        trailing_markers = sum(
            1
            for profile in profiles[ascii_prefix + control_hits :]
            if is_literal_marker(profile)
        )

        if trailing_markers == 0:
            return None

        notes = (
            f"ascii_prefix={ascii_prefix}",
            f"control_hits={control_hits}",
            f"markers={trailing_markers}",
            f"stackΔ={stack.change:+d}",
        )

        confidence = min(
            0.86,
            self.base_confidence
            + 0.04 * (ascii_prefix - 1)
            + 0.03 * control_hits
            + 0.02 * trailing_markers,
        )
        return SignatureMatch(self.name, self.category, confidence, notes)


class MarkerFenceReduceSignature(SignatureRule):
    """Recognise literal marker fences around ``00:69`` and reducers."""

    name = "marker_fence_reduce"
    category = "literal"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 6:
            return None

        labels = [profile.label for profile in profiles]
        operands = [profile.operand for profile in profiles]

        if labels[0] != "3D:30" or operands[0] != 0x3069:
            return None
        if labels[1] != "01:90":
            return None
        if labels[2] != "5E:29" or operands[2] != RET_MASK:
            return None
        if labels[3] != "ED:4D" or operands[3] != 0x4D0E:
            return None
        if labels[4] != "00:69" or operands[4] != 0x0190:
            return None

        reduce_idx = next(
            (idx for idx in range(5, len(profiles)) if profiles[idx].opcode == 0x04),
            None,
        )
        if reduce_idx is None:
            return None

        notes = (
            f"reduce_idx={reduce_idx}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.87, self.base_confidence + 0.04 * (len(profiles) - reduce_idx))
        return SignatureMatch(self.name, self.category, confidence, notes)


class LiteralZeroInitSignature(SignatureRule):
    """Identify literal loaders wrapped around a ``DE:ED`` zero initialiser."""

    name = "literal_zero_init"
    category = "literal"
    base_confidence = 0.56

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 5:
            return None

        if profiles[2].label != "DE:ED":
            return None

        if not all(profiles[idx].kind is InstructionKind.LITERAL for idx in (0, 1)):
            return None

        tail = profiles[3:]
        if len(tail) < 2 or not all(item.kind is InstructionKind.LITERAL for item in tail):
            return None

        zero_tail = [item for item in tail if item.operand == 0]
        if len(zero_tail) != len(tail):
            return None

        notes = (
            f"head_operands=({profiles[0].operand:#x},{profiles[1].operand:#x})",
            f"tail_literals={len(tail)}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.85, self.base_confidence + 0.04 * (len(tail) - 1))
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


class ModeSweepSignature(SignatureRule):
    """Detect uniform mode sweeps used for register initialisation."""

    name = "mode_sweep_block"
    category = "setup"
    base_confidence = 0.59

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        modes = {profile.mode for profile in profiles}
        if len(modes) != 1:
            return None

        (mode,) = modes
        if mode not in {0x4E, 0x4F}:
            return None

        distinct_opcodes = {profile.opcode for profile in profiles}
        if len(distinct_opcodes) < 3:
            return None

        notes = (
            f"mode=0x{mode:02X}",
            f"unique_opcodes={len(distinct_opcodes)}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.85, self.base_confidence + 0.04 * (len(profiles) - 3))
        return SignatureMatch(self.name, self.category, confidence, notes)


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


class LiteralMirrorReduceSignature(SignatureRule):
    """Detect mirrored ``0x6704``/``0x0067`` literal loops capped by reducers."""

    name = "literal_mirror_reduce_loop"
    category = "literal"
    base_confidence = 0.6
    _mirror_pairs = {(0x6704, 0x0067), (0x0067, 0x6704)}

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 6:
            return None

        triplets = []
        idx = 0
        while idx + 2 < len(profiles):
            first, second, third = profiles[idx : idx + 3]
            if (
                is_literal_like(first)
                and is_literal_like(second)
                and third.kind is InstructionKind.REDUCE
            ):
                triplets.append((idx, first.operand, second.operand))
                idx += 3
                while idx < len(profiles) and is_literal_marker(profiles[idx]):
                    idx += 1
            else:
                idx += 1

        if len(triplets) < 2:
            return None

        pair_counts = Counter((a, b) for _, a, b in triplets)
        dominant_pair, dominant_hits = pair_counts.most_common(1)[0]

        if dominant_pair not in self._mirror_pairs:
            return None

        notes = (
            f"runs={len(triplets)}",
            f"pair=0x{dominant_pair[0]:04X}/0x{dominant_pair[1]:04X}",
            f"stackΔ={stack.change:+d}",
        )

        confidence = min(
            0.9,
            self.base_confidence
            + 0.04 * (len(triplets) - 1)
            + 0.03 * (dominant_hits - 1),
        )
        return SignatureMatch(self.name, self.category, confidence, notes)


class StackLiftPairSignature(SignatureRule):
    """Detect the ``00:30`` → ``00:48`` stack lift micro pattern."""

    name = "stack_lift_pair"
    category = "literal"
    base_confidence = 0.55

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 2:
            return None
        if profiles[0].label != "00:30" or profiles[1].label != "00:48":
            return None

        notes = (
            "stack_lift_pair",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.8, self.base_confidence + 0.03)
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


class TailcallAsciiWrapperSignature(SignatureRule):
    """Match rare tailcall blocks with inline ASCII payload and wrappers."""

    name = "tailcall_ascii_wrapper"
    category = "call"
    base_confidence = 0.57
    _anchors = {"52:05", "32:29"}

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 5:
            return None

        tail_idx = next((idx for idx, profile in enumerate(profiles) if is_tailcall(profile)), None)
        if tail_idx is None or tail_idx >= len(profiles) - 2:
            return None

        labels = [profile.label for profile in profiles]
        if not self._anchors.issubset(labels[tail_idx + 1 :]):
            return None

        ascii_after = sum(
            1 for profile in profiles[tail_idx + 1 :] if profile.kind is InstructionKind.ASCII_CHUNK
        )
        if ascii_after == 0:
            return None

        branch_after = any(profile.kind is InstructionKind.BRANCH for profile in profiles[tail_idx + 1 :])
        if not branch_after:
            return None

        literal_prefix = sum(1 for profile in profiles[:tail_idx] if is_literal_like(profile))
        prefix_push = labels[0] == "03:00"

        notes = (
            f"tail_idx={tail_idx}",
            f"ascii_after={ascii_after}",
            f"literal_prefix={literal_prefix}",
            f"stackΔ={stack.change:+d}",
        )

        confidence = self.base_confidence + 0.04 * ascii_after
        if prefix_push:
            confidence += 0.03
        if literal_prefix:
            confidence += min(0.03, 0.01 * literal_prefix)
        return SignatureMatch(self.name, self.category, min(0.88, confidence), notes)


class JumpAsciiTailcallSignature(SignatureRule):
    """Recognise ``jump`` → ASCII → ``tailcall`` → ASCII chains."""

    name = "jump_ascii_tailcall"
    category = "call"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        first = profiles[0]
        if first.kind not in {InstructionKind.BRANCH, InstructionKind.CONTROL} and not first.label.startswith("22:"):
            return None

        if profiles[1].kind is not InstructionKind.ASCII_CHUNK:
            return None

        if not is_tailcall(profiles[2]):
            return None

        ascii_tail = sum(
            1 for profile in profiles[3:] if profile.kind is InstructionKind.ASCII_CHUNK
        )
        if ascii_tail == 0:
            return None

        notes = (
            "jump_idx=0",
            f"ascii_tail={ascii_tail}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.87, self.base_confidence + 0.05 * min(ascii_tail, 2))
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
            (
                idx
                for idx in range(marker_4b_idx + 1, len(profiles))
                if is_indirect_access(profiles[idx])
            ),
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
            (
                idx
                for idx in range(tail_idx + 1, len(profiles))
                if is_indirect_access(profiles[idx])
            ),
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


class TailcallReturnMarkerSignature(SignatureRule):
    """Match tailcalls with a literal marker epilogue after the return."""

    name = "tailcall_return_marker"
    category = "call"
    base_confidence = 0.61

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 5:
            return None

        tail_idx = next((idx for idx, profile in enumerate(profiles) if is_tailcall(profile)), None)
        if tail_idx is None or tail_idx >= len(profiles) - 2:
            return None

        return_profile = profiles[tail_idx + 1]
        if return_profile.kind is not InstructionKind.RETURN:
            return None

        suffix = profiles[tail_idx + 2 :]
        if not suffix:
            return None

        if any(
            profile.kind in {InstructionKind.CALL, InstructionKind.TAILCALL, InstructionKind.INDIRECT}
            or is_call_helper(profile)
            for profile in suffix
        ):
            return None

        if not all(is_literal_like(profile) for profile in suffix):
            return None

        literal_tail = list(suffix)

        if not literal_tail:
            return None

        prefix_literals = sum(1 for profile in profiles[:tail_idx] if is_literal_like(profile))
        notes = (
            f"tail_idx={tail_idx}",
            f"literal_tail={len(literal_tail)}",
            f"prefix_literals={prefix_literals}",
            f"stackΔ={stack.change:+d}",
        )

        confidence = self.base_confidence
        if prefix_literals:
            confidence += min(0.05, 0.02 * prefix_literals)
        if stack.change <= 0:
            confidence += 0.03
        return SignatureMatch(self.name, self.category, min(0.89, confidence), notes)


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
            (
                idx
                for idx in range(helper_idx + 1, len(profiles))
                if is_indirect_access(profiles[idx])
            ),
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


class ReturnTeardownMarkerSignature(SignatureRule):
    """Identify stack teardown blocks that emit a marker literal afterwards."""

    name = "return_teardown_marker"
    category = "return"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        teardown_idx = next(
            (idx for idx, profile in enumerate(profiles) if profile.kind is InstructionKind.STACK_TEARDOWN),
            None,
        )
        if teardown_idx is None:
            return None

        return_idx = next(
            (
                idx
                for idx in range(teardown_idx + 1, len(profiles))
                if profiles[idx].kind is InstructionKind.RETURN
            ),
            None,
        )
        if return_idx is None or return_idx >= len(profiles) - 1:
            return None

        literal_tail = [
            profile for profile in profiles[return_idx + 1 :] if is_literal_like(profile)
        ]
        if not literal_tail:
            return None

        notes = (
            f"teardown_idx={teardown_idx}",
            f"return_idx={return_idx}",
            f"literal_tail={len(literal_tail)}",
            f"stackΔ={stack.change:+d}",
        )

        confidence = min(
            0.86,
            self.base_confidence
            + 0.04 * len(literal_tail)
            + (0.03 if stack.change <= 0 else 0.0),
        )
        return SignatureMatch(self.name, self.category, confidence, notes)


class ReturnModeRibbonSignature(SignatureRule):
    """Recognise return sequences that stay within ``mode=0x5B``."""

    name = "return_mode_ribbon"
    category = "return"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        labels = [profile.label for profile in profiles]
        if labels[:3] != ["27:5B", "2A:5B", "30:5B"]:
            return None

        tail = profiles[3:]
        if not tail or any(profile.mode != 0x5B for profile in tail):
            return None

        notes = (
            "mode=0x5B",
            f"tail_span={len(tail)}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.88, self.base_confidence + 0.05 * len(tail))
        return SignatureMatch(self.name, self.category, confidence, notes)


class ReturnStackMarkerSignature(SignatureRule):
    """Recognise return chains guarded by ``5E:29`` / ``F0:4B`` markers."""

    name = "return_stack_marker_seq"
    category = "return"
    base_confidence = 0.62

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        labels = [profile.label for profile in profiles]
        try:
            idx_return = labels.index("30:6C")
        except ValueError:
            return None

        if idx_return < 2 or idx_return + 1 >= len(profiles):
            return None
        if labels[idx_return - 1] != "F0:4B" or labels[idx_return - 2] != "5E:29":
            return None
        if labels[idx_return + 1] != "01:F0":
            return None

        zero_tail = [
            profile
            for profile in profiles[idx_return + 2 :]
            if profile.kind is InstructionKind.LITERAL and profile.operand == 0
        ]
        if not zero_tail:
            return None

        notes = (
            f"return_idx={idx_return}",
            f"zero_tail={len(zero_tail)}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.9, self.base_confidence + 0.04 * len(zero_tail))
        return SignatureMatch(self.name, self.category, confidence, notes)


class ReturnBdCapsuleSignature(SignatureRule):
    """Detect ``C5:BD``/``AC:BD`` capsules surrounding return helpers."""

    name = "return_bd_capsule"
    category = "return"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        labels = [profile.label for profile in profiles]
        if labels[0] != "C5:BD" or labels[-1] != "AC:BD":
            return None

        required = {"30:69", "10:37", "0B:3D"}
        if not required.issubset(labels):
            return None

        notes = (
            f"capsule_span={len(profiles)}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = self.base_confidence
        if stack.change <= 0:
            confidence += 0.05
        return SignatureMatch(self.name, self.category, min(0.88, confidence), notes)


class PoisonReturnPrologSignature(SignatureRule):
    """Detect the ``FA:FF``/``41:DD``/``00:09`` prolog before returns."""

    name = "poison_return_prolog"
    category = "return"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        required = ["FA:FF", "41:DD", "00:09", "30:29"]
        labels = [profile.label for profile in profiles[:4]]
        if labels != required:
            return None

        notes = (
            "poison_capsule",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.86, self.base_confidence + 0.05)
        return SignatureMatch(self.name, self.category, confidence, notes)


class ReturnAsciiEpilogueSignature(SignatureRule):
    """Detect return tails that inject ASCII chunks before the follow-up push."""

    name = "return_ascii_epilogue"
    category = "return"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 6:
            return None

        if profiles[0].opcode != 0x04:
            return None
        if not is_call_helper(profiles[1]):
            return None
        if profiles[2].label != "77:00":
            return None
        if profiles[3].label != "01:84":
            return None
        if profiles[4].kind is not InstructionKind.ASCII_CHUNK:
            return None
        if profiles[5].kind not in {InstructionKind.LITERAL, InstructionKind.PUSH}:
            return None

        notes = (
            "ascii_tail",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.86, self.base_confidence + 0.04)
        return SignatureMatch(self.name, self.category, confidence, notes)


class B4SlotReturnSignature(SignatureRule):
    """Recognise ``20:00`` → ``01:B4`` slot prep followed by a guarded return."""

    name = "b4_slot_return"
    category = "return"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 6:
            return None

        labels = [profile.label for profile in profiles[:6]]
        if labels[0] != "20:00" or labels[1] != "01:B4" or labels[2] != "00:69":
            return None
        if not labels[3].startswith("27:"):
            return None
        if labels[4] != "02:66":
            return None
        if profiles[5].kind is not InstructionKind.RETURN:
            return None

        notes = (
            "b4_slot_sequence",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.87, self.base_confidence + 0.04)
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


class FanoutTeardownExtendedSignature(SignatureRule):
    """Match the extended fanout teardown guarded by ``1A:21``."""

    name = "fanout_teardown_ext"
    category = "call"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 7:
            return None

        labels = [profile.label for profile in profiles]
        if labels[0] != "1A:21":
            return None
        if labels[1] != "04:00":
            return None
        if profiles[2].opcode != 0x66:
            return None
        if labels[3] != "01:69":
            return None
        if not labels[4].startswith("27:"):
            return None
        if labels[5] != "04:66":
            return None
        if not is_call_helper(profiles[6]):
            return None

        notes = (
            "fanout_teardown_ext",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.87, self.base_confidence + 0.04)
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


class DoubleTailcallBranchSignature(SignatureRule):
    """Spot double tailcall blocks followed by branch validation."""

    name = "double_tailcall_branch"
    category = "call"
    base_confidence = 0.6

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 6:
            return None

        labels = [profile.label for profile in profiles]
        tail_indices = [idx for idx, profile in enumerate(profiles) if is_tailcall(profile)]
        if len(tail_indices) < 2:
            return None
        if tail_indices[1] != tail_indices[0] + 1:
            return None

        search_from = tail_indices[1] + 1
        for label in ("28:10", "2F:29", "2F:2C", "26:30"):
            try:
                idx = labels.index(label, search_from)
            except ValueError:
                return None
            if idx < search_from:
                return None
            search_from = idx + 1

        if "BD:00" not in labels[search_from - 1 :]:
            return None

        notes = (
            f"tail_indices={tuple(tail_indices[:2])}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(0.9, self.base_confidence + 0.04)
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

        if not any(is_indirect_access(profile) for profile in profiles[second_push + 1 :]):
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

        lead_idx = next(
            (idx for idx, profile in enumerate(profiles) if is_call_helper(profile)),
            None,
        )
        if lead_idx is None:
            return None

        marker_idx = next(
            (
                idx
                for idx, profile in enumerate(profiles[lead_idx + 1 :], start=lead_idx + 1)
                if is_indirect_access(profile)
            ),
            None,
        )
        if marker_idx is None:
            return None

        if lead_idx and any(
            profile.kind in {InstructionKind.RETURN, InstructionKind.TERMINATOR}
            for profile in profiles[:lead_idx]
        ):
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
            f"lead_idx={lead_idx}",
            f"marker_idx={marker_idx}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(
            0.85,
            self.base_confidence
            + 0.03 * (marker_idx - lead_idx)
            + 0.04 * (len(profiles) - marker_idx),
        )
        return SignatureMatch(self.name, self.category, confidence, notes)


class IndirectCallMiniSignature(SignatureRule):
    """Match short helper → indirect → literal call scaffolding."""

    name = "indirect_call_mini"
    category = "call"
    base_confidence = 0.57

    def match(
        self, profiles: Sequence[InstructionProfile], stack: StackSummary
    ) -> Optional[SignatureMatch]:
        if len(profiles) < 4:
            return None

        helper_idx = next(
            (idx for idx, profile in enumerate(profiles) if is_call_helper(profile)),
            None,
        )
        if helper_idx is None or helper_idx > 1:
            return None

        indirect_idx = next(
            (
                idx
                for idx in range(helper_idx + 1, len(profiles))
                if is_indirect_access(profiles[idx])
            ),
            None,
        )
        if indirect_idx is None:
            return None

        tail_slice = profiles[indirect_idx + 1 :]
        if not tail_slice:
            return None

        literal_tail = sum(1 for profile in tail_slice if is_literal_like(profile))
        arithmetic_hits = sum(
            1
            for profile in tail_slice
            if profile.kind in {InstructionKind.ARITHMETIC, InstructionKind.LOGICAL, InstructionKind.STACK_TEARDOWN}
        )

        if literal_tail == 0 or arithmetic_hits == 0:
            return None

        notes = (
            f"helper_idx={helper_idx}",
            f"indirect_idx={indirect_idx}",
            f"literal_tail={literal_tail}",
            f"arith_hits={arithmetic_hits}",
            f"stackΔ={stack.change:+d}",
        )
        confidence = min(
            0.84,
            self.base_confidence + 0.05 * min(literal_tail, 2) + 0.04 * min(arithmetic_hits, 2),
        )
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
            ReduceAsciiPrologSignature(),
            AsciiRunSignature(),
            AsciiControlClusterSignature(),
            HeaderAsciiCtrlSeqSignature(),
            ScriptHeaderPrologSignature(),
            ModeSweepSignature(),
            TableStoreSignature(),
            IndirectFetchSignature(),
            MarkerPairWithHeaderSignature(),
            AsciiPrologMarkerComboSignature(),
            AsciiWrapperEf48Signature(),
            AsciiReduceMarkerSignature(),
            MarkerFenceReduceSignature(),
            LiteralZeroInitSignature(),
            LiteralRunWithMarkersSignature(),
            LiteralMirrorReduceSignature(),
            LiteralReduceChainExSignature(),
            StackLiftPairSignature(),
            AsciiTailcallPatternSignature(),
            TailcallAsciiWrapperSignature(),
            JumpAsciiTailcallSignature(),
            AsciiIndirectTailcallSignature(),
            TailcallPostJumpSignature(),
            TailcallReturnMarkerSignature(),
            TailcallReturnComboSignature(),
            TailcallReturnIndirectSignature(),
            ReturnTeardownMarkerSignature(),
            ReturnModeRibbonSignature(),
            ReturnStackMarkerSignature(),
            ReturnBdCapsuleSignature(),
            PoisonReturnPrologSignature(),
            ReturnAsciiEpilogueSignature(),
            B4SlotReturnSignature(),
            CallprepAsciiDispatchSignature(),
            FanoutTeardownExtendedSignature(),
            FanoutTeardownSignature(),
            DoubleTailcallBranchSignature(),
            IndirectCallDualLiteralSignature(),
            IndirectCallMiniSignature(),
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
