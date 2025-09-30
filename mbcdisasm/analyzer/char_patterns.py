"""Pattern library specialised for ``_char`` data segments.

When decoding the `_char` module we often look for repeated structural motifs:
clusters of literal markers delimiting configuration tables, pairs of palindromic
words that act as sentinels or short ASCII banners announcing a state machine.
The :mod:`data_signatures` helpers annotate individual words with detectors, and
this module composes them into higher level patterns.

The library is intentionally small but easy to extend.  Each pattern knows the
expected length and carries a matcher function that inspects the enriched
profiles produced by :class:`InstructionProfile`.  The functions at the bottom of
the file expose convenience helpers for counting and formatting matches which
are used by :mod:`char_report` during report generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping, Sequence, Tuple

from .instruction_profile import InstructionProfile


@dataclass(frozen=True)
class CharPattern:
    """Small helper that encapsulates a structural pattern."""

    name: str
    length: int
    description: str
    matcher: Callable[[Sequence[InstructionProfile]], bool]

    def matches(self, window: Sequence[InstructionProfile]) -> bool:
        if len(window) < self.length:
            return False
        slice_profiles = window[: self.length]
        return self.matcher(slice_profiles)

    def describe(self) -> str:
        return f"{self.name} (len={self.length})"


def _detector_sequence(*detectors: str) -> Callable[[Sequence[InstructionProfile]], bool]:
    """Return a matcher that checks detector names sequentially."""

    def match(profiles: Sequence[InstructionProfile]) -> bool:
        if len(profiles) != len(detectors):
            return False
        for profile, detector in zip(profiles, detectors):
            if profile.traits.get("detector") != detector:
                return False
        return True

    return match


def _ascii_contains(keyword: str) -> Callable[[Sequence[InstructionProfile]], bool]:
    """Return a matcher that requires ``keyword`` in the concatenated text."""

    def match(profiles: Sequence[InstructionProfile]) -> bool:
        collected = []
        for profile in profiles:
            if profile.traits.get("detector") != "ascii_chunk":
                return False
            text = str(profile.traits.get("ascii_text", ""))
            collected.append(text)
        joined = "".join(collected).lower()
        return keyword.lower() in joined

    return match


def _raw_sequence(*values: int) -> Callable[[Sequence[InstructionProfile]], bool]:
    def match(profiles: Sequence[InstructionProfile]) -> bool:
        if len(profiles) != len(values):
            return False
        for profile, value in zip(profiles, values):
            if profile.word.raw != value:
                return False
        return True

    return match


CHAR_PATTERNS: Tuple[CharPattern, ...] = (
    CharPattern(
        name="config_header",
        length=2,
        description="Два последовательных ASCII-слова с путями params",
        matcher=_ascii_contains("params\\"),
    ),
    CharPattern(
        name="zero_block",
        length=4,
        description="Последовательность из четырёх нулевых слов",
        matcher=_raw_sequence(0, 0, 0, 0),
    ),
    CharPattern(
        name="palindrome_pair",
        length=2,
        description="Два подряд идущих палиндрома",
        matcher=_detector_sequence("palindrome", "palindrome"),
    ),
    CharPattern(
        name="literal_opcode_run",
        length=3,
        description="Три последовательных загрузчика opcode 0x00",
        matcher=_detector_sequence("literal_opcode00", "literal_opcode00", "literal_opcode00"),
    ),
    CharPattern(
        name="ascii_log",
        length=1,
        description="ASCII сообщение с маркером ERROR",
        matcher=_ascii_contains("error"),
    ),
    CharPattern(
        name="function_banner",
        length=1,
        description="Имя функции вида InitChar",
        matcher=_ascii_contains("initchar"),
    ),
    CharPattern(
        name="marker_cluster",
        length=3,
        description="Комбинация маркеров repeat16/char_marker",
        matcher=_detector_sequence("repeat16", "char_marker", "repeat16"),
    ),
)


def find_patterns(profiles: Sequence[InstructionProfile]) -> Tuple[str, ...]:
    matches: List[str] = []
    for start in range(len(profiles)):
        window = profiles[start:]
        for pattern in CHAR_PATTERNS:
            if pattern.matches(window):
                matches.append(pattern.name)
    return tuple(matches)


def summarise_patterns(profiles: Sequence[InstructionProfile]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for name in find_patterns(profiles):
        counts[name] = counts.get(name, 0) + 1
    return counts


def pattern_summary_text(profiles: Sequence[InstructionProfile]) -> str:
    summary = summarise_patterns(profiles)
    chunks = [f"{name}:{count}" for name, count in sorted(summary.items())]
    return ", ".join(chunks)


def list_pattern_names() -> Tuple[str, ...]:
    return tuple(pattern.name for pattern in CHAR_PATTERNS)


def contains_pattern(profiles: Sequence[InstructionProfile], name: str) -> bool:
    return name in find_patterns(profiles)
