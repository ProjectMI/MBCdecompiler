"""Utilities for reasoning about string literal sequences.

The reconstruction pipeline frequently encounters bytecode sequences that push
short ASCII fragments onto the stack.  Individually these fragments offer very
little context, however, when combined they often form human readable strings
such as function names, prompts or enumerated values.  This module centralises
the heuristics responsible for extracting identifiers from such strings and for
deriving stable names that can be reused by higher level reconstruction stages.

The implementation intentionally exposes small, composable helpers instead of a
single monolithic routine.  The high level reconstructor can mix-and-match the
building blocks depending on the situation (naming local variables, deriving
function names, annotating metadata, ...).  The helpers also provide rich
metadata that simplifies unit testing and keeps the heuristics deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "IdentifierToken",
    "NameCandidate",
    "StringSequenceProfile",
    "extract_identifier_tokens",
    "iter_token_windows",
    "sanitize_identifier",
    "sequence_base_candidates",
    "score_candidate",
    "select_best_candidate",
    "subtokenise_identifier",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentifierToken:
    """A token extracted from a larger string literal sequence.

    Attributes:
        text: Sanitised identifier that only contains characters valid in Lua
            identifiers.  The text is never empty.
        offset: Absolute bytecode offset corresponding to the instruction that
            produced the token.  The offset is aligned to match the first
            chunk that contributed to the token.
        chunk_index: Index of the originating string chunk inside the sequence.
        original: The raw substring extracted from the literal before
            sanitisation.  This helps debugging and unit testing.
    """

    text: str
    offset: int
    chunk_index: int
    original: str


@dataclass(frozen=True)
class NameCandidate:
    """Potential identifier produced by the heuristics."""

    identifier: str
    normalized: str
    score: Tuple[int, int, int, int, int, int, str]
    source_offset: int
    chunk_index: int


@dataclass(frozen=True)
class StringSequenceProfile:
    """Pre-computed information about a string literal sequence.

    The high level reconstructor frequently evaluates different naming
    strategies on the same sequence (naming locals, naming functions, metadata
    rendering).  Computing the helper tokens and derived statistics once and
    reusing them everywhere keeps the code simpler and reduces the amount of
    duplicated logic spread across modules.
    """

    text: str
    offsets: Tuple[int, ...]
    tokens: Tuple[IdentifierToken, ...]

    @property
    def start_offset(self) -> int:
        return self.offsets[0]

    @property
    def end_offset(self) -> int:
        return self.offsets[-1]

    @property
    def chunk_count(self) -> int:
        return len(self.offsets)

    def preview(self, limit: int = 80) -> str:
        if len(self.text) <= limit:
            return self.text
        if limit <= 3:
            return "..."
        return self.text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Identifier extraction utilities
# ---------------------------------------------------------------------------


_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def sanitize_identifier(text: str) -> Optional[str]:
    """Convert *text* into a Lua friendly identifier.

    Non alphanumeric characters are replaced with underscores and consecutive
    separators collapse into a single underscore.  Identifiers that start with a
    digit are prefixed with an underscore to keep them valid in Lua source.
    Empty identifiers return :data:`None`.
    """

    if not text:
        return None
    pieces: List[str] = []
    for char in text:
        if char.isalnum() or char == "_":
            pieces.append(char)
        else:
            pieces.append("_")
    candidate = "".join(pieces)
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if not candidate:
        return None
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    return candidate


def subtokenise_identifier(token: str) -> List[Tuple[int, str]]:
    """Split camelCase or snake_case identifiers into smaller pieces.

    Returns a list of ``(relative_index, substring)`` tuples.  Only substrings
    containing at least three characters are returned to avoid generating noisy
    names such as ``set`` or ``val``.  The helper is intentionally conservative
    – it focuses on strings that look like programmer provided names.
    """

    parts: List[Tuple[int, str]] = []
    start = 0
    for index in range(1, len(token)):
        if token[index].isupper() and token[index - 1].islower():
            if index - start >= 3:
                parts.append((start, token[start:index]))
            start = index
    if len(token) - start >= 3:
        parts.append((start, token[start:]))
    return parts


def _align_index(relative: int, limit: int) -> int:
    if relative > limit:
        return limit
    return relative


def extract_identifier_tokens(
    text: str, offsets: Sequence[int]
) -> List[IdentifierToken]:
    """Extract identifier shaped tokens from ``text``.

    The :class:`StringLiteralCollector` feeds us the *combined* string literal
    value as well as the offsets for each contributing chunk.  The helper walks
    the string, identifies candidate tokens and aligns them with the matching
    instruction offsets.  Alignment is important because it allows callers to
    reason about which chunk produced a particular identifier and to align stack
    values with the eventual names.
    """

    if not text:
        return []
    if not offsets:
        return []
    limit = len(offsets) - 1
    results: List[IdentifierToken] = []
    for match in _IDENTIFIER_PATTERN.finditer(text):
        start = match.start()
        # Chunk boundaries are 16 bit wide.  Ignore candidates that do not
        # align – they most likely straddle instructions and represent noise.
        if start % 2 != 0:
            continue
        chunk_index = start // 2
        chunk_index = _align_index(chunk_index, limit)
        original = match.group(0)
        sanitized = sanitize_identifier(original)
        if not sanitized:
            continue
        include_full = True
        underscore_index = original.find("_")
        if underscore_index != -1 and any(
            ch.isupper() for ch in original[underscore_index + 1 :]
        ):
            include_full = False
        if include_full:
            results.append(
                IdentifierToken(
                    text=sanitized,
                    offset=offsets[chunk_index],
                    chunk_index=chunk_index,
                    original=original,
                )
            )
        for rel_start, sub in subtokenise_identifier(original):
            if rel_start == 0 and include_full:
                continue
            absolute = start + rel_start
            if absolute % 2 != 0:
                continue
            sub_index = _align_index(absolute // 2, limit)
            sanitized_sub = sanitize_identifier(sub)
            if not sanitized_sub:
                continue
            results.append(
                IdentifierToken(
                    text=sanitized_sub,
                    offset=offsets[sub_index],
                    chunk_index=sub_index,
                    original=sub,
                )
            )
    return results


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def sequence_base_candidates(
    text: str, offsets: Sequence[int], *, stopwords: Iterable[str]
) -> List[NameCandidate]:
    """Return potential names for the entire string sequence."""

    lowered_stopwords = {word.lower() for word in stopwords}
    tokens = extract_identifier_tokens(text, offsets)
    if not tokens:
        return []
    frequency: Dict[str, int] = {}
    for token in tokens:
        lowered = token.text.lower()
        if lowered in lowered_stopwords:
            continue
        frequency[lowered] = frequency.get(lowered, 0) + 1
    candidates: List[NameCandidate] = []
    seen: set[str] = set()
    entry_offset = offsets[0] if offsets else 0
    for token in tokens:
        lowered = token.text.lower()
        if lowered in lowered_stopwords:
            continue
        if token.text in seen:
            continue
        count = frequency.get(lowered, 1)
        score = score_candidate(
            token.text,
            entry_offset=entry_offset,
            candidate_offset=token.offset,
            count=count,
        )
        candidates.append(
            NameCandidate(
                identifier=token.text,
                normalized=lowered,
                score=score,
                source_offset=token.offset,
                chunk_index=token.chunk_index,
            )
        )
        seen.add(token.text)
    return candidates


def score_candidate(
    identifier: str,
    *,
    entry_offset: int,
    candidate_offset: int,
    count: int,
) -> Tuple[int, int, int, int, int, int, str]:
    """Score a candidate identifier.

    The scoring rules mirror the historic heuristics from
    :mod:`mbcdisasm.highlevel` but live in a dedicated helper so they can be
    reused when naming locals.  Lower scores are considered better.
    """

    underscore_penalty = 0 if "_" in identifier else 1
    distance = abs(candidate_offset - entry_offset)
    lower = sum(char.islower() for char in identifier)
    upper = sum(char.isupper() for char in identifier)
    if lower and upper:
        case_penalty = 0
    elif lower:
        case_penalty = 1
    elif upper:
        case_penalty = 2
    else:
        case_penalty = 3
    digit_penalty = sum(char.isdigit() for char in identifier)
    length_penalty = len(identifier)
    tie_breaker = identifier.lower()
    return (
        -count,
        underscore_penalty,
        distance,
        case_penalty,
        digit_penalty,
        length_penalty,
        tie_breaker,
    )


def select_best_candidate(candidates: Sequence[NameCandidate]) -> Optional[NameCandidate]:
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate.score)


# ---------------------------------------------------------------------------
# Sliding window helpers
# ---------------------------------------------------------------------------


def iter_token_windows(tokens: Sequence[IdentifierToken], window: int) -> Iterator[List[IdentifierToken]]:
    """Yield lists containing up to ``window`` neighbouring tokens."""

    if window <= 0:
        raise ValueError("window must be positive")
    if not tokens:
        return iter(())
    for index in range(len(tokens)):
        end = min(len(tokens), index + window)
        yield list(tokens[index:end])


# ---------------------------------------------------------------------------
# Convenience helpers used by the high level reconstructor
# ---------------------------------------------------------------------------


def collect_sequence_profiles(
    texts: Iterable[str],
    offsets: Iterable[Sequence[int]],
) -> List[StringSequenceProfile]:
    """Build :class:`StringSequenceProfile` objects from raw components."""

    profiles: List[StringSequenceProfile] = []
    for text, sequence_offsets in zip(texts, offsets):
        tokens = extract_identifier_tokens(text, sequence_offsets)
        profiles.append(
            StringSequenceProfile(
                text=text,
                offsets=tuple(sequence_offsets),
                tokens=tuple(tokens),
            )
        )
    return profiles


def sequence_primary_identifier(
    profile: StringSequenceProfile,
    *,
    stopwords: Iterable[str],
) -> Optional[str]:
    """Return the best identifier for *profile* if one exists."""

    candidates = sequence_base_candidates(profile.text, profile.offsets, stopwords=stopwords)
    best = select_best_candidate(candidates)
    if best is None:
        return None
    return best.identifier


def chunk_name_hints(
    profile: StringSequenceProfile,
    chunk_texts: Sequence[str],
    *,
    stopwords: Iterable[str],
) -> List[Optional[str]]:
    """Derive naming hints for each chunk inside ``profile``.

    The result list is aligned with ``chunk_texts`` and contains either the
    preferred identifier or :data:`None` if no sensible name could be derived.
    The function intentionally keeps the heuristics lightweight – callers are
    free to post-process the hints (prefixing, applying fallbacks, enforcing
    uniqueness, ...).
    """

    lowered_stopwords = {word.lower() for word in stopwords}
    hints: List[Optional[str]] = [None for _ in chunk_texts]
    if not profile.tokens:
        base = sanitize_identifier(profile.text)
        if base and base.lower() not in lowered_stopwords:
            for index in range(len(chunk_texts)):
                suffix = "" if len(chunk_texts) == 1 else f"_{index + 1}"
                hints[index] = f"{base}{suffix}".strip("_")
        return hints
    by_chunk: Dict[int, List[IdentifierToken]] = {}
    for token in profile.tokens:
        by_chunk.setdefault(token.chunk_index, []).append(token)
    for chunk_index, tokens in by_chunk.items():
        best = select_best_candidate(
            [
                NameCandidate(
                    identifier=token.text,
                    normalized=token.text.lower(),
                    score=score_candidate(
                        token.text,
                        entry_offset=profile.start_offset,
                        candidate_offset=token.offset,
                        count=1,
                    ),
                    source_offset=token.offset,
                    chunk_index=token.chunk_index,
                )
                for token in tokens
                if token.text.lower() not in lowered_stopwords
            ]
        )
        if best is not None:
            hints[chunk_index] = best.identifier
    return hints

