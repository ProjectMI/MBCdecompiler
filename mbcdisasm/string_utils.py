"""Utility helpers for reasoning about textual string artefacts."""

from __future__ import annotations

import re

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


__all__ = [
    "DEFAULT_LUA_KEYWORDS",
    "DEFAULT_STOPWORDS",
    "IdentifierToken",
    "StringNameSuggester",
    "StackNameAllocator",
    "collapse_identifier",
    "extract_identifier_tokens",
]


# ---------------------------------------------------------------------------
# Identifier processing primitives


DEFAULT_LUA_KEYWORDS = {
    "and",
    "break",
    "do",
    "else",
    "elseif",
    "end",
    "false",
    "for",
    "function",
    "goto",
    "if",
    "in",
    "local",
    "nil",
    "not",
    "or",
    "repeat",
    "return",
    "then",
    "true",
    "until",
    "while",
}


DEFAULT_STOPWORDS = {"usage", "warning"}


_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass(frozen=True)
class IdentifierToken:
    """Description of an identifier-like token extracted from a string."""

    text: str
    offset: int


def collapse_identifier(text: str) -> Optional[str]:
    """Normalise *text* into a Lua identifier friendly form."""

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


def _split_identifier_subtokens(token: str) -> List[Tuple[int, str]]:
    """Break a composite identifier into camelCase components."""

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


def extract_identifier_tokens(
    text: str, offsets: Sequence[int]
) -> Iterable[IdentifierToken]:
    """Yield identifier tokens contained in *text*.

    ``offsets`` mirrors the instruction offsets that produced ``text``.  The
    function mirrors :func:`_identifier_tokens` from the original
    ``highlevel`` module but is shared to enable richer heuristics.
    """

    if not text:
        return []
    limit = len(offsets) - 1
    results: List[IdentifierToken] = []
    for match in _IDENTIFIER_PATTERN.finditer(text):
        start = match.start()
        if start % 2 != 0:
            continue
        chunk_index = start // 2
        if chunk_index > limit:
            chunk_index = limit
        token = match.group(0)
        underscore_index = token.find("_")
        include_full = True
        if underscore_index != -1 and any(
            ch.isupper() for ch in token[underscore_index + 1 :]
        ):
            include_full = False
        if include_full:
            results.append(IdentifierToken(token, offsets[chunk_index]))
        for rel_start, sub in _split_identifier_subtokens(token):
            if rel_start == 0 and include_full:
                continue
            absolute = start + rel_start
            if absolute % 2 != 0:
                continue
            sub_index = absolute // 2
            if sub_index > limit:
                sub_index = limit
            results.append(IdentifierToken(sub, offsets[sub_index]))
    return results


def _tokenise_words(text: str) -> Iterator[str]:
    for match in _IDENTIFIER_PATTERN.finditer(text):
        yield match.group(0)


# ---------------------------------------------------------------------------
# String naming heuristics


@dataclass(frozen=True)
class _NameCandidate:
    identifier: str
    lowered: str
    offset: int
    weight: int


class StringNameSuggester:
    """Generate descriptive names from detected textual artefacts."""

    def __init__(
        self,
        *,
        stopwords: Optional[Iterable[str]] = None,
        keywords: Optional[Iterable[str]] = None,
    ) -> None:
        self._stopwords = {word.lower() for word in (stopwords or DEFAULT_STOPWORDS)}
        self._keywords = set(keywords or DEFAULT_LUA_KEYWORDS)
        self._literal_cache: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    def function_name(
        self,
        sequences: Sequence["StringLiteralSequence"],
        entry_offset: int,
    ) -> Optional[str]:
        raw_candidates: List[_NameCandidate] = []
        frequency: Counter[str] = Counter()
        for sequence in sequences:
            text = sequence.text.strip()
            if text and not any(ch.isspace() for ch in text):
                sanitized = collapse_identifier(text)
                if sanitized and sanitized.lower() not in self._stopwords:
                    lowered = sanitized.lower()
                    raw_candidates.append(
                        _NameCandidate(sanitized, lowered, sequence.start_offset, 2)
                    )
                    frequency[lowered] += 1
            for token in extract_identifier_tokens(sequence.text, sequence.offsets):
                sanitized = collapse_identifier(token.text)
                if not sanitized:
                    continue
                lowered = sanitized.lower()
                if lowered in self._stopwords:
                    continue
                raw_candidates.append(
                    _NameCandidate(sanitized, lowered, token.offset, 1)
                )
                frequency[lowered] += 1
        if not raw_candidates:
            return None
        scored: List[Tuple[Tuple[int, int, int, int, int, str], str]] = []
        seen: set[str] = set()
        for candidate in raw_candidates:
            if candidate.identifier in seen:
                continue
            seen.add(candidate.identifier)
            count = frequency[candidate.lowered]
            score = self._score_candidate(candidate, entry_offset, count)
            scored.append((score, candidate.identifier))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        selected = scored[0][1]
        return self._avoid_keyword(selected)

    def literal_hint(self, text: str) -> Optional[str]:
        text = text.strip()
        if not text:
            return None
        cached = self._literal_cache.get(text)
        if cached is not None:
            return cached
        if not any(ch.isspace() for ch in text):
            sanitized = collapse_identifier(text)
            if sanitized and sanitized.lower() not in self._stopwords:
                result = self._avoid_keyword(sanitized)
                self._literal_cache[text] = result
                return result
        # Inspect identifier-like tokens, prioritising longer fragments.
        candidates: List[str] = []
        for token in _tokenise_words(text):
            sanitized = collapse_identifier(token)
            if not sanitized:
                continue
            lowered = sanitized.lower()
            if lowered in self._stopwords:
                continue
            candidates.append(sanitized)
        if not candidates:
            self._literal_cache[text] = None
            return None
        candidates.sort(key=lambda item: (-len(item), item.lower()))
        result = self._avoid_keyword(candidates[0])
        self._literal_cache[text] = result
        return result

    # ------------------------------------------------------------------
    def _score_candidate(
        self, candidate: _NameCandidate, entry_offset: int, count: int
    ) -> Tuple[int, int, int, int, int, str]:
        distance = max(0, candidate.offset - entry_offset)
        underscore_penalty = 0 if "_" in candidate.identifier else 1
        lower = sum(1 for ch in candidate.identifier if ch.islower())
        upper = sum(1 for ch in candidate.identifier if ch.isupper())
        if lower and upper:
            case_penalty = 0
        elif lower:
            case_penalty = 1
        elif upper:
            case_penalty = 2
        else:
            case_penalty = 3
        digit_penalty = sum(1 for ch in candidate.identifier if ch.isdigit())
        length_penalty = len(candidate.identifier)
        return (
            -max(count, candidate.weight),
            underscore_penalty,
            distance,
            case_penalty,
            digit_penalty,
            length_penalty,
            candidate.lowered,
        )

    def _avoid_keyword(self, identifier: str) -> str:
        lowered = identifier.lower()
        if lowered in self._keywords:
            return f"{identifier}_fn"
        return identifier


# ---------------------------------------------------------------------------
# Stack naming support


class StackNameAllocator:
    """Allocate unique stack variable names using textual hints."""

    def __init__(
        self,
        *,
        keywords: Optional[Iterable[str]] = None,
    ) -> None:
        self._used: set[str] = set()
        self._prefix_counters: Dict[str, int] = defaultdict(int)
        self._fallback_counter = 0
        self._keywords = set(keywords or DEFAULT_LUA_KEYWORDS)

    def allocate(self, prefix: str, hint: Optional[str] = None) -> str:
        candidates: List[str] = []
        if hint:
            sanitized = collapse_identifier(hint)
            if sanitized:
                candidates.append(self._ensure_unique(self._avoid_keyword(sanitized)))
        normalized_prefix = collapse_identifier(prefix) or prefix or "value"
        if not normalized_prefix:
            normalized_prefix = "value"
        index = self._prefix_counters[normalized_prefix]
        self._prefix_counters[normalized_prefix] = index + 1
        fallback = f"{normalized_prefix}_{index}"
        candidates.append(self._ensure_unique(self._avoid_keyword(fallback)))
        for candidate in candidates:
            if candidate:
                return candidate
        return self._ensure_unique(self._avoid_keyword(self._fallback()))

    # ------------------------------------------------------------------
    def reserve(self, name: str) -> str:
        sanitized = collapse_identifier(name)
        if not sanitized:
            sanitized = self._fallback()
        sanitized = self._avoid_keyword(sanitized)
        if sanitized in self._used:
            return self._ensure_unique(sanitized)
        self._used.add(sanitized)
        return sanitized

    # ------------------------------------------------------------------
    def _ensure_unique(self, base: str) -> str:
        if not base:
            return base
        if base not in self._used:
            self._used.add(base)
            return base
        index = 1
        while f"{base}_{index}" in self._used:
            index += 1
        candidate = f"{base}_{index}"
        self._used.add(candidate)
        return candidate

    def _avoid_keyword(self, identifier: str) -> str:
        if identifier.lower() in self._keywords:
            return f"{identifier}_value"
        return identifier

    def _fallback(self) -> str:
        name = f"value_{self._fallback_counter}"
        self._fallback_counter += 1
        return name

