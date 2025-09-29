"""Utilities that analyse string literal sequences for naming heuristics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .lua_literals import escape_lua_string


# ---------------------------------------------------------------------------
# data structures used by the high level reconstructor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentifierCandidate:
    """Normalised identifier candidate extracted from a string sequence."""

    name: str
    normalized: str
    occurrences: int
    first_offset: int
    source: str
    weight: int = 1

    def score(self, entry_offset: int) -> Tuple[int, int, int, int, int, str]:
        """Return a tuple used to rank candidates.

        The tuple mirrors the scoring function previously implemented in
        :mod:`mbcdisasm.highlevel` but incorporates the accumulated weight for a
        candidate.  Lower tuples are considered better.
        """

        count_score = -max(self.weight, self.occurrences)
        underscore_penalty = self.name.count("_")
        distance = abs(self.first_offset - entry_offset)
        case_penalty = 0 if self.name and self.name[0].islower() else 1
        digit_penalty = 1 if any(ch.isdigit() for ch in self.name) else 0
        length_penalty = len(self.name)
        tie_breaker = self.normalized
        return (
            count_score,
            underscore_penalty,
            distance,
            case_penalty,
            digit_penalty,
            length_penalty,
            tie_breaker,
        )


@dataclass(frozen=True)
class SequenceAnalysis:
    """Summary describing the heuristics for a string literal sequence."""

    text: str
    offsets: Tuple[int, ...]
    candidates: Tuple[IdentifierCandidate, ...] = field(default_factory=tuple)
    primary_identifier: Optional[str] = None
    categories: Tuple[str, ...] = field(default_factory=tuple)
    chunk_name_suggestions: Tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    notes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StringLiteralSequence:
    """Metadata describing a detected string literal run inside a block."""

    text: str
    offsets: Tuple[int, ...]
    chunk_names: Tuple[str, ...] = field(default_factory=tuple)
    candidates: Tuple[IdentifierCandidate, ...] = field(default_factory=tuple)
    primary_identifier: Optional[str] = None
    categories: Tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def start_offset(self) -> int:
        return self.offsets[0]

    @property
    def end_offset(self) -> int:
        return self.offsets[-1]

    def chunk_count(self) -> int:
        return len(self.offsets)

    def length(self) -> int:
        return len(self.text)

    def preview(self, limit: int = 80) -> str:
        if len(self.text) <= limit:
            return self.text
        if limit <= 3:
            return "..."
        return self.text[: limit - 3] + "..."

    @property
    def identifier_candidates(self) -> Tuple[str, ...]:
        if not self.candidates:
            return tuple()
        return tuple(candidate.name for candidate in self.candidates)

    def comment_lines(self) -> List[str]:
        """Return comment lines describing the sequence."""

        base = (
            "string literal sequence: "
            f"{escape_lua_string(self.text)}"
            f" (len={self.length()} chunks={self.chunk_count()})"
        )
        lines = [base]
        if self.primary_identifier:
            lines.append(f"identifier hint: {self.primary_identifier}")
        elif self.candidates:
            alt = ", ".join(candidate.name for candidate in self.candidates[:3])
            if alt:
                lines.append(f"identifier candidates: {alt}")
        if self.categories:
            lines.append("categories: " + ", ".join(self.categories))
        if self.notes:
            lines.extend(self.notes)
        return lines


class _CandidateBuilder:
    """Helper tracking statistics for a potential identifier."""

    def __init__(self, name: str, *, source: str) -> None:
        self.name = name
        self.normalized = name.lower()
        self.occurrences = 0
        self.first_offset = None
        self.weight = 0
        self.source = source

    def bump(self, *, weight: int, offset: int) -> None:
        self.occurrences += 1
        self.weight += weight
        if self.first_offset is None or offset < self.first_offset:
            self.first_offset = offset

    def build(self) -> IdentifierCandidate:
        return IdentifierCandidate(
            name=self.name,
            normalized=self.normalized,
            occurrences=self.occurrences,
            first_offset=self.first_offset or 0,
            source=self.source,
            weight=max(self.weight, 1),
        )


class StringAnalyzer:
    """Analyse string literal sequences and derive heuristic metadata."""

    def __init__(
        self,
        *,
        min_token_length: int = 3,
        max_identifier_length: int = 64,
        stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.min_token_length = min_token_length
        self.max_identifier_length = max_identifier_length
        self.stopwords = {word.lower() for word in (stopwords or _STRING_NAME_STOPWORDS)}

    # ------------------------------------------------------------------
    def analyse(
        self,
        text: str,
        offsets: Sequence[int],
        chunk_literals: Sequence[str],
        *,
        entry_offset: int,
    ) -> SequenceAnalysis:
        """Return a :class:`SequenceAnalysis` describing ``text``."""

        offsets_tuple = tuple(offsets)
        candidates = self._gather_candidates(text, offsets_tuple)
        primary = self._select_primary_identifier(candidates, entry_offset)
        categories = self._classify_categories(text, chunk_literals)
        suggestions = self._chunk_name_suggestions(
            primary, candidates, len(chunk_literals)
        )
        confidence = self._estimate_confidence(text, candidates, categories)
        notes = self._derive_notes(text, categories, candidates)
        return SequenceAnalysis(
            text=text,
            offsets=offsets_tuple,
            candidates=candidates,
            primary_identifier=primary,
            categories=tuple(categories),
            chunk_name_suggestions=suggestions,
            confidence=confidence,
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------
    def select_function_name(
        self,
        sequences: Sequence[StringLiteralSequence],
        *,
        entry_offset: int,
    ) -> Optional[str]:
        """Return the preferred function name derived from ``sequences``."""

        candidate_pool: Dict[str, IdentifierCandidate] = {}
        for sequence in sequences:
            for candidate in sequence.candidates:
                existing = candidate_pool.get(candidate.normalized)
                if existing is None or candidate.score(entry_offset) < existing.score(entry_offset):
                    candidate_pool[candidate.normalized] = candidate
            if sequence.primary_identifier:
                normalized = sequence.primary_identifier.lower()
                if normalized not in candidate_pool:
                    candidate_pool[normalized] = IdentifierCandidate(
                        name=sequence.primary_identifier,
                        normalized=normalized,
                        occurrences=1,
                        first_offset=sequence.start_offset,
                        source="sequence",
                        weight=2,
                    )
        if not candidate_pool:
            return None
        ranked = sorted(candidate_pool.values(), key=lambda cand: cand.score(entry_offset))
        selected = ranked[0].name
        if selected.lower() in _LUA_KEYWORDS:
            return f"{selected}_fn"
        return selected

    # ------------------------------------------------------------------
    def _gather_candidates(
        self,
        text: str,
        offsets: Tuple[int, ...],
    ) -> Tuple[IdentifierCandidate, ...]:
        builders: Dict[str, _CandidateBuilder] = {}
        base = self._sanitize_identifier(text)
        if base and base.lower() not in self.stopwords:
            builder = builders.setdefault(base.lower(), _CandidateBuilder(base, source="sequence"))
            builder.bump(weight=3, offset=offsets[0])
        for token, offset in _identifier_tokens(text, offsets):
            if len(token) < self.min_token_length:
                continue
            sanitized = self._sanitize_identifier(token)
            if not sanitized:
                continue
            normalized = sanitized.lower()
            if normalized in self.stopwords:
                continue
            builder = builders.setdefault(
                normalized,
                _CandidateBuilder(sanitized, source="token"),
            )
            builder.bump(weight=1, offset=offset)
        return tuple(builder.build() for builder in builders.values())

    def _select_primary_identifier(
        self,
        candidates: Sequence[IdentifierCandidate],
        entry_offset: int,
    ) -> Optional[str]:
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda cand: cand.score(entry_offset))
        return ranked[0].name

    def _classify_categories(
        self,
        text: str,
        chunk_literals: Sequence[str],
    ) -> List[str]:
        categories: List[str] = []
        stripped = text.strip()
        unique_chars = {ch for ch in text}
        if "\n" in text:
            categories.append("multiline")
        if any(ch in unique_chars for ch in {"%", "{", "}"}):
            categories.append("format")
        if "/" in text or "\\" in text:
            categories.append("path")
        if stripped.startswith("<") and stripped.endswith(">"):
            categories.append("markup")
        if stripped.startswith("[") and stripped.endswith("]"):
            categories.append("tag")
        if text.isupper() and len(text) >= 4:
            categories.append("uppercase")
        if text.islower() and len(text) >= 4:
            categories.append("lowercase")
        if any(char.isdigit() for char in text):
            categories.append("numeric")
        if text.endswith("?") or text.endswith("!"):
            categories.append("message")
        if " " in text:
            categories.append("phrase")
        if any(len(chunk) == 1 for chunk in chunk_literals):
            categories.append("chunked")
        return categories

    def _chunk_name_suggestions(
        self,
        primary: Optional[str],
        candidates: Sequence[IdentifierCandidate],
        chunk_count: int,
    ) -> Tuple[str, ...]:
        if chunk_count <= 0:
            return tuple()
        base = primary
        if not base and candidates:
            ranked = sorted(candidates, key=lambda cand: cand.weight, reverse=True)
            base = ranked[0].name
        if not base:
            return tuple()
        sanitized = self._sanitize_identifier(base)
        if not sanitized:
            return tuple()
        sanitized = sanitized.lower()
        if sanitized.startswith("string_"):
            prefix = sanitized
        else:
            prefix = f"{sanitized}_str"
        suggestions: List[str] = []
        for index in range(chunk_count):
            suffix = "" if index == 0 else f"_chunk{index + 1}"
            suggestions.append(f"{prefix}{suffix}")
        return tuple(suggestions)

    def _estimate_confidence(
        self,
        text: str,
        candidates: Sequence[IdentifierCandidate],
        categories: Sequence[str],
    ) -> float:
        length_factor = min(len(text) / self.max_identifier_length, 1.0)
        candidate_factor = min(len(candidates) / 6.0, 1.0)
        category_factor = min(len(categories) / 8.0, 1.0)
        confidence = 0.2 + 0.4 * length_factor + 0.25 * candidate_factor + 0.15 * category_factor
        return max(0.0, min(confidence, 1.0))

    def _derive_notes(
        self,
        text: str,
        categories: Sequence[str],
        candidates: Sequence[IdentifierCandidate],
    ) -> List[str]:
        notes: List[str] = []
        if categories:
            notes.append("classified as " + ", ".join(categories))
        entropy = _shannon_entropy(text)
        notes.append(f"entropy={entropy:.2f}")
        if candidates:
            top = sorted(candidates, key=lambda cand: cand.weight, reverse=True)[:3]
            listing = ", ".join(f"{cand.name}({cand.weight})" for cand in top)
            notes.append("candidates: " + listing)
        return notes

    def _sanitize_identifier(self, text: str) -> Optional[str]:
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
        if len(candidate) > self.max_identifier_length:
            candidate = candidate[: self.max_identifier_length]
        return candidate


_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _identifier_tokens(
    text: str,
    offsets: Sequence[int],
) -> Iterator[Tuple[str, int]]:
    if not text:
        return iter(())
    offset_tuple = tuple(offsets)
    limit = len(offset_tuple) - 1
    for match in _IDENTIFIER_PATTERN.finditer(text):
        start = match.start()
        if start % 2 != 0:
            continue
        chunk_index = start // 2
        if chunk_index > limit:
            chunk_index = limit
        token = match.group(0)
        yield token, offset_tuple[chunk_index]
        for rel_start, sub in _split_identifier_subtokens(token):
            absolute = start + rel_start
            if absolute % 2 != 0:
                continue
            sub_index = absolute // 2
            if sub_index > limit:
                sub_index = limit
            yield sub, offset_tuple[sub_index]


def _split_identifier_subtokens(token: str) -> List[Tuple[int, str]]:
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


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    histogram: Dict[str, int] = {}
    for char in text:
        histogram[char] = histogram.get(char, 0) + 1
    total = float(len(text))
    entropy = 0.0
    for count in histogram.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


_LUA_KEYWORDS = {
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

_STRING_NAME_STOPWORDS = {"usage", "warning"}


__all__ = [
    "IdentifierCandidate",
    "SequenceAnalysis",
    "StringLiteralSequence",
    "StringAnalyzer",
]

