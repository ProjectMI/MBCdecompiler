"""Utility helpers for keyword based classification.

The disassembler consumes a heterogeneous knowledge base that mixes English and
Russian descriptions.  Historically the classification layer used simple
substring checks which failed to account for Cyrillic spellings and for
morphological variations (``literal`` vs ``literals`` vs ``литерал``).  The
lexicon module centralises the normalisation logic and provides a registry of
synonyms that can be shared across the analyser components.

The goal of the lexicon is *not* to provide a full natural language
understanding layer.  Instead it offers a deterministic matcher that understands
common inflections and transliteration variants.  This keeps the implementation
fast, easy to reason about and fully transparent to reverse engineers studying
the emitted diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "KeywordLexicon",
    "KeywordMatch",
    "normalise",
    "tokenise",
    "default_lexicon",
]

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _strip_diacritics(text: str) -> str:
    """Return ``text`` without diacritic marks."""

    decomposed = unicodedata.normalize("NFKD", text)
    filtered = [char for char in decomposed if not unicodedata.combining(char)]
    return unicodedata.normalize("NFKC", "".join(filtered))


def normalise(text: str) -> str:
    """Return a normalised representation of ``text``.

    The function performs a number of lightweight transformations:

    * Unicode normalisation and diacritic stripping.
    * Lowercasing using :func:`str.casefold` which handles both ASCII and
      Cyrillic characters.
    * Replacement of punctuation symbols with whitespace.
    * Collapsing multiple whitespace characters into a single space.

    The end result is well suited for straightforward substring checks or for
    tokenisation.
    """

    text = _strip_diacritics(text)
    text = text.casefold()
    translation_table = dict.fromkeys(map(ord, "-_,.;:!?#/"), " ")
    text = text.translate(translation_table)
    text = " ".join(part for part in text.split() if part)
    return text


def tokenise(text: str) -> Tuple[str, ...]:
    """Split ``text`` into normalised tokens."""

    if not text:
        return tuple()
    return tuple(normalise(text).split())


# ---------------------------------------------------------------------------
# Lexicon data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeywordMatch:
    """Represents the result of a lexicon lookup."""

    label: str
    score: float
    evidence: Tuple[str, ...] = tuple()

    def describe(self) -> str:
        if not self.evidence:
            return f"{self.label}={self.score:.2f}"
        return f"{self.label}={self.score:.2f} ({', '.join(self.evidence)})"


@dataclass
class KeywordEntry:
    """Stores canonical tokens and their synonyms."""

    label: str
    tokens: Set[str] = field(default_factory=set)

    def add(self, *synonyms: str) -> None:
        for synonym in synonyms:
            if not synonym:
                continue
            for token in tokenise(synonym):
                self.tokens.add(token)

    def matches(self, text: str) -> bool:
        normalised = tokenise(text)
        return any(token in self.tokens for token in normalised)


class KeywordLexicon:
    """Lookup table for keyword driven classification."""

    def __init__(self) -> None:
        self._entries: Dict[str, KeywordEntry] = {}
        self._aliases: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # registration API
    # ------------------------------------------------------------------
    def register(self, label: str, *synonyms: str, alias: Optional[str] = None) -> None:
        """Add ``label`` to the lexicon with ``synonyms``.

        The optional ``alias`` argument can be used to reuse the synonym set from
        another label.  This is handy when a concept appears under two canonical
        names (for example ``literal`` and ``const``) but should be treated as the
        same group during classification.
        """

        label = label.strip().lower()
        entry = self._entries.setdefault(label, KeywordEntry(label=label))
        entry.add(*synonyms)
        if alias:
            self._aliases[label] = alias.strip().lower()

    def extend(self, mapping: Mapping[str, Sequence[str]]) -> None:
        for label, synonyms in mapping.items():
            self.register(label, *synonyms)

    # ------------------------------------------------------------------
    # query API
    # ------------------------------------------------------------------
    def aliases_for(self, label: str) -> Tuple[str, ...]:
        primary = self._aliases.get(label)
        if primary:
            return (primary,)
        return tuple()

    def entry(self, label: str) -> Optional[KeywordEntry]:
        return self._entries.get(label)

    def entries(self) -> Iterable[KeywordEntry]:
        return tuple(self._entries.values())

    def match(self, text: str, labels: Iterable[str]) -> Tuple[KeywordMatch, ...]:
        """Return matches for ``text`` restricted to ``labels``.

        The method iterates over ``labels`` and checks whether the corresponding
        synonym set is present in ``text``.  The score is currently a simple
        binary indicator (``1.0`` for a match) but the class is intentionally
        designed to accommodate future weighting schemes.
        """

        matches: list[KeywordMatch] = []
        normalised = normalise(text)
        for label in labels:
            entry = self.entry(label)
            if not entry:
                continue
            if any(token in entry.tokens for token in normalised.split()):
                matches.append(KeywordMatch(label=label, score=1.0, evidence=(text,)))
                continue
            for alias in self.aliases_for(label):
                alias_entry = self.entry(alias)
                if alias_entry and any(token in alias_entry.tokens for token in normalised.split()):
                    matches.append(KeywordMatch(label=label, score=1.0, evidence=(text,)))
                    break
        return tuple(matches)

    def detect(self, text: str) -> Tuple[KeywordMatch, ...]:
        """Return matches for all registered labels."""

        matches: list[KeywordMatch] = []
        normalised = normalise(text)
        for entry in self._entries.values():
            if any(token in entry.tokens for token in normalised.split()):
                matches.append(KeywordMatch(label=entry.label, score=1.0, evidence=(text,)))
        return tuple(matches)


# ---------------------------------------------------------------------------
# Default lexicon configuration
# ---------------------------------------------------------------------------

def default_lexicon() -> KeywordLexicon:
    """Return a :class:`KeywordLexicon` populated with common synonyms."""

    lexicon = KeywordLexicon()
    lexicon.extend(
        {
            "literal": [
                "literal",
                "literals",
                "const",
                "constant",
                "constants",
                "immediate",
                "константа",
                "константы",
                "литерал",
                "литералы",
                "значение",
            ],
            "ascii": [
                "ascii",
                "текст",
                "строка",
                "string",
                "inline",
            ],
            "marker": [
                "marker",
                "markers",
                "маркер",
                "маркеры",
                "resource",
                "ресурс",
                "ресурсный",
                "структурный",
                "comment",
                "annotation",
            ],
            "push": [
                "push",
                "stack push",
                "загрузка",
                "помещение",
                "stack",
                "помещает",
            ],
            "reduce": [
                "reduce",
                "reducer",
                "fold",
                "collapses",
                "редуктор",
                "сворачивает",
            ],
            "copy": [
                "copy",
                "duplicate",
                "dup",
                "копир",
                "дублир",
            ],
            "indirect": [
                "indirect",
                "table",
                "lookup",
                "косвенные",
                "таблица",
                "индексация",
                "slot",
                "слот",
            ],
            "table": [
                "table",
                "таблица",
                "табличн",
            ],
            "return": [
                "return",
                "terminator",
                "halt",
                "возврат",
                "окончание",
                "завершение",
            ],
            "terminator": [
                "terminator",
                "halt",
                "stop",
                "конец",
                "завершение",
            ],
            "call": [
                "call",
                "invoke",
                "вызов",
                "tailcall",
            ],
            "test": [
                "test",
                "branch",
                "условие",
                "проверка",
            ],
            "arithmetic": [
                "arith",
                "math",
                "матем",
                "арифм",
            ],
            "logical": [
                "logic",
                "boolean",
                "логик",
                "булев",
            ],
            "bitwise": [
                "bit",
                "бит",
                "разря",
            ],
            "stack_teardown": [
                "teardown",
                "drop",
                "clear",
                "pop",
                "tears",
                "сбрасыва",
                "разбор",
                "снимает",
            ],
            "meta": [
                "meta",
                "helper",
                "служеб",
                "вспомог",
                "описат",
            ],
        }
    )

    # Aliases allow us to reuse synonym sets without duplicating data.
    lexicon.register("const", alias="literal")
    lexicon.register("literal_marker", alias="marker")

    return lexicon


# Shared singleton used by the analyser.  The singleton makes it trivial to
# plug the lexicon into existing helper functions without threading additional
# configuration objects through every call site.
DEFAULT_LEXICON = default_lexicon()


def lookup_keywords(text: Optional[str], *labels: str) -> Tuple[KeywordMatch, ...]:
    """Utility wrapper that looks up ``labels`` within ``text``."""

    if not text:
        return tuple()
    return DEFAULT_LEXICON.match(text, labels)


def has_keyword(text: Optional[str], label: str) -> bool:
    """Return ``True`` if ``label`` is present in ``text``."""

    if not text:
        return False
    matches = lookup_keywords(text, label)
    return any(match.label == label for match in matches)
