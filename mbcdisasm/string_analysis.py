"""Utility helpers for analysing and classifying recovered string literals."""

from __future__ import annotations

import math
import re
import string

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

_PRINTABLE = set(string.printable)
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_BOUNDARY = re.compile(r"(?<![A-Z])[A-Z](?=[a-z])")


@dataclass(frozen=True)
class StringInsight:
    """Classification metadata describing an analysed string literal."""

    text: str
    tokens: Tuple[str, ...] = field(default_factory=tuple)
    classification: str = "unknown"
    confidence: float = 0.0
    hints: Tuple[str, ...] = field(default_factory=tuple)
    entropy: float = 0.0
    case_style: str = "neutral"
    printable_ratio: float = 1.0
    token_density: float = 0.0

    def has_hint(self, hint: str) -> bool:
        return hint in self.hints

    def token_score(self) -> float:
        if not self.tokens:
            return 0.0
        total = sum(len(token) for token in self.tokens)
        return total / (len(self.tokens) * 8.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": list(self.tokens),
            "classification": self.classification,
            "confidence": self.confidence,
            "hints": list(self.hints),
            "entropy": self.entropy,
            "case_style": self.case_style,
            "printable_ratio": self.printable_ratio,
            "token_density": self.token_density,
        }


class StringClassifier:
    """Derive human friendly insights about recovered string literals."""

    def __init__(
        self,
        *,
        min_token_length: int = 3,
        max_tokens: int = 24,
        stopwords: Sequence[str] = (),
    ) -> None:
        self._min_length = min_token_length
        self._max_tokens = max_tokens
        self._stopwords = {word.lower() for word in stopwords}

    # ------------------------------------------------------------------
    def classify(self, text: str) -> StringInsight:
        tokens = tuple(self._extract_tokens(text))
        tokens = tokens[: self._max_tokens]
        classification, confidence, hints, printable_ratio = self._evaluate(text, tokens)
        entropy = self._character_entropy(text)
        case_style = self._case_style(text)
        token_density = self._token_density(text, tokens)
        return StringInsight(
            text=text,
            tokens=tokens,
            classification=classification,
            confidence=confidence,
            hints=hints,
            entropy=entropy,
            case_style=case_style,
            printable_ratio=round(printable_ratio, 3),
            token_density=token_density,
        )

    # ------------------------------------------------------------------
    def _extract_tokens(self, text: str) -> Iterator[str]:
        for match in _WORD_PATTERN.finditer(text):
            token = match.group(0)
            if len(token) < self._min_length:
                continue
            lowered = token.lower()
            if lowered in self._stopwords:
                continue
            components = list(self._split_identifier(token))
            if components:
                for component in components:
                    lowered_component = component.lower()
                    if lowered_component not in self._stopwords:
                        yield lowered_component
                continue
            yield lowered

    def _split_identifier(self, token: str) -> Iterable[str]:
        if "_" in token and token.upper() != token:
            pieces = [piece for piece in token.split("_") if len(piece) >= self._min_length]
            if len(pieces) >= 2:
                return pieces
        matches = list(_CAMEL_BOUNDARY.finditer(token))
        if not matches:
            return []
        components: List[str] = []
        last = 0
        for match in matches:
            index = match.start()
            segment = token[last:index]
            if len(segment) >= self._min_length:
                components.append(segment)
            last = index
        tail = token[last:]
        if len(tail) >= self._min_length:
            components.append(tail)
        return components

    # ------------------------------------------------------------------
    def _evaluate(
        self, text: str, tokens: Sequence[str]
    ) -> Tuple[str, float, Tuple[str, ...], float]:
        stripped = text.strip()
        if not stripped:
            return "empty", 0.0, ("blank",), 1.0
        hints: List[str] = []
        ascii_ratio = self._ascii_ratio(text)
        if ascii_ratio < 0.6:
            hints.append("low-printable")
        if "\n" in text or "\r" in text:
            hints.append("multiline")
        if any(ch in text for ch in ("/", "\\")):
            hints.append("path")
        if "%" in text:
            hints.append("format")
        if "{" in text and "}" in text:
            hints.append("template")
        if any(ch.isdigit() for ch in stripped) and not any(ch.isalpha() for ch in stripped):
            hints.append("numeric")
        if stripped.endswith("?"):
            hints.append("question")
        if stripped.endswith("!"):
            hints.append("exclamation")
        classification = self._select_classification(stripped, tokens, hints, ascii_ratio)
        confidence = self._estimate_confidence(classification, hints, tokens)
        return classification, confidence, tuple(hints), ascii_ratio

    def _select_classification(
        self,
        text: str,
        tokens: Sequence[str],
        hints: Sequence[str],
        ascii_ratio: float,
    ) -> str:
        if "low-printable" in hints:
            return "binary"
        if "path" in hints:
            return "path"
        if "format" in hints or "template" in hints:
            return "format"
        if "numeric" in hints:
            return "numeric"
        if tokens and self._looks_like_identifier(text, tokens):
            return "identifier"
        if tokens and ("?" in hints or "exclamation" in hints):
            return "dialogue"
        if tokens and (" " in text or "multiline" in hints):
            return "sentence"
        if ascii_ratio > 0.95 and tokens:
            return "keyword"
        return "unknown"

    def _estimate_confidence(
        self,
        classification: str,
        hints: Sequence[str],
        tokens: Sequence[str],
    ) -> float:
        base = {
            "binary": 0.95,
            "path": 0.9,
            "format": 0.85,
            "numeric": 0.75,
            "identifier": 0.8,
            "dialogue": 0.6,
            "sentence": 0.7,
            "keyword": 0.5,
            "unknown": 0.4,
            "empty": 0.0,
        }[classification]
        modifier = 0.0
        if "question" in hints or "exclamation" in hints:
            modifier += 0.05
        if tokens:
            diversity = len({token for token in tokens})
            modifier += min(0.15, diversity / 40.0)
        modifier = min(modifier, 0.2)
        return round(base + modifier, 3)

    def _looks_like_identifier(self, text: str, tokens: Sequence[str]) -> bool:
        if len(tokens) == 1:
            token = tokens[0]
            if token.islower() and " " not in text:
                return True
            if token.isupper() and " " not in text:
                return True
        if " " in text or "-" in text:
            return False
        has_alpha = any(token.isalpha() for token in tokens)
        return has_alpha and len(tokens) <= 3

    def _ascii_ratio(self, text: str) -> float:
        if not text:
            return 1.0
        printable = sum(1 for char in text if char in _PRINTABLE)
        return printable / len(text)

    def _character_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        counts = {}
        for char in text:
            counts[char] = counts.get(char, 0) + 1
        total = len(text)
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        return round(entropy, 3)

    def _case_style(self, text: str) -> str:
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return "neutral"
        if all(char.isupper() for char in letters):
            return "upper"
        if all(char.islower() for char in letters):
            return "lower"
        if letters[0].isupper() and all(char.islower() for char in letters[1:]):
            return "title"
        return "mixed"

    def _token_density(self, text: str, tokens: Sequence[str]) -> float:
        if not text:
            return 0.0
        density = len(tokens) / len(text)
        return round(density, 3)


def summarise_insights(insights: Sequence[StringInsight]) -> Tuple[int, List[Tuple[str, int]]]:
    """Compute a histogram of classifications for collected insights."""

    histogram = {}
    for insight in insights:
        key = insight.classification
        histogram[key] = histogram.get(key, 0) + 1
    ordered = sorted(histogram.items(), key=lambda item: (-item[1], item[0]))
    return sum(histogram.values()), ordered


def token_histogram(
    insights: Sequence[StringInsight], limit: int = 10
) -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for insight in insights:
        counter.update(insight.tokens)
    return counter.most_common(limit)

