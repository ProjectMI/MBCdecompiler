"""Heuristics for classifying ASCII strings embedded in ``_char``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class StringClassification:
    """Result of :class:`StringClassifier` inference."""

    category: str
    tags: Tuple[str, ...] = field(default_factory=tuple)
    weight: float = 1.0
    reason: str = ""

    def describe(self) -> str:
        chunks = [self.category]
        if self.tags:
            chunks.append("(" + ", ".join(self.tags) + ")")
        if self.reason:
            chunks.append(f"[{self.reason}]")
        return " ".join(chunks)


class StringClassifier:
    """Categorise ASCII snippets extracted from ``_char`` resources."""

    _function_pattern = re.compile(r"^(?:[A-Z][a-z0-9]+){1,3}(?:[A-Z][a-z0-9]*)?$")
    _config_pattern = re.compile(r"params\\[\\w.]+", re.IGNORECASE)
    _animation_tokens = {
        "walk",
        "run",
        "shoot",
        "die",
        "sleep",
        "magic",
        "free",
        "axe",
        "sword",
        "mantra",
        "turnskin",
    }
    _log_markers = ("ERROR", "Created", "setting", "Can't open")

    def classify(self, text: str) -> StringClassification:
        trimmed = text.strip()
        if not trimmed:
            return StringClassification(category="empty", weight=0.1, reason="blank")

        lower = trimmed.lower()

        if self._config_pattern.search(trimmed):
            return StringClassification(category="config_path", tags=(trimmed,), weight=1.2, reason="matches params pattern")
        if lower.startswith("params") and "cfg" in lower:
            return StringClassification(category="config_path", tags=(trimmed,), weight=1.0, reason="params prefix")

        if any(marker.lower() in lower for marker in self._log_markers):
            return StringClassification(category="log_message", weight=0.9, reason="contains log marker")

        if "%" in trimmed:
            return StringClassification(category="format_string", weight=0.8, reason="printf placeholder")

        if self._function_pattern.match(trimmed):
            return StringClassification(category="function_name", tags=(trimmed,), weight=1.0, reason="camel case")

        if trimmed.isupper() and len(trimmed) <= 32:
            tokens = re.split(r"[^A-Z]+", trimmed)
            hits = [token.lower() for token in tokens if token]
            if any(token in self._animation_tokens for token in hits):
                return StringClassification(category="animation_state", tags=tuple(hits), weight=1.1, reason="animation token")

        if trimmed.startswith("_") and trimmed[1:].isalpha():
            return StringClassification(category="identifier", tags=(trimmed,), weight=0.7, reason="leading underscore")

        if any(ch.isdigit() for ch in trimmed):
            return StringClassification(category="mixed", weight=0.6, reason="contains digits")

        return StringClassification(category="text", weight=0.5, reason="fallback")


GLOBAL_STRING_CLASSIFIER = StringClassifier()
