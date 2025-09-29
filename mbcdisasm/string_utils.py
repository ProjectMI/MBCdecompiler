"""Utility helpers for dealing with printable strings and previews.

The Sphere runtime regularly stores ASCII payloads inside instruction streams
and standalone data segments.  Several reconstruction layers need to make
decisions based on how *textual* a blob of bytes looks – for example the inline
string collector prefers rendering comments only when the underlying data is
readable.  Historically each caller implemented their own tiny helper which
inevitably drifted apart whenever escaping rules evolved.

This module centralises the logic.  It intentionally mirrors the semantics used
by :mod:`mbcdisasm.inline_strings` so the behaviour stays consistent across
modules.  Everything operates on raw :class:`bytes` objects and only introduces
Unicode at the very edge when building human-readable previews.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

__all__ = [
    "PRINTABLE_ASCII",
    "is_printable_byte",
    "printable_ratio",
    "trim_ascii_suffix",
    "decode_latin1",
    "Preview",
    "build_preview",
    "chunk_preview",
]


# ---------------------------------------------------------------------------
# printable byte detection
# ---------------------------------------------------------------------------


PRINTABLE_ASCII: frozenset[int] = frozenset(range(0x20, 0x7F))


def is_printable_byte(value: int) -> bool:
    """Return ``True`` when ``value`` is a reasonably printable ASCII byte."""

    if value in PRINTABLE_ASCII:
        return True
    # Frequently encountered whitespace/control characters that are safe to
    # display as-is in Lua comments.  We leave the actual escaping decision to
    # the caller but expose the detection logic centrally.
    return value in {0x09, 0x0A, 0x0D}


def printable_ratio(data: bytes) -> float:
    """Return the fraction of bytes that look printable."""

    if not data:
        return 0.0
    printable = sum(1 for byte in data if is_printable_byte(byte))
    return printable / len(data)


def trim_ascii_suffix(data: bytes) -> bytes:
    """Strip trailing NUL and carriage-return characters used as terminators."""

    while data and data[-1] in (0, 0x0D):
        data = data[:-1]
    return data


def decode_latin1(data: bytes) -> str:
    """Decode ``data`` using Latin-1 with replacement for safety."""

    return data.decode("latin-1", "replace")


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preview:
    """Lightweight representation of a formatted preview."""

    text: str
    truncated: bool

    def __str__(self) -> str:  # pragma: no cover - trivial proxy
        return self.text


def build_preview(parts: Iterable[str], *, truncated: bool = False) -> Preview:
    """Construct a :class:`Preview` from a sequence of ``parts``."""

    text = "".join(parts)
    return Preview(text=text, truncated=truncated)


def _normalise_whitespace(text: str) -> Iterator[str]:
    for char in text:
        code = ord(char)
        if code == 0x0A:
            yield "\\n"
        elif code == 0x0D:
            yield "\\r"
        elif code == 0x09:
            yield "\\t"
        else:
            yield char


def chunk_preview(data: bytes, *, limit: int = 48) -> Preview:
    """Return a preview of ``data`` suitable for inline comments.

    Parameters
    ----------
    limit:
        Maximum number of decoded characters to include before truncating the
        preview.  The function emits an ellipsis when truncation happens so
        callers can surface the hint to the user.
    """

    stripped = trim_ascii_suffix(data)
    if not stripped:
        return Preview(text="<empty>", truncated=False)

    text = decode_latin1(stripped)
    if len(text) <= limit:
        return build_preview(_normalise_whitespace(text))

    visible = text[:limit]
    preview_text = list(_normalise_whitespace(visible))
    preview_text.append("…")
    return build_preview(preview_text, truncated=True)

