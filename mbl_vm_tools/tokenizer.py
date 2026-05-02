from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .vm_spec import VMWord, decode_words


@dataclass(frozen=True)
class Token:
    """Thin token wrapper over the authoritative VM word decoder."""

    offset: int
    kind: str
    size: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "kind": self.kind,
            "size": self.size,
            "payload": self.payload,
        }


def token_from_word(word: VMWord) -> Token:
    payload = dict(word.operands)
    if word.prefixes:
        payload["prefixes"] = list(word.prefixes)
    payload["terminal_kind"] = word.terminal_kind
    payload["decoder_rule"] = word.decoder_rule
    payload["raw_hex"] = word.raw.hex(" ")
    return Token(offset=word.offset, kind=word.kind, size=word.size, payload=payload)


def tokenize_stream(data: bytes, *, base_offset: int = 0) -> list[Token]:
    tokens = [token_from_word(word) for word in decode_words(data)]
    if base_offset:
        tokens = [Token(t.offset + base_offset, t.kind, t.size, dict(t.payload)) for t in tokens]
    return tokens
