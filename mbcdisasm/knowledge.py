"""Simplified opcode knowledge lookup support."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class OpcodeInfo:
    """Human-friendly annotation for a single opcode/mode combination."""

    mnemonic: str
    summary: Optional[str] = None


class KnowledgeBase:
    """Very small helper that resolves opcodes to manual annotations."""

    def __init__(self, annotations: Mapping[str, OpcodeInfo]):
        self._annotations: Dict[str, OpcodeInfo] = dict(annotations)

    @classmethod
    def load(cls, manual_path: Path) -> "KnowledgeBase":
        """Load a knowledge base from the provided manual annotation file."""

        resolved = manual_path
        if manual_path.is_dir():
            resolved = manual_path / "manual_annotations.json"

        if not resolved.exists():
            return cls({})

        data = json.loads(resolved.read_text("utf-8"))
        annotations: Dict[str, OpcodeInfo] = {}

        if isinstance(data, dict):
            for key, entry in data.items():
                if not isinstance(entry, Mapping):
                    continue
                mnemonic = str(entry.get("name") or key)
                summary = entry.get("summary")
                for label in entry.get("opcodes", []):
                    if not isinstance(label, str):
                        continue
                    normalized = _normalize_label(label)
                    if normalized is not None:
                        annotations[normalized] = OpcodeInfo(mnemonic=mnemonic, summary=summary)

        return cls(annotations)

    def lookup(self, label: str) -> Optional[OpcodeInfo]:
        """Return manual information for the requested opcode label."""

        return self._annotations.get(label.upper())


def _parse_component(component: str) -> int:
    """Parse a single opcode component which may be hex or decimal."""

    token = component.strip()
    if not token:
        raise ValueError("empty component")

    if token.lower().startswith("0x"):
        return int(token, 16)

    if any(ch in string.hexdigits[10:] for ch in token):
        return int(token, 16)

    return int(token, 10)


def _normalize_label(label: str) -> Optional[str]:
    """Convert mixed-format labels to canonical hexadecimal form."""

    token = label.strip()
    if not token:
        return None

    if ":" not in token:
        return token.upper()

    parts = token.split(":")
    if len(parts) != 2:
        return token.upper()

    try:
        opcode = _parse_component(parts[0])
        mode = _parse_component(parts[1])
    except ValueError:
        return token.upper()

    if not (0 <= opcode <= 0xFF and 0 <= mode <= 0xFF):
        return None

    return f"{opcode:02X}:{mode:02X}"
