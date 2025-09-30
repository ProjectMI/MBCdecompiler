"""Simplified opcode knowledge lookup support."""

from __future__ import annotations

import json
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
                    if isinstance(label, str):
                        annotations[label.upper()] = OpcodeInfo(mnemonic=mnemonic, summary=summary)

        return cls(annotations)

    def lookup(self, label: str) -> Optional[OpcodeInfo]:
        """Return manual information for the requested opcode label."""

        return self._annotations.get(label.upper())
