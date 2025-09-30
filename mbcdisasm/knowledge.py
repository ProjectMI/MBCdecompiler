"""Simplified opcode knowledge lookup support."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class OpcodeInfo:
    """Human-friendly annotation for a single opcode/mode combination.

    The historical tooling only tracked a mnemonic string and an optional
    summary.  The upcoming pipeline analyser requires richer metadata so the
    structure has been expanded to accommodate additional fields commonly found
    in the manual annotation database.  All fields are optional and default to
    ``None`` or an empty mapping which keeps the public surface backwards
    compatible for callers that only rely on the mnemonic.
    """

    mnemonic: str
    summary: Optional[str] = None
    control_flow: Optional[str] = None
    category: Optional[str] = None
    stack_delta: Optional[int] = None
    stack_push: Optional[int] = None
    stack_pop: Optional[int] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def stack_effect(self) -> Optional[int]:
        """Return the signed stack delta if it is explicitly defined."""

        if self.stack_delta is not None:
            return self.stack_delta
        if self.stack_push is not None or self.stack_pop is not None:
            pushed = self.stack_push or 0
            popped = self.stack_pop or 0
            return pushed - popped
        return None

    @classmethod
    def from_json(cls, mnemonic: str, entry: Mapping[str, Any]) -> "OpcodeInfo":
        """Create an :class:`OpcodeInfo` instance from a JSON entry."""

        summary = entry.get("summary")
        control_flow = entry.get("control_flow")
        category = entry.get("category")

        stack_delta = entry.get("stack_delta")
        stack_push = entry.get("stack_push")
        stack_pop = entry.get("stack_pop")

        attributes: Dict[str, Any] = {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "name",
                "mnemonic",
                "summary",
                "control_flow",
                "category",
                "opcodes",
                "stack_delta",
                "stack_push",
                "stack_pop",
            }
        }

        return cls(
            mnemonic=mnemonic,
            summary=summary,
            control_flow=control_flow,
            category=category,
            stack_delta=stack_delta,
            stack_push=stack_push,
            stack_pop=stack_pop,
            attributes=attributes,
        )


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
                info = OpcodeInfo.from_json(mnemonic, entry)
                for label in entry.get("opcodes", []):
                    if not isinstance(label, str):
                        continue
                    normalized = _normalize_label(label)
                    if normalized is not None:
                        annotations[normalized] = info

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
