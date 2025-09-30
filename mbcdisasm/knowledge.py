"""Simplified opcode knowledge lookup support."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


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
    """Resolve opcode/mode pairs to :class:`OpcodeInfo` entries.

    The original implementation only stored the exact labels defined in
    ``manual_annotations.json`` which meant that the knowledge base could not
    recognise opcode families expressed as ``"29:*"`` in the source document.
    Manual reverse engineering sessions – the ``_char`` script in particular –
    rely heavily on such families to describe large swathes of helper opcodes.
    The class therefore understands both precise labels (``"29:10"``) and
    wildcards (``"29:*"``).  Callers can ask for either; the lookup routine will
    first consult the explicit annotation table before falling back to a
    wildcard entry for the corresponding opcode.

    The loader keeps a secondary mapping of annotation names to ``OpcodeInfo``
    instances.  Wildcard definitions reference these names via the ``category``
    field which historically doubled as a human readable description.  The
    modern loader interprets this field as a link to an existing entry and
    clones its metadata.  This keeps the JSON format backwards compatible while
    dramatically reducing duplication: a single ``tailcall_dispatch`` entry can
    now describe all ``29:*`` modes instead of enumerating each one.
    """

    def __init__(
        self,
        annotations: Mapping[str, OpcodeInfo],
        *,
        wildcards: Optional[Mapping[int, OpcodeInfo]] = None,
        by_name: Optional[Mapping[str, OpcodeInfo]] = None,
    ) -> None:
        self._annotations: Dict[str, OpcodeInfo] = dict(annotations)
        self._wildcards: Dict[int, OpcodeInfo] = dict(wildcards or {})
        self._by_name: Dict[str, OpcodeInfo] = dict(by_name or {})

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
        by_name: Dict[str, OpcodeInfo] = {}
        wildcard_specs: Dict[int, Mapping[str, Any]] = {}

        if isinstance(data, dict):
            for key, entry in data.items():
                if not isinstance(entry, Mapping):
                    continue

                mnemonic = str(entry.get("name") or key)
                info = OpcodeInfo.from_json(mnemonic, entry)

                opcodes = entry.get("opcodes")
                if isinstance(opcodes, Iterable) and opcodes:
                    by_name[key] = info
                    for label in opcodes:
                        if not isinstance(label, str):
                            continue
                        normalized = _normalize_label(label)
                        if normalized is not None:
                            annotations[normalized] = info
                    continue

                opcode_value = _parse_wildcard_key(key)
                if opcode_value is None:
                    by_name[key] = info
                    continue

                wildcard_specs[opcode_value] = entry

        wildcard_annotations = _materialise_wildcards(wildcard_specs, by_name)
        return cls(annotations, wildcards=wildcard_annotations, by_name=by_name)

    def lookup(self, label: str) -> Optional[OpcodeInfo]:
        """Return manual information for the requested opcode label."""

        canonical = label.upper()
        info = self._annotations.get(canonical)
        if info is not None:
            return info

        opcode = _extract_opcode(canonical)
        if opcode is None:
            return None
        try:
            decimal_opcode = _parse_component(canonical.split(":", 1)[0])
        except ValueError:
            decimal_opcode = None
        if decimal_opcode is not None:
            info = self._wildcards.get(decimal_opcode)
            if info is not None:
                return info

        info = self._wildcards.get(opcode)
        if info is not None:
            return info

        return None

    def lookup_by_name(self, name: str) -> Optional[OpcodeInfo]:
        """Return an annotation by the entry name used in the JSON file."""

        return self._by_name.get(name)


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


def _extract_opcode(label: str) -> Optional[int]:
    """Return the opcode component encoded in ``label``.

    The helper accepts labels in the canonical ``"AA:BB"`` form and returns the
    integer value of the first component.  Invalid tokens yield ``None`` which
    allows callers to fall back to other lookup strategies without having to
    repeat the parsing logic.  Canonical labels emitted by
    :meth:`InstructionWord.label` always use hexadecimal so we fast-path that
    representation before falling back to the permissive
    :func:`_parse_component` parser.  This keeps wildcards such as ``"16:*"`` in
    sync with individual opcode lookups like ``"10:84"``.
    """

    if ":" not in label:
        return None

    opcode_text, _ = label.split(":", 1)

    try:
        if len(opcode_text) in {1, 2} and all(ch in string.hexdigits for ch in opcode_text):
            opcode = int(opcode_text, 16)
        else:
            opcode = _parse_component(opcode_text)
    except ValueError:
        return None

    if not (0 <= opcode <= 0xFF):
        return None
    return opcode


def _parse_wildcard_key(key: str) -> Optional[int]:
    """Return the opcode encoded in a ``"AA:*"`` wildcard key."""

    if ":" not in key:
        return None
    opcode_text, mode_text = key.split(":", 1)
    if mode_text.strip() != "*":
        return None
    try:
        opcode = _parse_component(opcode_text)
    except ValueError:
        return None
    if not (0 <= opcode <= 0xFF):
        return None
    return opcode


def _materialise_wildcards(
    wildcard_specs: Mapping[int, Mapping[str, Any]],
    by_name: Mapping[str, OpcodeInfo],
) -> Dict[int, OpcodeInfo]:
    """Translate wildcard entries into opcode-to-info mappings.

    ``wildcard_specs`` maps opcode integers to the raw JSON entries that
    describe the wildcard.  The routine attempts to resolve the ``category``
    field as a reference to another entry and, when successful, reuses the
    associated :class:`OpcodeInfo`.  If the reference cannot be resolved a fresh
    ``OpcodeInfo`` instance is built from the sparse data embedded in the
    wildcard entry.
    """

    resolved: Dict[int, OpcodeInfo] = {}
    for opcode, entry in wildcard_specs.items():
        target = entry.get("category")
        if isinstance(target, str) and target in by_name:
            resolved[opcode] = by_name[target]
            continue

        mnemonic = str(entry.get("name") or entry.get("mnemonic") or f"{opcode:02X}:*")
        info = OpcodeInfo.from_json(mnemonic, entry)
        resolved[opcode] = info
    return resolved
