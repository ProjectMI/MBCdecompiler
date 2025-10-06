"""Simplified opcode knowledge lookup support."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


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
        
        normalized_aliases = None
        for alias_key in ("operand_aliases", "operand_names"):
            raw_aliases = attributes.get(alias_key)
            if isinstance(raw_aliases, Mapping):
                normalized_aliases = _normalize_operand_aliases(raw_aliases)
                attributes[alias_key] = normalized_aliases
                break
        if normalized_aliases is not None:
            attributes.setdefault("operand_aliases", normalized_aliases)

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


@dataclass(frozen=True)
class CallSignatureEffect:
    """Description of an instruction folded into a call contract."""

    mnemonic: str
    operand: Optional[int] = None
    pops: int = 0
    operand_role: Optional[str] = None
    operand_alias: Optional[str] = None
    inherit_operand: bool = False
    inherit_alias: bool = False


@dataclass(frozen=True)
class CallSignaturePattern:
    """Sequence element consumed when assembling a call contract."""

    kind: str
    mnemonic: Optional[str] = None
    operand: Optional[int] = None
    optional: bool = False
    effect: Optional[CallSignatureEffect] = None
    cleanup_mask: Optional[int] = None
    tail: bool = False
    predicate: Optional[str] = None


@dataclass(frozen=True)
class CallSignature:
    """Contract describing how a helper call should appear in the IR."""

    target: int
    arity: Optional[int] = None
    returns: Optional[int] = None
    shuffle: Optional[int] = None
    shuffle_options: Tuple[int, ...] = tuple()
    cleanup_mask: Optional[int] = None
    cleanup: Tuple[CallSignatureEffect, ...] = tuple()
    prelude: Tuple[CallSignaturePattern, ...] = tuple()
    postlude: Tuple[CallSignaturePattern, ...] = tuple()
    tail: Optional[bool] = None


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
        addresses: Optional[Mapping[int, str]] = None,
        call_signatures: Optional[Mapping[int, CallSignature]] = None,
    ) -> None:
        self._annotations: Dict[str, OpcodeInfo] = dict(annotations)
        self._wildcards: Dict[int, OpcodeInfo] = dict(wildcards or {})
        self._by_name: Dict[str, OpcodeInfo] = dict(by_name or {})
        self._addresses: Dict[int, str] = dict(addresses or {})
        self._call_signatures: Dict[int, CallSignature] = dict(call_signatures or {})

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
        address_table: Dict[int, str] = {}
        call_signatures: Dict[int, CallSignature] = {}

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

        table_path = resolved.with_name("address_table.json")
        if table_path.exists():
            address_table = _load_address_table(table_path)

        signatures_path = resolved.with_name("call_signatures.json")
        if signatures_path.exists():
            call_signatures = _load_call_signatures(signatures_path)

        return cls(
            annotations,
            wildcards=wildcard_annotations,
            by_name=by_name,
            addresses=address_table,
            call_signatures=call_signatures,
        )

    def lookup(self, label: str) -> Optional[OpcodeInfo]:
        """Return manual information for the requested opcode label."""

        canonical = label.upper()
        info = self._annotations.get(canonical)
        if info is not None:
            return info

        opcode = _extract_opcode(canonical)
        if opcode is None:
            return None
        return self._wildcards.get(opcode)

    def lookup_by_name(self, name: str) -> Optional[OpcodeInfo]:
        """Return an annotation by the entry name used in the JSON file."""

        return self._by_name.get(name)

    def lookup_address(self, address: int) -> Optional[str]:
        """Resolve ``address`` to a symbolic name if it is known."""

        return self._addresses.get(int(address) & 0xFFFF)

    def call_signature(self, target: int) -> Optional[CallSignature]:
        """Return the call contract for ``target`` when available."""

        return self._call_signatures.get(int(target) & 0xFFFF)


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

def _normalize_operand_aliases(aliases: Mapping[Any, Any]) -> Dict[int, str]:
    """Convert mapping keys from JSON into integer operand values."""

    normalized: Dict[int, str] = {}
    for key, value in aliases.items():
        operand = _parse_operand_value(key)
        if operand is None:
            continue
        normalized[operand] = str(value)
    return normalized


def _parse_operand_value(key: Any) -> Optional[int]:
    """Interpret ``key`` as a hexadecimal or decimal operand value."""

    if isinstance(key, int):
        return key & 0xFFFF

    if isinstance(key, str):
        token = key.strip()
        if not token:
            return None
        if token.lower().startswith("0x"):
            try:
                return int(token, 16) & 0xFFFF
            except ValueError:
                return None
        if any(ch in string.hexdigits[10:] for ch in token):
            try:
                return int(token, 16) & 0xFFFF
            except ValueError:
                return None
        try:
            return int(token, 10) & 0xFFFF
        except ValueError:
            return None

    return None

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
    repeat the parsing logic.
    """

    if ":" not in label:
        return None
    opcode_text, _ = label.split(":", 1)
    try:
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


def _load_address_table(table_path: Path) -> Dict[int, str]:
    """Load a ``address`` → ``symbol`` mapping from ``table_path``."""

    try:
        raw = json.loads(table_path.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}

    if not isinstance(raw, Mapping):
        return {}

    table: Dict[int, str] = {}
    for key, value in raw.items():
        symbol = str(value).strip()
        if not symbol:
            continue
        parsed = _parse_operand_value(key)
        if parsed is None:
            continue
        table[parsed] = symbol
    return table


def _load_call_signatures(table_path: Path) -> Dict[int, CallSignature]:
    """Load ABI contracts for helper calls."""

    try:
        raw = json.loads(table_path.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}

    if not isinstance(raw, Mapping):
        return {}

    signatures: Dict[int, CallSignature] = {}
    for key, entry in raw.items():
        target = _parse_operand_value(key)
        if target is None:
            continue
        signature = _parse_call_signature_entry(target, entry)
        if signature is not None:
            signatures[target] = signature
    return signatures


def _parse_operand_value(key: Any) -> Optional[int]:
    """Interpret ``key`` as a hexadecimal or decimal operand value."""

    if isinstance(key, int):
        return key & 0xFFFF

    if isinstance(key, str):
        token = key.strip()
        if not token:
            return None
        if token.lower().startswith("0x"):
            try:
                return int(token, 16) & 0xFFFF
            except ValueError:
                return None
        if any(ch in string.hexdigits[10:] for ch in token):
            try:
                return int(token, 16) & 0xFFFF
            except ValueError:
                return None
        try:
            return int(token, 10) & 0xFFFF
        except ValueError:
            return None

    return None


def _parse_call_signature_entry(
    target: int, entry: Any
) -> Optional[CallSignature]:
    if not isinstance(entry, Mapping):
        return CallSignature(target=target)

    arity = _parse_optional_int(entry.get("arity"))
    returns = _parse_optional_int(entry.get("returns"))
    shuffle = _parse_optional_int(entry.get("shuffle"))

    shuffle_options: Tuple[int, ...] = tuple(
        value
        for value in (
            _parse_optional_int(candidate)
            for candidate in _ensure_sequence(entry.get("shuffle_options"))
        )
        if value is not None
    )

    cleanup_mask = _parse_optional_int(entry.get("cleanup_mask"))

    cleanup_effects: Tuple[CallSignatureEffect, ...] = tuple(
        effect
        for effect in (
            _parse_call_signature_effect(spec)
            for spec in _ensure_sequence(entry.get("cleanup"))
        )
        if effect is not None
    )

    prelude_patterns: Tuple[CallSignaturePattern, ...] = tuple(
        pattern
        for pattern in (
            _parse_call_signature_pattern(spec)
            for spec in _ensure_sequence(entry.get("prelude"))
        )
        if pattern is not None
    )

    postlude_patterns: Tuple[CallSignaturePattern, ...] = tuple(
        pattern
        for pattern in (
            _parse_call_signature_pattern(spec)
            for spec in _ensure_sequence(entry.get("postlude"))
        )
        if pattern is not None
    )

    tail_value = entry.get("tail")
    tail = tail_value if isinstance(tail_value, bool) else None

    return CallSignature(
        target=target,
        arity=arity,
        returns=returns,
        shuffle=shuffle,
        shuffle_options=shuffle_options,
        cleanup_mask=cleanup_mask,
        cleanup=cleanup_effects,
        prelude=prelude_patterns,
        postlude=postlude_patterns,
        tail=tail,
    )


def _parse_call_signature_effect(spec: Any) -> Optional[CallSignatureEffect]:
    if not isinstance(spec, Mapping):
        return None

    mnemonic = spec.get("mnemonic")
    if not isinstance(mnemonic, str) or not mnemonic:
        return None

    operand = _parse_optional_int(spec.get("operand"))
    pops_value = spec.get("pops")
    pops = _parse_optional_int(pops_value)
    if pops is None:
        pops = 0

    operand_role = spec.get("operand_role")
    if operand_role is not None:
        operand_role = str(operand_role)

    operand_alias = spec.get("operand_alias")
    if operand_alias is not None:
        operand_alias = str(operand_alias)

    inherit_operand = bool(spec.get("inherit_operand", False))
    inherit_alias = bool(spec.get("inherit_alias", False))

    return CallSignatureEffect(
        mnemonic=mnemonic,
        operand=operand,
        pops=pops,
        operand_role=operand_role,
        operand_alias=operand_alias,
        inherit_operand=inherit_operand,
        inherit_alias=inherit_alias,
    )


def _parse_call_signature_pattern(spec: Any) -> Optional[CallSignaturePattern]:
    if not isinstance(spec, Mapping):
        return None

    kind = spec.get("kind", "raw")
    if not isinstance(kind, str) or not kind:
        return None

    mnemonic = spec.get("mnemonic")
    if mnemonic is not None:
        mnemonic = str(mnemonic)

    operand = _parse_optional_int(spec.get("operand"))
    optional = bool(spec.get("optional", False))
    effect = _parse_call_signature_effect(spec.get("effect"))
    cleanup_mask = _parse_optional_int(spec.get("cleanup_mask"))
    tail_value = spec.get("tail")
    tail = bool(tail_value) if isinstance(tail_value, bool) else False
    predicate = spec.get("predicate")
    if predicate is not None:
        predicate = str(predicate)

    return CallSignaturePattern(
        kind=kind,
        mnemonic=mnemonic,
        operand=operand,
        optional=optional,
        effect=effect,
        cleanup_mask=cleanup_mask,
        tail=tail,
        predicate=predicate,
    )


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        base = 16 if token.lower().startswith("0x") else 10
        try:
            return int(token, base)
        except ValueError:
            return None
    return None


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    if value is None:
        return ()
    return (value,)
