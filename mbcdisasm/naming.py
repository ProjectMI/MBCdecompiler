"""Identifier normalisation and naming heuristics used across the project.

The decompiler needs to mint a large amount of temporary names while
reconstructing high level Lua.  Historically every subsystem rolled its own
helper which led to subtly different behaviour between the VM analysis layer
and the high level renderer.  The result were annoying inconsistencies – a
value named ``literal_0`` inside the VM trace could end up being called
``tmp_0`` once the high level reconstructor emitted Lua code.  Coordinating the
two implementations was brittle and made future improvements hard because every
change had to be mirrored manually.

This module centralises the logic for deriving human friendly identifiers.  It
exposes a small toolbox that sanitises arbitrary strings and extracts sensible
base names from :class:`~mbcdisasm.manual_semantics.InstructionSemantics`
objects.  Having a shared implementation keeps the heuristics consistent
between the VM analysis reports and the Lua emitter which in turn makes it much
easier to follow a value across the different stages of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .manual_semantics import InstructionSemantics

__all__ = [
    "sanitize_identifier",
    "NameAllocator",
    "derive_stack_symbol_name",
]


def sanitize_identifier(name: str, default: str = "value") -> str:
    """Return a Lua compatible identifier based on ``name``.

    The helper performs a couple of normalisation steps:

    * non-alphanumeric characters are replaced with underscores,
    * the identifier is forced to lower-case to keep the generated code
      stylistically consistent,
    * leading underscores are stripped so that we do not accidentally emit
      special Lua names like ``__index`` unless the input explicitly requests
      it,
    * identifiers that start with a digit gain a ``v_`` prefix because Lua does
      not allow numbers as the first character of a name,
    * finally, if everything collapses to an empty string we fall back to the
      provided ``default`` value.

    The behaviour mirrors the historical implementation from
    :mod:`mbcdisasm.vm_analysis` while being shared between all components.
    """

    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in name]
    if not cleaned:
        return default
    identifier = "".join(cleaned)
    while identifier.startswith("_"):
        identifier = identifier[1:]
    identifier = identifier.strip("_")
    if not identifier:
        identifier = default
    if identifier[0].isdigit():
        identifier = "v_" + identifier
    return identifier


@dataclass
class NameAllocator:
    """Utility class that hands out unique names for a chosen base.

    The allocator keeps an internal counter per base prefix.  The first request
    for ``literal`` returns the bare name while subsequent calls receive a
    numerical suffix.  This mirrors the common ``literal``, ``literal_1`` naming
    scheme that existing tooling expects.  The class intentionally keeps the API
    tiny – we only need :meth:`allocate` for now but wrapping it in an object
    allows future extensions such as reserving names or providing human readable
    hints without touching all call sites.
    """

    default: str = "value"
    _counters: Dict[str, int] = field(default_factory=dict)

    def allocate(self, base: str | None = None) -> str:
        base = sanitize_identifier(base or self.default, self.default)
        counter = self._counters.get(base, 0)
        self._counters[base] = counter + 1
        if counter == 0:
            return base
        return f"{base}_{counter}"


def derive_stack_symbol_name(
    semantics: InstructionSemantics, fallback: str = "value"
) -> str:
    """Return a readable base name for a stack value produced by ``semantics``.

    The heuristics mirror those used by the VM analysis reports.  Literal
    instructions win over comparison helpers which in turn win over generic
    call helpers.  Falling back to the mnemonic keeps the naming stable even for
    opcodes without a manual annotation.
    """

    if semantics.has_tag("literal"):
        base = "literal"
    elif semantics.has_tag("comparison"):
        base = "cmp"
    elif semantics.control_flow == "branch":
        base = "cond"
    else:
        base = semantics.vm_method or semantics.manual_name or semantics.mnemonic
    return sanitize_identifier(base or fallback, fallback)

