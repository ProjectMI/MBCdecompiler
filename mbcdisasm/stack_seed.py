"""Helpers for seeding :class:`StackDeltaModeler` instances."""

from __future__ import annotations

from typing import Optional, Sequence, Set

from .instruction import InstructionWord
from .knowledge import KnowledgeBase
from .stack_model import StackDeltaModeler


def seed_stack_modeler_from_knowledge(
    modeler: StackDeltaModeler,
    knowledge: KnowledgeBase,
    instructions: Sequence[InstructionWord],
    *,
    seeded_keys: Optional[Set[str]] = None,
) -> None:
    """Prime ``modeler`` with stack deltas learnt from ``knowledge``.

    ``seeded_keys`` may be supplied to avoid redundant lookups for opcode
    labels that have already been processed.  The set is mutated in-place when
    provided.
    """

    for instr in instructions:
        key = instr.label()
        if seeded_keys is not None and key in seeded_keys:
            continue
        if modeler.known_delta(key) is not None:
            if seeded_keys is not None:
                seeded_keys.add(key)
            continue
        estimate = knowledge.estimate_stack_delta(key)
        if estimate is None:
            continue
        modeler.seed_known_delta(key, float(estimate))
        if seeded_keys is not None:
            seeded_keys.add(key)
