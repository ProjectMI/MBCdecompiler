from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from .vm_spec import (
    VMWord,
    branch_operand_base_offset,
    branch_target_offset,
    signed_u16,
    terminal_atom_offset,
)


SEMANTIC_CONTRACT_VERSION = "vm-semantic-v1"
BRANCH_SEMANTIC_POLICY = (
    "Opcode 0x4A is a VM control jump; opcodes 0x4B/0x4C/0x4D are "
    "conditional VM branches. Predicate polarity and source-level condition "
    "meaning are intentionally not inferred here."
)


@dataclass(frozen=True)
class VMBranchSemantics:
    contract: str
    word_index: Optional[int]
    offset: int
    size: int
    terminal_op_offset: int
    operand_base_offset: int
    op: int
    prefixes_hex: list[str]
    encoded_offset: int
    signed_offset: int
    target_offset: int
    fallthrough_offset: Optional[int]
    branch_kind: str
    taken_edge_kind: str
    fallthrough_edge_kind: Optional[str]
    has_fallthrough_edge: bool
    predicate_source: Optional[str]
    predicate_polarity: Optional[str]
    stack_pop: Optional[int]
    confidence: float
    policy: str = BRANCH_SEMANTIC_POLICY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


UNCONDITIONAL_BRANCH_OPS = {0x4A}
CONDITIONAL_BRANCH_OPS = {0x4B, 0x4C, 0x4D}


def classify_branch(word: VMWord, *, source_word_index: Optional[int] = None) -> VMBranchSemantics:
    """Classify VM branch control-flow semantics without source-level lowering.

    This pass only labels edge behavior.  It does not decide whether a condition
    is ``true`` or ``false`` in source terms, and it does not transform regions
    into ``if``/``while`` nodes.
    """

    if word.terminal_kind != "BR":
        raise ValueError("classify_branch requires a BR VMWord")

    op = int(word.operands.get("op", -1) or -1) & 0xFF
    encoded = int(word.operands.get("off", 0) or 0) & 0xFFFF
    signed = signed_u16(encoded)
    target = branch_target_offset(word)
    fallthrough = int(word.offset) + int(word.size)
    word_index = source_word_index
    if word_index is None:
        word_index = int(word.index) if int(word.index) >= 0 else None

    if op in UNCONDITIONAL_BRANCH_OPS:
        branch_kind = "unconditional_jump"
        taken_edge_kind = "jump"
        fallthrough_edge_kind = None
        has_fallthrough_edge = False
        predicate_source = None
        predicate_polarity = None
        stack_pop: Optional[int] = 0
        confidence = 1.0
    elif op in CONDITIONAL_BRANCH_OPS:
        branch_kind = "conditional_branch"
        taken_edge_kind = "conditional_taken"
        fallthrough_edge_kind = "conditional_fallthrough"
        has_fallthrough_edge = True
        predicate_source = "prefix_chain_or_stack" if word.prefixes else "stack_top_or_flag"
        # Polarity is deliberately unresolved.  The VM can tell us which edge is
        # taken by the branch atom, but not yet whether that edge represents a
        # source-level true or false condition.
        predicate_polarity = "unresolved"
        stack_pop = None
        confidence = 0.95
    else:
        branch_kind = "unknown_branch"
        taken_edge_kind = "branch"
        fallthrough_edge_kind = "fallthrough"
        has_fallthrough_edge = True
        predicate_source = "unknown"
        predicate_polarity = "unknown"
        stack_pop = None
        confidence = 0.0

    return VMBranchSemantics(
        contract=SEMANTIC_CONTRACT_VERSION,
        word_index=word_index,
        offset=int(word.offset),
        size=int(word.size),
        terminal_op_offset=terminal_atom_offset(word),
        operand_base_offset=branch_operand_base_offset(word),
        op=op,
        prefixes_hex=[f"0x{p:02X}" for p in word.prefixes],
        encoded_offset=encoded,
        signed_offset=signed,
        target_offset=target,
        fallthrough_offset=fallthrough if has_fallthrough_edge else None,
        branch_kind=branch_kind,
        taken_edge_kind=taken_edge_kind,
        fallthrough_edge_kind=fallthrough_edge_kind,
        has_fallthrough_edge=has_fallthrough_edge,
        predicate_source=predicate_source,
        predicate_polarity=predicate_polarity,
        stack_pop=stack_pop,
        confidence=confidence,
    )
