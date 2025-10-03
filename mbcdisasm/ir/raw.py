"""Parse instruction words into a raw opcode stream suitable for normalisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from ..instruction import InstructionWord
from ..knowledge import KnowledgeBase
from ..analyzer.instruction_profile import InstructionKind, InstructionProfile


@dataclass
class RawInstruction:
    """Instruction decorated with stack metadata."""

    profile: InstructionProfile
    stack_delta: int
    annotations: List[str] = field(default_factory=list)

    @property
    def mnemonic(self) -> str:
        return self.profile.mnemonic

    @property
    def offset(self) -> int:
        return self.profile.word.offset


@dataclass
class RawBasicBlock:
    """Simple basic block containing a linear instruction range."""

    label: str
    instructions: List[RawInstruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    def add_instruction(self, instruction: RawInstruction) -> None:
        self.instructions.append(instruction)

    def add_successor(self, label: str) -> None:
        if label not in self.successors:
            self.successors.append(label)


@dataclass
class RawSegment:
    blocks: Sequence[RawBasicBlock]


def parse_raw_segment(
    words: Sequence[InstructionWord], knowledge: KnowledgeBase
) -> RawSegment:
    """Convert instruction ``words`` into :class:`RawBasicBlock` instances."""

    blocks: List[RawBasicBlock] = []
    current = RawBasicBlock(label=_make_label(len(blocks)))
    blocks.append(current)

    pending_annotations: List[str] = []

    for idx, word in enumerate(words):
        profile = InstructionProfile.from_word(word, knowledge)
        if profile.is_literal_marker():
            pending_annotations.append(profile.mnemonic)
            continue

        stack_hint = profile.estimated_stack_delta()
        instruction = RawInstruction(profile=profile, stack_delta=stack_hint.nominal)
        if pending_annotations:
            instruction.annotations.extend(pending_annotations)
            pending_annotations.clear()

        current.add_instruction(instruction)

        if _terminates_block(profile):
            successor_label = _make_label(len(blocks))
            if idx + 1 < len(words):
                next_block = RawBasicBlock(label=successor_label)
                blocks.append(next_block)
                current.add_successor(successor_label)
                current = next_block
        elif _starts_new_block(profile):
            successor_label = _make_label(len(blocks))
            next_block = RawBasicBlock(label=successor_label)
            blocks.append(next_block)
            current.add_successor(successor_label)
            current = next_block

    return RawSegment(blocks=blocks)


def _terminates_block(profile: InstructionProfile) -> bool:
    if profile.kind in {InstructionKind.RETURN, InstructionKind.TERMINATOR}:
        return True
    if profile.mnemonic in {"return_values"}:
        return True
    return False


def _starts_new_block(profile: InstructionProfile) -> bool:
    return profile.kind is InstructionKind.BRANCH or profile.mnemonic in {"test_branch", "testset_branch"}


def _make_label(index: int) -> str:
    return f"blk_{index:03d}"
