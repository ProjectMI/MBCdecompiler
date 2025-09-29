"""Control-flow graph construction utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .instruction import InstructionWord, WORD_SIZE, read_instructions
from .knowledge import KnowledgeBase
from .manual_semantics import AnnotatedInstruction, ManualSemanticAnalyzer
from .vm_analysis import estimate_stack_io
from .mbc import Segment


@dataclass
class BasicBlock:
    """A linear sequence of instructions without internal jumps."""

    start: int
    instructions: List[AnnotatedInstruction]
    successors: Set[int] = field(default_factory=set)
    predecessors: Set[int] = field(default_factory=set)

    @property
    def end(self) -> int:
        if not self.instructions:
            return self.start
        return self.instructions[-1].word.offset + WORD_SIZE


@dataclass
class ControlFlowGraph:
    """Container for the CFG associated with a segment."""

    segment: Segment
    blocks: Dict[int, BasicBlock]
    remainder: int = 0

    def block_order(self) -> List[BasicBlock]:
        return [self.blocks[offset] for offset in sorted(self.blocks)]

    def to_text(self) -> str:
        lines: List[str] = [
            f"segment {self.segment.index} cfg (start=0x{self.segment.start:06X})"
        ]
        for block in self.block_order():
            lines.append(
                f"  block 0x{block.start:06X} size={len(block.instructions)} succ={[hex(s) for s in sorted(block.successors)]}"
            )
            for instr in block.instructions:
                sem = instr.semantics
                inputs, outputs = estimate_stack_io(sem)
                lines.append(
                    f"    {instr.word.offset:08X}: {instr.word.raw:08X} "
                    f"op={instr.word.opcode:02X} mode={instr.word.mode:02X} "
                    f"operand=0x{instr.word.operand:04X}"
                    f" {sem.manual_name}"
                    f" inputs={inputs} outputs={outputs}"
                )
        if self.remainder:
            lines.append(f"  trailing bytes ignored: {self.remainder}")
        return "\n".join(lines) + "\n"


class ControlFlowGraphBuilder:
    """Create CFGs using heuristics supplied by the knowledge base."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        *,
        semantic_analyzer: Optional[ManualSemanticAnalyzer] = None,
    ) -> None:
        self.knowledge = knowledge
        self._semantic_analyzer = semantic_analyzer or ManualSemanticAnalyzer(knowledge)

    def build(
        self,
        segment: Segment,
        *,
        max_instructions: Optional[int] = None,
    ) -> ControlFlowGraph:
        raw_instructions, remainder = read_instructions(segment.data, segment.start)
        if max_instructions is not None:
            raw_instructions = raw_instructions[:max_instructions]

        if not raw_instructions:
            return ControlFlowGraph(segment, {}, remainder)

        annotated = [
            self._semantic_analyzer.describe_word(instr) for instr in raw_instructions
        ]

        block_starts = self._discover_block_starts(segment, annotated)
        blocks = self._materialise_blocks(annotated, block_starts)
        self._link_edges(segment, blocks)
        return ControlFlowGraph(segment, blocks, remainder)

    def _discover_block_starts(
        self,
        segment: Segment,
        instructions: Sequence[AnnotatedInstruction],
    ) -> Set[int]:
        starts: Set[int] = {instructions[0].word.offset}
        offsets = {instr.word.offset: idx for idx, instr in enumerate(instructions)}
        available_offsets = set(offsets)
        for idx, instr in enumerate(instructions):
            semantics = instr.semantics
            key = instr.label()
            hint = semantics.control_flow or self.knowledge.control_flow_hint(key)
            next_offset = instr.word.offset + WORD_SIZE
            if hint == "return" or hint == "stop":
                if idx + 1 < len(instructions):
                    starts.add(next_offset)
                continue
            if hint in {"jump", "branch", "call"}:
                target = self._resolve_target(segment, instr.word, available_offsets)
                if target is not None:
                    starts.add(target)
            # Even for unconditional jumps we keep the following block as a
            # potential entry so the graph remains connected for fallthrough
            # analysis or when annotations are imprecise.
            if idx + 1 < len(instructions):
                starts.add(next_offset)

        # Normalise to existing instruction offsets.
        return {offset for offset in starts if offset in offsets}

    def _materialise_blocks(
        self,
        instructions: Sequence[AnnotatedInstruction],
        starts: Set[int],
    ) -> Dict[int, BasicBlock]:
        ordered_starts = sorted(starts)
        index_by_offset = {instr.word.offset: idx for idx, instr in enumerate(instructions)}
        blocks: Dict[int, BasicBlock] = {}
        for pos, start in enumerate(ordered_starts):
            idx = index_by_offset[start]
            if pos + 1 < len(ordered_starts):
                next_start = ordered_starts[pos + 1]
                slice_end = index_by_offset[next_start]
            else:
                slice_end = len(instructions)
            block_instructions = list(instructions[idx:slice_end])
            blocks[start] = BasicBlock(start=start, instructions=block_instructions)
        return blocks

    def _link_edges(self, segment: Segment, blocks: Dict[int, BasicBlock]) -> None:
        offsets = sorted(blocks)
        available_offsets = {
            instr.word.offset
            for block in blocks.values()
            for instr in block.instructions
        }
        block_by_offset = blocks
        for offset in offsets:
            block = block_by_offset[offset]
            if not block.instructions:
                continue
            last = block.instructions[-1]
            semantics = last.semantics
            key = last.label()
            hint = semantics.control_flow or self.knowledge.control_flow_hint(key)
            next_offset = last.word.offset + WORD_SIZE
            if hint == "return" or hint == "stop":
                continue
            if hint in {"jump", "branch"}:
                target = self._resolve_target(segment, last.word, available_offsets)
                if target is not None and target in block_by_offset:
                    block.successors.add(target)
                    block_by_offset[target].predecessors.add(block.start)
            if hint == "branch":
                if next_offset in block_by_offset:
                    block.successors.add(next_offset)
                    block_by_offset[next_offset].predecessors.add(block.start)
            elif hint == "call":
                target = self._resolve_target(segment, last.word, available_offsets)
                if target is not None and target in block_by_offset:
                    block.successors.add(target)
                    block_by_offset[target].predecessors.add(block.start)
                if next_offset in block_by_offset:
                    block.successors.add(next_offset)
                    block_by_offset[next_offset].predecessors.add(block.start)
            elif hint == "jump":
                # Pure jumps only keep the explicit target edge. A fallthrough
                # edge would misrepresent the actual flow but the basic block
                # for the next offset is still materialised so alternative
                # analyses can use it if necessary.
                pass
            else:
                if next_offset in block_by_offset:
                    block.successors.add(next_offset)
                    block_by_offset[next_offset].predecessors.add(block.start)

    def _resolve_target(
        self,
        segment: Segment,
        instr: InstructionWord,
        available_offsets: Set[int],
    ) -> Optional[int]:
        hint = self.knowledge.flow_target_hint(instr.label()) or "segment"
        operand = instr.operand
        candidates: List[int]
        if hint == "absolute":
            candidates = [operand]
        elif hint == "word":
            candidates = [segment.start + operand * WORD_SIZE]
        elif hint == "relative":
            signed = operand if operand < 0x8000 else operand - 0x10000
            base = instr.offset + WORD_SIZE
            candidates = [base + signed, base + signed * WORD_SIZE]
        else:  # "segment" or unknown
            base = instr.offset + WORD_SIZE
            signed16 = operand if operand < 0x8000 else operand - 0x10000
            signed8 = operand & 0xFF
            if signed8 >= 0x80:
                signed8 -= 0x100

            candidates = []
            for signed in (signed8, signed16):
                for scale in (1, WORD_SIZE):
                    target = base + signed * scale
                    if target not in candidates:
                        candidates.append(target)

            abs_candidates = [
                segment.start + operand,
                segment.start + operand * WORD_SIZE,
            ]
            for target in abs_candidates:
                if target not in candidates:
                    candidates.append(target)

        def is_within_segment(target: int) -> bool:
            return segment.start <= target < segment.end

        preferred = [
            target
            for target in candidates
            if is_within_segment(target)
            and (target - segment.start) % WORD_SIZE == 0
            and target in available_offsets
        ]
        if preferred:
            return preferred[0]

        fallback_existing = [
            target
            for target in candidates
            if is_within_segment(target) and target in available_offsets
        ]
        if fallback_existing:
            return fallback_existing[0]

        for target in candidates:
            if is_within_segment(target):
                return target
        return None


def render_cfgs(
    graphs: Iterable[ControlFlowGraph],
) -> str:
    """Render multiple CFGs into a single textual blob."""

    lines: List[str] = []
    for graph in graphs:
        lines.append(graph.to_text().rstrip())
    return "\n\n".join(lines) + "\n"
