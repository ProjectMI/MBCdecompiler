"""Instruction listing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .instruction import InstructionWord, read_instructions
from .knowledge import KnowledgeBase
from .mbc import MbcContainer, Segment


class Disassembler:
    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge

    def generate_listing(
        self,
        container: MbcContainer,
        *,
        segment_indices: Optional[Sequence[int]] = None,
        max_instructions: Optional[int] = None,
    ) -> str:
        selection = tuple(segment_indices or [])
        lines: List[str] = []

        if selection:
            segment_map = {segment.index: segment for segment in container.segments()}
            for index in selection:
                segment = segment_map.get(index)
                if segment is None:
                    continue
                lines.extend(self._render_segment(segment, max_instructions))
        else:
            for segment in container.segments():
                lines.extend(self._render_segment(segment, max_instructions))
        return "\n".join(lines) + "\n"

    def _render_segment(self, segment: Segment, max_instructions: Optional[int]) -> List[str]:
        instructions, remainder = read_instructions(segment.data, segment.start)

        header = (
            f"; segment {segment.index} offset=0x{segment.start:06X} length={segment.length}"
        )
        lines = [header]
        for idx, instruction in enumerate(instructions):
            if max_instructions is not None and idx >= max_instructions:
                lines.append("; ... truncated ...")
                break
            lines.append(self._format_instruction(instruction))
        if remainder:
            lines.append(f"; trailing {remainder} byte(s) ignored")
        lines.append("")
        return lines

    def _format_instruction(self, instruction: InstructionWord) -> str:
        key = instruction.label()
        info = self.knowledge.lookup(key)

        mnemonic = info.mnemonic if info else f"op_{instruction.opcode:02X}_{instruction.mode:02X}"
        summary = f" ; {info.summary}" if info and info.summary else ""

        return (
            f"{instruction.offset:08X}: {instruction.raw:08X}    "
            f"{mnemonic:<24} op={instruction.opcode:02X} "
            f"mode={instruction.mode:02X} operand=0x{instruction.operand:04X}{summary}"
        )

    def write_listing(
        self,
        container: MbcContainer,
        output_path: Path,
        *,
        segment_indices: Optional[Sequence[int]] = None,
        max_instructions: Optional[int] = None,
    ) -> None:
        listing = self.generate_listing(
            container,
            segment_indices=segment_indices,
            max_instructions=max_instructions,
        )
        output_path.write_text(listing, "utf-8")
