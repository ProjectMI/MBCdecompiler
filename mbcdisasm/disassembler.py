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
        metadata = self.knowledge.instruction_metadata(key)

        stack_value = metadata.stack_delta
        stack_confidence = metadata.stack_confidence
        stack_origin = metadata.stack_source

        stack_detail = "stackΔ="
        if stack_value is not None:
            stack_detail += self._format_stack_delta(stack_value)
            extras = []
            if stack_origin:
                extras.append(stack_origin)
            if stack_confidence is not None:
                extras.append(f"{stack_confidence * 100:.0f}%")
            if metadata.stack_samples:
                extras.append(f"n={metadata.stack_samples}")
            if extras:
                stack_detail += f" ({', '.join(extras)})"
        else:
            stack_detail += "unknown"

        role = metadata.control_flow or "unknown"
        if metadata.control_flow and metadata.flow_target:
            role += f"→{metadata.flow_target}"
        role_detail = f"role={role}"

        if metadata.operand_hint:
            operand = metadata.operand_hint
            if metadata.operand_confidence is not None:
                operand += f" ({metadata.operand_confidence * 100:.0f}%)"
            operand_detail = f"operand≈{operand}"
        else:
            operand_detail = "operand≈unknown"

        details = [stack_detail, role_detail, operand_detail]
        if metadata.summary:
            details.append(metadata.summary)

        comment = " | ".join(details)

        return (
            f"{instruction.offset:08X}: {instruction.raw:08X}    "
            f"{metadata.mnemonic:<24} ; mode={instruction.mode:02X} "
            f"operand=0x{instruction.operand:04X} | {comment}"
        )

    @staticmethod
    def _format_stack_delta(delta: float) -> str:
        value = float(delta)
        if value.is_integer():
            return f"{value:+.0f}"
        return f"{value:+.1f}"

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
