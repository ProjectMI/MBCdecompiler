"""Instruction listing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .analyzer import PipelineAnalyzer
from .instruction import InstructionWord, read_instructions
from .knowledge import KnowledgeBase
from .mbc import MbcContainer, Segment


class Disassembler:
    """Render textual disassembly listings enriched with pipeline analysis."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        *,
        analyzer: Optional[PipelineAnalyzer] = None,
    ) -> None:
        self.knowledge = knowledge
        self.analyzer = analyzer or PipelineAnalyzer(knowledge)

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
        report = self.analyzer.analyse_segment(instructions) if self.analyzer else None

        header = (
            f"; segment {segment.index} offset=0x{segment.start:06X} length={segment.length}"
        )
        lines = [header]

        if report and report.statistics:
            stats = report.statistics
            dominant = stats.dominant_category() or "n/a"
            lines.append(
                "; pipeline stats: blocks={blocks} instructions={instr} stackΔ={delta:+d} dominant={dominant}".format(
                    blocks=stats.block_count,
                    instr=stats.instruction_count,
                    delta=stats.total_stack_delta,
                    dominant=dominant,
                )
            )
        if report and report.warnings:
            lines.append("; pipeline warnings:")
            for warning in report.warnings:
                lines.append(f";   - {warning}")

        block_map = self._index_blocks(report.blocks) if report else {}

        for idx, instruction in enumerate(instructions):
            block_lines = block_map.get(instruction.offset)
            if block_lines:
                lines.extend(block_lines)
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

    def _index_blocks(
        self, blocks: Sequence["PipelineBlock"]
    ) -> Dict[int, List[str]]:
        from .analyzer.report import PipelineBlock  # local import to avoid cycle

        indexed: Dict[int, List[str]] = {}
        for idx, block in enumerate(blocks):
            if not block.profiles:
                continue
            start_offset = block.profiles[0].word.offset
            indexed[start_offset] = self._format_block(idx, block)
        return indexed

    def _format_block(self, index: int, block: "PipelineBlock") -> List[str]:
        from .analyzer.report import PipelineBlock  # local import to avoid cycle

        pattern = block.pattern.pattern.name if block.pattern else "?"
        header = (
            f"; pipeline block {index + 1}: [{block.start_offset:08X}-{block.end_offset:08X}] "
            f"kind={block.kind.name} category={block.category} {block.stack.describe()} "
            f"len={len(block.profiles)} pattern={pattern} conf={block.confidence:.2f}"
        )
        lines = [header]
        note_lines = self._format_block_notes(block.notes)
        lines.extend(note_lines)
        return lines

    @staticmethod
    def _format_block_notes(notes: Iterable[str]) -> List[str]:
        formatted: List[str] = []
        for note in notes:
            message = note.strip()
            if not message:
                continue
            formatted.append(f";   note: {message}")
        return formatted

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
