"""High level entry points for the IR normaliser."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from ..instruction import InstructionWord
from ..knowledge import KnowledgeBase
from .model import IRProgram, NormalizerMetrics
from .normalizer import BlockNormaliser
from .raw import RawSegment, parse_raw_segment


def build_segment_ir(
    instructions: Sequence[InstructionWord], knowledge: KnowledgeBase
) -> IRProgram:
    """Normalise a segment worth of instructions into IR blocks."""

    raw = parse_raw_segment(instructions, knowledge)
    blocks = []
    metrics = NormalizerMetrics()
    for block in raw.blocks:
        normaliser = BlockNormaliser(block)
        normalised = normaliser.run()
        metrics.merge(normaliser.metrics)
        blocks.append(normalised)
    return IRProgram(blocks=blocks, metrics=metrics)


def render_program(program: IRProgram) -> str:
    """Render ``program`` into a human readable string."""

    return "\n".join(program.render()) + "\n"
