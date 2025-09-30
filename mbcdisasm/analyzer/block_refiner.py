"""Post-processing helpers for pipeline blocks.

The :class:`~mbcdisasm.analyzer.pipeline.PipelineAnalyzer` focuses on recognising
short instruction windows and assigning a coarse category.  In practice the
initial classification may leave small marker-only blocks labelled as
``unknown`` or assign a low confidence score to literal loaders that are wrapped
in descriptive metadata.  The block refiner performs a secondary pass that
stabilises those classifications and merges contextual information exposed by the
heuristic engine.

The refiner is intentionally conservative – it never invents new instructions or
rewrites the stack deltas.  Instead it adjusts the block metadata (category,
confidence and notes) to better reflect the observed instruction mix.  The logic
encoded here mirrors the heuristics used during manual reversing sessions and is
heavily documented so that contributors can audit the decision making process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .instruction_profile import InstructionKind
from .report import PipelineBlock

__all__ = ["BlockSignature", "RefinementSettings", "RefinementSummary", "BlockRefiner"]


# ---------------------------------------------------------------------------
# Lightweight signature describing the instruction mix of a block
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockSignature:
    """Summary statistics for a :class:`PipelineBlock`.

    The signature tracks how many instructions fall into each high level kind
    which allows the refiner to reason about the dominant behaviour of a block.
    Consumers can treat the signature as an immutable value object and derive
    additional metrics such as literal or marker density.
    """

    literal: int
    markers: int
    pushes: int
    ascii_chunks: int
    reducers: int
    returns: int
    controls: int
    unknown: int
    total: int

    @classmethod
    def from_block(cls, block: PipelineBlock) -> "BlockSignature":
        literal = markers = pushes = ascii_chunks = reducers = returns = controls = unknown = 0
        for profile in block.profiles:
            kind = profile.kind
            if kind is InstructionKind.LITERAL:
                literal += 1
            elif kind is InstructionKind.MARKER:
                markers += 1
            elif kind is InstructionKind.PUSH:
                pushes += 1
            elif kind is InstructionKind.ASCII_CHUNK:
                ascii_chunks += 1
            elif kind is InstructionKind.REDUCE:
                reducers += 1
            elif kind in {InstructionKind.RETURN, InstructionKind.TERMINATOR}:
                returns += 1
            elif kind in {InstructionKind.BRANCH, InstructionKind.CONTROL}:
                controls += 1
            elif kind is InstructionKind.UNKNOWN:
                unknown += 1
        total = len(block.profiles)
        return cls(
            literal=literal,
            markers=markers,
            pushes=pushes,
            ascii_chunks=ascii_chunks,
            reducers=reducers,
            returns=returns,
            controls=controls,
            unknown=unknown,
            total=total,
        )

    # ------------------------------------------------------------------
    # convenience metrics
    # ------------------------------------------------------------------
    def literal_density(self) -> float:
        if not self.total:
            return 0.0
        return (self.literal + self.pushes + self.ascii_chunks) / self.total

    def marker_density(self) -> float:
        if not self.total:
            return 0.0
        return self.markers / self.total

    def unknown_ratio(self) -> float:
        if not self.total:
            return 0.0
        return self.unknown / self.total

    def return_ratio(self) -> float:
        if not self.total:
            return 0.0
        return self.returns / self.total

    def describe(self) -> str:
        return (
            f"literal={self.literal} markers={self.markers} pushes={self.pushes} "
            f"ascii={self.ascii_chunks} reducers={self.reducers} returns={self.returns} "
            f"unknown={self.unknown}/{self.total}"
        )


# ---------------------------------------------------------------------------
# Refinement settings
# ---------------------------------------------------------------------------

@dataclass
class RefinementSettings:
    """Configuration object controlling the :class:`BlockRefiner`."""

    literal_threshold: float = 0.55
    marker_threshold: float = 0.4
    return_threshold: float = 0.2
    min_confidence: float = 0.45
    marker_confidence: float = 0.6
    literal_confidence: float = 0.65
    return_confidence: float = 0.7


class BlockRefiner:
    """Adjust block metadata to improve classification stability."""

    def __init__(self, settings: RefinementSettings | None = None) -> None:
        self.settings = settings or RefinementSettings()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def refine(self, blocks: Sequence[PipelineBlock]) -> Tuple[PipelineBlock, ...]:
        refined: List[PipelineBlock] = []
        total = len(blocks)
        for index, block in enumerate(blocks):
            signature = BlockSignature.from_block(block)
            previous = refined[index - 1] if index > 0 else None
            following = blocks[index + 1] if index + 1 < total else None
            self._reclassify(block, signature)
            self._stabilise_confidence(block, signature)
            self._propagate_context(block, signature, previous, following)
            refined.append(block)
        return tuple(refined)

    def summarise(self, blocks: Sequence[PipelineBlock]) -> RefinementSummary:
        literal = return_blocks = control = unknown = 0
        for block in blocks:
            if block.category == "literal":
                literal += 1
            elif block.category == "return":
                return_blocks += 1
            elif block.category == "control":
                control += 1
            elif block.category == "unknown":
                unknown += 1
        return RefinementSummary(
            literal_blocks=literal,
            return_blocks=return_blocks,
            control_blocks=control,
            unknown_blocks=unknown,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _reclassify(self, block: PipelineBlock, signature: BlockSignature) -> None:
        """Assign a better suited category based on the block signature."""

        if signature.total == 0:
            return

        literal_density = signature.literal_density()
        marker_density = signature.marker_density()
        return_density = signature.return_ratio()

        if marker_density >= self.settings.marker_threshold and literal_density >= 0.3:
            block.category = "literal"
            block.add_note("refiner: marker-dense literal block")
        elif literal_density >= self.settings.literal_threshold:
            block.category = "literal"
            block.add_note("refiner: literal density threshold exceeded")
        elif return_density >= self.settings.return_threshold or signature.returns:
            block.category = "return"
            block.add_note("refiner: return signature detected")
        elif signature.controls and block.category == "unknown":
            block.category = "control"
            block.add_note("refiner: promoted to control due to neighbouring opcodes")

        if signature.markers and block.category == "unknown":
            block.category = "literal"
            block.add_note("refiner: marker-only block interpreted as literal metadata")

    def _stabilise_confidence(self, block: PipelineBlock, signature: BlockSignature) -> None:
        """Ensure that the confidence matches the refined category."""

        baseline = max(self.settings.min_confidence, block.confidence)
        if block.category == "literal":
            target = max(baseline, self.settings.literal_confidence)
            if signature.marker_density() >= self.settings.marker_threshold:
                target = max(target, self.settings.marker_confidence)
            block.confidence = target
        elif block.category == "return":
            target = max(baseline, self.settings.return_confidence)
            block.confidence = target
        elif block.category == "control" and signature.controls:
            block.confidence = max(baseline, 0.55)
        else:
            block.confidence = baseline

        block.add_note(f"refiner: signature {signature.describe()}")

    def _propagate_context(
        self,
        block: PipelineBlock,
        signature: BlockSignature,
        previous: PipelineBlock | None,
        following: PipelineBlock | None,
    ) -> None:
        """Adjust block metadata using neighbouring information."""

        if previous and previous.category == block.category:
            block.confidence = max(block.confidence, previous.confidence * 0.95)
            block.add_note("refiner: confidence aligned with previous block")
        if following and following.category == block.category:
            block.confidence = max(block.confidence, following.confidence * 0.95)
            block.add_note("refiner: confidence aligned with next block")

        if (
            previous
            and following
            and previous.category == following.category == "literal"
            and block.category == "unknown"
            and signature.marker_density() > 0
        ):
            block.category = "literal"
            block.confidence = max(block.confidence, self.settings.literal_confidence)
            block.add_note("refiner: interpolated literal block between literal neighbours")
@dataclass(frozen=True)
class RefinementSummary:
    """Aggregated view of the refined block categories."""

    literal_blocks: int
    return_blocks: int
    control_blocks: int
    unknown_blocks: int

    def describe(self) -> str:
        return (
            f"literal={self.literal_blocks} return={self.return_blocks} "
            f"control={self.control_blocks} unknown={self.unknown_blocks}"
        )

