"""Normalize pipeline blocks into macro-level operations.

The pipeline analyser already recognises clusters of instructions that behave
like cohesive execution units – literal trains, call preparation helpers,
tailcall dispatchers and so on.  Downstream tooling, however, still has to sift
through the low-level instruction mix when building intermediate code.  The
``MacroNormalizer`` implemented in this module bridges that gap by collapsing
well-known patterns into *macro operations*.  These macros expose the intent of
the block ("tail dispatch", "build table", "predicate assignment") while hiding
the instruction bookkeeping that previously polluted corpus statistics.

The normaliser is intentionally conservative: only patterns that are stable
across all inspected `.mbc` corpora are promoted to macros.  Everything else is
folded into a generic ``raw_block`` descriptor so that callers can still build a
complete representation without second-guessing the heuristics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

from .instruction_profile import InstructionKind, InstructionProfile
from .report import PipelineBlock, PipelineReport


LiteralKinds = {
    InstructionKind.LITERAL,
    InstructionKind.ASCII_CHUNK,
    InstructionKind.PUSH,
    InstructionKind.TABLE_LOOKUP,
}


@dataclass(frozen=True)
class NormalizedOp:
    """Single macro-level operation produced by the normaliser."""

    macro: str
    operands: Tuple[str, ...] = tuple()
    sources: Tuple[str, ...] = tuple()
    notes: Tuple[str, ...] = tuple()


@dataclass(frozen=True)
class NormalizedBlock:
    """Container tying a :class:`PipelineBlock` to its macro operations."""

    block: PipelineBlock
    operations: Tuple[NormalizedOp, ...]


@dataclass(frozen=True)
class IntermediateSummary:
    """Aggregated statistics over normalised blocks."""

    operation_counts: Mapping[str, int]
    block_count: int

    def describe(self) -> str:
        parts = [f"blocks={self.block_count}"]
        for name, count in sorted(self.operation_counts.items()):
            parts.append(f"{name}:{count}")
        return " ".join(parts)


class MacroNormalizer:
    """Collapse low-level instruction streams into macro operations."""

    def normalize_block(self, block: PipelineBlock) -> NormalizedBlock:
        """Return macro operations describing ``block``."""

        operations: list[NormalizedOp] = []

        tail_macro = self._detect_tail_dispatch(block)
        if tail_macro is not None:
            return NormalizedBlock(block=block, operations=(tail_macro,))

        return_macro = self._detect_return(block)
        if return_macro is not None:
            return NormalizedBlock(block=block, operations=(return_macro,))

        literal_macro = self._detect_literal_build(block)
        if literal_macro is not None:
            operations.append(literal_macro)

        predicate_macro = self._detect_predicate(block)
        if predicate_macro is not None:
            operations.append(predicate_macro)

        operations.extend(self._detect_indirect_access(block))

        if not operations:
            operations.append(self._fallback_macro(block))

        return NormalizedBlock(block=block, operations=tuple(operations))

    def normalize_report(
        self, report: PipelineReport | Sequence[PipelineBlock]
    ) -> Tuple[NormalizedBlock, ...]:
        """Normalise all blocks contained in ``report``."""

        if isinstance(report, PipelineReport):
            blocks = report.blocks
        else:
            blocks = tuple(report)
        return tuple(self.normalize_block(block) for block in blocks)

    def summarise(self, blocks: Iterable[NormalizedBlock]) -> IntermediateSummary:
        """Return an :class:`IntermediateSummary` for ``blocks``."""

        counter: Counter[str] = Counter()
        count = 0
        for normalized in blocks:
            count += 1
            for operation in normalized.operations:
                counter[operation.macro] += 1
        return IntermediateSummary(operation_counts=dict(counter), block_count=count)

    def evaluate(self, report: PipelineReport) -> IntermediateSummary:
        """Convenience wrapper combining :meth:`normalize_report` and :meth:`summarise`."""

        normalized = self.normalize_report(report)
        return self.summarise(normalized)

    # ------------------------------------------------------------------
    # detection helpers
    # ------------------------------------------------------------------
    def _detect_tail_dispatch(self, block: PipelineBlock) -> NormalizedOp | None:
        labels = [profile.label for profile in block.profiles if self._is_tail(profile)]
        if not labels:
            return None

        operands = []
        if block.stack.change:
            operands.append(f"stackΔ={block.stack.change:+d}")
        notes = (block.category,) if block.category else tuple()
        return NormalizedOp(
            macro="tail_dispatch",
            operands=tuple(operands),
            sources=tuple(labels),
            notes=notes,
        )

    def _detect_return(self, block: PipelineBlock) -> NormalizedOp | None:
        labels = [
            profile.label
            for profile in block.profiles
            if profile.kind in {InstructionKind.RETURN, InstructionKind.TERMINATOR}
        ]
        if not labels:
            return None

        operands = []
        if block.stack.change:
            operands.append(f"stackΔ={block.stack.change:+d}")
        notes = (block.category,) if block.category else tuple()
        return NormalizedOp(
            macro="frame_return",
            operands=tuple(operands),
            sources=tuple(labels),
            notes=notes,
        )

    def _detect_literal_build(self, block: PipelineBlock) -> NormalizedOp | None:
        literal_profiles: list[InstructionProfile] = []
        reduce_labels: list[str] = []

        index = 0
        for profile in block.profiles:
            if profile.kind in LiteralKinds:
                literal_profiles.append(profile)
                index += 1
                continue
            break

        if len(literal_profiles) < 2:
            return None

        reduce_count = 0
        for profile in block.profiles[index:]:
            if profile.kind is InstructionKind.REDUCE:
                reduce_count += 1
                reduce_labels.append(profile.label)
            else:
                break

        if reduce_count == 0:
            return None

        has_push = any(profile.kind is InstructionKind.PUSH for profile in literal_profiles)
        literal_labels = [profile.label for profile in literal_profiles]

        if has_push:
            macro = "table_build"
        elif len(literal_profiles) == 2 and reduce_count == 1:
            macro = "tuple_build"
        else:
            macro = "array_build"

        operands = (
            f"width={len(literal_profiles)}",
            f"reduces={reduce_count}",
        )
        return NormalizedOp(
            macro=macro,
            operands=operands,
            sources=tuple(literal_labels + reduce_labels),
        )

    def _detect_predicate(self, block: PipelineBlock) -> NormalizedOp | None:
        test_indices = [idx for idx, profile in enumerate(block.profiles) if profile.kind is InstructionKind.TEST]
        if not test_indices:
            return None

        sources = [block.profiles[idx].label for idx in test_indices]
        operands: list[str] = [f"tests={len(test_indices)}"]

        first_test = test_indices[0]
        if first_test > 0:
            target = block.profiles[first_test - 1]
            if target.kind is InstructionKind.PUSH:
                operands.append(f"target={target.label}")

        return NormalizedOp(
            macro="predicate_assign",
            operands=tuple(operands),
            sources=tuple(sources),
        )

    def _detect_indirect_access(self, block: PipelineBlock) -> list[NormalizedOp]:
        operations: list[NormalizedOp] = []
        for profile in block.profiles:
            if not self._is_indirect(profile):
                continue
            zone = "frame" if profile.mode < 0x20 else "global"
            macro = f"{zone}_access"
            operands = (f"slot=0x{profile.mode:02X}",)
            operations.append(
                NormalizedOp(
                    macro=macro,
                    operands=operands,
                    sources=(profile.label,),
                )
            )
        return operations

    def _fallback_macro(self, block: PipelineBlock) -> NormalizedOp:
        operands = (
            f"category={block.category}",
            f"instr={len(block.profiles)}",
        )
        labels = tuple(profile.label for profile in block.profiles)
        return NormalizedOp(macro="raw_block", operands=operands, sources=labels)

    @staticmethod
    def _is_tail(profile: InstructionProfile) -> bool:
        if profile.kind is InstructionKind.TAILCALL:
            return True
        return profile.label.startswith("29:")

    @staticmethod
    def _is_indirect(profile: InstructionProfile) -> bool:
        if profile.kind in {
            InstructionKind.INDIRECT,
            InstructionKind.INDIRECT_LOAD,
            InstructionKind.INDIRECT_STORE,
        }:
            return True
        return profile.label.startswith("69:")


__all__ = [
    "MacroNormalizer",
    "NormalizedBlock",
    "NormalizedOp",
    "IntermediateSummary",
]
