"""Feature extraction helpers for pipeline blocks.

The :mod:`mbcdisasm.analyzer.pipeline` module is primarily concerned with
segmenting an instruction stream into coherent blocks.  Once a block has been
identified we still need to decide what it represents – literal loading, a call
helper prelude, a teardown sequence, and so on.  Historically this logic lived
in :func:`PipelineAnalyzer._classify_block` where it quickly became an unwieldy
collection of ad-hoc counters.  ``_char`` in particular exposed the weaknesses
of that approach: the script mixes hundreds of helper opcodes that are not yet
fully documented which meant that the majority of blocks defaulted to the
``UNKNOWN`` kind despite containing plenty of signal (tail calls, indirect table
fetches, literal trains).

This module factors the feature extraction into a dedicated helper class.  The
:class:`BlockFeatures` dataclass converts a sequence of
:class:`~mbcdisasm.analyzer.instruction_profile.InstructionProfile` objects into
a rich set of aggregate counters that capture the "shape" of the block.  The
pipeline can then make classification decisions based on these aggregates rather
than re-deriving the same information multiple times.  Centralising the logic
also makes it easier to reason about the heuristics: the class exposes
human-friendly accessors (``literal_ratio``, ``call_ratio`` and friends) and a
``summarise`` method that produces diagnostic strings suitable for attaching to
pipeline notes.

The feature extraction deliberately biases towards literal-heavy blocks.  This
mirrors the manual reverse engineering sessions that motivated the work on
``_char``: most of the script is concerned with building lookup tables and
resource descriptors, so we favour heuristics that can tell apart literal runs,
call scaffolding and stack teardown sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

from .instruction_profile import InstructionKind, InstructionProfile, summarise_profiles

# Kinds that contribute to the literal density of a block.  Table lookups are
# included because ``_char`` often loads an index from a literal immediately
# before issuing an indirect fetch.
_LITERAL_LIKE = {
    InstructionKind.LITERAL,
    InstructionKind.ASCII_CHUNK,
    InstructionKind.PUSH,
    InstructionKind.TABLE_LOOKUP,
}

# Kinds associated with control flow features.
_CALL_LIKE = {InstructionKind.CALL, InstructionKind.TAILCALL}
_RETURN_LIKE = {InstructionKind.RETURN, InstructionKind.TERMINATOR}
_BRANCH_LIKE = {InstructionKind.BRANCH, InstructionKind.CONTROL}
_COMPUTE_LIKE = {
    InstructionKind.ARITHMETIC,
    InstructionKind.BITWISE,
    InstructionKind.LOGICAL,
    InstructionKind.REDUCE,
}
_TEARDOWN_LIKE = {InstructionKind.STACK_TEARDOWN, InstructionKind.STACK_COPY}
_INDIRECT_LIKE = {InstructionKind.INDIRECT}


@dataclass(frozen=True)
class BlockFeatures:
    """Aggregate counters describing the instruction mix of a block."""

    total: int
    counts: Dict[InstructionKind, int]
    literal_like: int
    call_like: int
    return_like: int
    branch_like: int
    test_like: int
    compute_like: int
    teardown_like: int
    indirect_like: int
    meta_like: int
    unknown_like: int
    distinct_opcodes: int

    @classmethod
    def from_profiles(cls, profiles: Sequence[InstructionProfile]) -> "BlockFeatures":
        """Compute :class:`BlockFeatures` for ``profiles``."""

        counts = summarise_profiles(profiles)
        total = len(profiles)

        literal_like = sum(counts.get(kind, 0) for kind in _LITERAL_LIKE)
        call_like = sum(counts.get(kind, 0) for kind in _CALL_LIKE)
        return_like = sum(counts.get(kind, 0) for kind in _RETURN_LIKE)
        branch_like = sum(counts.get(kind, 0) for kind in _BRANCH_LIKE)
        test_like = counts.get(InstructionKind.TEST, 0)
        compute_like = sum(counts.get(kind, 0) for kind in _COMPUTE_LIKE)
        teardown_like = sum(counts.get(kind, 0) for kind in _TEARDOWN_LIKE)
        indirect_like = sum(counts.get(kind, 0) for kind in _INDIRECT_LIKE)
        meta_like = counts.get(InstructionKind.META, 0)
        unknown_like = counts.get(InstructionKind.UNKNOWN, 0)

        # ``_char`` frequently reuses the same helper opcode with different
        # modes.  Tracking the distinct opcode count gives the classifier a
        # cheap way to differentiate tight literal runs (usually one or two
        # opcodes repeated with varying operands) from busy helper sequences.
        distinct_opcodes = len({profile.word.opcode for profile in profiles})

        return cls(
            total=total,
            counts=dict(counts),
            literal_like=literal_like,
            call_like=call_like,
            return_like=return_like,
            branch_like=branch_like,
            test_like=test_like,
            compute_like=compute_like,
            teardown_like=teardown_like,
            indirect_like=indirect_like,
            meta_like=meta_like,
            unknown_like=unknown_like,
            distinct_opcodes=distinct_opcodes,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def ratio(self, value: int) -> float:
        if self.total == 0:
            return 0.0
        return value / self.total

    def literal_ratio(self) -> float:
        return self.ratio(self.literal_like)

    def call_ratio(self) -> float:
        return self.ratio(self.call_like)

    def return_ratio(self) -> float:
        return self.ratio(self.return_like)

    def branch_ratio(self) -> float:
        return self.ratio(self.branch_like + self.test_like)

    def compute_ratio(self) -> float:
        return self.ratio(self.compute_like)

    def indirect_ratio(self) -> float:
        return self.ratio(self.indirect_like)

    def meta_ratio(self) -> float:
        return self.ratio(self.meta_like)

    def unknown_ratio(self) -> float:
        return self.ratio(self.unknown_like)

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def summarise(self) -> Tuple[str, ...]:
        """Return human-readable summaries for diagnostic notes."""

        summary = [
            f"literal_like={self.literal_like}/{self.total}",
            f"call_like={self.call_like}/{self.total}",
            f"return_like={self.return_like}/{self.total}",
            f"branch_like={self.branch_like + self.test_like}/{self.total}",
            f"compute_like={self.compute_like}/{self.total}",
            f"indirect_like={self.indirect_like}/{self.total}",
            f"meta_like={self.meta_like}/{self.total}",
        ]
        if self.unknown_like:
            summary.append(f"unknown_like={self.unknown_like}/{self.total}")
        summary.append(f"distinct_opcodes={self.distinct_opcodes}")
        return tuple(summary)

    def emphasise(self, kinds: Iterable[InstructionKind]) -> int:
        """Return the combined count for ``kinds``."""

        return sum(self.counts.get(kind, 0) for kind in kinds)


__all__ = ["BlockFeatures"]
