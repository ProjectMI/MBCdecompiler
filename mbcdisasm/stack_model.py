"""Stack delta modelling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from .instruction import InstructionWord


@dataclass(frozen=True)
class StackDeltaEstimate:
    """Describes the inferred stack effect for a single instruction."""

    key: str
    delta: Optional[float]
    confidence: float
    source: str = "unknown"


class StackDeltaModeler:
    """Derive stack deltas using lightweight modelling heuristics."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.6,
        max_delta: float = 6.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_delta = max_delta
        self._known: Dict[str, float] = {}
        self._samples: Dict[str, List[float]] = {}
        self._locked: Set[str] = set()

    def model_segment(
        self,
        instructions: Sequence[InstructionWord],
        *,
        initial_stack: float = 0.0,
    ) -> List[StackDeltaEstimate]:
        """Return stack delta estimates for ``instructions``.

        The model first aggregates previously learned deltas.  If a segment
        contains exactly one opcode/mode pair without a known delta, we deduce
        its effect by enforcing that the overall stack change equals zero.  All
        remaining instructions are analysed using a bit-count heuristic that
        interprets mode nibbles as push/pop masks.
        """

        known_total = 0.0
        unknown_counts: Dict[str, int] = {}
        provisional: List[StackDeltaEstimate] = []

        for instr in instructions:
            key = instr.label()
            known = self._known.get(key)
            if known is not None:
                estimate = StackDeltaEstimate(
                    key=key, delta=known, confidence=0.95, source="derived"
                )
                provisional.append(estimate)
                known_total += known
                continue

            estimate = self._heuristic_estimate(instr)
            provisional.append(estimate)
            if self.is_confident(estimate) and estimate.delta is not None:
                known_total += float(estimate.delta)
            else:
                unknown_counts[key] = unknown_counts.get(key, 0) + 1

        if unknown_counts:
            unsolved = [key for key in unknown_counts if key not in self._known]
            if len(unsolved) == 1 and abs(known_total) > 1e-9:
                key = unsolved[0]
                count = unknown_counts[key]
                if count:
                    inferred = (-known_total) / count
                    if self._is_reasonable(inferred):
                        self._record_known(key, inferred)
                        provisional = [
                            StackDeltaEstimate(
                                key=item.key,
                                delta=inferred if item.key == key else item.delta,
                                confidence=0.95 if item.key == key else item.confidence,
                                source="derived" if item.key == key else item.source,
                            )
                            for item in provisional
                        ]

        estimates: List[StackDeltaEstimate] = []
        for estimate in provisional:
            if estimate.source != "derived" and self.is_confident(estimate):
                # Promote high-confidence heuristic results for future segments.
                self._record_known(estimate.key, float(estimate.delta))
            estimates.append(estimate)

        return estimates

    def estimate_instruction(self, instr: InstructionWord) -> StackDeltaEstimate:
        """Return a stack delta estimate for ``instr`` without updating state."""

        key = instr.label()
        known = self._known.get(key)
        if known is not None:
            return StackDeltaEstimate(key=key, delta=known, confidence=0.95, source="derived")
        return self._heuristic_estimate(instr)

    def known_delta(self, key: str) -> Optional[float]:
        """Return the learned stack delta for ``key`` if available."""

        return self._known.get(key)

    def seed_known_delta(self, key: str, delta: float) -> None:
        """Prime the model with a trusted stack delta for ``key``."""

        value = float(delta)
        self._known[key] = value
        self._samples[key] = [value]
        self._locked.add(key)

    def is_confident(self, estimate: StackDeltaEstimate) -> bool:
        """Return :data:`True` if ``estimate`` should be treated as reliable."""

        return estimate.delta is not None and estimate.confidence >= self.confidence_threshold

    def _record_known(self, key: str, delta: float) -> None:
        if not self._is_reasonable(delta):
            return
        if key in self._locked:
            known = self._known.get(key)
            if known is None or abs(known - delta) > 1e-6:
                # Preserve the seeded value when heuristics disagree.  Allow
                # follow-up samples that reaffirm the seed to accumulate.
                return
            # Accept additional samples that match the trusted value closely.
            delta = known
        samples = self._samples.setdefault(key, [])
        samples.append(delta)
        average = sum(samples) / len(samples)
        self._known[key] = average

    def _is_reasonable(self, delta: float) -> bool:
        return abs(delta) <= self.max_delta

    def _heuristic_estimate(self, instr: InstructionWord) -> StackDeltaEstimate:
        mode = instr.mode
        push_mask = mode & 0x0F
        pop_mask = (mode >> 4) & 0x0F
        pushes = self._bitcount(push_mask)
        pops = self._bitcount(pop_mask)
        if pushes or pops:
            delta = float(pushes - pops)
            return StackDeltaEstimate(
                key=instr.label(),
                delta=delta,
                confidence=0.65,
                source="mode-bitcount",
            )
        if instr.operand == 0:
            # Neutral operations still expose an estimate but with low confidence
            return StackDeltaEstimate(
                key=instr.label(),
                delta=0.0,
                confidence=0.3,
                source="neutral",
            )
        return StackDeltaEstimate(key=instr.label(), delta=None, confidence=0.0, source="unknown")

    @staticmethod
    def _bitcount(value: int) -> int:
        return bin(value & 0xF).count("1")

    def clone(self) -> "StackDeltaModeler":
        """Return a deep copy of the current model state."""

        clone = StackDeltaModeler(
            confidence_threshold=self.confidence_threshold,
            max_delta=self.max_delta,
        )
        clone._known = dict(self._known)
        clone._samples = {key: list(values) for key, values in self._samples.items()}
        clone._locked = set(self._locked)
        return clone
