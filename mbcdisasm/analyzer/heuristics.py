"""Auxiliary heuristics used to refine pipeline classification.

The heuristics implemented here mirror the multi-layered approach described in
the project notes:

1.  Strict stack invariants – every block is evaluated with a running stack
    counter that exposes the cumulative delta and the minimum depth touched by
    the instructions.
2.  Local opcode templates – lightweight pattern detectors that flag literal
    trains, inline ASCII chunk reducers and call frame initialisers.
3.  Control context – ties the block to the surrounding control flow
    instructions and provides hints on whether we are looking at operand
    preparation, indirect loads or teardown sequences.
4.  Indirect detection – identifies table lookups and other two-stage operand
    fetches by examining opcode combinations and stack deltas.
5.  Gap analysis – computes how much of the linear instruction stream is covered
    by recognised pipelines and tags suspicious leftovers for manual review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from .instruction_profile import InstructionKind, InstructionProfile, filter_profiles
from .stack import StackSummary


@dataclass(frozen=True)
class LocalFeature:
    """Single heuristic feature produced during analysis."""

    name: str
    score: float
    evidence: Tuple[str, ...] = tuple()

    def describe(self) -> str:
        parts = [f"{self.name}={self.score:.2f}"]
        if self.evidence:
            parts.append("(" + ", ".join(self.evidence) + ")")
        return " ".join(parts)


@dataclass(frozen=True)
class HeuristicReport:
    """Collection of heuristic features."""

    features: Tuple[LocalFeature, ...]
    stack_delta: int
    stack_range: Tuple[int, int]
    confidence: float

    def feature_map(self) -> Mapping[str, LocalFeature]:
        return {feature.name: feature for feature in self.features}

    def describe(self) -> str:
        chunks = [feature.describe() for feature in self.features]
        chunks.append(f"stackΔ={self.stack_delta:+d}")
        chunks.append(f"range=({self.stack_range[0]:+d},{self.stack_range[1]:+d})")
        chunks.append(f"conf={self.confidence:.2f}")
        return " ".join(chunks)


@dataclass
class HeuristicSettings:
    """Configuration options for the heuristic engine."""

    literal_weight: float = 0.2
    push_weight: float = 0.15
    test_weight: float = 0.1
    call_weight: float = 0.3
    return_weight: float = 0.25
    indirect_weight: float = 0.2
    max_literal_gap: int = 2
    max_call_gap: int = 3


class HeuristicEngine:
    """Generate heuristic reports for instruction blocks."""

    def __init__(self, settings: Optional[HeuristicSettings] = None) -> None:
        self.settings = settings or HeuristicSettings()

    def analyse(
        self,
        profiles: Sequence[InstructionProfile],
        stack: StackSummary,
        *,
        previous: Optional[InstructionProfile] = None,
        following: Optional[InstructionProfile] = None,
    ) -> HeuristicReport:
        features: List[LocalFeature] = []

        features.extend(self._stack_features(stack))
        features.extend(self._literal_features(profiles))
        features.extend(self._call_features(profiles, following))
        features.extend(self._return_features(profiles, following))
        features.extend(self._indirect_features(profiles))
        features.extend(self._context_features(previous, following))

        confidence = sum(feature.score for feature in features)
        confidence = max(0.0, min(1.0, confidence))

        return HeuristicReport(
            features=tuple(features),
            stack_delta=stack.change,
            stack_range=(stack.minimum, stack.maximum),
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # individual feature extractors
    # ------------------------------------------------------------------
    def _stack_features(self, stack: StackSummary) -> List[LocalFeature]:
        features: List[LocalFeature] = []
        features.append(LocalFeature(name="stack_delta", score=self._score_stack_delta(stack)))
        if stack.uncertain:
            features.append(LocalFeature(name="stack_uncertain", score=-0.2))
        spread = stack.maximum - stack.minimum
        if spread <= 1:
            features.append(LocalFeature(name="stack_stable", score=0.1, evidence=("spread<=1",)))
        else:
            features.append(LocalFeature(name="stack_spread", score=-0.05, evidence=(f"spread={spread}",)))
        return features

    def _score_stack_delta(self, stack: StackSummary) -> float:
        if stack.change == 0:
            return 0.05
        if stack.change > 0:
            return 0.1
        return 0.0

    def _literal_features(self, profiles: Sequence[InstructionProfile]) -> List[LocalFeature]:
        literal_like = filter_profiles(
            profiles,
            {
                InstructionKind.LITERAL,
                InstructionKind.ASCII_CHUNK,
                InstructionKind.PUSH,
                InstructionKind.TABLE_LOOKUP,
            },
        )
        features: List[LocalFeature] = []
        if literal_like:
            weight = self.settings.literal_weight * len(literal_like)
            evidence = tuple(profile.label for profile in literal_like[:3])
            features.append(LocalFeature(name="literal_chain", score=weight, evidence=evidence))
        gaps = self._literal_gaps(profiles)
        if gaps:
            for gap in gaps:
                features.append(LocalFeature(name="literal_gap", score=-0.05, evidence=(str(gap),)))
        return features

    def _literal_gaps(self, profiles: Sequence[InstructionProfile]) -> List[int]:
        gaps: List[int] = []
        run = 0
        for profile in profiles:
            if profile.kind in {InstructionKind.LITERAL, InstructionKind.ASCII_CHUNK, InstructionKind.PUSH}:
                run += 1
                continue
            if run:
                if run <= self.settings.max_literal_gap:
                    gaps.append(run)
                run = 0
        if run and run <= self.settings.max_literal_gap:
            gaps.append(run)
        return gaps

    def _call_features(
        self,
        profiles: Sequence[InstructionProfile],
        following: Optional[InstructionProfile],
    ) -> List[LocalFeature]:
        features: List[LocalFeature] = []
        call_like = filter_profiles(
            profiles,
            {
                InstructionKind.CALL,
                InstructionKind.TAILCALL,
                InstructionKind.META,
            },
        )
        if call_like:
            evidence = tuple(profile.label for profile in call_like)
            features.append(
                LocalFeature(
                    name="call_helper",
                    score=self.settings.call_weight * len(call_like),
                    evidence=evidence,
                )
            )
            if following and following.kind is InstructionKind.BRANCH:
                features.append(LocalFeature(name="call_followed_by_branch", score=0.05))
        return features

    def _return_features(
        self,
        profiles: Sequence[InstructionProfile],
        following: Optional[InstructionProfile],
    ) -> List[LocalFeature]:
        features: List[LocalFeature] = []
        teardown = filter_profiles(profiles, {InstructionKind.STACK_TEARDOWN})
        returns = filter_profiles(profiles, {InstructionKind.RETURN, InstructionKind.TERMINATOR})
        if teardown:
            features.append(
                LocalFeature(
                    name="stack_teardown",
                    score=self.settings.return_weight * len(teardown),
                    evidence=tuple(profile.label for profile in teardown),
                )
            )
        if returns:
            features.append(
                LocalFeature(
                    name="return_sequence",
                    score=self.settings.return_weight * len(returns),
                    evidence=tuple(profile.label for profile in returns),
                )
            )
        if following and following.kind is InstructionKind.TERMINATOR:
            features.append(LocalFeature(name="terminator_followup", score=0.05))
        return features

    def _indirect_features(self, profiles: Sequence[InstructionProfile]) -> List[LocalFeature]:
        features: List[LocalFeature] = []
        base = filter_profiles(profiles, {InstructionKind.PUSH, InstructionKind.LITERAL})
        lookup = filter_profiles(profiles, {InstructionKind.INDIRECT, InstructionKind.TABLE_LOOKUP})
        if base and lookup:
            features.append(
                LocalFeature(
                    name="indirect_pattern",
                    score=self.settings.indirect_weight,
                    evidence=(base[0].label, lookup[-1].label),
                )
            )
        return features

    def _context_features(
        self,
        previous: Optional[InstructionProfile],
        following: Optional[InstructionProfile],
    ) -> List[LocalFeature]:
        features: List[LocalFeature] = []
        if previous and previous.is_control():
            features.append(LocalFeature(name="pre_control", score=0.05, evidence=(previous.label,)))
        if following and following.is_control():
            features.append(LocalFeature(name="post_control", score=0.05, evidence=(following.label,)))
        return features
