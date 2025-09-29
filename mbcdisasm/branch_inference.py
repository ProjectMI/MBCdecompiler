"""Heuristics that infer branch semantics for opcode/mode pairs.

The historical toolchain relied entirely on manually curated annotations to
identify control-flow instructions.  While this worked for the handful of
archaeologically studied opcodes, it prevented new dumps from recovering a
meaningful control-flow graph until the knowledge base had been massaged by
hand.  This module introduces a lightweight inference engine that inspects the
available metadata – mnemonic names, stack annotations, operand hints and
observed usage statistics – to recognise conditional branches, unconditional
jumps and fallthrough operations without requiring explicit manual entries.

The heuristics are intentionally conservative.  They prioritise avoiding false
positives that would fragment the CFG over exhaustively detecting every exotic
branching construct.  The engine exposes detailed reasoning metadata so callers
can surface the applied heuristics to operators when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from .knowledge import KnowledgeBase, OpcodeProfile


@dataclass(frozen=True)
class BranchInference:
    """Structured description of inferred control-flow semantics."""

    control_flow: str
    flow_target: Optional[str]
    confidence: float
    reasons: Tuple[str, ...]


class BranchInferenceEngine:
    """Infer control-flow semantics for opcode/mode combinations."""

    _KEYWORDS = {
        "branch": "branch",
        "jump": "jump",
        "goto": "jump",
        "jumps": "jump",
        "loop": "branch",
        "test": "branch",
        "cond": "branch",
        "tf": "branch",
    }

    _RELATIVE_OPERAND_HINTS = {
        "relative",
        "relative_word",
        "relative_word:tiny",
        "word:relative",
        "word:relative:tiny",
        "segment_relative",
    }

    _ABSOLUTE_OPERAND_HINTS = {
        "absolute",
        "absolute_word",
        "segment",
        "segment_word",
        "word",
    }

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge
        self._profile_cache: dict[str, Optional[OpcodeProfile]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(
        self,
        key: str,
        *,
        mnemonic: str,
        manual_name: Optional[str],
        summary: Optional[str],
        tags: Iterable[str],
        operand_hint: Optional[str],
        stack_inputs: Optional[int],
        stack_outputs: Optional[int],
    ) -> Optional[BranchInference]:
        """Return inferred control-flow semantics for ``key`` if possible."""

        lowered_tags = {tag.lower() for tag in tags}
        reasons: List[str] = []

        direct = self._keyword_match(mnemonic, manual_name, summary)
        if direct:
            reasons.extend(direct)

        if "branch" in lowered_tags:
            reasons.append("tag:branch")
        if "jump" in lowered_tags:
            reasons.append("tag:jump")

        profile = self._profile_for(key)
        operand_hint = (operand_hint or "").lower() or None

        if operand_hint in self._RELATIVE_OPERAND_HINTS:
            reasons.append(f"operand:{operand_hint}")
        elif operand_hint in self._ABSOLUTE_OPERAND_HINTS:
            reasons.append(f"operand:{operand_hint}")

        if profile is not None:
            profile_reason = self._inspect_profile(profile)
            if profile_reason:
                reasons.append(profile_reason)

        # Stack heuristics: conditional branches pop at least one value and
        # typically do not produce outputs.
        if stack_inputs is not None and stack_inputs > 0 and (stack_outputs or 0) == 0:
            reasons.append(f"stack:{stack_inputs}in")

        if not reasons:
            return None

        # Confidence calculation emphasises direct textual matches and
        # operand hints, with stack/profile cues acting as supporting signals.
        score = 0.0
        for reason in reasons:
            if reason.startswith("keyword:") or reason.startswith("tag:"):
                score += 0.4
            elif reason.startswith("operand:"):
                score += 0.25
            elif reason.startswith("profile:"):
                score += 0.2
            elif reason.startswith("stack:"):
                score += 0.15
            else:
                score += 0.1

        confidence = min(score, 0.95)

        control_flow = "branch"
        if any(reason.startswith("keyword:jump") for reason in reasons) or any(
            reason.startswith("tag:jump") for reason in reasons
        ):
            control_flow = "jump"

        flow_target = None
        if operand_hint in self._RELATIVE_OPERAND_HINTS:
            flow_target = "relative"
        elif operand_hint in self._ABSOLUTE_OPERAND_HINTS:
            flow_target = "absolute"

        return BranchInference(
            control_flow=control_flow,
            flow_target=flow_target,
            confidence=confidence,
            reasons=tuple(sorted(set(reasons))),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _profile_for(self, key: str) -> Optional[OpcodeProfile]:
        cached = self._profile_cache.get(key)
        if cached is not None or key in self._profile_cache:
            return cached
        try:
            profile = self.knowledge.get_profile(key)
        except Exception:  # pragma: no cover - defensive guard
            profile = None
        self._profile_cache[key] = profile
        return profile

    def _keyword_match(
        self,
        mnemonic: str,
        manual_name: Optional[str],
        summary: Optional[str],
    ) -> List[str]:
        text_sources: List[Tuple[str, Optional[str]]] = [
            ("mnemonic", mnemonic),
            ("name", manual_name),
            ("summary", summary),
        ]
        reasons: List[str] = []
        for label, value in text_sources:
            if not value:
                continue
            lowered = value.lower()
            for keyword, category in self._KEYWORDS.items():
                if keyword in lowered:
                    reasons.append(f"keyword:{category}:{label}")
        return reasons

    def _inspect_profile(self, profile: OpcodeProfile) -> Optional[str]:
        operand_types = profile.operand_types
        if not operand_types:
            return None
        total_operands = sum(operand_types.values())
        if total_operands == 0:
            return None
        rel_hits = sum(
            count
            for hint, count in operand_types.items()
            if hint.lower() in self._RELATIVE_OPERAND_HINTS
        )
        abs_hits = sum(
            count
            for hint, count in operand_types.items()
            if hint.lower() in self._ABSOLUTE_OPERAND_HINTS
        )
        if rel_hits / total_operands >= 0.4:
            return f"profile:relative:{rel_hits}/{total_operands}"
        if abs_hits / total_operands >= 0.4:
            return f"profile:absolute:{abs_hits}/{total_operands}"
        return None


def merge_inference(
    base: Optional[BranchInference], inferred: Optional[BranchInference]
) -> Optional[BranchInference]:
    """Combine two branch inference results preferring explicit data."""

    if base is None:
        return inferred
    if inferred is None:
        return base
    if base.confidence >= inferred.confidence:
        return base
    reasons = tuple(sorted(set(base.reasons + inferred.reasons)))
    flow_target = inferred.flow_target or base.flow_target
    control_flow = inferred.control_flow or base.control_flow
    return BranchInference(
        control_flow=control_flow,
        flow_target=flow_target,
        confidence=inferred.confidence,
        reasons=reasons,
    )

