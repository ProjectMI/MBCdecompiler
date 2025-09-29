"""Heuristics for deriving control-flow semantics for opcode/mode pairs.

The original implementation in :mod:`knowledge` relies on the manual
annotations to state whether an instruction performs a branch, jump or call.
While convenient for curated samples, in practice the data quickly drifts out
of date and a substantial amount of bytecode ends up without a meaningful
``control_flow`` tag.  The rest of the pipeline then assumes the instruction is
straight-line which in turn prevents higher level passes from discovering
structured control flow such as ``if`` statements or loops.

To improve the situation we introduce a small rule engine that analyses the
available metadata for each opcode/mode pair and derives a best-effort
classification.  The implementation intentionally favours transparency over raw
performance; each heuristic returns a :class:`FlowEvidence` record explaining
why it believes the instruction represents a particular control-flow kind.  The
classifier collates the evidence and exposes the winning result through a
:class:`FlowDescriptor` structure that callers can cache and reuse.

The heuristics are designed to be composable and easy to extend.  They rely on
signals that are already present in the knowledge base such as manual tags,
annotation summaries, stack deltas and operand usage statistics.  When manual
data is contradictory the classifier keeps track of the confidence score so the
rest of the system can decide whether to blindly trust the inferred result or
fall back to conservative behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# A small hand-curated list of keywords that strongly hint at a control-flow
# operation.  The classifier uses the same vocabulary across several heuristics
# to keep the behaviour predictable and the implementation straightforward to
# audit.  The mapping stores tuples of ``(kind, confidence, weight)`` where
# ``kind`` is one of the recognised control-flow classes, ``confidence`` is the
# baseline certainty provided by the keyword and ``weight`` is the relative
# importance when aggregating evidence from multiple sources.
_KEYWORD_HINTS: Mapping[str, Tuple[str, float, float]] = {
    "branch": ("branch", 0.9, 1.0),
    "jump": ("jump", 0.85, 0.9),
    "jmp": ("jump", 0.8, 0.8),
    "goto": ("jump", 0.8, 0.7),
    "call": ("call", 0.9, 0.9),
    "invoke": ("call", 0.8, 0.8),
    "return": ("return", 0.95, 1.0),
    "ret": ("return", 0.8, 0.8),
    "exit": ("stop", 0.85, 0.8),
    "halt": ("stop", 0.9, 0.9),
    "stop": ("stop", 0.9, 0.9),
    "loop": ("branch", 0.6, 0.6),
    "test": ("branch", 0.55, 0.55),
    "compare": ("branch", 0.6, 0.65),
    "conditional": ("branch", 0.7, 0.7),
    "switch": ("branch", 0.8, 0.75),
    "case": ("branch", 0.65, 0.65),
}


@dataclass(frozen=True)
class FlowEvidence:
    """Stores the contribution of a single heuristic.

    Attributes
    ----------
    source:
        Short identifier for the heuristic (for example ``"manual"`` or
        ``"summary-keyword"``).
    kind:
        Suggested control-flow classification.
    confidence:
        Value in the range ``[0.0, 1.0]`` expressing how certain the heuristic
        is about the proposed ``kind``.  Higher numbers indicate a stronger
        signal.
    weight:
        Optional multiplier describing how much the evidence should sway the
        overall result.  When in doubt use ``1.0``.
    note:
        Human readable explanation of the reasoning.  The classifier preserves
        the most influential notes so callers can surface them in diagnostics or
        debug logs.
    """

    source: str
    kind: Optional[str]
    confidence: float
    weight: float
    note: str


@dataclass(frozen=True)
class FlowDescriptor:
    """Final control-flow classification for an opcode/mode pair."""

    key: str
    declared: Optional[str]
    kind: Optional[str]
    confidence: float
    primary_note: Optional[str]
    evidence: Tuple[FlowEvidence, ...]

    def explain(self) -> str:
        """Return a human readable justification for the classification."""

        if not self.evidence:
            return "no control-flow signals detected"
        parts = [
            f"{item.source}: {item.kind or 'fallthrough'} ({item.confidence:.2f})"
            for item in self.evidence
        ]
        return ", ".join(parts)


class FlowHeuristic:
    """Base class for flow heuristics.

    Instances receive the opcode ``key`` and the manual annotation payload and
    return a list of :class:`FlowEvidence` entries.  Returning an empty list is
    perfectly acceptable and signals that the heuristic did not observe any
    meaningful data.
    """

    def evaluate(
        self,
        key: str,
        annotation: Mapping[str, object],
        metadata: Mapping[str, object],
        profile: Optional[Mapping[str, object]],
    ) -> List[FlowEvidence]:
        raise NotImplementedError


class ManualTagHeuristic(FlowHeuristic):
    """Use explicit ``control_flow`` annotations and tags."""

    _CONTROL_FLOW_VALUES = {"jump", "branch", "call", "return", "stop"}

    def evaluate(
        self,
        key: str,
        annotation: Mapping[str, object],
        metadata: Mapping[str, object],
        profile: Optional[Mapping[str, object]],
    ) -> List[FlowEvidence]:
        evidence: List[FlowEvidence] = []

        declared = annotation.get("control_flow")
        if isinstance(declared, str):
            normalised = declared.lower().strip()
            if normalised in self._CONTROL_FLOW_VALUES:
                note = f"manual annotation declares {normalised}"
                evidence.append(
                    FlowEvidence(
                        source="manual",
                        kind=normalised,
                        confidence=1.0,
                        weight=1.5,
                        note=note,
                    )
                )
            elif normalised == "fallthrough":
                evidence.append(
                    FlowEvidence(
                        source="manual",
                        kind=None,
                        confidence=0.6,
                        weight=1.2,
                        note="manual annotation marks instruction as fallthrough",
                    )
                )

        tags = annotation.get("tags")
        if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
            for raw_tag in tags:
                if not isinstance(raw_tag, str):
                    continue
                tag = raw_tag.lower()
                if tag in self._CONTROL_FLOW_VALUES:
                    evidence.append(
                        FlowEvidence(
                            source="manual-tag",
                            kind=tag,
                            confidence=0.9,
                            weight=1.0,
                            note=f"manual tag {tag}",
                        )
                    )
                elif tag in {"conditional", "loop"}:
                    evidence.append(
                        FlowEvidence(
                            source="manual-tag",
                            kind="branch",
                            confidence=0.75,
                            weight=0.9,
                            note=f"manual tag {tag}",
                        )
                    )

        return evidence


class KeywordHeuristic(FlowHeuristic):
    """Scan human-readable fields for indicative keywords."""

    _FIELDS = ("name", "summary", "description", "comment", "notes", "category")

    def evaluate(
        self,
        key: str,
        annotation: Mapping[str, object],
        metadata: Mapping[str, object],
        profile: Optional[Mapping[str, object]],
    ) -> List[FlowEvidence]:
        evidence: List[FlowEvidence] = []
        seen: Dict[str, float] = {}

        def handle_text(text: str, source: str) -> None:
            lower = text.lower()
            for keyword, (kind, confidence, weight) in _KEYWORD_HINTS.items():
                if keyword in lower:
                    previous = seen.get(kind, 0.0)
                    if confidence > previous:
                        seen[kind] = confidence
                    evidence.append(
                        FlowEvidence(
                            source=source,
                            kind=kind,
                            confidence=confidence,
                            weight=weight,
                            note=f"keyword '{keyword}'",
                        )
                    )

        manual_name = annotation.get("name")
        if isinstance(manual_name, str):
            handle_text(manual_name, "name")

        for field in self._FIELDS:
            raw = annotation.get(field)
            if isinstance(raw, str):
                handle_text(raw, field)

        mnemonic = metadata.get("mnemonic")
        if isinstance(mnemonic, str):
            handle_text(mnemonic, "mnemonic")

        return evidence


class OperandHeuristic(FlowHeuristic):
    """Treat operands that reference control-flow targets as evidence."""

    _TARGET_HINTS = {
        "absolute": "jump",
        "word": "jump",
        "relative": "branch",
        "segment": "branch",
        "table": "branch",
    }

    def evaluate(
        self,
        key: str,
        annotation: Mapping[str, object],
        metadata: Mapping[str, object],
        profile: Optional[Mapping[str, object]],
    ) -> List[FlowEvidence]:
        evidence: List[FlowEvidence] = []
        flow_target = annotation.get("flow_target")
        if isinstance(flow_target, str):
            normalised = flow_target.strip().lower()
            if normalised in self._TARGET_HINTS:
                kind = self._TARGET_HINTS[normalised]
                confidence = 0.75 if kind == "branch" else 0.7
                evidence.append(
                    FlowEvidence(
                        source="flow-target",
                        kind=kind,
                        confidence=confidence,
                        weight=1.0,
                        note=f"declared flow target {normalised}",
                    )
                )

        operand_hint = annotation.get("operand_hint")
        if isinstance(operand_hint, str):
            lower = operand_hint.lower()
            for keyword, (kind, confidence, weight) in _KEYWORD_HINTS.items():
                if keyword in lower and kind in {"jump", "branch", "call"}:
                    evidence.append(
                        FlowEvidence(
                            source="operand",
                            kind=kind,
                            confidence=max(0.55, confidence - 0.15),
                            weight=weight,
                            note=f"operand hint contains '{keyword}'",
                        )
                    )

        return evidence


class StackDeltaHeuristic(FlowHeuristic):
    """Use stack delta statistics to infer calls and returns."""

    def evaluate(
        self,
        key: str,
        annotation: Mapping[str, object],
        metadata: Mapping[str, object],
        profile: Optional[Mapping[str, object]],
    ) -> List[FlowEvidence]:
        evidence: List[FlowEvidence] = []

        stack_delta = annotation.get("stack_delta")
        if stack_delta is None:
            stack_delta = metadata.get("stack_delta")

        try:
            numeric = float(stack_delta) if stack_delta is not None else None
        except (TypeError, ValueError):
            numeric = None

        if numeric is None:
            return evidence

        if numeric < -0.9:
            note = f"stack delta {numeric:.2f} suggests call (consumes operands)"
            evidence.append(
                FlowEvidence(
                    source="stack-delta",
                    kind="call",
                    confidence=0.55,
                    weight=0.5,
                    note=note,
                )
            )
        elif numeric > 0.9:
            note = f"stack delta {numeric:.2f} suggests return (produces values)"
            evidence.append(
                FlowEvidence(
                    source="stack-delta",
                    kind="return",
                    confidence=0.5,
                    weight=0.45,
                    note=note,
                )
            )

        return evidence


class ProfileHeuristic(FlowHeuristic):
    """Leverage dynamic observation histograms for rare cases."""

    def evaluate(
        self,
        key: str,
        annotation: Mapping[str, object],
        metadata: Mapping[str, object],
        profile: Optional[Mapping[str, object]],
    ) -> List[FlowEvidence]:
        evidence: List[FlowEvidence] = []
        if not profile:
            return evidence

        following = profile.get("following")
        if isinstance(following, Mapping):
            jump_like = sum(
                count
                for next_key, count in following.items()
                if isinstance(next_key, str) and next_key.startswith("00:")
            )
            total = sum(count for count in following.values() if isinstance(count, (int, float)))
            if total and jump_like / total >= 0.85:
                evidence.append(
                    FlowEvidence(
                        source="profile-following",
                        kind="jump",
                        confidence=0.6,
                        weight=0.6,
                        note="profile frequently followed by entry opcodes",
                    )
                )

        preceding = profile.get("preceding")
        if isinstance(preceding, Mapping):
            branch_markers = sum(
                count
                for prev_key, count in preceding.items()
                if isinstance(prev_key, str)
                and any(keyword in prev_key.lower() for keyword in ("test", "cmp"))
            )
            total_prev = sum(
                count for count in preceding.values() if isinstance(count, (int, float))
            )
            if total_prev and branch_markers / total_prev >= 0.5:
                evidence.append(
                    FlowEvidence(
                        source="profile-preceding",
                        kind="branch",
                        confidence=0.55,
                        weight=0.5,
                        note="profile often follows comparisons",
                    )
                )

        return evidence


class FlowClassifier:
    """Combine heuristics to classify opcodes without explicit annotations."""

    def __init__(
        self,
        manual_data: Mapping[str, Mapping[str, object]],
        metadata_index: Mapping[str, Mapping[str, object]],
        profiles: Mapping[str, Mapping[str, object]],
    ) -> None:
        self._manual_data = manual_data
        self._metadata_index = metadata_index
        self._profiles = profiles
        self._cache: Dict[str, FlowDescriptor] = {}
        self._heuristics: Tuple[FlowHeuristic, ...] = (
            ManualTagHeuristic(),
            KeywordHeuristic(),
            OperandHeuristic(),
            StackDeltaHeuristic(),
            ProfileHeuristic(),
        )

    def descriptor(self, key: str) -> FlowDescriptor:
        """Return the cached descriptor for ``key``."""

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        annotation = self._manual_data.get(key, {})
        metadata = self._metadata_index.get(key, {})
        profile = self._profiles.get(key)

        evidence: List[FlowEvidence] = []
        for heuristic in self._heuristics:
            try:
                evidence.extend(heuristic.evaluate(key, annotation, metadata, profile))
            except Exception as exc:  # pragma: no cover - defensive safety net
                evidence.append(
                    FlowEvidence(
                        source=heuristic.__class__.__name__,
                        kind=None,
                        confidence=0.0,
                        weight=0.0,
                        note=f"heuristic error: {exc}",
                    )
                )

        descriptor = self._select_best(key, annotation.get("control_flow"), evidence)
        self._cache[key] = descriptor
        return descriptor

    def update_metadata(self, key: str, payload: Mapping[str, object]) -> None:
        """Update the metadata view for ``key`` and invalidate cached results."""

        self._metadata_index[key] = dict(payload)
        self._cache.pop(key, None)

    def update_profile(self, key: str, payload: Optional[Mapping[str, object]]) -> None:
        """Refresh the statistical profile associated with ``key``."""

        if payload is None:
            self._profiles.pop(key, None)
        else:
            self._profiles[key] = dict(payload)
        self._cache.pop(key, None)

    def _select_best(
        self,
        key: str,
        declared: Optional[object],
        evidence: Iterable[FlowEvidence],
    ) -> FlowDescriptor:
        declared_str = str(declared).lower() if isinstance(declared, str) else None
        best_kind: Optional[str] = None
        best_score = 0.0
        best_note: Optional[str] = None
        collected: List[FlowEvidence] = []

        for item in evidence:
            collected.append(item)
            if item.kind is None:
                continue
            score = item.confidence * max(item.weight, 0.0)
            if score > best_score:
                best_score = score
                best_kind = item.kind
                best_note = item.note

        if declared_str in {"jump", "branch", "call", "return", "stop"}:
            # Manual annotations win regardless of other heuristics.  This keeps
            # behaviour compatible with existing curated data while still allowing
            # heuristics to contribute when the manual entry is ambiguous.
            best_kind = declared_str
            best_score = max(best_score, 1.0)
            best_note = "manual annotation"

        if best_kind is None and declared_str not in {None, "", "fallthrough"}:
            best_kind = declared_str
            best_note = "manual annotation (unrecognised value)"
            best_score = max(best_score, 0.7)

        descriptor = FlowDescriptor(
            key=key,
            declared=declared_str,
            kind=best_kind,
            confidence=min(max(best_score, 0.0), 1.0),
            primary_note=best_note,
            evidence=tuple(collected),
        )
        return descriptor

