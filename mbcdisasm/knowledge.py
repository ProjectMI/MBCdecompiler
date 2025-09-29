"""Knowledge base support for opcode statistics and annotations."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from .flow_analysis import FlowClassifier, FlowDescriptor


def _normalize_stack_delta_key(value: object) -> object:
    """Normalise stack delta keys while preserving fractional values."""

    if isinstance(value, str):
        if value == "unknown":
            return "unknown"
        try:
            numeric = float(value)
        except ValueError:
            return value
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        return numeric
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return value
    if isinstance(value, int):
        return value
    return value


def _coerce_stack_delta_numeric(value: object) -> Optional[float]:
    """Return a numeric stack delta if possible, otherwise :data:`None`."""

    normalized = _normalize_stack_delta_key(value)
    if isinstance(normalized, (int, float)):
        numeric = float(normalized)
        if math.isnan(numeric):
            return None
        return numeric
    return None


@dataclass(frozen=True)
class ProfileAssessment:
    """Summary of how a new observation matches the stored knowledge."""

    key: str
    status: str
    existing_count: int
    new_count: int
    notes: str = ""

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "existing_count": self.existing_count,
            "new_count": self.new_count,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AnnotationUpdate:
    """Record describing an automatic change to the knowledge annotations."""

    key: str
    field: str
    old_value: object
    new_value: object
    confidence: float
    samples: int
    reason: str

    def to_json(self) -> dict:
        return {
            "field": self.field,
            "old": self.old_value,
            "new": self.new_value,
            "confidence": self.confidence,
            "samples": self.samples,
            "reason": self.reason,
        }


@dataclass
class OpcodeProfile:
    key: str
    count: int = 0
    stack_deltas: Counter = field(default_factory=Counter)
    operand_types: Counter = field(default_factory=Counter)
    preceding: Counter = field(default_factory=Counter)
    following: Counter = field(default_factory=Counter)

    def to_json(self) -> dict:
        return {
            "count": self.count,
            "stack_deltas": {str(k): int(v) for k, v in self.stack_deltas.items()},
            "operand_types": dict(self.operand_types),
            "preceding": dict(self.preceding),
            "following": dict(self.following),
        }

    @classmethod
    def from_json(cls, key: str, data: Mapping[str, object]) -> "OpcodeProfile":
        profile = cls(key)
        profile.count = int(data.get("count", 0))
        stack_deltas: Counter = Counter()
        for raw_key, raw_value in data.get("stack_deltas", {}).items():
            value = int(raw_value)
            key_obj = _normalize_stack_delta_key(raw_key)
            if key_obj == "unknown":
                stack_deltas["unknown"] += value
            elif isinstance(key_obj, (int, float)):
                stack_deltas[key_obj] += value
            else:
                stack_deltas[str(key_obj)] += value

        profile.stack_deltas = stack_deltas
        profile.operand_types = Counter({str(k): int(v) for k, v in data.get("operand_types", {}).items()})
        profile.preceding = Counter({str(k): int(v) for k, v in data.get("preceding", {}).items()})
        profile.following = Counter({str(k): int(v) for k, v in data.get("following", {}).items()})
        return profile


@dataclass(frozen=True)
class StackObservation:
    """Summary of stack delta evidence gathered for an opcode profile."""

    key: str
    total_samples: int
    known_samples: int
    unknown_samples: int
    dominant: Optional[float]
    confidence: Optional[float]

    @property
    def unknown_ratio(self) -> Optional[float]:
        if self.total_samples == 0:
            return None
        return self.unknown_samples / self.total_samples

    def to_json(self) -> dict:
        payload: dict = {
            "total_samples": self.total_samples,
            "known_samples": self.known_samples,
            "unknown_samples": self.unknown_samples,
        }
        if self.dominant is not None:
            payload["dominant"] = self.dominant
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload

    @classmethod
    def from_profile(cls, profile: "OpcodeProfile") -> "StackObservation":
        total = int(sum(profile.stack_deltas.values()))
        filtered = KnowledgeBase._filter_known_stack_deltas(profile.stack_deltas)
        known = int(sum(filtered.values()))
        unknown = int(profile.stack_deltas.get("unknown", 0))
        if filtered and total:
            dominant, count = filtered.most_common(1)[0]
            dominant_value = float(dominant)
            confidence = count / total
        else:
            dominant_value = None
            confidence = None
        return cls(
            key=profile.key,
            total_samples=total,
            known_samples=known,
            unknown_samples=unknown,
            dominant=dominant_value,
            confidence=confidence,
        )


@dataclass
class ReviewTask:
    """Actionable follow-up derived from aggregated analysis evidence."""

    key: str
    reason: str
    samples: int
    missing_annotations: List[str] = field(default_factory=list)
    stack_unknown_ratio: Optional[float] = None
    stack_dominant: Optional[float] = None
    stack_confidence: Optional[float] = None
    operand_hint: Optional[str] = None
    operand_confidence: Optional[float] = None
    top_preceding: List[Tuple[str, int]] = field(default_factory=list)
    top_following: List[Tuple[str, int]] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> dict:
        payload = {
            "key": self.key,
            "reason": self.reason,
            "samples": self.samples,
            "missing_annotations": list(self.missing_annotations),
        }
        if self.stack_unknown_ratio is not None:
            payload["stack_unknown_ratio"] = self.stack_unknown_ratio
        if self.stack_dominant is not None:
            payload["stack_dominant"] = self.stack_dominant
        if self.stack_confidence is not None:
            payload["stack_confidence"] = self.stack_confidence
        if self.operand_hint is not None:
            payload["operand_hint"] = self.operand_hint
        if self.operand_confidence is not None:
            payload["operand_confidence"] = self.operand_confidence
        if self.top_preceding:
            payload["top_preceding"] = [list(item) for item in self.top_preceding]
        if self.top_following:
            payload["top_following"] = [list(item) for item in self.top_following]
        if self.notes:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "ReviewTask":
        key = str(data.get("key", ""))
        reason = str(data.get("reason", ""))
        samples = int(data.get("samples", 0))
        missing_raw = data.get("missing_annotations", [])
        if isinstance(missing_raw, Sequence):
            missing = [str(item) for item in missing_raw]
        else:
            missing = []

        def _optional_float(value: object) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _parse_pairs(payload: object) -> List[Tuple[str, int]]:
            pairs: List[Tuple[str, int]] = []
            if isinstance(payload, Sequence):
                for entry in payload:
                    if isinstance(entry, Sequence) and len(entry) >= 2:
                        name = str(entry[0])
                        try:
                            count = int(entry[1])
                        except (TypeError, ValueError):
                            continue
                        pairs.append((name, count))
            return pairs

        return cls(
            key=key,
            reason=reason,
            samples=samples,
            missing_annotations=missing,
            stack_unknown_ratio=_optional_float(data.get("stack_unknown_ratio")),
            stack_dominant=_optional_float(data.get("stack_dominant")),
            stack_confidence=_optional_float(data.get("stack_confidence")),
            operand_hint=(
                str(data.get("operand_hint"))
                if data.get("operand_hint") is not None
                else None
            ),
            operand_confidence=_optional_float(data.get("operand_confidence")),
            top_preceding=_parse_pairs(data.get("top_preceding")),
            top_following=_parse_pairs(data.get("top_following")),
            notes=str(data.get("notes", "")) if data.get("notes") is not None else "",
        )

    def merge(self, other: "ReviewTask") -> bool:
        """Merge ``other`` into this task and return True if new data was added."""

        changed = False
        if other.samples > self.samples:
            self.samples = other.samples
            changed = True

        combined = sorted(set(self.missing_annotations) | set(other.missing_annotations))
        if combined != sorted(self.missing_annotations):
            self.missing_annotations = combined
            changed = True

        def _prefer_float(
            existing: Optional[float],
            candidate: Optional[float],
            *,
            prefer_small: bool = False,
        ) -> Tuple[Optional[float], bool]:
            if candidate is None:
                return existing, False
            if existing is None:
                return candidate, True
            if prefer_small:
                if candidate < existing:
                    return candidate, True
            else:
                if candidate > existing:
                    return candidate, True
            return existing, False

        self.stack_unknown_ratio, flag = _prefer_float(
            self.stack_unknown_ratio, other.stack_unknown_ratio, prefer_small=True
        )
        changed = changed or flag

        self.stack_confidence, flag = _prefer_float(
            self.stack_confidence, other.stack_confidence
        )
        changed = changed or flag

        self.operand_confidence, flag = _prefer_float(
            self.operand_confidence, other.operand_confidence
        )
        changed = changed or flag

        if self.stack_dominant is None and other.stack_dominant is not None:
            self.stack_dominant = other.stack_dominant
            changed = True
        if self.operand_hint is None and other.operand_hint is not None:
            self.operand_hint = other.operand_hint
            changed = True

        if other.top_preceding and other.top_preceding != self.top_preceding:
            self.top_preceding = list(other.top_preceding)
            changed = True
        if other.top_following and other.top_following != self.top_following:
            self.top_following = list(other.top_following)
            changed = True

        if other.notes and other.notes != self.notes:
            self.notes = other.notes
            changed = True

        return changed


@dataclass
class MergeReport:
    """Outcome of a knowledge merge pass with feedback for operators."""

    updates: List[AnnotationUpdate]
    review_tasks: List[ReviewTask]
    stack_observations: List[StackObservation]
    assessments: List[ProfileAssessment]

    def conflicts(self) -> List[ProfileAssessment]:
        return [a for a in self.assessments if a.status == "conflict"]

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return bool(self.updates or self.review_tasks)


@dataclass(frozen=True)
class InstructionMetadata:
    """Human-oriented description assembled from the stored knowledge."""

    mnemonic: str
    stack_delta: Optional[float]
    stack_confidence: Optional[float]
    stack_samples: Optional[int]
    stack_source: Optional[str]
    operand_hint: Optional[str]
    operand_confidence: Optional[float]
    control_flow: Optional[str]
    flow_target: Optional[str]
    summary: Optional[str]
    declared_flow: Optional[str]
    flow_confidence: float
    flow_reason: Optional[str]


@dataclass(frozen=True)
class SemanticSignature:
    """Compact representation of behavioural evidence for an opcode key."""

    key: str
    stack_delta: Optional[float]
    stack_confidence: Optional[float]
    operand_hint: Optional[str]
    operand_confidence: Optional[float]
    preceding: Dict[str, float]
    following: Dict[str, float]
    flow_target: Optional[str]


@dataclass(frozen=True)
class SemanticMatch:
    """Description of a semantic naming match applied to a profile."""

    key: str
    prototype: str
    score: float
    channels: Dict[str, str]
    applied_fields: Tuple[str, ...]


@dataclass(frozen=True)
class SemanticNamingReport:
    """Outcome of a semantic post-analysis pass."""

    updates: List[AnnotationUpdate]
    matches: List[SemanticMatch]


class KnowledgeBase:
    """On-disk knowledge base with light conflict tracking."""

    SCHEMA_VERSION = 1
    SEMANTIC_MIN_PROFILE_SAMPLES = 4
    SEMANTIC_MIN_SCORE = 0.75
    SEMANTIC_SCORE_MARGIN = 0.08
    SEMANTIC_NEIGHBOUR_THRESHOLD = 0.35
    SEMANTIC_STACK_WEIGHT = 0.4
    SEMANTIC_OPERAND_WEIGHT = 0.3
    SEMANTIC_CONTEXT_WEIGHT = 0.2
    SEMANTIC_FLOW_WEIGHT = 0.1
    _MigrationFunc = Callable[[MutableMapping[str, object]], MutableMapping[str, object]]
    _MIGRATIONS: Dict[int, _MigrationFunc] = {}

    def __init__(self, path: Path, data: MutableMapping[str, object]) -> None:
        self.path = path
        self._data = data
        self._profiles: Dict[str, OpcodeProfile] = {}
        for key, payload in data.get("opcode_modes", {}).items():
            self._profiles[key] = OpcodeProfile.from_json(key, payload)
        self._annotations = data.setdefault("annotations", {})
        self._manual_reference: Dict[str, Mapping[str, object]] = {}
        self._merge_manual_annotations()
        self._flow_metadata_index: Dict[str, Mapping[str, object]] = {}
        self._refresh_flow_metadata_index()
        self._flow_classifier = FlowClassifier(
            self._manual_reference,
            self._flow_metadata_index,
            {key: profile.to_json() for key, profile in self._profiles.items()},
        )
        self._review_tasks: Dict[Tuple[str, str], ReviewTask] = {}
        for entry in data.get("review_tasks", []):
            if isinstance(entry, Mapping):
                try:
                    task = ReviewTask.from_json(entry)
                except (TypeError, ValueError):
                    continue
                self._review_tasks[(task.key, task.reason)] = task

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"schema": cls.SCHEMA_VERSION, "opcode_modes": {}}
        schema = data.get("schema")
        if schema != cls.SCHEMA_VERSION:
            data = cls._upgrade_schema(data, schema)
        return cls(path, data)

    def get_profile(self, key: str) -> OpcodeProfile:
        profile = self._profiles.get(key)
        if profile is None:
            profile = OpcodeProfile(key)
            self._profiles[key] = profile
            self._data.setdefault("opcode_modes", {})[key] = profile.to_json()
            if hasattr(self, "_flow_classifier"):
                self._flow_classifier.update_profile(key, profile.to_json())
        return profile

    def assess_profile(self, profile: OpcodeProfile) -> ProfileAssessment:
        existing = self._profiles.get(profile.key)
        if existing is None or existing.count == 0:
            return ProfileAssessment(
                key=profile.key,
                status="unknown",
                existing_count=0,
                new_count=profile.count,
                notes="no prior observations",
            )

        existing_dom, existing_ratio = self._dominant(existing.stack_deltas)
        new_dom, new_ratio = self._dominant(profile.stack_deltas)
        existing_avg = self._average_stack_delta(existing.stack_deltas)
        new_avg = self._average_stack_delta(profile.stack_deltas)

        ann = self._annotations.get(profile.key, {})
        annotated_delta = ann.get("stack_delta")
        if annotated_delta is not None:
            try:
                expected = float(annotated_delta)
            except (TypeError, ValueError):
                expected = existing_avg if existing_avg is not None else existing_dom
        else:
            expected = existing_avg if existing_avg is not None else existing_dom
        observed = new_avg if new_avg is not None else new_dom

        if expected is None or observed is None:
            status = "undetermined"
            notes = "insufficient stack delta samples"
        elif self._stack_delta_close(expected, observed):
            if abs(existing_ratio - new_ratio) <= 0.2:
                status = "confirmed"
                notes = (
                    "dominant stack delta "
                    f"{self._format_stack_delta(expected)} stable"
                )
            else:
                status = "partial"
                notes = (
                    "dominant stack delta "
                    f"{self._format_stack_delta(expected)} but distribution shifted "
                    f"({existing_ratio:.2f}→{new_ratio:.2f})"
                )
        else:
            existing_known = sum(
                self._filter_known_stack_deltas(existing.stack_deltas).values()
            )
            new_known = sum(self._filter_known_stack_deltas(profile.stack_deltas).values())
            if existing_known < 3 and new_known >= max(3, existing_known * 2):
                status = "partial"
                notes = (
                    "stack delta adjusted "
                    f"{self._format_stack_delta(expected)}→"
                    f"{self._format_stack_delta(observed)}"
                    f" based on {new_known} new samples"
                )
            else:
                status = "conflict"
                notes = (
                    "expected stack delta "
                    f"{self._format_stack_delta(expected)}, observed "
                    f"{self._format_stack_delta(observed)}"
                )

        return ProfileAssessment(
            key=profile.key,
            status=status,
            existing_count=existing.count,
            new_count=profile.count,
            notes=notes,
        )

    def estimate_stack_delta(self, key: str) -> Optional[float]:
        ann = self._annotations.get(key, {})
        if "stack_delta" in ann:
            return float(ann["stack_delta"])
        profile = self._profiles.get(key)
        if not profile or not profile.stack_deltas:
            return None
        known = self._filter_known_stack_deltas(profile.stack_deltas)
        if not known:
            return None
        total = sum(known.values())
        weighted = sum(delta * count for delta, count in known.items())
        if total:
            return weighted / total
        return None

    def describe(self, key: str) -> str:
        return self.instruction_metadata(key).mnemonic

    def instruction_metadata(self, key: str) -> InstructionMetadata:
        """Return a consolidated human-facing description for ``key``."""

        ann = self._annotations.get(key, {})
        mnemonic = str(ann.get("name", f"opcode_{key.replace(':', '_')}"))

        stack_delta = self.estimate_stack_delta(key)
        stack_confidence: Optional[float] = None
        stack_samples: Optional[int] = None
        stack_source: Optional[str] = None

        profile = self._profiles.get(key)
        if profile is not None:
            filtered = self._filter_known_stack_deltas(profile.stack_deltas)
            if filtered:
                stack_samples = sum(filtered.values())
                _, ratio = self._dominant(profile.stack_deltas)
                if ratio:
                    stack_confidence = ratio

        if stack_confidence is None:
            confidence = ann.get("stack_confidence")
            if confidence is not None:
                stack_confidence = float(confidence)
        if stack_samples is None:
            samples = ann.get("stack_samples")
            if samples is not None:
                stack_samples = int(samples)
        source_hint = ann.get("stack_source")
        if source_hint:
            stack_source = str(source_hint)
        elif key in self._manual_reference and "stack_delta" in self._manual_reference[key]:
            stack_source = "manual"
        elif stack_delta is not None and stack_samples:
            stack_source = "derived"

        operand_hint, operand_confidence = self._dominant_operand_type(profile)
        hint_override = ann.get("operand_hint")
        if hint_override:
            operand_hint = str(hint_override)
            confidence_override = ann.get("operand_confidence")
            if confidence_override is not None:
                operand_confidence = float(confidence_override)
            else:
                operand_confidence = None

        control_flow = self.control_flow_hint(key)
        flow_descriptor = self.control_flow_descriptor(key)
        flow_target = self.flow_target_hint(key)

        summary: Optional[str] = None
        for field in ("summary", "description", "comment", "notes"):
            value = ann.get(field)
            if value:
                summary = str(value)
                break

        return InstructionMetadata(
            mnemonic=mnemonic,
            stack_delta=stack_delta,
            stack_confidence=stack_confidence,
            stack_samples=stack_samples,
            stack_source=stack_source,
            operand_hint=operand_hint,
            operand_confidence=operand_confidence,
            control_flow=control_flow,
            flow_target=flow_target,
            summary=summary,
            declared_flow=flow_descriptor.declared,
            flow_confidence=flow_descriptor.confidence,
            flow_reason=flow_descriptor.primary_note,
        )

    def manual_annotation(self, key: str) -> Mapping[str, object]:
        """Return the raw manual annotation payload for ``key`` if available."""

        manual = self._manual_reference.get(key)
        if manual is not None:
            return dict(manual)
        ann = self._annotations.get(key)
        if isinstance(ann, Mapping):
            return {str(field): value for field, value in ann.items()}
        return {}

    def control_flow_hint(self, key: str) -> Optional[str]:
        """Return the inferred control-flow semantics for ``key``."""

        return self.control_flow_descriptor(key).kind

    def control_flow_descriptor(self, key: str) -> FlowDescriptor:
        """Return the detailed control-flow descriptor for ``key``."""

        self._ensure_flow_entry(key)
        return self._flow_classifier.descriptor(key)

    def flow_target_hint(self, key: str) -> Optional[str]:
        """Return the strategy used to resolve branch targets for ``key``.

        Opcodes that alter control flow may encode their target addresses in a
        variety of formats (absolute offsets, word indices, relative jumps and
        so on).  The knowledge base can store a ``flow_target`` hint so graph
        builders can resolve destinations heuristically.  Recognised values are
        ``absolute`` (use the operand verbatim), ``segment`` (operand is
        relative to the start of the segment), ``word`` (word index relative to
        the segment start) and ``relative`` (offset from the next instruction).
        Unknown entries yield :data:`None` which callers may interpret as a
        best-effort guess.
        """

        ann = self._annotations.get(key, {})
        value = ann.get("flow_target")
        if value is None:
            return None
        return str(value)

    def merge_profiles(
        self,
        profiles: Iterable[OpcodeProfile],
        *,
        min_samples: int = 6,
        confidence_threshold: float = 0.75,
    ) -> MergeReport:
        """Fold new profile observations into the persistent store.

        In addition to updating the raw histograms this method also performs
        self-learning passes that refresh annotations and queue manual review
        tasks when the gathered evidence is actionable.
        """

        store = self._data.setdefault("opcode_modes", {})
        updates: List[AnnotationUpdate] = []
        review_tasks: List[ReviewTask] = []
        observations: List[StackObservation] = []
        assessments: List[ProfileAssessment] = []
        for profile in profiles:
            assessment = self.assess_profile(profile)
            assessments.append(assessment)

            existing = self._profiles.get(profile.key)
            if existing is None:
                self._profiles[profile.key] = profile
                store[profile.key] = profile.to_json()
                target = profile
            else:
                existing.count += profile.count
                existing.stack_deltas.update(profile.stack_deltas)
                existing.operand_types.update(profile.operand_types)
                existing.preceding.update(profile.preceding)
                existing.following.update(profile.following)
                store[profile.key] = existing.to_json()
                target = existing

            if hasattr(self, "_flow_classifier"):
                self._flow_classifier.update_profile(target.key, target.to_json())

            observation = StackObservation.from_profile(target)
            observations.append(observation)

            update = self._update_stack_annotation(
                target,
                min_samples=min_samples,
                confidence_threshold=confidence_threshold,
            )
            if update:
                updates.append(update)

            operand_update = self._update_operand_annotation(
                target,
                min_samples=min_samples,
                confidence_threshold=confidence_threshold,
            )
            if operand_update:
                updates.append(operand_update)

            for task in self._derive_review_tasks(
                target,
                assessment,
                observation,
                min_samples=min_samples,
                confidence_threshold=confidence_threshold,
            ):
                stored = self._enqueue_review_task(task)
                if stored is not None:
                    review_tasks.append(stored)

        return MergeReport(
            updates=updates,
            review_tasks=review_tasks,
            stack_observations=observations,
            assessments=assessments,
        )

    def render_merge_report(
        self,
        report: MergeReport,
        *,
        min_samples: int = 6,
        confidence_threshold: float = 0.75,
    ) -> str:
        """Return a human-readable summary for ``report``.

        The generated text focuses on entries that require operator
        attention—conflicts, partial matches and newly applied stack delta
        updates.  For each entry the summary displays the annotated and
        observed stack deltas, the number of previously known versus new
        samples and a suggested follow-up action.
        """

        if not report.assessments and not report.updates:
            return ""

        observation_map = {item.key: item for item in report.stack_observations}
        stack_updates: Dict[str, AnnotationUpdate] = {}
        for update in report.updates:
            if update.field == "stack_delta":
                stack_updates[update.key] = update

        review_reasons: Dict[str, str] = {}
        for task in report.review_tasks:
            if task.reason:
                review_reasons.setdefault(task.key, task.reason)

        def _format_delta(value: object) -> str:
            numeric = _coerce_stack_delta_numeric(value)
            if numeric is not None:
                return self._format_stack_delta(numeric)
            if value is None:
                return "—"
            return str(value)

        def _recommendation(
            key: str,
            assessment: ProfileAssessment,
            observation: Optional[StackObservation],
        ) -> str:
            update = stack_updates.get(key)
            if assessment.status == "conflict" or review_reasons.get(key) == "stack_conflict":
                return "инициировать ручной просмотр трассировки"
            if update is not None:
                return "принять корректировку"

            if observation is not None:
                missing = max(0, int(min_samples) - int(observation.known_samples))
            else:
                missing = int(min_samples)
            if missing <= 0:
                missing = max(1, int(round(min_samples * (1.0 - confidence_threshold))))
            return f"отложить до набора ещё {missing} наблюдений"

        ranked_entries: List[Tuple[Tuple[int, str], List[str]]] = []
        for assessment in report.assessments:
            key = assessment.key
            observation = observation_map.get(key)
            update = stack_updates.get(key)

            if update is None and assessment.status == "confirmed":
                continue

            annotated = self._annotations.get(key, {}).get("stack_delta")
            observed = observation.dominant if observation is not None else None

            observed_samples: str
            if observation is not None:
                observed_samples = (
                    f"известных {observation.known_samples}/{observation.total_samples}"
                )
                confidence = (
                    f", уверенность {observation.confidence:.2f}"
                    if observation.confidence is not None
                    else ""
                )
            else:
                observed_samples = "известных —"
                confidence = ""

            details = [
                f"- {key}: аннотированная Δ={_format_delta(annotated)} → наблюдённая Δ={_format_delta(observed)}",
                (
                    "    выборка: прошлые={existing} новых={new} {samples}{confidence}"
                ).format(
                    existing=int(assessment.existing_count),
                    new=int(assessment.new_count),
                    samples=observed_samples,
                    confidence=confidence,
                ),
                f"    статус: {assessment.status} – {assessment.notes}",
                f"    рекомендация: {_recommendation(key, assessment, observation)}",
            ]

            if update is not None:
                details.insert(
                    3,
                    f"    обновление: {update.old_value if update.old_value is not None else '—'} → {update.new_value}"
                    f" ({update.reason}, {update.samples} сэмплов, доверие {update.confidence:.2f})",
                )

            if review_reasons.get(key):
                details.append(f"    очередь обзора: {review_reasons[key]}")

            severity = {
                "conflict": 0,
                "partial": 1,
                "undetermined": 2,
                "unknown": 3,
            }.get(assessment.status, 4)
            if update is not None:
                severity = min(severity, 1)

            ranked_entries.append(((severity, key), details))

        if not ranked_entries:
            return ""

        lines = ["Сводка обновления знаний:"]
        for _, entry_lines in sorted(ranked_entries, key=lambda item: item[0]):
            lines.extend(entry_lines)

        return "\n".join(lines)

    def record_annotation(self, key: str, **updates: object) -> None:
        key_str = str(key)
        normalized = {str(field): value for field, value in updates.items()}
        ann = self._annotations.setdefault(key_str, {})
        ann.update(normalized)

        manual_entry = self._manual_reference.get(key_str)
        if manual_entry is None:
            manual_entry = {}
            self._manual_reference[key_str] = manual_entry
        manual_entry.update(normalized)

    def pending_review_tasks(self) -> List[ReviewTask]:
        return sorted(
            self._review_tasks.values(), key=lambda item: (item.key, item.reason)
        )

    def resolve_review_task(self, key: str, reason: Optional[str] = None) -> int:
        """Remove stored review tasks matching ``key`` and ``reason``.

        Parameters
        ----------
        key:
            Opcode/mode identifier whose review tasks should be cleared.
        reason:
            Optional reason filter.  When omitted all tasks for ``key`` are
            removed.  When provided only matching tasks are affected.

        Returns
        -------
        int
            Number of queued tasks that were removed.
        """

        if reason is None:
            candidates = [handle for handle in self._review_tasks if handle[0] == key]
        else:
            handle = (key, reason)
            candidates = [handle] if handle in self._review_tasks else []

        removed = 0
        for handle in candidates:
            if self._review_tasks.pop(handle, None) is not None:
                removed += 1
        return removed

    @staticmethod
    def _filter_known_stack_deltas(counter: Counter) -> Counter:
        filtered: Counter = Counter()
        for delta, count in counter.items():
            numeric = _coerce_stack_delta_numeric(delta)
            if numeric is None:
                continue
            filtered[numeric] += int(count)
        return filtered

    @classmethod
    def _dominant(cls, counter: Counter) -> tuple[Optional[float], float]:
        filtered = cls._filter_known_stack_deltas(counter)
        if not filtered:
            return None, 0.0
        delta, count = filtered.most_common(1)[0]
        total = sum(filtered.values())
        if total == 0:
            return None, 0.0
        return float(delta), count / total

    @staticmethod
    def _stack_delta_close(a: float, b: float) -> bool:
        return math.isclose(float(a), float(b), rel_tol=0.05, abs_tol=0.5)

    @classmethod
    def _average_stack_delta(cls, counter: Counter) -> Optional[float]:
        filtered = cls._filter_known_stack_deltas(counter)
        if not filtered:
            return None
        total = sum(filtered.values())
        if total == 0:
            return None
        weighted = sum(float(delta) * count for delta, count in filtered.items())
        return weighted / total

    @staticmethod
    def _format_stack_delta(delta: float) -> str:
        value = float(delta)
        if not math.isfinite(value):
            return str(delta)
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"

    @staticmethod
    def _dominant_operand_type(
        profile: Optional[OpcodeProfile],
    ) -> tuple[Optional[str], Optional[float]]:
        if profile is None or not profile.operand_types:
            return None, None
        total = sum(profile.operand_types.values())
        if not total:
            return None, None
        operand, count = profile.operand_types.most_common(1)[0]
        return str(operand), count / total

    def _top_relationships(self, counter: Counter, *, limit: int = 3) -> List[Tuple[str, int]]:
        return [
            (str(key), int(value))
            for key, value in counter.most_common(limit)
            if value > 0
        ]

    def _update_stack_annotation(
        self,
        profile: OpcodeProfile,
        *,
        min_samples: int,
        confidence_threshold: float,
    ) -> Optional[AnnotationUpdate]:
        filtered = self._filter_known_stack_deltas(profile.stack_deltas)
        total_samples = sum(filtered.values())
        if total_samples < min_samples:
            return None

        dominant, ratio = self._dominant(profile.stack_deltas)
        if dominant is None or ratio < confidence_threshold:
            return None

        existing_ann = self._annotations.get(profile.key)
        ann_is_mutable = isinstance(existing_ann, MutableMapping)
        ann: MutableMapping[str, object]
        if ann_is_mutable:
            ann = cast(MutableMapping[str, object], existing_ann)
        else:
            ann = {}
        previous = ann.get("stack_delta")
        previous_numeric = _coerce_stack_delta_numeric(previous)
        previous_confidence = float(ann.get("stack_confidence", 0.0))
        previous_samples = int(ann.get("stack_samples", 0))

        if (
            previous_numeric is not None
            and self._stack_delta_close(previous_numeric, dominant)
            and previous_samples >= total_samples
            and abs(previous_confidence - ratio) < 1e-9
        ):
            # Nothing new to report.
            return None

        dominant_numeric = float(dominant)
        if math.isfinite(dominant_numeric) and dominant_numeric.is_integer():
            stored_dominant: object = int(dominant_numeric)
        else:
            stored_dominant = dominant_numeric

        ann.update(
            {
                "stack_delta": stored_dominant,
                "stack_confidence": float(ratio),
                "stack_samples": int(total_samples),
                "stack_source": "derived",
            }
        )
        if not ann_is_mutable:
            self._annotations[profile.key] = ann

        reason = "initial" if previous is None else "revised"
        return AnnotationUpdate(
            key=profile.key,
            field="stack_delta",
            old_value=previous,
            new_value=stored_dominant,
            confidence=float(ratio),
            samples=int(total_samples),
            reason=reason,
        )

    def _update_operand_annotation(
        self,
        profile: OpcodeProfile,
        *,
        min_samples: int,
        confidence_threshold: float,
    ) -> Optional[AnnotationUpdate]:
        if not profile.operand_types:
            return None
        total_samples = int(sum(profile.operand_types.values()))
        if total_samples < max(1, min_samples // 2):
            return None
        operand, ratio = self._dominant_operand_type(profile)
        if operand is None or ratio is None:
            return None
        if ratio < confidence_threshold:
            return None

        existing_ann = self._annotations.get(profile.key)
        ann_is_mutable = isinstance(existing_ann, MutableMapping)
        ann: MutableMapping[str, object]
        if ann_is_mutable:
            ann = cast(MutableMapping[str, object], existing_ann)
        else:
            ann = {}
        previous = ann.get("operand_hint")
        previous_confidence = float(ann.get("operand_confidence", 0.0))
        previous_samples = int(ann.get("operand_samples", 0))

        if (
            previous == operand
            and previous_samples >= total_samples
            and abs(previous_confidence - ratio) < 1e-9
        ):
            return None

        ann.update(
            {
                "operand_hint": operand,
                "operand_confidence": float(ratio),
                "operand_samples": int(total_samples),
            }
        )
        if not ann_is_mutable:
            self._annotations[profile.key] = ann

        reason = "initial" if previous is None else "revised"
        return AnnotationUpdate(
            key=profile.key,
            field="operand_hint",
            old_value=previous,
            new_value=operand,
            confidence=float(ratio),
            samples=int(total_samples),
            reason=reason,
        )

    def _enqueue_review_task(self, task: ReviewTask) -> Optional[ReviewTask]:
        key = (task.key, task.reason)
        existing = self._review_tasks.get(key)
        if existing is None:
            stored = ReviewTask.from_json(task.to_json())
            self._review_tasks[key] = stored
            return stored
        if existing.merge(task):
            return existing
        return None

    def _derive_review_tasks(
        self,
        profile: OpcodeProfile,
        assessment: ProfileAssessment,
        observation: StackObservation,
        *,
        min_samples: int,
        confidence_threshold: float,
    ) -> List[ReviewTask]:
        tasks: List[ReviewTask] = []
        ann = self._annotations.get(profile.key, {})
        operand_hint, operand_confidence = self._dominant_operand_type(profile)

        missing_fields: List[str] = []
        if "name" not in ann:
            missing_fields.append("name")
        if "stack_delta" not in ann:
            missing_fields.append("stack_delta")
        if "control_flow" not in ann and (profile.preceding or profile.following):
            missing_fields.append("control_flow")
        # Manual annotations may intentionally omit operand hints; do not
        # treat this as an actionable error.

        reason: Optional[str] = None
        notes = ""
        if assessment.status == "conflict":
            reason = "stack_conflict"
            notes = assessment.notes
        else:
            unknown_ratio = observation.unknown_ratio
            if (
                unknown_ratio is not None
                and unknown_ratio > (1.0 - confidence_threshold)
                and profile.count >= max(min_samples, 3)
            ):
                reason = "stack_unknown"
                if "stack_delta" not in missing_fields:
                    missing_fields.append("stack_delta")
            elif missing_fields and profile.count >= max(min_samples, 3):
                reason = "missing_annotations"

        if reason:
            tasks.append(
                ReviewTask(
                    key=profile.key,
                    reason=reason,
                    samples=int(profile.count),
                    missing_annotations=sorted(dict.fromkeys(missing_fields)),
                    stack_unknown_ratio=observation.unknown_ratio,
                    stack_dominant=observation.dominant,
                    stack_confidence=observation.confidence,
                    operand_hint=operand_hint,
                    operand_confidence=operand_confidence,
                    top_preceding=self._top_relationships(profile.preceding),
                    top_following=self._top_relationships(profile.following),
                    notes=notes,
                )
            )

        return tasks

    def save(self) -> None:
        payload = dict(self._data)
        payload["opcode_modes"] = {
            key: profile.to_json() for key, profile in self._profiles.items()
        }

        cleaned_annotations: Dict[str, object] = {}
        stale_keys: List[str] = []
        for key, value in self._annotations.items():
            if isinstance(value, Mapping):
                if value:
                    cleaned_annotations[key] = dict(value)
                else:
                    stale_keys.append(key)
            else:
                cleaned_annotations[key] = value
        if stale_keys:
            for key in stale_keys:
                self._annotations.pop(key, None)
        payload["annotations"] = cleaned_annotations
        payload["review_tasks"] = [
            task.to_json()
            for task in sorted(
                self._review_tasks.values(), key=lambda item: (item.key, item.reason)
            )
        ]
        payload["schema"] = self.SCHEMA_VERSION
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")

    def _merge_manual_annotations(self) -> None:
        self._manual_reference = {}

        manual_path = self.path.with_name("manual_annotations.json")
        if not manual_path.exists():
            return
        try:
            manual_data = json.loads(manual_path.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manual annotations file {manual_path} contains invalid JSON"
            ) from exc

        if not isinstance(manual_data, Mapping):
            raise ValueError(
                f"manual annotations file {manual_path} must contain an object"
            )

        for key, payload in manual_data.items():
            if not isinstance(payload, Mapping):
                continue
            key_str = str(key)
            normalized = {str(field): value for field, value in payload.items()}
            self._manual_reference[key_str] = normalized
            if normalized:
                existing_ann = self._annotations.get(key_str)
                ann_is_mutable = isinstance(existing_ann, MutableMapping)
                ann: MutableMapping[str, object]
                if ann_is_mutable:
                    ann = cast(MutableMapping[str, object], existing_ann)
                else:
                    ann = {}
                ann.update(normalized)
                if "stack_delta" in normalized:
                    ann.setdefault("stack_source", "manual")
                if not ann_is_mutable:
                    self._annotations[key_str] = ann
        self._refresh_flow_metadata_index()
        self._flow_classifier = FlowClassifier(
            self._manual_reference,
            self._flow_metadata_index,
            {key: profile.to_json() for key, profile in self._profiles.items()},
        )

    def _refresh_flow_metadata_index(self) -> None:
        """Synchronise the metadata view consumed by the flow classifier."""

        index: Dict[str, Dict[str, object]] = {}
        for key in self._annotations:
            index[key] = self._build_flow_metadata_payload(key)
        self._flow_metadata_index = index
        classifier = getattr(self, "_flow_classifier", None)
        if classifier is not None:
            for key, payload in index.items():
                classifier.update_metadata(key, payload)

    def _build_flow_metadata_payload(self, key: str) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        annotation = self._annotations.get(key)
        if isinstance(annotation, Mapping):
            payload.update(annotation)
        manual = self._manual_reference.get(key)
        if isinstance(manual, Mapping):
            for field, value in manual.items():
                payload.setdefault(field, value)
        payload.setdefault("mnemonic", payload.get("name", f"opcode_{key}"))
        return payload

    def _ensure_flow_entry(self, key: str) -> None:
        payload = self._build_flow_metadata_payload(key)
        existing = self._flow_metadata_index.get(key)
        if existing != payload:
            self._flow_metadata_index[key] = payload
            if hasattr(self, "_flow_classifier"):
                self._flow_classifier.update_metadata(key, payload)
        profile = self._profiles.get(key)
        if hasattr(self, "_flow_classifier"):
            profile_payload = profile.to_json() if profile is not None else None
            self._flow_classifier.update_profile(key, profile_payload)

    def apply_semantic_annotations(
        self,
        *,
        min_samples: int = 6,
        neighbour_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> SemanticNamingReport:
        if neighbour_threshold is None:
            neighbour_threshold = self.SEMANTIC_NEIGHBOUR_THRESHOLD
        if score_threshold is None:
            score_threshold = self.SEMANTIC_MIN_SCORE

        prototypes = self._semantic_prototypes(
            min_samples=max(1, min_samples // 2),
            neighbour_threshold=neighbour_threshold,
        )
        if not prototypes:
            return SemanticNamingReport(updates=[], matches=[])

        updates: List[AnnotationUpdate] = []
        matches: List[SemanticMatch] = []
        required_samples = max(min_samples, self.SEMANTIC_MIN_PROFILE_SAMPLES)

        for profile in sorted(self._profiles.values(), key=lambda item: item.key):
            if profile.count < required_samples:
                continue
            if profile.key in self._manual_reference:
                continue

            existing_ann = self._annotations.get(profile.key)
            ann_is_mutable = isinstance(existing_ann, MutableMapping)
            ann: MutableMapping[str, object]
            if ann_is_mutable:
                ann = cast(MutableMapping[str, object], existing_ann)
            else:
                ann = {}

            if "name" in ann:
                continue

            signature = self._semantic_signature(
                profile,
                neighbour_threshold=neighbour_threshold,
                annotation=ann,
            )

            match = self._select_semantic_match(
                signature,
                prototypes,
                score_threshold=score_threshold,
            )
            if match is None:
                continue

            prototype_key, payload, score, channel_details = match
            applied_fields: List[str] = []
            semantic_reason = f"semantic_match:{prototype_key}"

            if "stack_delta" not in ann and payload.get("stack_delta") is not None:
                ann["stack_delta"] = payload["stack_delta"]
                ann["stack_source"] = "manual"
                applied_fields.append("stack_delta")
                updates.append(
                    AnnotationUpdate(
                        key=profile.key,
                        field="stack_delta",
                        old_value=None,
                        new_value=payload.get("stack_delta"),
                        confidence=float(score),
                        samples=int(profile.count),
                        reason=semantic_reason,
                    )
                )
            if "operand_hint" not in ann and payload.get("operand_hint") is not None:
                ann["operand_hint"] = payload["operand_hint"]
                applied_fields.append("operand_hint")
                updates.append(
                    AnnotationUpdate(
                        key=profile.key,
                        field="operand_hint",
                        old_value=None,
                        new_value=payload.get("operand_hint"),
                        confidence=float(score),
                        samples=int(profile.count),
                        reason=semantic_reason,
                    )
                )

            for field in (
                "name",
                "summary",
                "description",
                "comment",
                "notes",
                "control_flow",
                "flow_target",
            ):
                value = payload.get(field)
                if value is None or field in ann:
                    continue
                ann[field] = value
                applied_fields.append(field)
                updates.append(
                    AnnotationUpdate(
                        key=profile.key,
                        field=field,
                        old_value=None,
                        new_value=value,
                        confidence=float(score),
                        samples=int(profile.count),
                        reason=semantic_reason,
                    )
                )

            previous_manual_source = ann.get("manual_source")
            if previous_manual_source != prototype_key:
                ann["manual_source"] = prototype_key
                applied_fields.append("manual_source")
                updates.append(
                    AnnotationUpdate(
                        key=profile.key,
                        field="manual_source",
                        old_value=previous_manual_source,
                        new_value=prototype_key,
                        confidence=float(score),
                        samples=int(profile.count),
                        reason=semantic_reason,
                    )
                )

            prev_basis_raw = ann.get("semantic_basis")
            if isinstance(prev_basis_raw, Mapping):
                previous_basis: object = dict(prev_basis_raw)
            else:
                previous_basis = prev_basis_raw
            basis = {
                "prototype": prototype_key,
                "score": round(float(score), 3),
                "channels": dict(channel_details),
            }
            if previous_basis != basis:
                ann["semantic_basis"] = basis
                applied_fields.append("semantic_basis")
                updates.append(
                    AnnotationUpdate(
                        key=profile.key,
                        field="semantic_basis",
                        old_value=previous_basis,
                        new_value=basis,
                        confidence=float(score),
                        samples=int(profile.count),
                        reason=semantic_reason,
                    )
                )

            if applied_fields and not ann_is_mutable:
                self._annotations[profile.key] = ann

            if applied_fields:
                matches.append(
                    SemanticMatch(
                        key=profile.key,
                        prototype=prototype_key,
                        score=float(score),
                        channels=dict(channel_details),
                        applied_fields=tuple(sorted(set(applied_fields))),
                    )
                )

        return SemanticNamingReport(updates=updates, matches=matches)

    def _semantic_signature(
        self,
        profile: OpcodeProfile,
        *,
        neighbour_threshold: float,
        override: Optional[Mapping[str, object]] = None,
        annotation: Optional[Mapping[str, object]] = None,
    ) -> SemanticSignature:
        dominant_delta, dominant_ratio = self._dominant(profile.stack_deltas)
        stack_confidence = dominant_ratio if dominant_ratio else None

        if override:
            override_delta = _coerce_stack_delta_numeric(override.get("stack_delta"))
            if override_delta is not None:
                dominant_delta = float(override_delta)
                stack_confidence = 1.0

        if dominant_delta is None and annotation is not None:
            override_delta = _coerce_stack_delta_numeric(annotation.get("stack_delta"))
            if override_delta is not None:
                dominant_delta = float(override_delta)
                try:
                    raw_conf = annotation.get("stack_confidence")
                    stack_confidence = float(raw_conf) if raw_conf is not None else 1.0
                except (TypeError, ValueError):
                    stack_confidence = 1.0

        operand_hint, operand_ratio = self._dominant_operand_type(profile)
        operand_confidence = operand_ratio if operand_ratio else None

        if override and override.get("operand_hint") is not None:
            operand_hint = str(override.get("operand_hint"))
            operand_confidence = 1.0

        if operand_hint is None and annotation is not None:
            operand_override = annotation.get("operand_hint")
            if operand_override is not None:
                operand_hint = str(operand_override)
                try:
                    raw_conf = annotation.get("operand_confidence")
                    operand_confidence = (
                        float(raw_conf) if raw_conf is not None else operand_confidence or 1.0
                    )
                except (TypeError, ValueError):
                    operand_confidence = operand_confidence or 1.0

        flow_target: Optional[str] = None
        if override and override.get("flow_target") is not None:
            flow_target = str(override.get("flow_target"))
        elif annotation is not None and annotation.get("flow_target") is not None:
            flow_target = str(annotation.get("flow_target"))

        preceding = self._stable_neighbour_ratios(
            profile.preceding, threshold=neighbour_threshold
        )
        following = self._stable_neighbour_ratios(
            profile.following, threshold=neighbour_threshold
        )

        return SemanticSignature(
            key=profile.key,
            stack_delta=dominant_delta,
            stack_confidence=stack_confidence,
            operand_hint=operand_hint,
            operand_confidence=operand_confidence,
            preceding=preceding,
            following=following,
            flow_target=flow_target,
        )

    @staticmethod
    def _stable_neighbour_ratios(
        counter: Counter, *, threshold: float
    ) -> Dict[str, float]:
        total = sum(counter.values())
        if total <= 0:
            return {}
        stable: Dict[str, float] = {}
        for key, count in counter.items():
            if count <= 0:
                continue
            ratio = count / total
            if ratio >= threshold:
                stable[str(key)] = ratio
        return stable

    def _semantic_prototypes(
        self,
        *,
        min_samples: int,
        neighbour_threshold: float,
    ) -> Dict[str, Tuple[SemanticSignature, Mapping[str, object]]]:
        prototypes: Dict[str, Tuple[SemanticSignature, Mapping[str, object]]] = {}
        for key, payload in self._manual_reference.items():
            profile = self._profiles.get(key)
            if profile is None:
                profile = OpcodeProfile(key)
            if profile.count < min_samples and not payload:
                continue
            annotation_raw = self._annotations.get(key)
            annotation = annotation_raw if isinstance(annotation_raw, Mapping) else None
            signature = self._semantic_signature(
                profile,
                neighbour_threshold=neighbour_threshold,
                override=payload,
                annotation=annotation,
            )
            prototypes[key] = (signature, payload)
        return prototypes

    def _select_semantic_match(
        self,
        signature: SemanticSignature,
        prototypes: Mapping[str, Tuple[SemanticSignature, Mapping[str, object]]],
        *,
        score_threshold: float,
    ) -> Optional[Tuple[str, Mapping[str, object], float, Dict[str, str]]]:
        candidates: List[Tuple[float, Dict[str, str], str, Mapping[str, object]]] = []
        for proto_key, (proto_signature, payload) in prototypes.items():
            similarity = self._semantic_similarity(signature, proto_signature)
            if similarity is None:
                continue
            score, channels = similarity
            if score < score_threshold:
                continue
            candidates.append((score, channels, proto_key, payload))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > 1 and (
            candidates[0][0] - candidates[1][0] < self.SEMANTIC_SCORE_MARGIN
        ):
            return None

        top_score, top_channels, top_key, top_payload = candidates[0]
        return top_key, top_payload, top_score, dict(top_channels)

    def _semantic_similarity(
        self,
        candidate: SemanticSignature,
        prototype: SemanticSignature,
    ) -> Optional[Tuple[float, Dict[str, str]]]:
        total_weight = 0.0
        weighted_score = 0.0
        details: Dict[str, str] = {}

        stack_result = self._stack_similarity(
            candidate.stack_delta,
            candidate.stack_confidence,
            prototype.stack_delta,
            prototype.stack_confidence,
        )
        if stack_result is not None:
            score, detail, consider_weight = stack_result
            if consider_weight:
                total_weight += self.SEMANTIC_STACK_WEIGHT
                weighted_score += score * self.SEMANTIC_STACK_WEIGHT
            details["stack"] = detail

        operand_result = self._operand_similarity(
            candidate.operand_hint,
            candidate.operand_confidence,
            prototype.operand_hint,
            prototype.operand_confidence,
        )
        if operand_result is not None:
            score, detail = operand_result
            total_weight += self.SEMANTIC_OPERAND_WEIGHT
            weighted_score += score * self.SEMANTIC_OPERAND_WEIGHT
            details["operand"] = detail

        context_result = self._context_similarity(
            candidate.preceding,
            candidate.following,
            prototype.preceding,
            prototype.following,
        )
        if context_result is not None:
            score, detail = context_result
            total_weight += self.SEMANTIC_CONTEXT_WEIGHT
            weighted_score += score * self.SEMANTIC_CONTEXT_WEIGHT
            details["context"] = detail

        flow_result = self._flow_similarity(
            candidate.flow_target,
            prototype.flow_target,
        )
        if flow_result is not None:
            score, detail = flow_result
            total_weight += self.SEMANTIC_FLOW_WEIGHT
            weighted_score += score * self.SEMANTIC_FLOW_WEIGHT
            details["flow"] = detail

        if total_weight == 0.0:
            return None

        return weighted_score / total_weight, details

    def _stack_similarity(
        self,
        candidate_delta: Optional[float],
        candidate_confidence: Optional[float],
        prototype_delta: Optional[float],
        prototype_confidence: Optional[float],
    ) -> Optional[Tuple[float, str, bool]]:
        if candidate_delta is None and prototype_delta is None:
            return None
        if candidate_delta is None or prototype_delta is None:
            fmt = lambda value: self._format_stack_delta(value) if value is not None else "—"
            detail = f"Δ={fmt(candidate_delta)} vs {fmt(prototype_delta)}"
            return 0.0, detail, False

        diff = abs(candidate_delta - prototype_delta)
        closeness = max(0.0, 1.0 - min(diff, 1.0))
        if not self._stack_delta_close(candidate_delta, prototype_delta):
            closeness *= 0.25

        min_conf = min(
            float(candidate_confidence or 0.0),
            float(prototype_confidence or 0.0),
        )
        score = closeness * (0.5 + 0.5 * min(1.0, min_conf))
        detail = (
            f"Δ={self._format_stack_delta(candidate_delta)} vs "
            f"{self._format_stack_delta(prototype_delta)} (Δ={diff:.2f}, conf≈{min_conf:.2f})"
        )
        return score, detail, True

    def _operand_similarity(
        self,
        candidate_operand: Optional[str],
        candidate_confidence: Optional[float],
        prototype_operand: Optional[str],
        prototype_confidence: Optional[float],
    ) -> Optional[Tuple[float, str]]:
        if candidate_operand is None and prototype_operand is None:
            return None
        if candidate_operand is None or prototype_operand is None:
            detail = f"operand {candidate_operand or '—'} vs {prototype_operand or '—'}"
            return 0.0, detail
        if candidate_operand != prototype_operand:
            detail = f"operand {candidate_operand} vs {prototype_operand}"
            return 0.0, detail

        min_conf = min(
            float(candidate_confidence or 0.0),
            float(prototype_confidence or 0.0),
        )
        score = 0.5 + 0.5 * min_conf
        detail = f"operand={candidate_operand} (conf≈{min_conf:.2f})"
        return score, detail

    def _context_similarity(
        self,
        candidate_pre: Dict[str, float],
        candidate_post: Dict[str, float],
        prototype_pre: Dict[str, float],
        prototype_post: Dict[str, float],
    ) -> Optional[Tuple[float, str]]:
        components: List[str] = []
        scores: List[float] = []

        if candidate_pre and prototype_pre:
            score, detail = self._neighbour_similarity(candidate_pre, prototype_pre)
            components.append(f"L{detail}")
            scores.append(score)

        if candidate_post and prototype_post:
            score, detail = self._neighbour_similarity(candidate_post, prototype_post)
            components.append(f"R{detail}")
            scores.append(score)

        if not scores:
            return None

        combined_score = sum(scores) / len(scores)
        return combined_score, "; ".join(components)

    def _neighbour_similarity(
        self, candidate: Dict[str, float], prototype: Dict[str, float]
    ) -> Tuple[float, str]:
        if not candidate or not prototype:
            return 0.0, "0/0"

        candidate_keys = set(candidate)
        prototype_keys = set(prototype)
        intersection = candidate_keys & prototype_keys
        union = candidate_keys | prototype_keys
        if not union:
            return 0.0, "0/0"
        if not intersection:
            return 0.0, f"0/{len(union)}"

        base = len(intersection) / len(union)
        ratio_alignment = sum(
            1.0 - min(abs(candidate[key] - prototype[key]), 1.0)
            for key in intersection
        ) / len(intersection)
        score = base * (0.5 + 0.5 * ratio_alignment)
        detail = f"{len(intersection)}/{len(union)} {','.join(sorted(intersection))}"
        return score, detail

    def _flow_similarity(
        self,
        candidate_flow: Optional[str],
        prototype_flow: Optional[str],
    ) -> Optional[Tuple[float, str]]:
        if candidate_flow is None and prototype_flow is None:
            return None
        if candidate_flow is None or prototype_flow is None:
            return 0.0, f"flow {candidate_flow or '—'} vs {prototype_flow or '—'}"
        if candidate_flow == prototype_flow:
            return 1.0, f"flow={candidate_flow}"
        return 0.0, f"flow {candidate_flow}->{prototype_flow}"

    @staticmethod
    def _relationship_similarity(counter_a: Counter, counter_b: Counter) -> float:
        if not counter_a or not counter_b:
            return 0.0
        keys_a = {key for key, value in counter_a.items() if value > 0}
        keys_b = {key for key, value in counter_b.items() if value > 0}
        if not keys_a or not keys_b:
            return 0.0
        intersection = len(keys_a & keys_b)
        union = len(keys_a | keys_b)
        if union == 0:
            return 0.0
        return intersection / union

    @classmethod
    def _upgrade_schema(
        cls, data: MutableMapping[str, object], schema: Optional[object]
    ) -> MutableMapping[str, object]:
        if schema is None:
            current_version = 0
        elif isinstance(schema, int):
            current_version = schema
        else:
            raise ValueError(
                "Knowledge base schema is invalid: expected integer or missing value"
            )

        if current_version > cls.SCHEMA_VERSION:
            raise ValueError(
                "Knowledge base schema version "
                f"{current_version} is newer than supported {cls.SCHEMA_VERSION}"
            )

        upgraded: MutableMapping[str, object] = data
        while current_version < cls.SCHEMA_VERSION:
            migrator = cls._MIGRATIONS.get(current_version)
            if migrator is None:
                raise ValueError(
                    f"No migration path from schema {current_version} to {cls.SCHEMA_VERSION}"
                )
            upgraded = migrator(upgraded)
            current_version += 1

        upgraded["schema"] = cls.SCHEMA_VERSION
        return upgraded

    @staticmethod
    def _migrate_v0_to_v1(data: MutableMapping[str, object]) -> MutableMapping[str, object]:
        migrated: MutableMapping[str, object] = dict(data)

        opcode_modes_raw = migrated.get("opcode_modes")
        if isinstance(opcode_modes_raw, Mapping):
            opcode_modes = {
                str(key): dict(value) if isinstance(value, Mapping) else {}
                for key, value in opcode_modes_raw.items()
            }
        else:
            legacy_profiles = migrated.get("profiles")
            if isinstance(legacy_profiles, Mapping):
                opcode_modes = {
                    str(key): dict(value) if isinstance(value, Mapping) else {}
                    for key, value in legacy_profiles.items()
                }
            else:
                opcode_modes = {}
        migrated["opcode_modes"] = opcode_modes
        migrated.pop("profiles", None)

        annotations = migrated.get("annotations")
        if not isinstance(annotations, Mapping):
            migrated["annotations"] = {}
        else:
            migrated["annotations"] = dict(annotations)

        return migrated


KnowledgeBase._MIGRATIONS = {0: KnowledgeBase._migrate_v0_to_v1}
