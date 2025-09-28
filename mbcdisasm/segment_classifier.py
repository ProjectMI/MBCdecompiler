"""Heuristic, multi-factor segment classification utilities."""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

from .adb import SegmentDescriptor
from .instruction import WORD_SIZE, read_instructions
from .knowledge import KnowledgeBase

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .cfg import ControlFlowGraphBuilder
    from .emulator import Emulator
    from .mbc import Segment


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AutoLabel:
    """Lightweight summary of an automatically inferred label."""

    classification: str
    confidence: float
    reason: str


class SegmentClassifier:
    """Combine statistical features, CFG analysis and emulation feedback."""

    PROFILE_VERSION = 1

    def __init__(
        self,
        knowledge: Optional[KnowledgeBase] = None,
        *,
        opcode_ngram: int = 3,
        entropy_window: int = 64,
        learning_rate: float = 0.25,
        max_training_instructions: int = 160,
        weights_profile: Optional[Path] = None,
    ) -> None:
        self.knowledge = knowledge
        self.opcode_ngram = max(1, opcode_ngram)
        self.entropy_window = max(8, entropy_window)
        self.learning_rate = learning_rate
        self.max_training_instructions = max(16, max_training_instructions)
        self.weights: Dict[str, float] = {
            "instruction_density": 1.9,
            "opcode_entropy": 1.3,
            "ngram_diversity": 1.1,
            "byte_entropy": 0.8,
            "entropy_std": 0.5,
            "mode_diversity": 0.4,
            "ascii_ratio": -1.4,
            "zero_ratio": -1.1,
            "max_byte_fraction": -0.9,
            "remainder_ratio": -0.6,
        }
        self.bias = -0.6
        self._cfg_builder: Optional["ControlFlowGraphBuilder"] = None
        self._emulator: Optional["Emulator"] = None
        self._uncertainty_level = 0.0
        self._cooling_decay = 0.9
        self._max_uncertainty = 6.0
        self.profile_path = self._resolve_profile_path(weights_profile)
        self._profile_dirty = False
        if self.profile_path is not None:
            self._load_profile()

    def classify(self, descriptor: SegmentDescriptor, data: bytes) -> str:
        """Return a best-effort classification label for ``data``."""

        if not data:
            return "empty"

        self._decay_uncertainty()

        remainder = len(data) % WORD_SIZE
        usable = len(data) - remainder
        if usable <= 0:
            ascii_ratio = sum(32 <= b < 127 for b in data) / len(data)
            return "strings" if ascii_ratio > 0.85 else "blob"

        instructions, _ = read_instructions(data, descriptor.start)
        features = self._extract_features(data, instructions, remainder)
        auto_label = self._auto_label(descriptor, data, instructions, features)
        if auto_label is not None:
            self._train(features, auto_label)

        probability = self._score_to_confidence(self._score(features))
        classification = self._decide_label(
            features,
            probability,
            auto_label,
            len(instructions),
        )
        uncertain = self._log_if_uncertain(
            descriptor, probability, features, auto_label, classification
        )
        self._update_uncertainty(probability, uncertain)
        return classification

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def _extract_features(
        self,
        data: bytes,
        instructions: Sequence[object],
        remainder: int,
    ) -> Dict[str, float]:
        length = len(data)
        ascii_ratio = sum(32 <= b < 127 for b in data) / length
        zero_ratio = data.count(0) / length
        byte_counts = Counter(data)
        max_byte_fraction = max(byte_counts.values()) / length if byte_counts else 0.0
        byte_entropy_raw = self._entropy_from_counts(byte_counts.values(), length)
        entropy_profile = self._entropy_profile(data)
        entropy_std_raw = (
            statistics.pstdev(entropy_profile) if len(entropy_profile) > 1 else 0.0
        )

        instruction_count = len(instructions)
        instruction_density = (
            min(1.0, (instruction_count * WORD_SIZE) / length) if length else 0.0
        )

        opcode_entropy_norm = 0.0
        opcode_entropy_raw = 0.0
        ngram_diversity = 0.0
        mode_diversity = 0.0
        if instruction_count:
            opcodes = [getattr(instr, "opcode") for instr in instructions]
            modes = [getattr(instr, "mode") for instr in instructions]
            opcode_counts = Counter(opcodes)
            opcode_entropy_raw = self._entropy_from_counts(
                opcode_counts.values(), instruction_count
            )
            if len(opcode_counts) > 1:
                opcode_entropy_norm = opcode_entropy_raw / math.log2(
                    min(256, len(opcode_counts))
                )
            ngram_diversity = self._ngram_diversity(opcodes)
            mode_diversity = len(set(modes)) / instruction_count

        return {
            "ascii_ratio": ascii_ratio,
            "zero_ratio": zero_ratio,
            "max_byte_fraction": max_byte_fraction,
            "byte_entropy": byte_entropy_raw / 8.0,
            "byte_entropy_raw": byte_entropy_raw,
            "entropy_std": entropy_std_raw / 4.0,
            "entropy_std_raw": entropy_std_raw,
            "instruction_density": instruction_density,
            "opcode_entropy": opcode_entropy_norm,
            "opcode_entropy_raw": opcode_entropy_raw,
            "ngram_diversity": ngram_diversity,
            "mode_diversity": mode_diversity,
            "remainder_ratio": remainder / length if length else 0.0,
            "instruction_count": float(instruction_count),
            "length": float(length),
        }

    def _entropy_profile(self, data: bytes) -> Sequence[float]:
        if not data:
            return [0.0]
        window = min(self.entropy_window, len(data))
        if window <= 0:
            return [0.0]
        profile = []
        for offset in range(0, len(data), window):
            chunk = data[offset : offset + window]
            counts = Counter(chunk)
            profile.append(self._entropy_from_counts(counts.values(), len(chunk)))
        return profile

    def _entropy_from_counts(self, counts: Iterable[int], total: int) -> float:
        if total <= 0:
            return 0.0
        entropy = 0.0
        for count in counts:
            if count <= 0:
                continue
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy

    def _ngram_diversity(self, opcodes: Sequence[int]) -> float:
        n = min(self.opcode_ngram, len(opcodes))
        if n <= 0 or len(opcodes) < n:
            return 0.0
        total = len(opcodes) - n + 1
        unique = {
            tuple(opcodes[idx : idx + n])
            for idx in range(total)
        }
        return len(unique) / total if total else 0.0

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------
    def _auto_label(
        self,
        descriptor: SegmentDescriptor,
        data: bytes,
        instructions: Sequence[object],
        features: Dict[str, float],
    ) -> Optional[_AutoLabel]:
        if self.knowledge is None or not instructions:
            return None

        cfg_builder, emulator = self._ensure_tooling()
        if cfg_builder is None or emulator is None:
            return None

        sample = min(self.max_training_instructions, len(instructions))
        segment = self._build_segment(descriptor, data)
        cfg = cfg_builder.build(segment, max_instructions=sample)
        block_count = len(cfg.blocks)
        edge_count = sum(len(block.successors) for block in cfg.blocks.values())
        branching_blocks = sum(
            1 for block in cfg.blocks.values() if len(block.successors) > 1
        )

        report = emulator.simulate_segment(segment, max_instructions=sample)
        reliable = report.is_reliable and report.total_instructions >= min(4, sample)

        if reliable:
            flow_density = edge_count / max(1, block_count)
            if block_count >= 2 and (flow_density >= 0.9 or branching_blocks >= 1):
                return _AutoLabel("code", 0.9, "reliable-flow")
            if flow_density >= 0.7 and report.unknown_delta_ratio < 0.2:
                return _AutoLabel("code", 0.8, "stable-flow")

        if not reliable:
            if report.total_instructions == 0:
                if features["ascii_ratio"] > 0.75:
                    return _AutoLabel("strings", 0.75, "no-instr-ascii")
                if features["zero_ratio"] > 0.45:
                    return _AutoLabel("tables", 0.7, "no-instr-zero")
            if report.unknown_delta_ratio > 0.7 and block_count <= 1:
                if features["ascii_ratio"] > 0.65:
                    return _AutoLabel("strings", 0.7, "unknown-ascii")
                if features["zero_ratio"] > 0.5:
                    return _AutoLabel("tables", 0.7, "unknown-zero")

        return None

    def _score(self, features: Dict[str, float]) -> float:
        score = self.bias
        for key, weight in self.weights.items():
            score += weight * features.get(key, 0.0)
        return score

    def _score_to_confidence(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-score))

    def _decide_label(
        self,
        features: Dict[str, float],
        probability: float,
        auto_label: Optional[_AutoLabel],
        instruction_count: int,
    ) -> str:
        ascii_ratio = features["ascii_ratio"]
        zero_ratio = features["zero_ratio"]
        entropy_raw = features["byte_entropy_raw"]
        instruction_density = features["instruction_density"]
        remainder_ratio = features["remainder_ratio"]

        if auto_label is not None and auto_label.confidence >= 0.65:
            return auto_label.classification

        if probability >= 0.62 or (
            probability >= 0.55
            and instruction_density > 0.4
            and features["opcode_entropy"] > 0.3
        ):
            return "code"

        if instruction_density < 0.2 and probability < 0.45:
            if ascii_ratio > 0.7:
                return "strings"
            if zero_ratio > 0.4 and entropy_raw < 3.0:
                return "tables"
            if entropy_raw < 2.0 or remainder_ratio > 0.3:
                return "blob"

        if ascii_ratio > 0.85 and probability < 0.55:
            return "strings"

        if zero_ratio > 0.5 and entropy_raw < 3.5:
            return "tables"

        if probability <= 0.35:
            return "blob"

        if auto_label is not None:
            return auto_label.classification

        if (
            instruction_density >= 0.3
            and entropy_raw >= 3.0
            and features["opcode_entropy"] >= 0.35
        ):
            return "code"

        return "blob"

    def _has_confident_data_features(
        self, features: Dict[str, float], classification: str
    ) -> bool:
        ascii_ratio = features["ascii_ratio"]
        zero_ratio = features["zero_ratio"]
        entropy_raw = features["byte_entropy_raw"]
        opcode_entropy = features["opcode_entropy"]
        instruction_density = features["instruction_density"]
        max_byte_fraction = features["max_byte_fraction"]
        entropy_std = features["entropy_std_raw"]

        if classification == "strings":
            return ascii_ratio >= 0.7 and entropy_raw <= 5.0

        if classification == "tables":
            dominant_zero = zero_ratio >= 0.45
            compact_distribution = entropy_raw <= 4.0 or max_byte_fraction >= 0.3
            return dominant_zero and compact_distribution

        if classification == "blob":
            low_opcode_var = opcode_entropy <= 0.1 and instruction_density >= 0.75
            low_entropy = entropy_raw <= 3.0
            has_dominant_byte = max_byte_fraction >= 0.22 and entropy_std <= 0.25
            return low_opcode_var or low_entropy or has_dominant_byte

        return False

    def _log_if_uncertain(
        self,
        descriptor: SegmentDescriptor,
        probability: float,
        features: Dict[str, float],
        auto_label: Optional[_AutoLabel],
        classification: str,
    ) -> bool:
        if classification == "code":
            return False
        if auto_label is not None and auto_label.confidence >= 0.5:
            return False
        if self._has_confident_data_features(features, classification):
            return False
        if 0.4 <= probability <= 0.6:
            logger.warning(
                "Segment %d classification uncertain: prob=%.2f ascii=%.2f zero=%.2f entropy=%.2f",
                descriptor.index,
                probability,
                features["ascii_ratio"],
                features["zero_ratio"],
                features["byte_entropy_raw"],
            )
            return True
        return False

    def _train(self, features: Dict[str, float], label: _AutoLabel) -> None:
        target = 1 if label.classification == "code" else 0
        score = self._score(features)
        prediction = 1 if score >= 0.0 else 0
        error = target - prediction
        if error == 0:
            return
        adjustment = (
            self.learning_rate
            * label.confidence
            * error
            * self._cooling_multiplier()
        )
        for key in self.weights:
            self.weights[key] += adjustment * features.get(key, 0.0)
        self.bias += adjustment
        self._profile_dirty = True

    def _ensure_tooling(self) -> Tuple[Optional["ControlFlowGraphBuilder"], Optional["Emulator"]]:
        if self.knowledge is None:
            return None, None
        if self._cfg_builder is None:
            from .cfg import ControlFlowGraphBuilder

            self._cfg_builder = ControlFlowGraphBuilder(self.knowledge)
        if self._emulator is None:
            from .emulator import Emulator

            self._emulator = Emulator(self.knowledge)
        return self._cfg_builder, self._emulator

    def _build_segment(self, descriptor: SegmentDescriptor, data: bytes) -> "Segment":
        from .mbc import Segment

        return Segment(descriptor, data, "code")

    def save_profile(self) -> None:
        """Persist the adaptive weights alongside the knowledge base."""

        if self.profile_path is None or not self._profile_dirty:
            return
        payload = {
            "version": self.PROFILE_VERSION,
            "weights": self.weights,
            "bias": self.bias,
            "uncertainty_level": self._uncertainty_level,
        }
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")
        self._profile_dirty = False

    # ------------------------------------------------------------------
    # Internal helpers for persistence and cooling
    # ------------------------------------------------------------------
    def _resolve_profile_path(self, explicit: Optional[Path]) -> Optional[Path]:
        if explicit is not None:
            return explicit
        if self.knowledge is None:
            return None
        base = self.knowledge.path if hasattr(self.knowledge, "path") else None
        if base is None:
            return None
        return base.with_name(f"{base.stem}_segment_classifier.json")

    def _load_profile(self) -> None:
        if self.profile_path is None or not self.profile_path.exists():
            return
        try:
            payload = json.loads(self.profile_path.read_text("utf-8"))
        except (OSError, ValueError, TypeError):
            logger.warning("failed to load segment classifier profile from %s", self.profile_path)
            return
        if payload.get("version") != self.PROFILE_VERSION:
            logger.warning(
                "segment classifier profile %s has incompatible version", self.profile_path
            )
            return
        weights = payload.get("weights")
        bias = payload.get("bias")
        if isinstance(weights, dict):
            try:
                self.weights = {str(k): float(v) for k, v in weights.items()}
            except (TypeError, ValueError):
                logger.warning(
                    "segment classifier profile %s contains invalid weights", self.profile_path
                )
                return
        if isinstance(bias, (int, float)):
            self.bias = float(bias)
        uncertainty = payload.get("uncertainty_level")
        if isinstance(uncertainty, (int, float)) and math.isfinite(uncertainty):
            self._uncertainty_level = max(0.0, float(uncertainty))
        self._profile_dirty = False

    def _decay_uncertainty(self) -> None:
        if self._uncertainty_level <= 0.0:
            return
        self._uncertainty_level *= self._cooling_decay
        if self._uncertainty_level < 1e-3:
            self._uncertainty_level = 0.0

    def _update_uncertainty(self, probability: float, uncertain: bool) -> None:
        if not uncertain:
            return
        closeness = max(0.0, 0.5 - abs(probability - 0.5)) * 2.0
        self._uncertainty_level = min(
            self._max_uncertainty,
            self._uncertainty_level + closeness,
        )

    def _cooling_multiplier(self) -> float:
        if self._uncertainty_level <= 0.0:
            return 1.0
        return 1.0 / (1.0 + self._uncertainty_level)

