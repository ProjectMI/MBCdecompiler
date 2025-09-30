"""Diagnostic helpers for pipeline analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .report import PipelineBlock


@dataclass
class DiagnosticEntry:
    """Single diagnostic item."""

    severity: str
    message: str
    block: PipelineBlock
    hints: Tuple[str, ...] = tuple()

    def describe(self) -> str:
        prefix = {"info": "ℹ", "warning": "⚠", "error": "✖"}.get(self.severity, "•")
        return f"{prefix} 0x{self.block.start_offset:08X}: {self.message}"


@dataclass
class DiagnosticSummary:
    """Aggregate view of diagnostic severities."""

    info: int = 0
    warning: int = 0
    error: int = 0

    def register(self, severity: str) -> None:
        if severity == "info":
            self.info += 1
        elif severity == "warning":
            self.warning += 1
        elif severity == "error":
            self.error += 1

    def describe(self) -> str:
        return f"info={self.info} warning={self.warning} error={self.error}"


@dataclass
class DiagnosticReport:
    """Collection of diagnostic entries."""

    entries: Tuple[DiagnosticEntry, ...]
    summary: DiagnosticSummary

    def filter(self, severity: str) -> Tuple[DiagnosticEntry, ...]:
        return tuple(entry for entry in self.entries if entry.severity == severity)

    def describe(self) -> str:
        lines = ["Diagnostics:", "  summary=" + self.summary.describe()]
        for entry in self.entries:
            lines.append("  " + entry.describe())
        return "\n".join(lines)


@dataclass
class DiagnosticSettings:
    """Parameters steering diagnostic generation."""

    low_confidence_threshold: float = 0.35
    stack_spread_threshold: int = 4
    literal_mismatch_threshold: float = -0.05
    max_entries: int = 64


class DiagnosticBuilder:
    """Produce diagnostics from analysed pipeline blocks."""

    def __init__(self, settings: Optional[DiagnosticSettings] = None) -> None:
        self.settings = settings or DiagnosticSettings()

    def evaluate(self, blocks: Sequence[PipelineBlock]) -> DiagnosticReport:
        entries: List[DiagnosticEntry] = []
        summary = DiagnosticSummary()
        for block in blocks:
            block_entries = self._evaluate_block(block)
            entries.extend(block_entries)
            for entry in block_entries:
                summary.register(entry.severity)
            if len(entries) >= self.settings.max_entries:
                break
        return DiagnosticReport(entries=tuple(entries), summary=summary)

    def _evaluate_block(self, block: PipelineBlock) -> List[DiagnosticEntry]:
        entries: List[DiagnosticEntry] = []
        if block.confidence < self.settings.low_confidence_threshold:
            entries.append(
                DiagnosticEntry(
                    severity="warning",
                    message=f"low confidence classification ({block.confidence:.2f})",
                    block=block,
                    hints=tuple(block.notes),
                )
            )
        spread = block.stack.maximum - block.stack.minimum
        if spread >= self.settings.stack_spread_threshold:
            entries.append(
                DiagnosticEntry(
                    severity="info",
                    message=f"wide stack range ({spread})",
                    block=block,
                    hints=tuple(block.notes),
                )
            )
        for note in block.notes:
            if "literal" in note and "reduced" in note:
                entries.append(
                    DiagnosticEntry(
                        severity="info",
                        message="literal block reduced stack",
                        block=block,
                        hints=(note,),
                    )
                )
        return entries
