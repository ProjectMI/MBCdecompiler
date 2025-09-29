"""Static stack profile analysis for IR programs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .ir import IRBlock, IRProgram
from .vm_analysis import estimate_stack_io


@dataclass(frozen=True)
class BlockStackProfile:
    """Describes the inferred stack requirements for a single block."""

    block_start: int
    instruction_count: int
    entry_requirement: int
    min_depth: int
    max_depth: int
    net_delta: int
    warnings: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "block_start": self.block_start,
            "instruction_count": self.instruction_count,
            "entry_requirement": self.entry_requirement,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "net_delta": self.net_delta,
            "warnings": list(self.warnings),
        }

    def render(self) -> List[str]:
        header = (
            f"block 0x{self.block_start:06X}: instr={self.instruction_count} "
            f"entry={self.entry_requirement} delta={self.net_delta}"
        )
        lines = [header]
        lines.append(
            f"  depth range: min={self.min_depth} max={self.max_depth}"
        )
        if self.warnings:
            for warning in self.warnings:
                lines.append(f"  warning: {warning}")
        return lines


@dataclass(frozen=True)
class StackProfileSummary:
    """Aggregated statistics derived from block profiles."""

    total_blocks: int
    blocks_with_entry: int
    max_entry_requirement: int
    max_depth: int
    min_depth: int
    positive_deltas: int
    negative_deltas: int

    def to_dict(self) -> dict:
        return {
            "total_blocks": self.total_blocks,
            "blocks_with_entry": self.blocks_with_entry,
            "max_entry_requirement": self.max_entry_requirement,
            "max_depth": self.max_depth,
            "min_depth": self.min_depth,
            "positive_deltas": self.positive_deltas,
            "negative_deltas": self.negative_deltas,
        }

    def render_lines(self) -> List[str]:
        return [
            f"blocks analysed: {self.total_blocks}",
            f"blocks requiring entry stack: {self.blocks_with_entry}",
            f"maximum entry requirement: {self.max_entry_requirement}",
            f"depth bounds: min={self.min_depth} max={self.max_depth}",
            f"net deltas: positive={self.positive_deltas} negative={self.negative_deltas}",
        ]


@dataclass
class StackProfileReport:
    """Composite report combining per-block profiles with a summary."""

    segment_index: int
    profiles: List[BlockStackProfile]
    summary: StackProfileSummary

    def to_dict(self) -> dict:
        return {
            "segment_index": self.segment_index,
            "summary": self.summary.to_dict(),
            "profiles": [profile.to_dict() for profile in self.profiles],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render_text(self) -> str:
        lines = [f"segment {self.segment_index} stack profile"]
        lines.extend(self.summary.render_lines())
        lines.append("")
        for profile in self.profiles:
            lines.extend(profile.render())
        return "\n".join(lines).rstrip() + "\n"


class StackProfileAnalyzer:
    """Inspect IR blocks and infer stack usage characteristics."""

    def __init__(self, program: IRProgram) -> None:
        self._program = program

    def analyse(self) -> List[BlockStackProfile]:
        profiles = [self._analyse_block(block) for block in self._iter_blocks()]
        return profiles

    def summary(self, profiles: Optional[Sequence[BlockStackProfile]] = None) -> StackProfileSummary:
        profiles = list(profiles or self.analyse())
        if not profiles:
            return StackProfileSummary(0, 0, 0, 0, 0, 0, 0)
        blocks_with_entry = sum(1 for profile in profiles if profile.entry_requirement > 0)
        max_entry = max((profile.entry_requirement for profile in profiles), default=0)
        max_depth = max((profile.max_depth for profile in profiles), default=0)
        min_depth = min((profile.min_depth for profile in profiles), default=0)
        positive_deltas = sum(1 for profile in profiles if profile.net_delta > 0)
        negative_deltas = sum(1 for profile in profiles if profile.net_delta < 0)
        return StackProfileSummary(
            total_blocks=len(profiles),
            blocks_with_entry=blocks_with_entry,
            max_entry_requirement=max_entry,
            max_depth=max_depth,
            min_depth=min_depth,
            positive_deltas=positive_deltas,
            negative_deltas=negative_deltas,
        )

    def report(self) -> StackProfileReport:
        profiles = self.analyse()
        return StackProfileReport(
            segment_index=self._program.segment_index,
            profiles=profiles,
            summary=self.summary(profiles),
        )

    def _iter_blocks(self) -> Iterable[IRBlock]:
        for start in sorted(self._program.blocks):
            yield self._program.blocks[start]

    def _analyse_block(self, block: IRBlock) -> BlockStackProfile:
        depth = 0
        min_depth = 0
        max_depth = 0
        warnings: List[str] = []
        for instruction in block.instructions:
            semantics = instruction.semantics
            inputs, outputs = estimate_stack_io(semantics)
            depth -= inputs
            if depth < min_depth:
                min_depth = depth
            depth += outputs
            if depth > max_depth:
                max_depth = depth
        entry_requirement = max(0, -min_depth)
        if entry_requirement > 0:
            warnings.append(f"requires {entry_requirement} value(s) on entry")
        if depth < 0:
            warnings.append(f"net negative delta {depth}")
        elif depth > 0:
            warnings.append(f"net positive delta {depth}")
        return BlockStackProfile(
            block_start=block.start,
            instruction_count=len(block.instructions),
            entry_requirement=entry_requirement,
            min_depth=min_depth,
            max_depth=max_depth,
            net_delta=depth,
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def analyse_stack_profiles(program: IRProgram) -> List[BlockStackProfile]:
    """Return stack profiles for every block in ``program``."""

    return StackProfileAnalyzer(program).analyse()


def build_stack_profile_report(program: IRProgram) -> StackProfileReport:
    """Return a :class:`StackProfileReport` for ``program``."""

    return StackProfileAnalyzer(program).report()


def build_stack_profile_reports(programs: Iterable[IRProgram]) -> List[StackProfileReport]:
    """Return reports for each program in ``programs``."""

    return [build_stack_profile_report(program) for program in programs]


def render_stack_profiles(profiles: Sequence[BlockStackProfile]) -> str:
    """Render profiles to a human readable string."""

    sections: List[str] = []
    for profile in profiles:
        sections.append("\n".join(profile.render()))
    return "\n\n".join(section.rstrip() for section in sections if section) + "\n"


def render_stack_profile_report(report: StackProfileReport) -> str:
    """Render a :class:`StackProfileReport`."""

    return report.render_text()


def render_stack_profile_reports(reports: Iterable[StackProfileReport]) -> str:
    """Render multiple stack profile reports."""

    sections = [report.render_text().rstrip() for report in reports]
    return "\n\n".join(section for section in sections if section) + "\n"


def stack_profiles_to_json(profiles: Sequence[BlockStackProfile], *, indent: int = 2) -> str:
    """Serialise ``profiles`` to JSON."""

    payload = [profile.to_dict() for profile in profiles]
    return json.dumps(payload, indent=indent, sort_keys=True)


def stack_profile_reports_to_json(
    reports: Iterable[StackProfileReport], *, indent: int = 2
) -> str:
    """Serialise :class:`StackProfileReport` objects to JSON."""

    payload = [report.to_dict() for report in reports]
    return json.dumps(payload, indent=indent, sort_keys=True)


def render_stack_summary(summary: StackProfileSummary) -> str:
    """Render a :class:`StackProfileSummary` as text."""

    return "\n".join(summary.render_lines()) + "\n"


def stack_profiles_to_csv(profiles: Sequence[BlockStackProfile]) -> str:
    """Return a CSV representation of the provided profiles."""

    header = "block_start,instructions,entry_requirement,min_depth,max_depth,net_delta"
    rows = [header]
    for profile in profiles:
        rows.append(
            f"0x{profile.block_start:06X},{profile.instruction_count},{profile.entry_requirement},"
            f"{profile.min_depth},{profile.max_depth},{profile.net_delta}"
        )
    return "\n".join(rows) + "\n"


def write_stack_profile_reports(
    reports: Iterable[StackProfileReport], path: Path, *, encoding: str = "utf-8"
) -> None:
    """Write rendered stack profile reports to ``path``."""

    text = render_stack_profile_reports(reports)
    path.write_text(text, encoding)


def filter_profiles_by_entry_requirement(
    profiles: Sequence[BlockStackProfile], *, minimum: int
) -> List[BlockStackProfile]:
    """Return profiles whose entry requirement meets ``minimum``."""

    return [profile for profile in profiles if profile.entry_requirement >= minimum]


def filter_profiles_by_delta(
    profiles: Sequence[BlockStackProfile], *, positive: Optional[bool] = None
) -> List[BlockStackProfile]:
    """Filter profiles based on their net delta direction."""

    result: List[BlockStackProfile] = []
    for profile in profiles:
        if positive is True and profile.net_delta > 0:
            result.append(profile)
        elif positive is False and profile.net_delta < 0:
            result.append(profile)
        elif positive is None and profile.net_delta == 0:
            result.append(profile)
    return result

