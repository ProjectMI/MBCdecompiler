"""Generate readable summaries for helper usage collected during reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .lua_formatter import HelperRegistry, HelperSignature, MethodSignature


@dataclass
class HelperUsageEntry:
    """Describe how often a helper has been referenced."""

    name: str
    summary: str
    count: int
    signature: HelperSignature

    def render_line(self) -> str:
        descriptor = self.summary or self.signature.summary or "<no description>"
        if self.count == 1:
            return f"{self.name}: {descriptor}"
        return f"{self.name} (x{self.count}): {descriptor}"


@dataclass
class HelperReport:
    """Aggregated helper usage covering global functions and struct methods."""

    functions: List[HelperUsageEntry] = field(default_factory=list)
    methods: List[HelperUsageEntry] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.functions and not self.methods

    def summary_lines(self, *, limit: int = 6) -> List[str]:
        if self.is_empty():
            return []
        lines = ["helper usage:"]
        if self.functions:
            lines.append("- functions:")
            lines.extend(_render_usage(self.functions, limit))
        if self.methods:
            lines.append("- methods:")
            lines.extend(_render_usage(self.methods, limit))
        return lines


def build_helper_report(registry: HelperRegistry) -> HelperReport:
    """Construct a :class:`HelperReport` from ``registry`` usage data."""

    report = HelperReport()
    for signature, count in registry.function_usage():
        if count <= 0:
            continue
        report.functions.append(
            HelperUsageEntry(
                name=signature.name,
                summary=signature.summary,
                count=count,
                signature=signature,
            )
        )
    for signature, count in registry.method_usage():
        if count <= 0:
            continue
        display_name = f"{signature.struct}:{signature.method}"
        report.methods.append(
            HelperUsageEntry(
                name=display_name,
                summary=signature.summary,
                count=count,
                signature=signature,
            )
        )
    return report


def _render_usage(entries: Sequence[HelperUsageEntry], limit: int) -> List[str]:
    lines: List[str] = []
    for entry in entries[:limit]:
        lines.append(f"  - {entry.render_line()}")
    remaining = len(entries) - limit
    if remaining > 0:
        lines.append(f"  - ... ({remaining} additional helpers)")
    return lines


__all__ = ["HelperUsageEntry", "HelperReport", "build_helper_report"]
