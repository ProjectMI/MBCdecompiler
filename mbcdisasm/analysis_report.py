"""High level analysis reporting helpers for reconstructed Lua functions.

This module takes the structured metadata collected during high level
reconstruction and produces human readable as well as machine readable
summaries.  The intent is to allow operators to inspect trends across large
batches of segments without manually digging through the generated Lua output.
The text report focuses on a narrative overview whereas the JSON payload aims
to be convenient for further automated processing.

The report intentionally mirrors many of the comment blocks emitted alongside
the generated Lua but aggregates the information across functions and normalise
it into easily digestible structures.  Consumers can decide whether they prefer
to embed the text report as an appendix or ingest the JSON document into their
own analysis pipelines.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .highlevel import (
    FunctionMetadata,
    HighLevelFunction,
    InstructionTraceInfo,
    StackEvent,
)


@dataclass
class HelperUsageEntry:
    """Aggregated usage data for a single helper function or method."""

    kind: str
    name: str
    count: int


@dataclass
class InstructionTraceSummary:
    """Compact representation of an instruction trace entry."""

    offset: int
    mnemonic: str
    summary: str
    usages: Sequence[str] = field(default_factory=list)

    @classmethod
    def from_trace(cls, trace: InstructionTraceInfo) -> "InstructionTraceSummary":
        usages = [f"{role}: {comment}" for role, comment in trace.usages]
        return cls(offset=trace.offset, mnemonic=trace.mnemonic, summary=trace.summary, usages=tuple(usages))

    def to_dict(self) -> Dict[str, object]:
        return {
            "offset": self.offset,
            "mnemonic": self.mnemonic,
            "summary": self.summary,
            "usages": list(self.usages),
        }


@dataclass
class StackEventSummary:
    """Normalised view of a :class:`StackEvent`."""

    action: str
    value: str
    origin: Optional[int]
    comment: Optional[str]
    depth_before: int
    depth_after: int

    @classmethod
    def from_event(cls, event: StackEvent) -> "StackEventSummary":
        return cls(
            action=event.action,
            value=event.value,
            origin=event.origin,
            comment=event.comment,
            depth_before=event.depth_before,
            depth_after=event.depth_after,
        )

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "action": self.action,
            "value": self.value,
            "depth_before": self.depth_before,
            "depth_after": self.depth_after,
        }
        if self.origin is not None:
            payload["origin"] = self.origin
        if self.comment:
            payload["comment"] = self.comment
        return payload


@dataclass
class FunctionAnalysis:
    """Structured description of a reconstructed function."""

    name: str
    block_count: int
    instruction_count: int
    helper_calls: int
    branch_count: int
    literal_count: int
    warnings: Sequence[str]
    helper_usage: Sequence[HelperUsageEntry]
    value_comment_summary: Sequence[str]
    stack_event_summary: Sequence[str]
    instruction_trace: Sequence[InstructionTraceSummary]
    stack_events: Sequence[StackEventSummary]
    stack_depth_min: int
    stack_depth_max: int
    stack_depth_final: int
    stack_underflows: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "block_count": self.block_count,
            "instruction_count": self.instruction_count,
            "helper_calls": self.helper_calls,
            "branch_count": self.branch_count,
            "literal_count": self.literal_count,
            "warnings": list(self.warnings),
            "helper_usage": [entry.__dict__ for entry in self.helper_usage],
            "value_comment_summary": list(self.value_comment_summary),
            "stack_event_summary": list(self.stack_event_summary),
            "instruction_trace": [entry.to_dict() for entry in self.instruction_trace],
            "stack_events": [event.to_dict() for event in self.stack_events],
            "stack_depth": {
                "min": self.stack_depth_min,
                "max": self.stack_depth_max,
                "final": self.stack_depth_final,
                "underflows": self.stack_underflows,
            },
        }

    def render_text(self) -> str:
        lines = [f"Function: {self.name}"]
        lines.append(
            "  blocks={block} instructions={instr} helpers={helpers} branches={branches} literals={literals}".format(
                block=self.block_count,
                instr=self.instruction_count,
                helpers=self.helper_calls,
                branches=self.branch_count,
                literals=self.literal_count,
            )
        )
        lines.append(
            "  stack: min={min} max={max} final={final} underflows={underflows}".format(
                min=self.stack_depth_min,
                max=self.stack_depth_max,
                final=self.stack_depth_final,
                underflows=self.stack_underflows,
            )
        )
        if self.warnings:
            lines.append("  warnings:")
            for warning in self.warnings:
                lines.append(f"    - {warning}")
        if self.helper_usage:
            lines.append("  helper usage:")
            for entry in self.helper_usage:
                lines.append(f"    - {entry.kind}: {entry.name} ×{entry.count}")
        if self.value_comment_summary:
            lines.append("  value provenance:")
            for line in self.value_comment_summary:
                lines.append(f"    {line}")
        if self.stack_event_summary:
            lines.append("  stack summary:")
            for line in self.stack_event_summary:
                lines.append(f"    {line}")
        if self.instruction_trace:
            lines.append("  instruction trace:")
            for trace in self.instruction_trace:
                lines.append(
                    f"    - 0x{trace.offset:06X} {trace.mnemonic} :: {trace.summary or 'no summary'}"
                )
                for usage in trace.usages:
                    lines.append(f"      • {usage}")
        return "\n".join(lines)


@dataclass
class ModuleSummary:
    """Aggregated overview for a collection of analyses."""

    function_count: int
    total_instructions: int
    total_branches: int
    total_helpers: int
    total_literals: int
    warning_count: int
    helper_usage: Sequence[HelperUsageEntry]
    value_comment_totals: Sequence[str]
    stack_event_totals: Sequence[str]
    warning_details: Sequence[str]
    max_stack_depth: int
    min_stack_depth: int
    nonzero_final_depths: int
    underflow_functions: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "functions": self.function_count,
            "instructions": self.total_instructions,
            "branches": self.total_branches,
            "helper_calls": self.total_helpers,
            "literal_instructions": self.total_literals,
            "warnings": self.warning_count,
            "helper_usage": [entry.__dict__ for entry in self.helper_usage],
            "value_comments": list(self.value_comment_totals),
            "stack_events": list(self.stack_event_totals),
            "warning_details": list(self.warning_details),
            "stack_depth": {
                "max": self.max_stack_depth,
                "min": self.min_stack_depth,
                "nonzero_final": self.nonzero_final_depths,
                "underflow_functions": self.underflow_functions,
            },
        }

    def render_text(self) -> List[str]:
        lines = ["Module summary:"]
        lines.append(f"  functions: {self.function_count}")
        lines.append(f"  instructions: {self.total_instructions}")
        lines.append(f"  literal instructions: {self.total_literals}")
        lines.append(f"  helper calls: {self.total_helpers}")
        lines.append(f"  branches: {self.total_branches}")
        lines.append(f"  warnings: {self.warning_count}")
        lines.append(
            "  stack depth: max={max} min={min} unfinished={unfinished} underflows={underflows}".format(
                max=self.max_stack_depth,
                min=self.min_stack_depth,
                unfinished=self.nonzero_final_depths,
                underflows=self.underflow_functions,
            )
        )
        if self.helper_usage:
            lines.append("  top helpers:")
            for entry in self.helper_usage[:8]:
                lines.append(f"    - {entry.kind}: {entry.name} ×{entry.count}")
        if self.value_comment_totals:
            lines.append("  frequent value provenance notes:")
            for line in self.value_comment_totals[:6]:
                lines.append(f"    - {line}")
        if self.stack_event_totals:
            lines.append("  common stack events:")
            for line in self.stack_event_totals[:6]:
                lines.append(f"    - {line}")
        if self.warning_details:
            lines.append("  warning locations:")
            for detail in self.warning_details[:6]:
                lines.append(f"    - {detail}")
        return lines

    def render_markdown(self) -> List[str]:
        lines = ["## Module summary", ""]
        lines.append(
            "* Functions: {functions}\n* Instructions: {instr}\n* Literal instructions: {literals}\n* Helper calls: {helpers}\n* Branches: {branches}\n* Warnings: {warnings}\n* Stack max depth: {max_depth}\n* Stack min depth: {min_depth}\n* Functions with non-zero final depth: {unfinished}\n* Functions with underflow: {underflows}".format(
                functions=self.function_count,
                instr=self.total_instructions,
                literals=self.total_literals,
                helpers=self.total_helpers,
                branches=self.total_branches,
                warnings=self.warning_count,
                max_depth=self.max_stack_depth,
                min_depth=self.min_stack_depth,
                unfinished=self.nonzero_final_depths,
                underflows=self.underflow_functions,
            )
        )
        if self.helper_usage:
            lines.append("")
            lines.append("### Top helpers")
            for entry in self.helper_usage[:12]:
                lines.append(f"- **{entry.kind}** `{entry.name}` ×{entry.count}")
        if self.value_comment_totals:
            lines.append("")
            lines.append("### Frequent value provenance notes")
            for line in self.value_comment_totals[:10]:
                lines.append(f"- {line}")
        if self.stack_event_totals:
            lines.append("")
            lines.append("### Common stack events")
            for line in self.stack_event_totals[:10]:
                lines.append(f"- {line}")
        lines.append("")
        lines.append("### Stack depth overview")
        lines.append(
            "* Max depth: {max}\n* Min depth: {min}\n* Functions with non-zero final depth: {unfinished}\n* Functions with underflow: {underflows}".format(
                max=self.max_stack_depth,
                min=self.min_stack_depth,
                unfinished=self.nonzero_final_depths,
                underflows=self.underflow_functions,
            )
        )
        if self.warning_details:
            lines.append("")
            lines.append("### Warning locations")
            for detail in self.warning_details[:10]:
                lines.append(f"- {detail}")
        lines.append("")
        return lines


def _collect_helper_usage(metadata: FunctionMetadata) -> List[HelperUsageEntry]:
    entries: List[HelperUsageEntry] = []
    for kind, helpers in metadata.helper_usage.items():
        for name, count in helpers.items():
            entries.append(HelperUsageEntry(kind=kind, name=name, count=count))
    entries.sort(key=lambda item: (-item.count, item.kind, item.name))
    return entries


def build_function_analysis(function: HighLevelFunction) -> FunctionAnalysis:
    metadata = function.metadata
    helper_usage = _collect_helper_usage(metadata)
    value_summary = metadata.value_comment_summary_lines()
    stack_summary = metadata.stack_event_summary_lines()
    instruction_trace = [InstructionTraceSummary.from_trace(trace) for trace in metadata.instruction_trace.values()]
    instruction_trace.sort(key=lambda item: item.offset)
    stack_events = [StackEventSummary.from_event(event) for event in metadata.stack_events]
    return FunctionAnalysis(
        name=function.name,
        block_count=metadata.block_count,
        instruction_count=metadata.instruction_count,
        helper_calls=metadata.helper_calls,
        branch_count=metadata.branch_count,
        literal_count=metadata.literal_count,
        warnings=tuple(metadata.warnings),
        helper_usage=helper_usage,
        value_comment_summary=tuple(value_summary),
        stack_event_summary=tuple(stack_summary),
        instruction_trace=tuple(instruction_trace),
        stack_events=tuple(stack_events),
        stack_depth_min=metadata.stack_depth_min,
        stack_depth_max=metadata.stack_depth_max,
        stack_depth_final=metadata.stack_depth_final,
        stack_underflows=metadata.stack_underflows,
    )


def build_analysis(functions: Sequence[HighLevelFunction]) -> List[FunctionAnalysis]:
    return [build_function_analysis(function) for function in functions]


def build_module_summary(analysis: Sequence[FunctionAnalysis]) -> ModuleSummary:
    total_instr = sum(entry.instruction_count for entry in analysis)
    total_branches = sum(entry.branch_count for entry in analysis)
    total_helpers = sum(entry.helper_calls for entry in analysis)
    total_literals = sum(entry.literal_count for entry in analysis)
    warning_count = sum(len(entry.warnings) for entry in analysis)
    helper_counter: Dict[tuple[str, str], int] = {}
    for entry in analysis:
        for usage in entry.helper_usage:
            key = (usage.kind, usage.name)
            helper_counter[key] = helper_counter.get(key, 0) + usage.count
    helper_entries = [HelperUsageEntry(kind=kind, name=name, count=count) for (kind, name), count in helper_counter.items()]
    helper_entries.sort(key=lambda item: (-item.count, item.kind, item.name))
    value_counter: Dict[str, int] = {}
    stack_counter: Dict[str, int] = {}
    warning_details: List[str] = []
    for entry in analysis:
        for line in entry.value_comment_summary:
            key = line.lstrip("- ")
            value_counter[key] = value_counter.get(key, 0) + 1
        for line in entry.stack_event_summary:
            key = line.lstrip("- ")
            stack_counter[key] = stack_counter.get(key, 0) + 1
        for warning in entry.warnings:
            warning_details.append(f"{entry.name}: {warning}")
    value_totals = [f"{text} ×{count}" for text, count in sorted(value_counter.items(), key=lambda item: (-item[1], item[0]))]
    stack_totals = [f"{text} ×{count}" for text, count in sorted(stack_counter.items(), key=lambda item: (-item[1], item[0]))]
    warning_details.sort()
    max_stack_depth = max((entry.stack_depth_max for entry in analysis), default=0)
    min_stack_depth = min((entry.stack_depth_min for entry in analysis), default=0)
    nonzero_final = sum(1 for entry in analysis if entry.stack_depth_final != 0)
    underflow_functions = sum(
        1 for entry in analysis if entry.stack_depth_min < 0 or entry.stack_underflows > 0
    )
    return ModuleSummary(
        function_count=len(analysis),
        total_instructions=total_instr,
        total_branches=total_branches,
        total_helpers=total_helpers,
        total_literals=total_literals,
        warning_count=warning_count,
        helper_usage=tuple(helper_entries),
        value_comment_totals=tuple(value_totals),
        stack_event_totals=tuple(stack_totals),
        warning_details=tuple(warning_details),
        max_stack_depth=max_stack_depth,
        min_stack_depth=min_stack_depth,
        nonzero_final_depths=nonzero_final,
        underflow_functions=underflow_functions,
    )


def render_text_report(analysis: Sequence[FunctionAnalysis]) -> str:
    module_summary = build_module_summary(analysis)
    lines: List[str] = module_summary.render_text()
    lines.append("")
    for entry in analysis:
        lines.append(entry.render_text())
        lines.append("")
    return "\n".join(lines).rstrip()


def render_markdown_report(analysis: Sequence[FunctionAnalysis]) -> str:
    lines = ["# Reconstruction Analysis", ""]
    module_summary = build_module_summary(analysis)
    lines.extend(module_summary.render_markdown())
    for entry in analysis:
        lines.append(f"## {entry.name}")
        lines.append("")
        lines.append(
            "* Blocks: {block}\n* Instructions: {instr}\n* Helper calls: {helpers}\n* Branches: {branches}\n* Literal instructions: {literals}".format(
                block=entry.block_count,
                instr=entry.instruction_count,
                helpers=entry.helper_calls,
                branches=entry.branch_count,
                literals=entry.literal_count,
            )
        )
        if entry.warnings:
            lines.append("")
            lines.append("### Warnings")
            for warning in entry.warnings:
                lines.append(f"- {warning}")
        if entry.helper_usage:
            lines.append("")
            lines.append("### Helper usage")
            for usage in entry.helper_usage:
                lines.append(f"- **{usage.kind}** `{usage.name}` ×{usage.count}")
        if entry.value_comment_summary:
            lines.append("")
            lines.append("### Value provenance")
            for line in entry.value_comment_summary:
                lines.append(f"- {line}")
        if entry.stack_event_summary:
            lines.append("")
            lines.append("### Stack summary")
            for line in entry.stack_event_summary:
                lines.append(f"- {line}")
        if entry.instruction_trace:
            lines.append("")
            lines.append("### Instruction trace")
            for trace in entry.instruction_trace:
                bullet = f"- 0x{trace.offset:06X} `{trace.mnemonic}`"
                if trace.summary:
                    bullet += f" — {trace.summary}"
                lines.append(bullet)
                for usage in trace.usages:
                    lines.append(f"  - {usage}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def analysis_to_json(analysis: Sequence[FunctionAnalysis]) -> str:
    module_summary = build_module_summary(analysis)
    payload = {
        "module": module_summary.to_dict(),
        "functions": [entry.to_dict() for entry in analysis],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def write_text_report(analysis: Sequence[FunctionAnalysis], path: str) -> None:
    text = render_text_report(analysis)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def write_markdown_report(analysis: Sequence[FunctionAnalysis], path: str) -> None:
    text = render_markdown_report(analysis)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json_report(analysis: Sequence[FunctionAnalysis], path: str) -> None:
    text = analysis_to_json(analysis)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def render_csv_report(analysis: Sequence[FunctionAnalysis]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "name",
            "blocks",
            "instructions",
            "helper_calls",
            "branches",
            "literal_instructions",
            "warnings",
        ]
    )
    for entry in analysis:
        writer.writerow(
            [
                entry.name,
                entry.block_count,
                entry.instruction_count,
                entry.helper_calls,
                entry.branch_count,
                entry.literal_count,
                " | ".join(entry.warnings),
            ]
        )
    return buffer.getvalue().rstrip("\n")


def write_csv_report(analysis: Sequence[FunctionAnalysis], path: str) -> None:
    text = render_csv_report(analysis)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text + "\n")


def render_module_summary_text(analysis: Sequence[FunctionAnalysis]) -> str:
    summary = build_module_summary(analysis)
    return "\n".join(summary.render_text())


def render_helper_usage_csv(analysis: Sequence[FunctionAnalysis]) -> str:
    summary = build_module_summary(analysis)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["kind", "name", "count"])
    for entry in summary.helper_usage:
        writer.writerow([entry.kind, entry.name, entry.count])
    return buffer.getvalue().rstrip("\n")


def write_helper_usage_csv(analysis: Sequence[FunctionAnalysis], path: str) -> None:
    text = render_helper_usage_csv(analysis)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text + "\n")


def filter_warning_functions(analysis: Sequence[FunctionAnalysis]) -> List[FunctionAnalysis]:
    return [entry for entry in analysis if entry.warnings]


def render_warning_report(analysis: Sequence[FunctionAnalysis]) -> str:
    filtered = filter_warning_functions(analysis)
    if not filtered:
        return "No warnings recorded."
    lines = ["Warning summary:", ""]
    for entry in filtered:
        lines.append(f"Function {entry.name}:")
        for warning in entry.warnings:
            lines.append(f"  - {warning}")
        lines.append("")
    return "\n".join(lines).rstrip()


def write_warning_report(analysis: Sequence[FunctionAnalysis], path: str) -> None:
    text = render_warning_report(analysis)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")

