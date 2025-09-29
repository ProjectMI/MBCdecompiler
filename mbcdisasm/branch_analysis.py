"""Branch analysis helpers for the low level Lua reconstruction pipeline.

The classic disassembly view emitted by :mod:`mbcdisasm.ast` lists each
instruction in linear order.  While this is useful for quick inspection it
doesn't provide any context about the control-flow decisions that shape the
program.  The high level reconstruction stage (`mbcdisasm.highlevel`) embeds
its own structuring logic but a large portion of the tooling still operates on
the lightweight :class:`~mbcdisasm.ir.IRProgram` representation.  This module
fills that gap by analysing branch instructions and exposing a stable data
model that downstream consumers can reuse.

The implementation intentionally keeps the heuristics separate from the
existing CFG builder.  Doing so makes it easier to iterate on branch handling
without disturbing the rest of the pipeline and also means the resulting data
can be rendered independently (for example when producing debugging reports).

The public surface of the module mirrors the design of the other helper
packages in :mod:`mbcdisasm` – small dataclasses are used to describe computed
facts and lightweight renderer helpers are provided for textual and JSON
serialisation.  A large portion of the logic is dedicated to gathering context
for each branch: we track which basic block emitted the instruction, classify
the edge roles (``true``/``false``/``fallthrough``) and apply a couple of
heuristics to detect loop back-edges.  The metadata is intentionally rich so it
can be fed directly into diagnostics or higher level transforms without having
to recompute the expensive bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .ir import IRBlock, IRInstruction, IRProgram


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchEdge:
    """Describes a single outgoing edge for a branch instruction.

    Attributes
    ----------
    role:
        A short string describing the semantic role of the edge.  Recognised
        values are ``"true"``, ``"false"`, ``"fallthrough"`, ``"call"`` and
        ``"return"``.  Consumers should treat unknown values as opaque strings
        to remain forward compatible with future heuristics.
    target:
        The offset of the destination block.  ``None`` indicates that execution
        leaves the current function (for example due to a ``return`` or a
        ``stop`` instruction).
    is_backward:
        :data:`True` when the edge points to a block located before the source
        block.  This usually signifies a loop back-edge which is useful when
        reconstructing high level structures.
    is_fallthrough:
        Whether the edge corresponds to natural fallthrough into the next block
        in address order.  Even unconditional jumps report fallthrough edges so
        that consumers can tell whether the CFG builder kept the successor for
        analysis purposes.
    comment:
        Additional notes about the edge.  Presently used to surface cases where
        the heuristics had to guess the role or when the branch is only
        partially resolved.
    """

    role: str
    target: Optional[int]
    is_backward: bool
    is_fallthrough: bool
    comment: Optional[str] = None

    def describe(self) -> str:
        """Return a human readable summary for debugging output."""

        pieces = [self.role]
        if self.target is None:
            pieces.append("<exit>")
        else:
            pieces.append(f"0x{self.target:06X}")
        if self.is_backward:
            pieces.append("back")
        if self.is_fallthrough:
            pieces.append("fallthrough")
        if self.comment:
            pieces.append(f"note={self.comment}")
        return " ".join(pieces)


@dataclass(frozen=True)
class BranchDescriptor:
    """Summary describing how a branch instruction behaves."""

    segment_index: int
    block_start: int
    instruction_offset: int
    mnemonic: str
    classification: str
    edges: Tuple[BranchEdge, ...]
    comment: Optional[str] = None

    def to_dict(self) -> dict:
        """Return a JSON serialisable dictionary."""

        return {
            "segment_index": self.segment_index,
            "block_start": self.block_start,
            "instruction_offset": self.instruction_offset,
            "mnemonic": self.mnemonic,
            "classification": self.classification,
            "edges": [
                {
                    "role": edge.role,
                    "target": edge.target,
                    "is_backward": edge.is_backward,
                    "is_fallthrough": edge.is_fallthrough,
                    **({"comment": edge.comment} if edge.comment else {}),
                }
                for edge in self.edges
            ],
            **({"comment": self.comment} if self.comment else {}),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return a JSON formatted representation."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render_text(self) -> List[str]:
        """Pretty print the descriptor as multiple lines."""

        header = (
            f"0x{self.instruction_offset:08X} {self.mnemonic}"
            f" [{self.classification}]"
        )
        if self.comment:
            header += f"  -- {self.comment}"
        lines = [header]
        for edge in self.edges:
            lines.append(f"    -> {edge.describe()}")
        return lines


@dataclass
class BranchGraph:
    """Aggregate branch descriptors for an :class:`IRProgram`."""

    segment_index: int
    descriptors: Dict[int, BranchDescriptor]

    def __post_init__(self) -> None:
        self.descriptors = dict(sorted(self.descriptors.items()))

    def __iter__(self) -> Iterator[BranchDescriptor]:
        return iter(self.descriptors.values())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.descriptors)

    def get(self, offset: int) -> Optional[BranchDescriptor]:
        return self.descriptors.get(offset)

    def to_dict(self) -> dict:
        return {
            "segment_index": self.segment_index,
            "branches": {
                f"0x{offset:08X}": descriptor.to_dict()
                for offset, descriptor in self.descriptors.items()
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render_text(self) -> str:
        sections: List[str] = [f"segment {self.segment_index} branches"]
        for descriptor in self:
            sections.extend(descriptor.render_text())
        return "\n".join(sections) + "\n"

    # ------------------------------------------------------------------
    def statistics(self) -> "BranchStatistics":
        """Return aggregated statistics for the graph."""

        total = len(self.descriptors)
        conditional = 0
        jumps = 0
        calls = 0
        returns = 0
        stops = 0
        backward = 0
        fallthrough_edges = 0
        true_edges = 0
        false_edges = 0
        call_edges = 0

        for descriptor in self:
            kind = descriptor.classification
            if kind == "branch":
                conditional += 1
            elif kind == "jump":
                jumps += 1
            elif kind == "call":
                calls += 1
            elif kind == "return":
                returns += 1
            elif kind == "stop":
                stops += 1
            for edge in descriptor.edges:
                if edge.is_backward:
                    backward += 1
                if edge.is_fallthrough:
                    fallthrough_edges += 1
                if edge.role == "true":
                    true_edges += 1
                elif edge.role == "false":
                    false_edges += 1
                elif edge.role == "call":
                    call_edges += 1

        return BranchStatistics(
            total=total,
            conditional=conditional,
            jumps=jumps,
            calls=calls,
            returns=returns,
            stops=stops,
            backward_edges=backward,
            fallthrough_edges=fallthrough_edges,
            true_edges=true_edges,
            false_edges=false_edges,
            call_edges=call_edges,
        )

    def filter(
        self,
        *,
        classification: Optional[str] = None,
        roles: Optional[Sequence[str]] = None,
    ) -> Iterator[BranchDescriptor]:
        """Yield descriptors matching ``classification`` and/or ``roles``."""

        allowed_roles = set(roles or ())
        for descriptor in self:
            if classification and descriptor.classification != classification:
                continue
            if allowed_roles:
                if not any(edge.role in allowed_roles for edge in descriptor.edges):
                    continue
            yield descriptor

    def timeline(self) -> List["BranchTimelineEntry"]:
        """Return a chronologically ordered list of branch entries."""

        entries: List[BranchTimelineEntry] = []
        for ordinal, descriptor in enumerate(self, start=1):
            entry = BranchTimelineEntry.from_descriptor(ordinal, descriptor)
            entries.append(entry)
        return entries


# ---------------------------------------------------------------------------
# Aggregate summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchStatistics:
    """Numeric summary of the control-flow patterns observed in a graph."""

    total: int
    conditional: int
    jumps: int
    calls: int
    returns: int
    stops: int
    backward_edges: int
    fallthrough_edges: int
    true_edges: int
    false_edges: int
    call_edges: int

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "conditional": self.conditional,
            "jumps": self.jumps,
            "calls": self.calls,
            "returns": self.returns,
            "stops": self.stops,
            "backward_edges": self.backward_edges,
            "fallthrough_edges": self.fallthrough_edges,
            "true_edges": self.true_edges,
            "false_edges": self.false_edges,
            "call_edges": self.call_edges,
        }

    def render_lines(self) -> List[str]:
        """Return a list of formatted summary lines."""

        return [
            f"total branches: {self.total}",
            f"  conditional : {self.conditional}",
            f"  jumps       : {self.jumps}",
            f"  calls       : {self.calls}",
            f"  returns     : {self.returns}",
            f"  stops       : {self.stops}",
            f"edges: true={self.true_edges} false={self.false_edges} call={self.call_edges}",
            f"  fallthrough : {self.fallthrough_edges}",
            f"  backward    : {self.backward_edges}",
        ]

    def headline(self) -> str:
        """Return a condensed single line summary."""

        return (
            f"branches={self.total} cond={self.conditional} jump={self.jumps} "
            f"return={self.returns} back_edges={self.backward_edges}"
        )


@dataclass(frozen=True)
class BranchTimelineEntry:
    """Chronological view of a branch descriptor."""

    ordinal: int
    offset: int
    block_start: int
    mnemonic: str
    classification: str
    edges: Tuple[BranchEdge, ...]
    comment: Optional[str]

    @classmethod
    def from_descriptor(
        cls, ordinal: int, descriptor: BranchDescriptor
    ) -> "BranchTimelineEntry":
        return cls(
            ordinal=ordinal,
            offset=descriptor.instruction_offset,
            block_start=descriptor.block_start,
            mnemonic=descriptor.mnemonic,
            classification=descriptor.classification,
            edges=descriptor.edges,
            comment=descriptor.comment,
        )

    def to_dict(self) -> dict:
        return {
            "ordinal": self.ordinal,
            "offset": self.offset,
            "block_start": self.block_start,
            "mnemonic": self.mnemonic,
            "classification": self.classification,
            "comment": self.comment,
            "edges": [
                {
                    "role": edge.role,
                    "target": edge.target,
                    "is_backward": edge.is_backward,
                    "is_fallthrough": edge.is_fallthrough,
                    **({"comment": edge.comment} if edge.comment else {}),
                }
                for edge in self.edges
            ],
        }

    def render(self) -> str:
        edge_desc = ", ".join(edge.describe() for edge in self.edges)
        comment = f"  -- {self.comment}" if self.comment else ""
        return (
            f"#{self.ordinal:03d} 0x{self.offset:08X} {self.mnemonic} "
            f"[{self.classification}] -> {edge_desc}{comment}"
        )


# ---------------------------------------------------------------------------
# High level reports
# ---------------------------------------------------------------------------


@dataclass
class BranchReport:
    """Composite view combining statistics and descriptors."""

    segment_index: int
    graph: BranchGraph
    statistics: BranchStatistics
    timeline: Tuple[BranchTimelineEntry, ...]

    @classmethod
    def from_graph(cls, graph: BranchGraph) -> "BranchReport":
        stats = graph.statistics()
        timeline = tuple(graph.timeline())
        return cls(
            segment_index=graph.segment_index,
            graph=graph,
            statistics=stats,
            timeline=timeline,
        )

    def to_dict(self) -> dict:
        return {
            "segment_index": self.segment_index,
            "statistics": self.statistics.to_dict(),
            "graph": self.graph.to_dict(),
            "timeline": [entry.to_dict() for entry in self.timeline],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render_text(self) -> str:
        header = [f"segment {self.segment_index} branch report"]
        header.extend(self.statistics.render_lines())
        header.append("")
        header.append(self.graph.render_text().rstrip())
        if self.timeline:
            header.append("")
            header.append("timeline:")
            for entry in self.timeline:
                header.append("  " + entry.render())
        return "\n".join(header).rstrip() + "\n"


def render_branch_summary(graph: BranchGraph) -> str:
    """Return a textual summary including statistics and descriptors."""

    report = BranchReport.from_graph(graph)
    return report.render_text()


def render_branch_timeline(entries: Sequence[BranchTimelineEntry]) -> str:
    """Render a sequence of timeline entries."""

    lines = [entry.render() for entry in entries]
    return "\n".join(lines) + ("\n" if lines else "")


def branch_statistics_to_json(stats: BranchStatistics, *, indent: int = 2) -> str:
    """Return a JSON formatted representation of ``stats``."""

    return json.dumps(stats.to_dict(), indent=indent, sort_keys=True)


def build_branch_report(program: IRProgram) -> BranchReport:
    """Analyse ``program`` and return a :class:`BranchReport`."""

    return BranchReport.from_graph(BranchAnalyzer(program).analyse())


def build_branch_reports(programs: Iterable[IRProgram]) -> List[BranchReport]:
    """Construct branch reports for each program in ``programs``."""

    reports: List[BranchReport] = []
    for program in programs:
        reports.append(build_branch_report(program))
    return reports


def render_branch_reports(reports: Iterable[BranchReport]) -> str:
    """Render multiple :class:`BranchReport` objects."""

    sections = [report.render_text().rstrip() for report in reports]
    return "\n\n".join(section for section in sections if section) + "\n"


def branch_reports_to_json(reports: Iterable[BranchReport], *, indent: int = 2) -> str:
    """Serialise ``reports`` to JSON."""

    payload = [report.to_dict() for report in reports]
    return json.dumps(payload, indent=indent, sort_keys=True)


def write_branch_reports(
    reports: Iterable[BranchReport], path: Path, *, encoding: str = "utf-8"
) -> None:
    """Render ``reports`` and write them to ``path``."""

    text = render_branch_reports(reports)
    path.write_text(text, encoding)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


class BranchAnalyzer:
    """Inspect an :class:`IRProgram` and classify branch instructions."""

    def __init__(self, program: IRProgram) -> None:
        self._program = program
        self._block_order = sorted(program.blocks)
        self._next_block: Dict[int, Optional[int]] = {}
        for index, start in enumerate(self._block_order):
            if index + 1 < len(self._block_order):
                self._next_block[start] = self._block_order[index + 1]
            else:
                self._next_block[start] = None

    # ------------------------------------------------------------------
    def analyse(self) -> BranchGraph:
        """Analyse the program and return a :class:`BranchGraph`."""

        descriptors: Dict[int, BranchDescriptor] = {}
        for start in self._block_order:
            descriptor = self._analyse_block(start)
            if descriptor is not None:
                descriptors[descriptor.instruction_offset] = descriptor
        return BranchGraph(self._program.segment_index, descriptors)

    # ------------------------------------------------------------------
    def _analyse_block(self, start: int) -> Optional[BranchDescriptor]:
        block = self._program.blocks[start]
        if not block.instructions:
            return None
        instr = block.instructions[-1]
        semantics = instr.semantics
        edges = list(block.successors)
        next_block = self._next_block.get(start)

        classification = semantics.control_flow or getattr(instr, "control_flow", None)
        if classification is None:
            if len(edges) <= 1:
                return None
            classification = "unknown"

        builder = _BranchDescriptorBuilder(
            segment_index=self._program.segment_index,
            block=block,
            instruction=instr,
            classification=classification,
            successors=edges,
            next_block=next_block,
        )
        descriptor = builder.build()
        return descriptor


@dataclass
class _BranchDescriptorBuilder:
    """Internal helper used by :class:`BranchAnalyzer`."""

    segment_index: int
    block: IRBlock
    instruction: IRInstruction
    classification: str
    successors: Sequence[int]
    next_block: Optional[int]
    _edges: List[BranchEdge] = field(default_factory=list)
    _notes: List[str] = field(default_factory=list)

    def build(self) -> BranchDescriptor:
        method = getattr(self, f"_handle_{self.classification}", None)
        if callable(method):
            method()
        else:
            self._handle_generic()
        comment = "; ".join(self._notes) if self._notes else None
        return BranchDescriptor(
            segment_index=self.segment_index,
            block_start=self.block.start,
            instruction_offset=self.instruction.offset,
            mnemonic=self.instruction.semantics.manual_name,
            classification=self.classification,
            edges=tuple(self._edges),
            comment=comment,
        )

    # ------------------------------------------------------------------
    def _handle_branch(self) -> None:
        fallthrough = self._pick_fallthrough()
        if fallthrough is None:
            self._notes.append("no fallthrough successor identified")
        branch_target = self._pick_branch_target(fallthrough)
        if branch_target is None:
            self._notes.append("missing explicit branch target")

        if branch_target is not None:
            self._emit_edge("true", branch_target, is_fallthrough=False)
        if fallthrough is not None:
            self._emit_edge("false", fallthrough, is_fallthrough=True)

    def _handle_jump(self) -> None:
        target = self._pick_branch_target(None)
        if target is None and self.successors:
            target = self.successors[0]
            self._notes.append("using first successor as jump target")
        if target is not None:
            self._emit_edge("jump", target, is_fallthrough=False)
        if self.next_block is not None:
            self._emit_edge("fallthrough", self.next_block, is_fallthrough=True)

    def _handle_call(self) -> None:
        target = self._pick_branch_target(None)
        if target is not None:
            self._emit_edge("call", target, is_fallthrough=False)
        if self.next_block is not None:
            self._emit_edge("return", self.next_block, is_fallthrough=True)

    def _handle_return(self) -> None:
        self._emit_edge("return", None, is_fallthrough=False)

    def _handle_stop(self) -> None:
        self._emit_edge("stop", None, is_fallthrough=False)

    def _handle_generic(self) -> None:
        for target in self.successors:
            role = "fallthrough" if target == self.next_block else "edge"
            self._emit_edge(role, target, is_fallthrough=(target == self.next_block))
        if not self.successors and self.next_block is not None:
            self._emit_edge("fallthrough", self.next_block, is_fallthrough=True)

    # ------------------------------------------------------------------
    def _pick_branch_target(self, fallthrough: Optional[int]) -> Optional[int]:
        for target in self.successors:
            if target != fallthrough:
                return target
        return None

    def _pick_fallthrough(self) -> Optional[int]:
        if self.next_block is None:
            return None
        if self.next_block in self.successors:
            return self.next_block
        if not self.successors:
            return self.next_block
        return None

    def _emit_edge(self, role: str, target: Optional[int], *, is_fallthrough: bool) -> None:
        is_backward = False
        if target is not None:
            is_backward = target < self.block.start
        edge = BranchEdge(
            role=role,
            target=target,
            is_backward=is_backward,
            is_fallthrough=is_fallthrough,
        )
        self._edges.append(edge)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_branch_graph(graph: BranchGraph) -> str:
    """Return a textual representation for ``graph``."""

    return graph.render_text()


def branch_graph_to_dict(graph: BranchGraph) -> dict:
    """Return a JSON serialisable payload describing ``graph``."""

    return graph.to_dict()


def branch_graph_to_json(graph: BranchGraph, *, indent: int = 2) -> str:
    """Return a JSON formatted representation of ``graph``."""

    return graph.to_json(indent=indent)


def describe_branches(program: IRProgram) -> BranchGraph:
    """Convenience helper that wraps :class:`BranchAnalyzer` for quick use."""

    return BranchAnalyzer(program).analyse()

