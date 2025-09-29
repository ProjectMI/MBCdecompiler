"""Utility helpers that summarise control-flow structure for IR programs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .ir import IRBlock, IRProgram


def _format_addresses(values: Sequence[int], limit: int = 6) -> str:
    if not values:
        return ""
    entries = [f"0x{value:06X}" for value in list(values)[:limit]]
    if len(values) > limit:
        entries.append("…")
    return ", ".join(entries)


def _address_lines(values: Sequence[int], limit: int = 6) -> List[str]:
    if not values:
        return []
    lines: List[str] = []
    for index, value in enumerate(values):
        if index >= limit:
            remaining = len(values) - limit
            if remaining > 0:
                lines.append(f"- ... ({remaining} additional blocks omitted)")
            break
        lines.append(f"- 0x{value:06X}")
    return lines


@dataclass
class BlockFlowInfo:
    """Detailed description of how a single block connects to others."""

    start: int
    successors: List[int]
    predecessors: List[int]
    is_loop_header: bool = False
    is_unreachable: bool = False
    is_entry: bool = False
    is_branch: bool = False
    is_merge: bool = False
    is_exit: bool = False
    critical_successors: List[int] = field(default_factory=list)

    def describe(self) -> str:
        succ = ", ".join(f"0x{value:06X}" for value in self.successors) or "<none>"
        pred = len(self.predecessors)
        flags: List[str] = []
        if self.is_loop_header:
            flags.append("loop")
        if self.is_unreachable:
            flags.append("unreachable")
        if self.is_entry:
            flags.append("entry")
        if self.is_branch:
            flags.append("branch")
        if self.is_merge:
            flags.append("merge")
        if self.is_exit:
            flags.append("exit")
        flag_text = f" ({', '.join(flags)})" if flags else ""
        critical = ""
        if self.critical_successors:
            crit = ", ".join(f"0x{value:06X}" for value in self.critical_successors)
            critical = f" critical=[{crit}]"
        return f"0x{self.start:06X} -> [{succ}] pred={pred}{flag_text}{critical}"

    def to_dict(self) -> dict:
        return {
            "start": f"0x{self.start:06X}",
            "successors": [f"0x{value:06X}" for value in self.successors],
            "predecessors": [f"0x{value:06X}" for value in self.predecessors],
            "is_loop_header": self.is_loop_header,
            "is_unreachable": self.is_unreachable,
            "is_entry": self.is_entry,
            "is_branch": self.is_branch,
            "is_merge": self.is_merge,
            "is_exit": self.is_exit,
            "critical_successors": [
                f"0x{value:06X}" for value in self.critical_successors
            ],
        }


@dataclass
class ControlFlowMetrics:
    """Aggregated metrics describing the control-flow graph of a program."""

    block_count: int
    entry_points: List[int]
    loop_headers: List[int]
    unreachable: List[int]
    reachable_blocks: int
    exit_blocks: List[int]
    max_successors: int
    max_predecessors: int
    branch_blocks: List[int] = field(default_factory=list)
    merge_blocks: List[int] = field(default_factory=list)
    critical_edges: List[Tuple[int, int]] = field(default_factory=list)
    cyclomatic_complexity: int = 0
    connected_components: int = 0
    average_successors: float = 0.0
    edge_count: int = 0
    block_details: Dict[int, BlockFlowInfo] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        lines = ["control flow summary:"]
        lines.append(f"- blocks: {self.block_count}")
        lines.append(f"- reachable blocks: {self.reachable_blocks}")
        if self.entry_points:
            lines.append(
                "- entry points: "
                f"{len(self.entry_points)} ({_format_addresses(self.entry_points)})"
            )
        lines.append(f"- loop headers: {len(self.loop_headers)}")
        lines.append(f"- unreachable blocks: {len(self.unreachable)}")
        lines.append(f"- exits: {len(self.exit_blocks)}")
        lines.append(f"- branch points: {len(self.branch_blocks)}")
        lines.append(f"- merge points: {len(self.merge_blocks)}")
        lines.append(f"- critical edges: {len(self.critical_edges)}")
        lines.append(f"- max successors: {self.max_successors}")
        lines.append(f"- max predecessors: {self.max_predecessors}")
        if self.edge_count:
            lines.append(f"- edges: {self.edge_count}")
        if self.connected_components:
            lines.append(f"- components: {self.connected_components}")
        if self.cyclomatic_complexity:
            lines.append(
                f"- cyclomatic complexity: {self.cyclomatic_complexity}"
            )
        if self.average_successors:
            lines.append(
                f"- average successors per block: {self.average_successors:.2f}"
            )
        if self.loop_headers:
            lines.append(
                f"- loop header blocks: {_format_addresses(self.loop_headers)}"
            )
        if self.unreachable:
            lines.append(
                f"- unreachable blocks: {_format_addresses(self.unreachable)}"
            )
        if self.exit_blocks:
            lines.append(
                f"- exit blocks: {_format_addresses(self.exit_blocks)}"
            )
        return lines

    def to_dict(self) -> dict:
        return {
            "block_count": self.block_count,
            "entry_points": [f"0x{value:06X}" for value in self.entry_points],
            "loop_headers": [f"0x{value:06X}" for value in self.loop_headers],
            "unreachable": [f"0x{value:06X}" for value in self.unreachable],
            "reachable_blocks": self.reachable_blocks,
            "exit_blocks": [f"0x{value:06X}" for value in self.exit_blocks],
            "max_successors": self.max_successors,
            "max_predecessors": self.max_predecessors,
            "branch_blocks": [f"0x{value:06X}" for value in self.branch_blocks],
            "merge_blocks": [f"0x{value:06X}" for value in self.merge_blocks],
            "critical_edges": [
                {"from": f"0x{src:06X}", "to": f"0x{dst:06X}"}
                for src, dst in self.critical_edges
            ],
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "connected_components": self.connected_components,
            "average_successors": self.average_successors,
            "edge_count": self.edge_count,
            "blocks": {
                f"0x{start:06X}": info.to_dict()
                for start, info in self.block_details.items()
            },
        }

    def block_lines(self, limit: int = 8) -> List[str]:
        lines: List[str] = []
        for index, start in enumerate(sorted(self.block_details)):
            if index >= limit:
                remaining = len(self.block_details) - limit
                if remaining > 0:
                    lines.append(f"- ... ({remaining} additional blocks omitted)")
                break
            info = self.block_details[start]
            lines.append(f"- {info.describe()}")
        return lines

    def branch_lines(self, limit: int = 6) -> List[str]:
        return _address_lines(self.branch_blocks, limit=limit)

    def merge_lines(self, limit: int = 6) -> List[str]:
        return _address_lines(self.merge_blocks, limit=limit)

    def exit_lines(self, limit: int = 6) -> List[str]:
        return _address_lines(self.exit_blocks, limit=limit)

    def critical_edge_lines(self, limit: int = 6) -> List[str]:
        if not self.critical_edges:
            return []
        lines: List[str] = []
        for index, (src, dst) in enumerate(self.critical_edges):
            if index >= limit:
                remaining = len(self.critical_edges) - limit
                if remaining > 0:
                    lines.append(f"- ... ({remaining} additional edges omitted)")
                break
            lines.append(f"- 0x{src:06X} -> 0x{dst:06X}")
        return lines


def _compute_predecessors(blocks: Dict[int, IRBlock]) -> Dict[int, Set[int]]:
    predecessors: Dict[int, Set[int]] = {start: set() for start in blocks}
    for start, block in blocks.items():
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].add(start)
    return predecessors


def _reachable_blocks(blocks: Dict[int, IRBlock], entry: int) -> Set[int]:
    visited: Set[int] = set()
    queue: deque[int] = deque([entry])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        block = blocks.get(current)
        if block is None:
            continue
        for successor in block.successors:
            if successor not in visited and successor in blocks:
                queue.append(successor)
    return visited


def _detect_loop_headers(blocks: Dict[int, IRBlock], predecessors: Dict[int, Set[int]]) -> List[int]:
    headers: List[int] = []
    for start, block in blocks.items():
        for successor in block.successors:
            if successor <= start:
                headers.append(successor)
                continue
            if successor in predecessors and start in predecessors[successor]:
                headers.append(successor)
    headers = sorted(set(headers))
    return headers


def _connected_components(
    blocks: Dict[int, IRBlock],
    nodes: Set[int],
    predecessors: Dict[int, Set[int]],
) -> List[Set[int]]:
    if not nodes:
        return []
    remaining = set(nodes)
    components: List[Set[int]] = []
    while remaining:
        start = remaining.pop()
        queue: deque[int] = deque([start])
        component: Set[int] = set()
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            block = blocks.get(current)
            if block is None:
                continue
            neighbours = set(block.successors)
            neighbours.update(predecessors.get(current, set()))
            for neighbour in neighbours:
                if neighbour in nodes and neighbour not in component:
                    queue.append(neighbour)
        components.append(component)
        remaining.difference_update(component)
    return components


def summarise_control_flow(program: IRProgram) -> ControlFlowMetrics:
    blocks = program.blocks
    if not blocks:
        return ControlFlowMetrics(
            block_count=0,
            entry_points=[],
            loop_headers=[],
            unreachable=[],
            reachable_blocks=0,
            exit_blocks=[],
            max_successors=0,
            max_predecessors=0,
        )

    entry = min(blocks)
    predecessors = _compute_predecessors(blocks)
    reachable = _reachable_blocks(blocks, entry)
    entry_points = [
        start for start, preds in predecessors.items() if not preds and start in reachable
    ]
    reachable = _reachable_blocks(blocks, entry)
    unreachable = sorted(start for start in blocks if start not in reachable)
    loop_headers = _detect_loop_headers(blocks, predecessors)
    max_successors = max(
        (len(block.successors) for block in blocks.values()),
        default=0,
    )
    max_predecessors = max((len(preds) for preds in predecessors.values()), default=0)
    exit_blocks = sorted(
        start for start in reachable if not blocks[start].successors
    )
    branch_blocks = sorted(
        start for start in reachable if len(blocks[start].successors) > 1
    )
    merge_blocks = sorted(
        start for start in reachable if len(predecessors.get(start, ())) > 1
    )
    critical_edges: List[Tuple[int, int]] = []
    block_details: Dict[int, BlockFlowInfo] = {}
    loop_set = set(loop_headers)
    unreachable_set = set(unreachable)
    reachable_set = set(reachable)
    entry_set = set(entry_points)
    merge_set = set(merge_blocks)
    branch_set = set(branch_blocks)
    for start, block in blocks.items():
        succ = [value for value in block.successors if value in blocks]
        preds = sorted(predecessors.get(start, ()))
        critical = [
            value
            for value in succ
            if start in branch_set and value in merge_set
        ]
        if critical:
            critical_edges.extend((start, value) for value in critical)
        block_details[start] = BlockFlowInfo(
            start=start,
            successors=sorted(succ),
            predecessors=preds,
            is_loop_header=start in loop_set,
            is_unreachable=start in unreachable_set,
            is_entry=start in entry_set,
            is_branch=start in branch_set,
            is_merge=start in merge_set,
            is_exit=start in exit_blocks,
            critical_successors=critical,
        )

    components = _connected_components(blocks, reachable_set, predecessors)
    edge_count = sum(
        len([succ for succ in blocks[start].successors if succ in reachable_set])
        for start in reachable_set
    )
    cyclomatic = edge_count - len(reachable_set) + 2 * len(components)
    average_successors = (
        float(edge_count) / len(reachable_set)
        if reachable_set
        else 0.0
    )

    return ControlFlowMetrics(
        block_count=len(blocks),
        entry_points=sorted(entry_points),
        loop_headers=loop_headers,
        unreachable=unreachable,
        reachable_blocks=len(reachable_set),
        exit_blocks=exit_blocks,
        max_successors=max_successors,
        max_predecessors=max_predecessors,
        branch_blocks=branch_blocks,
        merge_blocks=merge_blocks,
        critical_edges=sorted(critical_edges),
        cyclomatic_complexity=cyclomatic,
        connected_components=len(components),
        average_successors=average_successors,
        edge_count=edge_count,
        block_details=block_details,
    )


def render_control_flow_summary(metrics: ControlFlowMetrics) -> List[str]:
    lines = metrics.summary_lines()
    block_lines = metrics.block_lines()
    if block_lines:
        lines.append("block edges:")
        lines.extend(block_lines)
    branch_lines = metrics.branch_lines()
    if branch_lines:
        lines.append("branch points:")
        lines.extend(branch_lines)
    merge_lines = metrics.merge_lines()
    if merge_lines:
        lines.append("merge points:")
        lines.extend(merge_lines)
    exit_lines = metrics.exit_lines()
    if exit_lines:
        lines.append("exit blocks:")
        lines.extend(exit_lines)
    critical_lines = metrics.critical_edge_lines()
    if critical_lines:
        lines.append("critical edges:")
        lines.extend(critical_lines)
    return lines


__all__ = [
    "BlockFlowInfo",
    "ControlFlowMetrics",
    "summarise_control_flow",
    "render_control_flow_summary",
]
