"""Stack seed planning utilities.

The original high level reconstructor defaulted to creating ad-hoc placeholder
variables every time the symbolic stack underflowed.  While adequate for quick
experiments it produced Lua output that was difficult to read: placeholders were
named ``stack_0``, ``stack_1`` and so on, with no attempt at describing their
origin or ensuring that the same logical value was referenced consistently
across multiple blocks.  This module implements a light-weight data-flow
analysis that predicts the minimum stack depth required at the entry of each
basic block.  The resulting plan is used to seed the high level stack with
stable names (``arg_0``, ``block_XXXX_slot_0``) which dramatically reduces the
number of synthetic placeholders and provides a clearer narrative for the Lua
reader.

The analysis is intentionally conservative: it does not attempt to resolve
complex control-flow joins or correlate stack slots with concrete VM registers.
Instead it focuses on three goals:

* ensure each block starts with enough symbolic values to avoid most underflows
  during reconstruction;
* produce deterministic, human friendly names for the synthetic values;
* keep the implementation approachable so it can act as a foundation for more
  sophisticated stack modelling in the future.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Mapping, Tuple, Set

from .ir import IRBlock, IRProgram
from .knowledge import KnowledgeBase
from .manual_semantics import InstructionSemantics
from .vm_analysis import estimate_stack_io
from .lua_ast import NameExpr, LuaExpression


@dataclass(frozen=True)
class BlockStackProfile:
    """Summarises how a block manipulates the VM stack."""

    start: int
    required_inputs: int
    net_delta: int
    max_depth: int


@dataclass(frozen=True)
class UnderflowEvent:
    """Record describing where the simulator detected a stack underflow."""

    block_start: int
    offset: int
    deficit: int
    mnemonic: str


@dataclass
class StackSeedPlan:
    """Collection of symbolic stack seeds for every block."""

    seeds: Dict[int, Tuple[str, ...]]
    placeholder_bases: Dict[int, str]

    def seed_expressions(self, block_start: int) -> List[LuaExpression]:
        names = self.seeds.get(block_start, ())
        return [NameExpr(name) for name in names]

    def placeholder_expression(self, block_start: int, counter: int) -> LuaExpression:
        base = self.placeholder_bases.get(block_start, "stack")
        return NameExpr(f"{base}_{counter}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "seeds": {start: list(names) for start, names in self.seeds.items()},
            "placeholder_bases": dict(self.placeholder_bases),
        }

    def blocks(self) -> List[int]:
        return sorted(self.seeds)


class StackSeedAnalyzer:
    """Analyse an :class:`IRProgram` and derive stack seed names."""

    def __init__(self, program: IRProgram, knowledge: KnowledgeBase) -> None:
        self.program = program
        self.knowledge = knowledge
        self.max_seed_depth = 16.0
        self._profiles: Dict[int, BlockStackProfile] = {}
        self._entry_depths: Dict[int, float] = {}
        self._underflows: List[UnderflowEvent] = []

    def build_plan(self) -> StackSeedPlan:
        profiles = {start: self._profile_block(block) for start, block in self.program.blocks.items()}
        self._profiles = profiles
        entry = min(self.program.blocks) if self.program.blocks else 0
        entry_depths = self._solve_entry_depths(profiles, entry)
        self._entry_depths = entry_depths
        self._underflows = self._simulate_underflow(entry_depths)
        seeds: Dict[int, Tuple[str, ...]] = {}
        bases: Dict[int, str] = {}
        for start, depth in entry_depths.items():
            base = "arg" if start == entry else f"block_{start:04X}_slot"
            count = max(0, int(round(depth)))
            seeds[start] = tuple(f"{base}_{index}" for index in range(count))
            bases[start] = base
        # Include unreachable blocks so placeholders remain deterministic.
        for start in self.program.blocks:
            if start not in seeds:
                base = f"block_{start:04X}_slot"
                seeds[start] = tuple()
                bases[start] = base
        return StackSeedPlan(seeds=seeds, placeholder_bases=bases)

    # ------------------------------------------------------------------
    def _profile_block(self, block: IRBlock) -> BlockStackProfile:
        depth = 0
        min_depth = 0
        max_depth = 0
        for instr in block.instructions:
            inputs, outputs = self._instruction_stack_io(instr.semantics)
            depth -= inputs
            min_depth = min(min_depth, depth)
            depth += outputs
            max_depth = max(max_depth, depth)
        required_inputs = int(max(0, -min_depth))
        net_delta = int(depth)
        return BlockStackProfile(
            start=block.start,
            required_inputs=required_inputs,
            net_delta=net_delta,
            max_depth=int(max_depth),
        )

    def _instruction_stack_io(self, semantics: InstructionSemantics) -> Tuple[int, int]:
        inputs, outputs = estimate_stack_io(semantics)
        return int(inputs), int(outputs)

    def _solve_entry_depths(
        self, profiles: Mapping[int, BlockStackProfile], entry: int
    ) -> Dict[int, float]:
        if not profiles:
            return {}

        entry_depths: Dict[int, float] = {entry: float(profiles[entry].required_inputs)}
        pending: Deque[int] = deque([entry])
        predecessor_map = self._compute_predecessors()

        while pending:
            block_start = pending.popleft()
            profile = profiles[block_start]
            previous_depth = entry_depths.get(block_start, float(profile.required_inputs))
            entry_depth = max(previous_depth, float(profile.required_inputs))
            entry_depth = min(entry_depth, self.max_seed_depth)
            exit_depth = entry_depth + profile.net_delta

            entry_depths[block_start] = entry_depth

            block = self.program.blocks.get(block_start)
            if block is None:
                continue

            for successor in block.successors:
                successor_profile = profiles.get(successor)
                if successor_profile is None:
                    continue
                candidate = max(float(successor_profile.required_inputs), exit_depth)
                candidate = min(candidate, self.max_seed_depth)
                previous = entry_depths.get(successor)
                if previous is None or candidate > previous + 1e-9:
                    entry_depths[successor] = candidate
                    pending.append(successor)

            # Revisit predecessors when the entry depth increases, ensuring that
            # loops converge.
            if previous_depth + 1e-9 < entry_depth:
                for predecessor in predecessor_map.get(block_start, ()):  # type: ignore[arg-type]
                    if predecessor not in pending:
                        pending.append(predecessor)

        return entry_depths

    def _compute_predecessors(self) -> Dict[int, Set[int]]:
        mapping: Dict[int, Set[int]] = {start: set() for start in self.program.blocks}
        for start, block in self.program.blocks.items():
            for successor in block.successors:
                if successor in mapping:
                    mapping[successor].add(start)
        return mapping

    def _simulate_underflow(self, entry_depths: Mapping[int, float]) -> List[UnderflowEvent]:
        events: List[UnderflowEvent] = []
        if not self.program.blocks:
            return events
        entry = min(self.program.blocks)
        pending: Deque[int] = deque([entry])
        visited: Set[int] = set()
        depth_map: Dict[int, float] = dict(entry_depths)

        while pending:
            block_start = pending.popleft()
            block = self.program.blocks.get(block_start)
            if block is None or block_start in visited:
                continue
            visited.add(block_start)
            profile = self._profiles.get(block_start)
            default_depth = float(profile.required_inputs) if profile else 0.0
            current_depth = depth_map.get(block_start, default_depth)
            for instr in block.instructions:
                inputs, outputs = self._instruction_stack_io(instr.semantics)
                if current_depth < inputs:
                    deficit = int(round(inputs - current_depth))
                    events.append(
                        UnderflowEvent(
                            block_start=block.start,
                            offset=instr.offset,
                            deficit=max(1, deficit),
                            mnemonic=instr.semantics.mnemonic,
                        )
                    )
                    current_depth = 0.0
                else:
                    current_depth -= inputs
                current_depth += outputs
            for successor in block.successors:
                successor_depth = depth_map.get(successor)
                if successor_depth is None or current_depth > successor_depth + 1e-9:
                    depth_map[successor] = current_depth
                    pending.append(successor)
        return events

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def diagnostics(self) -> List["StackSeedDiagnostics"]:
        """Return a detailed breakdown of the computed seed plan."""

        if not self._profiles or not self._entry_depths:
            self.build_plan()
        data: List[StackSeedDiagnostics] = []
        for start, profile in sorted(self._profiles.items()):
            entry_depth = self._entry_depths.get(start, float(profile.required_inputs))
            entry_depth = min(entry_depth, self.max_seed_depth)
            exit_depth = entry_depth + profile.net_delta
            underflow_count = sum(1 for event in self._underflows if event.block_start == start)
            data.append(
                StackSeedDiagnostics(
                    block_start=start,
                    required_inputs=profile.required_inputs,
                    net_delta=profile.net_delta,
                    entry_depth=entry_depth,
                    exit_depth=exit_depth,
                    max_depth=profile.max_depth,
                    underflows=underflow_count,
                )
            )
        return data

    def underflows(self) -> List[UnderflowEvent]:
        if not self._entry_depths and not self._profiles:
            self.build_plan()
        return list(self._underflows)


@dataclass(frozen=True)
class StackSeedDiagnostics:
    """Per-block diagnostic entry generated by :class:`StackSeedAnalyzer`."""

    block_start: int
    required_inputs: int
    net_delta: int
    entry_depth: float
    exit_depth: float
    max_depth: int
    underflows: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "block_start": self.block_start,
            "required_inputs": self.required_inputs,
            "net_delta": self.net_delta,
            "entry_depth": self.entry_depth,
            "exit_depth": self.exit_depth,
            "max_depth": self.max_depth,
            "underflows": self.underflows,
        }


def render_seed_plan(plan: StackSeedPlan, analyzer: StackSeedAnalyzer) -> str:
    """Return a human readable description of the generated seed plan."""

    lines = ["stack seed plan:"]
    diagnostics = analyzer.diagnostics()
    for entry in diagnostics:
        block_label = f"0x{entry.block_start:06X}"
        seeds = ", ".join(plan.seeds.get(entry.block_start, ())) or "<none>"
        lines.append(
            f"  {block_label}: entry={entry.entry_depth:.1f} exit={entry.exit_depth:.1f} "
            f"required={entry.required_inputs} underflows={entry.underflows} seeds=[{seeds}]"
        )
    if not diagnostics:
        lines.append("  <no blocks>")
    return "\n".join(lines) + "\n"


def build_stack_seed_plan(program: IRProgram, knowledge: KnowledgeBase) -> StackSeedPlan:
    """Convenience wrapper that returns a :class:`StackSeedPlan`."""

    analyzer = StackSeedAnalyzer(program, knowledge)
    return analyzer.build_plan()


@dataclass(frozen=True)
class StackSeedReport:
    """Composite report containing the plan and associated diagnostics."""

    plan: StackSeedPlan
    diagnostics: Tuple[StackSeedDiagnostics, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "diagnostics": [entry.to_dict() for entry in self.diagnostics],
        }

    def render(self) -> str:
        lines = ["stack seed diagnostics:"]
        for entry in self.diagnostics:
            label = f"0x{entry.block_start:06X}"
            lines.append(
                f"  {label}: required={entry.required_inputs} entry={entry.entry_depth:.1f} "
                f"exit={entry.exit_depth:.1f} delta={entry.net_delta} underflows={entry.underflows}"
            )
        if not self.diagnostics:
            lines.append("  <empty>")
        return "\n".join(lines) + "\n"

    def summary_lines(self) -> List[str]:
        total_blocks = len(self.plan.seeds)
        seeded = sum(1 for names in self.plan.seeds.values() if names)
        max_seed = max((len(names) for names in self.plan.seeds.values()), default=0)
        lines = [
            f"blocks analysed: {total_blocks}",
            f"blocks with seeds: {seeded}",
            f"maximum seeds per block: {max_seed}",
        ]
        if self.diagnostics:
            avg_entry = sum(entry.entry_depth for entry in self.diagnostics) / len(self.diagnostics)
            lines.append(f"average entry depth: {avg_entry:.2f}")
            total_underflows = sum(entry.underflows for entry in self.diagnostics)
            lines.append(f"total underflows: {total_underflows}")
        return lines


def build_stack_seed_report(program: IRProgram, knowledge: KnowledgeBase) -> StackSeedReport:
    """Return both the plan and diagnostics in a single structure."""

    analyzer = StackSeedAnalyzer(program, knowledge)
    plan = analyzer.build_plan()
    diagnostics = tuple(analyzer.diagnostics())
    return StackSeedReport(plan=plan, diagnostics=diagnostics)


def render_stack_seed_report(program: IRProgram, knowledge: KnowledgeBase) -> str:
    """Return a verbose text representation of the seed report."""

    report = build_stack_seed_report(program, knowledge)
    lines = ["stack seed report:"]
    lines.extend(f"- {line}" for line in report.summary_lines())
    lines.append("")
    lines.append(report.render().rstrip())
    return "\n".join(lines) + "\n"


def stack_seed_report_to_json(report: StackSeedReport, *, indent: int = 2) -> str:
    """Serialise ``report`` to a JSON string.

    The helper is primarily intended for debugging and regression tests where
    capturing the full state of the stack seed planner is desirable.  The
    output is stable thanks to ``sort_keys`` so that golden files are easy to
    maintain.
    """

    import json

    return json.dumps(report.to_dict(), indent=indent, sort_keys=True)


def filter_seed_diagnostics(
    report: StackSeedReport, *, min_required: int = 1
) -> List[StackSeedDiagnostics]:
    """Return diagnostics entries that require at least ``min_required`` inputs.

    Operators can use the result to focus on problematic blocks, for example
    by filtering for entries that require stack state to be materialised before
    analysis begins.
    """

    return [entry for entry in report.diagnostics if entry.required_inputs >= min_required]
