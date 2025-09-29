"""Stack analysis utilities that reduce placeholder churn during translation.

The initial version of the high level reconstructor treated the VM stack as an
opaque structure.  Whenever an instruction consumed more values than the model
had tracked so far the stack implementation materialised a synthetic placeholder
and recorded an ``underflow`` warning.  Large functions featuring nested
branches or multiple logical entry points frequently triggered these fallbacks,
polluting the recovered source code with meaningless identifiers.

This module analyses IR programs to estimate the stack requirements of each
basic block.  The :class:`StackBalanceAnalyzer` derives per-block profiles that
describe how many values are consumed before new ones are produced and computes
the minimal depth needed to execute the block without underflow.  The resulting
:class:`StackSeedPlan` provides consistent symbolic names that can be used to
seed :class:`~mbcdisasm.highlevel.HighLevelStack` instances and to service
occasional underflows with human-readable placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

from .ir import IRBlock, IRProgram
from .lua_ast import NameExpr


@dataclass(frozen=True)
class StackStep:
    """Intermediate accounting step while traversing a block."""

    offset: int
    inputs: int
    outputs: int
    depth_before: int
    depth_after: int


@dataclass
class BlockStackProfile:
    """Captures the relative stack usage for a basic block."""

    start: int
    steps: List[StackStep] = field(default_factory=list)
    min_depth: int = 0
    max_depth: int = 0
    net_delta: int = 0
    required_entry: int = 0

    def describe(self) -> List[str]:
        lines = [
            f"block 0x{self.start:06X}: netΔ={self.net_delta} required={self.required_entry}",
            f"  min_depth={self.min_depth} max_depth={self.max_depth}",
        ]
        for step in self.steps:
            lines.append(
                "  "
                + (
                    f"@0x{step.offset:06X} before={step.depth_before} "
                    f"after={step.depth_after} inputs={step.inputs} outputs={step.outputs}"
                )
            )
        return lines


@dataclass(frozen=True)
class StackSeedPlan:
    """Plan describing how to initialise the high level stack."""

    initial_depth: int
    slot_names: Tuple[str, ...]
    block_entry_depths: Mapping[int, int]
    profiles: Mapping[int, BlockStackProfile]
    fallback_prefix: str = "stack_seed"

    def initial_values(self) -> List[NameExpr]:
        """Return the expressions that should seed the symbolic stack."""

        return [NameExpr(name) for name in self.slot_names]

    def fallback_name(self, index: int) -> NameExpr:
        """Return the expression used for additional underflow events."""

        name = f"{self.fallback_prefix}_{index}"
        return NameExpr(name)


class StackUnderflowProvider:
    """Callable object used by :class:`HighLevelStack` to service underflows."""

    def __init__(self, plan: StackSeedPlan) -> None:
        self._plan = plan
        self._extra_index = 0

    def seed(self) -> List[NameExpr]:
        return self._plan.initial_values()

    def __call__(self) -> Tuple[NameExpr, str]:
        expr = self._plan.fallback_name(self._extra_index)
        self._extra_index += 1
        return expr, f"underflow satisfied by synthetic {expr.name}"


class StackBalanceAnalyzer:
    """Analyse IR programs to derive stack seeding hints."""

    def __init__(self, program: IRProgram) -> None:
        self._program = program

    def build_profiles(self) -> Dict[int, BlockStackProfile]:
        profiles: Dict[int, BlockStackProfile] = {}
        for start, block in self._program.blocks.items():
            profiles[start] = self._profile_block(block)
        return profiles

    def plan(self, *, fallback_prefix: str = "stack_seed") -> StackSeedPlan:
        profiles = self.build_profiles()
        entry = min(self._program.blocks)
        entry_depths = self._propagate_depths(entry, profiles)
        initial_depth = max(entry_depths.values()) if entry_depths else 0
        slot_names = tuple(f"arg_{index}" for index in range(initial_depth))
        return StackSeedPlan(
            initial_depth=initial_depth,
            slot_names=slot_names,
            block_entry_depths=entry_depths,
            profiles=profiles,
            fallback_prefix=fallback_prefix,
        )

    # ------------------------------------------------------------------
    def _profile_block(self, block: IRBlock) -> BlockStackProfile:
        profile = BlockStackProfile(start=block.start)
        depth = 0
        min_depth = 0
        max_depth = 0
        for instr in block.instructions:
            before = depth
            depth -= instr.stack_inputs
            min_depth = min(min_depth, depth)
            depth += instr.stack_outputs
            max_depth = max(max_depth, depth)
            profile.steps.append(
                StackStep(
                    offset=instr.offset,
                    inputs=instr.stack_inputs,
                    outputs=instr.stack_outputs,
                    depth_before=before,
                    depth_after=depth,
                )
            )
        profile.min_depth = min_depth
        profile.max_depth = max_depth
        profile.net_delta = depth
        profile.required_entry = abs(min_depth)
        return profile

    def _propagate_depths(
        self, entry: int, profiles: Mapping[int, BlockStackProfile]
    ) -> Dict[int, int]:
        pending: List[int] = [entry]
        entry_depths: Dict[int, int] = {entry: profiles[entry].required_entry}
        while pending:
            start = pending.pop(0)
            profile = profiles[start]
            required = max(entry_depths[start], profile.required_entry)
            if entry_depths[start] != required:
                entry_depths[start] = required
            exit_depth = required + profile.net_delta
            for succ in self._program.blocks[start].successors:
                next_profile = profiles.get(succ)
                if next_profile is None:
                    continue
                next_entry = max(exit_depth, next_profile.required_entry)
                previous = entry_depths.get(succ)
                if previous is None or next_entry > previous:
                    entry_depths[succ] = next_entry
                    if succ not in pending:
                        pending.append(succ)
        return entry_depths

