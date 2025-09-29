"""Stack flow diagnostics for intermediate representation programs.

The high-level reconstructor previously relied purely on the symbolic stack
contained inside :mod:`mbcdisasm.highlevel`.  While that approach worked for
straight-line code it was difficult to reason about complex control-flow graphs
where multiple predecessors fed into a single block.  The new diagnostics module
performs a lightweight data-flow analysis directly on the IR representation.  It
provides rich metadata about entry/exit stack depths, underflow events and
structural anomalies that the renderer can surface in the generated Lua code.

The implementation deliberately favours debuggability over raw performance.  All
intermediate states are preserved as dataclasses which makes it easy for tests
and future tooling to inspect the modelling decisions.  The analyser is entirely
self-contained and does not rely on the high-level reconstruction layer which
keeps it useful for command-line utilities or future review tools.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ir import IRBlock, IRProgram
from .vm_analysis import estimate_stack_io


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StackEvent:
    """Single noteworthy occurrence while simulating a block."""

    offset: int
    message: str

    def describe(self) -> str:
        return f"0x{self.offset:06X}: {self.message}"

    def to_dict(self) -> dict:
        return {"offset": self.offset, "message": self.message}


@dataclass
class InstructionStackState:
    """Detailed stack information for a single instruction."""

    offset: int
    mnemonic: str
    entry_depth: int
    exit_depth: int
    consumed: int
    produced: int

    def describe(self) -> str:
        return (
            f"0x{self.offset:06X} {self.mnemonic}: depth {self.entry_depth}" \
            f"→{self.exit_depth} (consumed={self.consumed}, produced={self.produced})"
        )

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "mnemonic": self.mnemonic,
            "entry_depth": self.entry_depth,
            "exit_depth": self.exit_depth,
            "consumed": self.consumed,
            "produced": self.produced,
        }


@dataclass
class BlockStackState:
    """Result of simulating a single IR block."""

    start: int
    entry_tokens: List[str]
    exit_tokens: List[str]
    entry_depth: int
    exit_depth: int
    max_depth: int
    underflow_events: List[StackEvent] = field(default_factory=list)
    instruction_states: List[InstructionStackState] = field(default_factory=list)

    def underflow_count(self) -> int:
        return len(self.underflow_events)

    def describe(self) -> List[str]:
        lines = [
            (
                f"block 0x{self.start:06X}: entry={self.entry_depth} "
                f"exit={self.exit_depth} max={self.max_depth}"
            )
        ]
        if self.underflow_events:
            lines.append(f"  underflows: {len(self.underflow_events)}")
            for event in self.underflow_events[:3]:
                lines.append(f"    {event.describe()}")
            remaining = len(self.underflow_events) - 3
            if remaining > 0:
                lines.append(f"    ... ({remaining} more)")
        return lines

    def instruction_lines(self, limit: int = 5) -> List[str]:
        if not self.instruction_states:
            return []
        lines: List[str] = []
        for state in self.instruction_states[:limit]:
            lines.append(f"  {state.describe()}")
        remaining = len(self.instruction_states) - limit
        if remaining > 0:
            lines.append(f"  ... ({remaining} more instructions)")
        return lines

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "entry_tokens": list(self.entry_tokens),
            "exit_tokens": list(self.exit_tokens),
            "entry_depth": self.entry_depth,
            "exit_depth": self.exit_depth,
            "max_depth": self.max_depth,
            "underflows": [event.to_dict() for event in self.underflow_events],
            "instructions": [state.to_dict() for state in self.instruction_states],
        }


@dataclass
class StackDiagnostics:
    """Aggregated stack information for an entire program."""

    block_states: Dict[int, BlockStackState]
    max_depth: int
    total_underflows: int

    def summary_lines(self) -> List[str]:
        return [
            "stack analysis:",
            f"- blocks analysed: {len(self.block_states)}",
            f"- max stack depth: {self.max_depth}",
            f"- underflow events: {self.total_underflows}",
        ]

    def block_lines(self, limit: int = 6) -> List[str]:
        lines: List[str] = []
        for index, start in enumerate(sorted(self.block_states)):
            if index >= limit:
                remaining = len(self.block_states) - limit
                if remaining > 0:
                    lines.append(f"- ... ({remaining} additional blocks omitted)")
                break
            state = self.block_states[start]
            summary = (
                f"- block 0x{start:06X}: entry={state.entry_depth} "
                f"exit={state.exit_depth} max={state.max_depth}"
            )
            if state.underflow_events:
                summary += f" underflows={len(state.underflow_events)}"
            lines.append(summary)
        return lines

    def detailed_lines(self, limit: int = 2) -> List[str]:
        """Return verbose underflow descriptions grouped per block."""

        lines: List[str] = []
        for start in sorted(self.block_states):
            state = self.block_states[start]
            if not state.underflow_events:
                continue
            lines.append(f"- block 0x{start:06X} underflows:")
            for event in state.underflow_events[:limit]:
                lines.append(f"  {event.describe()}")
            remaining = len(state.underflow_events) - limit
            if remaining > 0:
                lines.append(f"  ... ({remaining} more)")
        return lines

    def instruction_trace_lines(
        self, *, block_limit: int = 2, instruction_limit: int = 4
    ) -> List[str]:
        lines: List[str] = []
        for index, start in enumerate(sorted(self.block_states)):
            if index >= block_limit:
                break
            state = self.block_states[start]
            trace = state.instruction_lines(limit=instruction_limit)
            if not trace:
                continue
            lines.append(f"- block 0x{start:06X} instruction trace:")
            lines.extend(trace)
        return lines

    def to_dict(self) -> dict:
        return {
            "blocks": {
                f"0x{start:06X}": state.to_dict()
                for start, state in self.block_states.items()
            },
            "max_depth": self.max_depth,
            "total_underflows": self.total_underflows,
        }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


class _TokenGenerator:
    """Generate stable symbolic tokens for stack simulation."""

    def __init__(self) -> None:
        self._counter = 0
        self._value_cache: Dict[Tuple[int, int, int], str] = {}
        self._phi_cache: Dict[Tuple[int, int], str] = {}

    def _next(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def value(self, block_start: int, offset: int, index: int) -> str:
        key = (block_start, offset, index)
        cached = self._value_cache.get(key)
        if cached is None:
            cached = self._next("value")
            self._value_cache[key] = cached
        return cached

    def phi(self, block_start: int, index: int) -> str:
        key = (block_start, index)
        cached = self._phi_cache.get(key)
        if cached is None:
            cached = self._next("phi")
            self._phi_cache[key] = cached
        return cached


def analyze_stack(program: IRProgram) -> StackDiagnostics:
    """Perform a symbolic stack walk for ``program``."""

    blocks = program.blocks
    if not blocks:
        return StackDiagnostics({}, 0, 0)

    order = sorted(blocks)
    entry = order[0]

    token_seeds: Dict[int, List[str]] = {entry: []}
    block_states: Dict[int, BlockStackState] = {}
    worklist: deque[int] = deque([entry])
    processed: Dict[int, List[str]] = {}
    tokens = _TokenGenerator()

    while worklist:
        start = worklist.popleft()
        entry_tokens = token_seeds.get(start, [])
        previous = processed.get(start)
        if previous is not None and previous == entry_tokens:
            continue
        processed[start] = list(entry_tokens)

        block = blocks[start]
        state, exit_tokens = _simulate_block(block, entry_tokens, tokens)
        block_states[start] = state

        for successor in block.successors:
            if successor not in blocks:
                continue
            merged, changed = _merge_token_sequences(
                token_seeds.get(successor), exit_tokens, tokens, successor
            )
            if merged is not None:
                token_seeds[successor] = merged
            if changed:
                worklist.append(successor)
            elif successor not in processed:
                worklist.append(successor)

    max_depth = max((state.max_depth for state in block_states.values()), default=0)
    total_underflows = sum(state.underflow_count() for state in block_states.values())
    return StackDiagnostics(block_states, max_depth, total_underflows)


def _simulate_block(
    block: IRBlock,
    entry_tokens: Sequence[str],
    tokens: _TokenGenerator,
) -> Tuple[BlockStackState, List[str]]:
    stack: List[str] = list(entry_tokens)
    events: List[StackEvent] = []
    max_depth = len(stack)
    instruction_states: List[InstructionStackState] = []

    for instruction in block.instructions:
        inputs, outputs = estimate_stack_io(instruction.semantics)
        available = len(stack)
        entry_depth = available
        consumed = inputs
        produced = outputs
        if inputs > available:
            events.append(
                StackEvent(
                    instruction.offset,
                    f"pop {inputs} values from stack of size {available}",
                )
            )
            stack.clear()
            consumed = available
        else:
            for _ in range(inputs):
                stack.pop()
        produced = [
            tokens.value(block.start, instruction.offset, index)
            for index in range(outputs)
        ]
        stack.extend(produced)
        if len(stack) > max_depth:
            max_depth = len(stack)
        instruction_states.append(
            InstructionStackState(
                offset=instruction.offset,
                mnemonic=instruction.mnemonic,
                entry_depth=entry_depth,
                exit_depth=len(stack),
                consumed=consumed,
                produced=produced,
            )
        )

    state = BlockStackState(
        start=block.start,
        entry_tokens=list(entry_tokens),
        exit_tokens=list(stack),
        entry_depth=len(entry_tokens),
        exit_depth=len(stack),
        max_depth=max_depth,
        underflow_events=events,
        instruction_states=instruction_states,
    )
    return state, stack


def _merge_token_sequences(
    existing: Optional[List[str]],
    incoming: List[str],
    tokens: _TokenGenerator,
    successor: int,
) -> Tuple[Optional[List[str]], bool]:
    if existing is None:
        return list(incoming), True

    changed = False
    length = max(len(existing), len(incoming))
    merged: List[str] = []
    for index in range(length):
        lhs = existing[index] if index < len(existing) else None
        rhs = incoming[index] if index < len(incoming) else None
        lhs_kind = _token_kind(lhs)
        rhs_kind = _token_kind(rhs)

        if lhs is None and rhs is None:
            token = tokens.phi(successor, index)
        elif lhs is None:
            token = rhs if rhs is not None else tokens.phi(successor, index)
        elif rhs is None:
            if lhs_kind == "phi":
                token = lhs
            else:
                token = tokens.phi(successor, index)
        elif lhs == rhs:
            token = lhs
        elif lhs_kind == "phi" and rhs_kind != "phi":
            token = rhs
        elif rhs_kind == "phi" and lhs_kind != "phi":
            token = lhs
        else:
            token = tokens.phi(successor, index)
        current = existing[index] if index < len(existing) else None
        if token != current:
            changed = True
        merged.append(token)

    if len(merged) != len(existing):
        changed = True
    return merged, changed


def _token_kind(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    if token.startswith("value_"):
        return "value"
    if token.startswith("phi_"):
        return "phi"
    return "unknown"


# ---------------------------------------------------------------------------
# Rendering helpers for external callers
# ---------------------------------------------------------------------------


def render_stack_diagnostics(diagnostics: StackDiagnostics) -> List[str]:
    """Return a flattened comment-friendly representation."""

    lines = diagnostics.summary_lines()
    block_lines = diagnostics.block_lines()
    if block_lines:
        lines.append("block overview:")
        lines.extend(block_lines)
    detail_lines = diagnostics.detailed_lines()
    if detail_lines:
        lines.append("underflow details:")
        lines.extend(detail_lines)
    trace_lines = diagnostics.instruction_trace_lines()
    if trace_lines:
        lines.append("instruction traces:")
        lines.extend(trace_lines)
    return lines


__all__ = [
    "StackDiagnostics",
    "BlockStackState",
    "StackEvent",
    "InstructionStackState",
    "analyze_stack",
    "render_stack_diagnostics",
]
