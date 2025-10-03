"""Deterministic finite automata support for pipeline pattern matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .patterns import PatternRegistry, PipelinePattern, PatternMatch
from .stack import StackEvent


@dataclass
class Transition:
    """Represents an automaton transition."""

    target: int
    token_index: int


@dataclass
class AutomatonState:
    """A single DFA state."""

    index: int
    transitions: Dict[int, Transition] = field(default_factory=dict)
    accepting: List[int] = field(default_factory=list)

    def add_transition(self, symbol: int, transition: Transition) -> None:
        self.transitions[symbol] = transition


@dataclass
class AutomatonMatch:
    """Match result produced by the DFA."""

    pattern_index: int
    start: int
    end: int
    score: float


@dataclass
class AutomatonTrace:
    """Trace produced while the DFA walks through a sequence."""

    start: int
    end: int
    states: Tuple[int, ...]
    matches: Tuple[AutomatonMatch, ...]

    def describe(self) -> str:
        path = "->".join(str(state) for state in self.states)
        return f"trace[{self.start}:{self.end}] path={path} matches={len(self.matches)}"


class DeterministicAutomaton:
    """Simple DFA for scanning instruction streams."""

    def __init__(self, registry: PatternRegistry) -> None:
        self.registry = registry
        self.patterns: Tuple[PipelinePattern, ...] = tuple(registry)
        self.states: List[AutomatonState] = []
        self._build_states()

    def _build_states(self) -> None:
        self.states = [AutomatonState(index=0)]
        for idx, pattern in enumerate(self.patterns):
            self._insert_pattern(idx, pattern)

    def _insert_pattern(self, index: int, pattern: PipelinePattern) -> None:
        state = self.states[0]
        for token_idx, _token in enumerate(pattern.tokens):
            key = self._transition_key(index, token_idx)
            if key not in state.transitions:
                next_state = AutomatonState(index=len(self.states))
                self.states.append(next_state)
                state.add_transition(key, Transition(target=next_state.index, token_index=token_idx))
            transition = state.transitions[key]
            state = self.states[transition.target]
        state.accepting.append(index)

    def _transition_key(self, pattern_index: int, token_index: int) -> int:
        return (pattern_index << 8) | token_index

    # ------------------------------------------------------------------
    # high level APIs
    # ------------------------------------------------------------------
    def iter_matches(self, events: Sequence[StackEvent]) -> Iterator[AutomatonMatch]:
        """Yield all matches found in ``events``."""

        for start in range(len(events)):
            for pattern_index, pattern in enumerate(self.patterns):
                end = start + len(pattern.tokens)
                if end > len(events):
                    continue
                slice_events = events[start:end]
                match = pattern.match(slice_events)
                if match is None:
                    continue
                if not pattern.allow_extra and len(slice_events) != len(pattern.tokens):
                    yield AutomatonMatch(pattern_index=pattern_index, start=start, end=end, score=match.score)
                else:
                    # allow extra instructions until a control boundary is hit
                    extra_end = end
                    while extra_end < len(events):
                        next_event = events[extra_end]
                        if next_event.profile.is_control():
                            break
                        extended = events[start : extra_end + 1]
                        extended_match = pattern.match(extended)
                        if extended_match is None:
                            break
                        extra_end += 1
                        match = extended_match
                    yield AutomatonMatch(pattern_index=pattern_index, start=start, end=extra_end, score=match.score)

    def best_match(self, events: Sequence[StackEvent]) -> Optional[PatternMatch]:
        """Return the best scoring :class:`PatternMatch` for ``events``."""

        best: Optional[PatternMatch] = None
        for pattern in self.patterns:
            match = pattern.match(events)
            if match is None:
                continue
            if best is None or match.score > best.score:
                best = match
        return best

    def scan(self, events: Sequence[StackEvent], *, window: Optional[int] = None) -> Tuple[AutomatonMatch, ...]:
        """Return all matches found in ``events`` within ``window`` steps."""

        matches: List[AutomatonMatch] = []
        limit = window or len(events)
        for start in range(len(events)):
            if start >= limit:
                break
            for match in self.iter_matches(events[start: start + limit]):
                matches.append(
                    AutomatonMatch(
                        pattern_index=match.pattern_index,
                        start=start + match.start,
                        end=start + match.end,
                        score=match.score,
                    )
                )
        return tuple(matches)

    def trace(self, events: Sequence[StackEvent]) -> AutomatonTrace:
        """Produce a trace that records state transitions for ``events``."""

        states = [0]
        matches: List[AutomatonMatch] = []
        for match in self.iter_matches(events):
            matches.append(match)
            states.append(match.end)
        return AutomatonTrace(start=0, end=len(events), states=tuple(states), matches=tuple(matches))
