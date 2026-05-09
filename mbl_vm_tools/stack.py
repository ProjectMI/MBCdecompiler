from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .control_flow import FunctionCFG, ModuleCFGReport, analyze_module, word_to_dict
from .parser import MBCModule
from .vm_spec import VMWord, call_script_target_offset, is_lower_operand_atom, word_shape_signature

RETURN_TERMINALS = {"RETURN_PAIR", "END"}
BRANCH_FRAME_EFFECT_MODES = ("clear", "keep", "pop_candidates")


def _bit_count(value: int) -> int:
    return int(value).bit_count()


@dataclass(frozen=True)
class StackState:
    """Two-layer abstract VM stack state.

    ``value`` is the persistent VM value stack approximation.
    ``frame`` is the local lower-operand frame created by decoded literal/ref/
    record atoms.  Frame atoms do not flow as persistent stack values; calls and
    branches may bind or clear them, but they are tracked separately from values.
    """

    rows: tuple[int, ...]
    value_cap: int = 64
    frame_cap: int = 32

    @classmethod
    def initial(cls, *, value_cap: int = 64, frame_cap: int = 32, value_depth: int = 0, frame_depth: int = 0) -> "StackState":
        rows = [0] * (value_cap + 1)
        rows[min(value_cap, max(0, value_depth))] = 1 << min(frame_cap, max(0, frame_depth))
        return cls(tuple(rows), value_cap, frame_cap)

    @classmethod
    def empty_state(cls, *, value_cap: int = 64, frame_cap: int = 32) -> "StackState":
        return cls(tuple([0] * (value_cap + 1)), value_cap, frame_cap)

    @property
    def empty(self) -> bool:
        return not any(self.rows)

    @property
    def pair_count(self) -> int:
        return sum(_bit_count(row) for row in self.rows)

    @property
    def hit_value_cap(self) -> bool:
        return bool(self.rows and self.rows[self.value_cap])

    @property
    def hit_frame_cap(self) -> bool:
        mask = 1 << self.frame_cap
        return any(row & mask for row in self.rows)

    @property
    def max_value_depth(self) -> int:
        for i in range(len(self.rows) - 1, -1, -1):
            if self.rows[i]:
                return i
        return -1

    @property
    def min_value_depth(self) -> int:
        for i, row in enumerate(self.rows):
            if row:
                return i
        return -1

    @property
    def max_frame_depth(self) -> int:
        best = -1
        for row in self.rows:
            if row:
                best = max(best, row.bit_length() - 1)
        return best

    @property
    def min_frame_depth(self) -> int:
        best: int | None = None
        for row in self.rows:
            if row:
                depth = (row & -row).bit_length() - 1
                best = depth if best is None else min(best, depth)
        return -1 if best is None else best

    def join(self, other: "StackState") -> "StackState":
        if self.value_cap != other.value_cap or self.frame_cap != other.frame_cap:
            raise ValueError("cannot join StackState values with different caps")
        return StackState(tuple(a | b for a, b in zip(self.rows, other.rows)), self.value_cap, self.frame_cap)

    def iter_pairs(self, limit: int | None = None) -> Iterable[tuple[int, int]]:
        emitted = 0
        for value_depth, row in enumerate(self.rows):
            bits = row
            frame_depth = 0
            while bits:
                if bits & 1:
                    yield value_depth, frame_depth
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return
                bits >>= 1
                frame_depth += 1

    def push_frame(self, count: int = 1) -> "StackState":
        if count <= 0 or self.empty:
            return self
        mask = (1 << (self.frame_cap + 1)) - 1
        rows: list[int] = []
        for row in self.rows:
            shifted = (row << count) & mask
            if row >> max(0, self.frame_cap - count + 1):
                shifted |= 1 << self.frame_cap
            rows.append(shifted)
        return StackState(tuple(rows), self.value_cap, self.frame_cap)

    def clear_frame(self) -> "StackState":
        rows = [1 if row else 0 for row in self.rows]
        return StackState(tuple(rows), self.value_cap, self.frame_cap)

    def push_value_range(self, lo: int, hi: int) -> "StackState":
        if self.empty:
            return self
        lo = max(0, int(lo))
        hi = max(lo, int(hi))
        rows = [0] * (self.value_cap + 1)
        for value_depth, row in enumerate(self.rows):
            if not row:
                continue
            for count in range(lo, hi + 1):
                new_value_depth = min(self.value_cap, value_depth + count)
                rows[new_value_depth] |= row
        return StackState(tuple(rows), self.value_cap, self.frame_cap)

    def consume_call_frame(self, encoded_count: int, *, return_min: int, return_max: int, collect_sources: bool = False) -> tuple["StackState", bool, bool, Counter[str]]:
        """Consume a call frame from local operands first, with value fallback.

        The VM corpus falsifies the old model where every lower atom is a
        persistent push.  It also falsifies a strict adjacent-frame-only model.
        The stable abstraction is therefore a split: the encoded frame count can
        be satisfied by some mix of current lower frame atoms and persistent
        values.  If no split is possible, the call hypothesis is falsified.
        """

        encoded_count = max(0, int(encoded_count))
        return_min = max(0, int(return_min))
        return_max = max(return_min, int(return_max))
        source_shapes: Counter[str] = Counter()
        rows = [0] * (self.value_cap + 1)

        # A zero-count call cannot be a consumer of the local operand frame by
        # construction.  The corpus repeatedly uses frame=0 native calls between
        # lower operand atoms and a later non-zero call; clearing here produces
        # false definite underflows.
        if encoded_count == 0:
            for value_depth, frame_depth in self.iter_pairs():
                if collect_sources:
                    source_shapes["frame=0:value=0"] += 1
                for produced in range(return_min, return_max + 1):
                    new_value = min(self.value_cap, value_depth + produced)
                    rows[new_value] |= 1 << frame_depth
            out = StackState(tuple(rows), self.value_cap, self.frame_cap)
            return out, False, out.empty, source_shapes

        some_underflow = False
        for value_depth, frame_depth in self.iter_pairs():
            valid_for_pair = False
            max_frame_take = min(frame_depth, encoded_count)
            for frame_take in range(max_frame_take + 1):
                value_take = encoded_count - frame_take
                if value_depth < value_take:
                    continue
                valid_for_pair = True
                if collect_sources:
                    source_shapes[f"frame={frame_take}:value={value_take}"] += 1
                base_value = value_depth - value_take
                new_frame = frame_depth - frame_take
                for produced in range(return_min, return_max + 1):
                    new_value = min(self.value_cap, base_value + produced)
                    rows[new_value] |= 1 << new_frame
            if not valid_for_pair:
                some_underflow = True
        out = StackState(tuple(rows), self.value_cap, self.frame_cap)
        return out, some_underflow, out.empty, source_shapes

    def pop_frame_candidates(self, candidates: tuple[int, ...]) -> tuple["StackState", bool, bool]:
        rows = [0] * (self.value_cap + 1)
        some_underflow = False
        for value_depth, frame_depth in self.iter_pairs():
            valid = False
            for count in candidates:
                count = max(0, int(count))
                if frame_depth >= count:
                    rows[value_depth] |= 1 << (frame_depth - count)
                    valid = True
            if not valid:
                some_underflow = True
        out = StackState(tuple(rows), self.value_cap, self.frame_cap)
        return out, some_underflow, out.empty

    def depths_at_return(self, limit: int = 4096) -> list[tuple[int, int]]:
        return list(self.iter_pairs(limit))

    def range_key(self) -> str:
        return (
            f"value={self.min_value_depth}..{self.max_value_depth}"
            f"/frame={self.min_frame_depth}..{self.max_frame_depth}"
            + (":vcap" if self.hit_value_cap else "")
            + (":fcap" if self.hit_frame_cap else "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_min": self.min_value_depth,
            "value_max": self.max_value_depth,
            "frame_min": self.min_frame_depth,
            "frame_max": self.max_frame_depth,
            "pair_count": self.pair_count,
            "hit_value_cap": self.hit_value_cap,
            "hit_frame_cap": self.hit_frame_cap,
            "pairs": list(self.iter_pairs(48)),
        }


@dataclass(frozen=True)
class StackIssue:
    severity: str
    code: str
    function_name: str
    offset: Optional[int] = None
    target: Optional[int] = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StackHypothesis:
    """One falsifiable low-level stack/effect hypothesis."""

    name: str = "two_layer_value_stack_operand_frame"
    value_cap: int = 64
    frame_cap: int = 32
    native_return_min: int = 0
    native_return_max: int = 2
    script_return_min: int = 0
    script_return_max: int = 2
    branch_frame_effect: str = "clear"
    branch_pop_candidates: tuple[int, ...] = (0, 1)
    start_with_declared_arity: bool = False
    collect_call_source_shapes: bool = False

    def __post_init__(self) -> None:
        if self.branch_frame_effect not in BRANCH_FRAME_EFFECT_MODES:
            raise ValueError(f"unsupported branch_frame_effect: {self.branch_frame_effect}")

    @property
    def max_depth(self) -> int:
        # Backward-compatible alias for reports/tools that still read this name.
        return self.value_cap

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["branch_pop_candidates"] = list(self.branch_pop_candidates)
        payload["model"] = "two_layer_operand_frame_and_persistent_value_stack"
        return payload


@dataclass
class FunctionABI:
    name: str
    symbol: str
    absolute_start: int
    declared_param_arity: Optional[int]
    prologue_kind: Optional[str]
    prologue_raw_hex: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionStackReport:
    function_name: str
    symbol: str
    span: Optional[tuple[int, int]]
    declared_param_arity: Optional[int]
    prologue_kind: Optional[str]
    cfg_node_count: int
    analyzed_node_count: int
    max_observed_value_depth: int
    max_observed_operand_frame_depth: int
    hit_value_cap: bool
    hit_operand_frame_cap: bool
    return_states: Counter[tuple[str, int, int]]
    branch_input_ranges: Counter[str]
    branch_underflow_evidence: Counter[str]
    call_input_ranges: Counter[str]
    call_underflow_evidence: Counter[str]
    call_frame_source_shapes: Counter[str]
    issues: list[StackIssue]

    @property
    def hard_error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "symbol": self.symbol,
            "span": self.span,
            "declared_param_arity": self.declared_param_arity,
            "prologue_kind": self.prologue_kind,
            "cfg_node_count": self.cfg_node_count,
            "analyzed_node_count": self.analyzed_node_count,
            "max_observed_value_depth": self.max_observed_value_depth,
            "max_observed_operand_frame_depth": self.max_observed_operand_frame_depth,
            "hit_value_cap": self.hit_value_cap,
            "hit_operand_frame_cap": self.hit_operand_frame_cap,
            "return_states": {f"{term}:value={value}:frame={frame}": count for (term, value, frame), count in self.return_states.items()},
            "branch_input_ranges": dict(self.branch_input_ranges),
            "branch_underflow_evidence": dict(self.branch_underflow_evidence),
            "call_input_ranges": dict(self.call_input_ranges),
            "call_underflow_evidence": dict(self.call_underflow_evidence),
            "call_frame_source_shapes_top": dict(self.call_frame_source_shapes.most_common(16)),
            "hard_error_count": self.hard_error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues[:32]],
        }


@dataclass
class ModuleStackReport:
    module_path: str
    hypothesis: StackHypothesis
    function_reports: list[FunctionStackReport]
    function_abis: list[FunctionABI]
    call_script_target_relations: Counter[str]
    call_script_arity_relations: Counter[str]
    call_script_arity_mismatches: list[dict[str, Any]]
    native_call_shapes: Counter[str]

    @property
    def summary(self) -> dict[str, Any]:
        issue_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        branch_inputs: Counter[str] = Counter()
        branch_underflow: Counter[str] = Counter()
        call_inputs: Counter[str] = Counter()
        call_underflow: Counter[str] = Counter()
        call_sources: Counter[str] = Counter()
        return_states: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        functions_with_errors = 0
        functions_with_warnings = 0
        functions_hitting_value_cap = 0
        functions_hitting_frame_cap = 0

        for report in self.function_reports:
            totals["functions"] += 1
            totals["cfg_nodes"] += report.cfg_node_count
            totals["analyzed_nodes"] += report.analyzed_node_count
            totals["max_observed_value_depth"] = max(totals["max_observed_value_depth"], report.max_observed_value_depth)
            totals["max_observed_operand_frame_depth"] = max(totals["max_observed_operand_frame_depth"], report.max_observed_operand_frame_depth)
            if report.declared_param_arity is not None:
                totals["functions_with_declared_arity"] += 1
            if report.hit_value_cap:
                functions_hitting_value_cap += 1
            if report.hit_operand_frame_cap:
                functions_hitting_frame_cap += 1
            if report.hard_error_count:
                functions_with_errors += 1
            if report.warning_count:
                functions_with_warnings += 1
            branch_inputs.update(report.branch_input_ranges)
            branch_underflow.update(report.branch_underflow_evidence)
            call_inputs.update(report.call_input_ranges)
            call_underflow.update(report.call_underflow_evidence)
            call_sources.update(report.call_frame_source_shapes)
            for (term, value, frame), count in report.return_states.items():
                return_states[f"{term}:value={value}:frame={frame}"] += count
            for issue in report.issues:
                issue_counts[issue.code] += 1
                severity_counts[issue.severity] += 1

        return {
            "module_path": self.module_path,
            "hypothesis": self.hypothesis.to_dict(),
            "totals": dict(totals),
            "functions_hitting_value_cap": functions_hitting_value_cap,
            "functions_hitting_operand_frame_cap": functions_hitting_frame_cap,
            "functions_with_errors": functions_with_errors,
            "functions_with_warnings": functions_with_warnings,
            "call_script_target_relations": dict(self.call_script_target_relations),
            "call_script_arity_relations": dict(self.call_script_arity_relations),
            "call_script_arity_mismatch_count": len(self.call_script_arity_mismatches),
            "call_script_arity_mismatch_examples": self.call_script_arity_mismatches[:16],
            "native_call_shapes_top": dict(self.native_call_shapes.most_common(32)),
            "branch_input_ranges_top": dict(branch_inputs.most_common(32)),
            "branch_underflow_evidence": dict(branch_underflow),
            "call_input_ranges_top": dict(call_inputs.most_common(32)),
            "call_underflow_evidence_top": dict(call_underflow.most_common(32)),
            "call_frame_source_shapes_top": dict(call_sources.most_common(32)),
            "return_states_top": dict(return_states.most_common(32)),
            "issue_counts": dict(issue_counts),
            "severity_counts": dict(severity_counts),
            "notable_functions": [
                report.to_summary_dict()
                for report in self.function_reports
                if report.hard_error_count or report.warning_count or report.hit_value_cap or report.hit_operand_frame_cap
            ][:32],
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return self.summary


def _declared_abi_from_cfg(cfg: FunctionCFG) -> FunctionABI:
    absolute_start = int(cfg.span[0]) if cfg.span else 0
    declared_arity: Optional[int] = None
    prologue_kind: Optional[str] = None
    prologue_raw_hex: Optional[str] = None
    first = cfg.linear_words[0] if cfg.linear_words else None
    if first is not None:
        prologue_kind = first.terminal_kind
        prologue_raw_hex = first.raw.hex(" ")
        if first.terminal_kind in {"AGG", "AGG0"}:
            declared_arity = int(first.operands.get("arity") or 0)
    return FunctionABI(
        name=cfg.function_name,
        symbol=cfg.symbol,
        absolute_start=absolute_start,
        declared_param_arity=declared_arity,
        prologue_kind=prologue_kind,
        prologue_raw_hex=prologue_raw_hex,
    )


def _branch_key(word: VMWord) -> str:
    op = int(word.operands.get("op", -1) or -1) & 0xFF
    prefix = ".".join(f"{int(p):02X}" for p in word.prefixes) if word.prefixes else "-"
    return f"BR_0x{op:02X}/pfx={prefix}"


def _initial_state_for_root(cfg: FunctionCFG, hypothesis: StackHypothesis) -> StackState:
    value_depth = 0
    if hypothesis.start_with_declared_arity and cfg.linear_words and cfg.linear_words[0].terminal_kind in {"AGG", "AGG0"}:
        value_depth = int(cfg.linear_words[0].operands.get("arity") or 0)
    return StackState.initial(value_cap=hypothesis.value_cap, frame_cap=hypothesis.frame_cap, value_depth=value_depth, frame_depth=0)


def _apply_word_effect(
    word: VMWord,
    state: StackState,
    *,
    hypothesis: StackHypothesis,
    function_name: str,
    offset: int,
    issues: list[StackIssue],
    branch_input_ranges: Counter[str],
    branch_underflow_evidence: Counter[str],
    call_input_ranges: Counter[str],
    call_underflow_evidence: Counter[str],
    call_frame_source_shapes: Counter[str],
) -> StackState:
    terminal = word.terminal_kind

    if is_lower_operand_atom(word):
        return state.push_frame(1)

    if terminal in {"BARE_U32", "AGG", "AGG0", "MARK", "NOP", "UNKNOWN"}:
        return state

    if terminal in {"CALL_NATIVE", "CALL_SCRIPT"}:
        encoded_count = int(word.operands.get("argc", 0) or 0)
        shape = word_shape_signature(word)
        ret_min, ret_max = (
            (hypothesis.native_return_min, hypothesis.native_return_max)
            if terminal == "CALL_NATIVE"
            else (hypothesis.script_return_min, hypothesis.script_return_max)
        )
        after_call, some_underflow, definite_underflow, source_shapes = state.consume_call_frame(
            encoded_count,
            return_min=ret_min,
            return_max=ret_max,
            collect_sources=False,
        )
        if definite_underflow:
            return StackState.empty_state(value_cap=hypothesis.value_cap, frame_cap=hypothesis.frame_cap)
        return after_call

    if terminal == "BR":
        key = _branch_key(word)
        mode = hypothesis.branch_frame_effect
        if mode == "keep":
            return state
        if mode == "clear":
            return state.clear_frame()
        if mode == "pop_candidates":
            out, some_underflow, definite_underflow = state.pop_frame_candidates(hypothesis.branch_pop_candidates)
            if definite_underflow:
                return StackState.empty_state(value_cap=hypothesis.value_cap, frame_cap=hypothesis.frame_cap)
            return out
        raise ValueError(f"unsupported branch_frame_effect: {mode}")

    # Returns are handled by caller because they terminate a region.
    return state


def analyze_function_stack(cfg: FunctionCFG, *, hypothesis: StackHypothesis = StackHypothesis()) -> FunctionStackReport:
    abi = _declared_abi_from_cfg(cfg)
    issues: list[StackIssue] = []
    branch_input_ranges: Counter[str] = Counter()
    branch_underflow_evidence: Counter[str] = Counter()
    call_input_ranges: Counter[str] = Counter()
    call_underflow_evidence: Counter[str] = Counter()
    call_frame_source_shapes: Counter[str] = Counter()
    return_states: Counter[tuple[str, int, int]] = Counter()

    if not cfg.nodes:
        return FunctionStackReport(
            cfg.function_name,
            cfg.symbol,
            cfg.span,
            abi.declared_param_arity,
            abi.prologue_kind,
            0,
            0,
            0,
            0,
            False,
            False,
            return_states,
            branch_input_ranges,
            branch_underflow_evidence,
            call_input_ranges,
            call_underflow_evidence,
            call_frame_source_shapes,
            issues,
        )

    successors: dict[int, list[int]] = defaultdict(list)
    for edge in cfg.edges:
        if edge.dst is not None and edge.kind != "code_ref_reference":
            successors[int(edge.src)].append(int(edge.dst))

    roots = {0, *cfg.code_ref_region_roots}
    states: dict[int, StackState] = {}
    queue: deque[int] = deque()
    root_state = _initial_state_for_root(cfg, hypothesis)
    for root in sorted(roots):
        if root in cfg.nodes:
            states[root] = root_state
            queue.append(root)

    max_observed_value_depth = 0
    max_observed_frame_depth = 0
    hit_value_cap = False
    hit_frame_cap = False
    steps = 0
    max_steps = max(1000, len(cfg.nodes) * 512)
    while queue and steps < max_steps:
        offset = queue.popleft()
        steps += 1
        node = cfg.nodes[offset]
        state = states[offset]
        max_observed_value_depth = max(max_observed_value_depth, state.max_value_depth)
        max_observed_frame_depth = max(max_observed_frame_depth, state.max_frame_depth)
        hit_value_cap = hit_value_cap or state.hit_value_cap
        hit_frame_cap = hit_frame_cap or state.hit_frame_cap

        if node.word.terminal_kind in RETURN_TERMINALS:
            continue

        out_state = _apply_word_effect(
            node.word,
            state,
            hypothesis=hypothesis,
            function_name=cfg.function_name,
            offset=offset,
            issues=issues,
            branch_input_ranges=branch_input_ranges,
            branch_underflow_evidence=branch_underflow_evidence,
            call_input_ranges=call_input_ranges,
            call_underflow_evidence=call_underflow_evidence,
            call_frame_source_shapes=call_frame_source_shapes,
        )
        max_observed_value_depth = max(max_observed_value_depth, out_state.max_value_depth)
        max_observed_frame_depth = max(max_observed_frame_depth, out_state.max_frame_depth)
        hit_value_cap = hit_value_cap or out_state.hit_value_cap
        hit_frame_cap = hit_frame_cap or out_state.hit_frame_cap

        if out_state.empty:
            continue
        for dst in successors.get(offset, []):
            old = states.get(dst)
            if old is None:
                states[dst] = out_state
                queue.append(dst)
            else:
                joined = old.join(out_state)
                if joined.rows != old.rows:
                    states[dst] = joined
                    queue.append(dst)

    if steps >= max_steps and queue:
        issues.append(
            StackIssue(
                "warning",
                "abstract_stack_step_limit",
                cfg.function_name,
                message="Two-layer stack propagation hit the per-function step limit; report is partial.",
                details={"step_limit": max_steps, "pending_nodes": len(queue)},
            )
        )

    # Diagnostics are collected only after the fixed point.  During propagation
    # a temporarily shallow incoming state may later be joined with a valid one;
    # reporting underflow eagerly would turn invalid abstract paths into false
    # hard errors.
    for offset, state in sorted(states.items()):
        word = cfg.nodes[offset].word
        terminal = word.terminal_kind
        if terminal in RETURN_TERMINALS:
            for value_depth, frame_depth in state.depths_at_return(4096):
                return_states[(terminal, value_depth, frame_depth)] += 1
        elif terminal in {"CALL_NATIVE", "CALL_SCRIPT"}:
            encoded_count = int(word.operands.get("argc", 0) or 0)
            shape = word_shape_signature(word)
            call_input_ranges[f"{shape}:input={state.range_key()}"] += 1
            ret_min, ret_max = (
                (hypothesis.native_return_min, hypothesis.native_return_max)
                if terminal == "CALL_NATIVE"
                else (hypothesis.script_return_min, hypothesis.script_return_max)
            )
            _out, some_underflow, definite_underflow, source_shapes = state.consume_call_frame(
                encoded_count,
                return_min=ret_min,
                return_max=ret_max,
                collect_sources=hypothesis.collect_call_source_shapes,
            )
            if hypothesis.collect_call_source_shapes:
                for source_key, count in source_shapes.items():
                    call_frame_source_shapes[f"{shape}:{source_key}"] += count
            if definite_underflow:
                issues.append(
                    StackIssue(
                        "error",
                        "call_frame_definite_underflow",
                        cfg.function_name,
                        offset,
                        message="Final joined input state cannot satisfy the encoded call frame count.",
                        details={
                            "terminal": terminal,
                            "encoded_frame_count": encoded_count,
                            "input_state": state.to_dict(),
                            "word": word_to_dict(word),
                        },
                    )
                )
                call_underflow_evidence[f"{shape}:definite"] += 1
            elif some_underflow:
                call_underflow_evidence[f"{shape}:some_paths"] += 1
        elif terminal == "BR":
            key = _branch_key(word)
            branch_input_ranges[f"{key}:input={state.range_key()}"] += 1
            if hypothesis.branch_frame_effect == "pop_candidates":
                _out, some_underflow, definite_underflow = state.pop_frame_candidates(hypothesis.branch_pop_candidates)
                if definite_underflow:
                    issues.append(
                        StackIssue(
                            "error",
                            "branch_frame_definite_underflow",
                            cfg.function_name,
                            offset,
                            message="Final joined input state cannot satisfy configured branch operand-frame pop candidates.",
                            details={
                                "branch": key,
                                "input_state": state.to_dict(),
                                "candidates": list(hypothesis.branch_pop_candidates),
                                "word": word_to_dict(word),
                            },
                        )
                    )
                    branch_underflow_evidence[f"{key}:definite"] += 1
                elif some_underflow:
                    branch_underflow_evidence[f"{key}:some_paths"] += 1

    return FunctionStackReport(
        function_name=cfg.function_name,
        symbol=cfg.symbol,
        span=cfg.span,
        declared_param_arity=abi.declared_param_arity,
        prologue_kind=abi.prologue_kind,
        cfg_node_count=len(cfg.nodes),
        analyzed_node_count=len(states),
        max_observed_value_depth=max_observed_value_depth,
        max_observed_operand_frame_depth=max_observed_frame_depth,
        hit_value_cap=hit_value_cap,
        hit_operand_frame_cap=hit_frame_cap,
        return_states=return_states,
        branch_input_ranges=branch_input_ranges,
        branch_underflow_evidence=branch_underflow_evidence,
        call_input_ranges=call_input_ranges,
        call_underflow_evidence=call_underflow_evidence,
        call_frame_source_shapes=call_frame_source_shapes,
        issues=issues,
    )


def _classify_absolute_call_target(module: MBCModule, target: int, abi_by_start: dict[int, FunctionABI]) -> tuple[str, dict[str, Any]]:
    code_size = module.get_real_code_size()
    if target < 0 or target >= code_size:
        return "out_of_range", {"code_size": code_size}
    slot = module.initial_slot_for_offset(target)
    if slot is not None:
        return "initial_slot" if slot.valid else "invalid_initial_slot", {"slot": slot.to_dict()}
    abi = abi_by_start.get(target)
    if abi is not None:
        return "definition_entry", {"callee_abi": abi.to_dict()}
    return "code_offset", {"code_size": code_size}


def _validate_reachable_script_calls(
    module: MBCModule,
    cfg_report: ModuleCFGReport,
    abi_by_start: dict[int, FunctionABI],
) -> tuple[Counter[str], Counter[str], list[dict[str, Any]]]:
    target_relations: Counter[str] = Counter()
    arity_relations: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []

    for cfg in cfg_report.function_cfgs:
        if cfg.span is None:
            continue
        function_start = int(cfg.span[0])
        for node in sorted(cfg.nodes.values(), key=lambda n: n.offset):
            word = node.word
            if word.terminal_kind != "CALL_SCRIPT":
                continue
            encoded_count = int(word.operands.get("argc", 0) or 0)
            local_target = call_script_target_offset(word)
            absolute_target = function_start + local_target
            relation, details = _classify_absolute_call_target(module, absolute_target, abi_by_start)
            target_relations[relation] += 1
            if relation == "definition_entry":
                abi_payload = details["callee_abi"]
                declared = abi_payload.get("declared_param_arity")
                if declared is None:
                    arity_relations["definition_entry:callee_arity_unknown"] += 1
                elif int(declared) == encoded_count:
                    arity_relations["definition_entry:encoded_count_matches_declared_arity"] += 1
                else:
                    arity_relations["definition_entry:encoded_count_differs_from_declared_arity"] += 1
                    if len(mismatches) < 128:
                        mismatches.append({
                            "caller_function": cfg.function_name,
                            "caller_span": cfg.span,
                            "call_offset": node.offset,
                            "call_source_relation": node.relation.relation,
                            "encoded_frame_count": encoded_count,
                            "absolute_target": absolute_target,
                            "callee_name": abi_payload.get("name"),
                            "callee_symbol": abi_payload.get("symbol"),
                            "declared_param_arity": declared,
                            "previous_linear_word_shape": _previous_linear_word_shape(cfg, node.offset),
                            "word": word_to_dict(word),
                        })
            else:
                arity_relations[f"{relation}:arity_not_applicable"] += 1
    return target_relations, arity_relations, mismatches


def _previous_linear_word_shape(cfg: FunctionCFG, offset: int) -> str | None:
    prev: VMWord | None = None
    for word in cfg.linear_words:
        if int(word.offset) >= int(offset):
            break
        prev = word
    return word_shape_signature(prev) if prev is not None else None


def analyze_module_stack(
    module: MBCModule,
    *,
    cfg_report: ModuleCFGReport | None = None,
    hypothesis: StackHypothesis = StackHypothesis(),
) -> ModuleStackReport:
    cfg_report = analyze_module(module) if cfg_report is None else cfg_report
    function_abis = [_declared_abi_from_cfg(cfg) for cfg in cfg_report.function_cfgs]
    abi_by_start = {abi.absolute_start: abi for abi in function_abis if abi.absolute_start is not None}

    function_reports = [analyze_function_stack(cfg, hypothesis=hypothesis) for cfg in cfg_report.function_cfgs]
    target_relations, arity_relations, mismatches = _validate_reachable_script_calls(module, cfg_report, abi_by_start)

    native_shapes: Counter[str] = Counter()
    for cfg in cfg_report.function_cfgs:
        for node in cfg.nodes.values():
            if node.word.terminal_kind == "CALL_NATIVE":
                native_shapes[word_shape_signature(node.word)] += 1

    return ModuleStackReport(
        module_path=str(module.path),
        hypothesis=hypothesis,
        function_reports=function_reports,
        function_abis=function_abis,
        call_script_target_relations=target_relations,
        call_script_arity_relations=arity_relations,
        call_script_arity_mismatches=mismatches,
        native_call_shapes=native_shapes,
    )


def summarize_corpus_stack(paths: Iterable[str | Path], *, hypothesis: StackHypothesis = StackHypothesis()) -> dict[str, Any]:
    module_count = 0
    totals: Counter[str] = Counter()
    call_target_relations: Counter[str] = Counter()
    call_arity_relations: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    branch_underflow: Counter[str] = Counter()
    call_underflow: Counter[str] = Counter()
    return_states: Counter[str] = Counter()
    native_shapes: Counter[str] = Counter()
    call_sources: Counter[str] = Counter()
    mismatch_examples: list[dict[str, Any]] = []
    notable_modules: list[dict[str, Any]] = []
    functions_hitting_value_cap = 0
    functions_hitting_frame_cap = 0

    for path in paths:
        module = MBCModule(path, collect_auxiliary=False)
        cfg_report = analyze_module(module)
        stack_report = analyze_module_stack(module, cfg_report=cfg_report, hypothesis=hypothesis)
        module_count += 1
        module_issue_counts: Counter[str] = Counter()
        module_severity_counts: Counter[str] = Counter()
        module_functions_with_errors = 0
        module_functions_with_warnings = 0
        module_functions_hitting_value_cap = 0
        module_functions_hitting_frame_cap = 0

        call_target_relations.update(stack_report.call_script_target_relations)
        call_arity_relations.update(stack_report.call_script_arity_relations)
        native_shapes.update(stack_report.native_call_shapes)

        for report in stack_report.function_reports:
            totals["functions"] += 1
            totals["cfg_nodes"] += report.cfg_node_count
            totals["analyzed_nodes"] += report.analyzed_node_count
            totals["max_observed_value_depth"] = max(totals["max_observed_value_depth"], report.max_observed_value_depth)
            totals["max_observed_operand_frame_depth"] = max(totals["max_observed_operand_frame_depth"], report.max_observed_operand_frame_depth)
            if report.declared_param_arity is not None:
                totals["functions_with_declared_arity"] += 1
            if report.hit_value_cap:
                functions_hitting_value_cap += 1
                module_functions_hitting_value_cap += 1
            if report.hit_operand_frame_cap:
                functions_hitting_frame_cap += 1
                module_functions_hitting_frame_cap += 1
            if report.hard_error_count:
                module_functions_with_errors += 1
            if report.warning_count:
                module_functions_with_warnings += 1
            branch_underflow.update(report.branch_underflow_evidence)
            call_underflow.update(report.call_underflow_evidence)
            call_sources.update(report.call_frame_source_shapes)
            for (term, value, frame), count in report.return_states.items():
                return_states[f"{term}:value={value}:frame={frame}"] += count
            for issue in report.issues:
                issue_counts[issue.code] += 1
                severity_counts[issue.severity] += 1
                module_issue_counts[issue.code] += 1
                module_severity_counts[issue.severity] += 1

        if stack_report.call_script_arity_mismatches and len(mismatch_examples) < 128:
            for item in stack_report.call_script_arity_mismatches:
                if len(mismatch_examples) >= 128:
                    break
                enriched = dict(item)
                enriched["module_path"] = str(path)
                mismatch_examples.append(enriched)
        if module_functions_with_errors or module_functions_with_warnings or stack_report.call_script_arity_mismatches:
            if len(notable_modules) < 64:
                notable_modules.append({
                    "module_path": str(path),
                    "functions_with_errors": module_functions_with_errors,
                    "functions_with_warnings": module_functions_with_warnings,
                    "functions_hitting_value_cap": module_functions_hitting_value_cap,
                    "functions_hitting_operand_frame_cap": module_functions_hitting_frame_cap,
                    "call_script_arity_mismatch_count": len(stack_report.call_script_arity_mismatches),
                    "issue_counts": dict(module_issue_counts),
                    "severity_counts": dict(module_severity_counts),
                })

    return {
        "module_count": module_count,
        "hypothesis": hypothesis.to_dict(),
        "totals": dict(totals),
        "functions_hitting_value_cap": functions_hitting_value_cap,
        "functions_hitting_operand_frame_cap": functions_hitting_frame_cap,
        "call_script_target_relations": dict(call_target_relations),
        "call_script_arity_relations": dict(call_arity_relations),
        "call_script_arity_difference_count": call_arity_relations.get("definition_entry:encoded_count_differs_from_declared_arity", 0),
        "call_script_arity_difference_examples": mismatch_examples,
        "branch_underflow_evidence_top": dict(branch_underflow.most_common(64)),
        "call_underflow_evidence_top": dict(call_underflow.most_common(64)),
        "call_frame_source_shapes_top": dict(call_sources.most_common(64)),
        "return_states_top": dict(return_states.most_common(64)),
        "native_call_shapes_top": dict(native_shapes.most_common(64)),
        "issue_counts": dict(issue_counts),
        "severity_counts": dict(severity_counts),
        "notable_modules": notable_modules,
    }

def _merge_corpus_stack_summaries(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    module_count = 0
    totals: Counter[str] = Counter()
    call_target_relations: Counter[str] = Counter()
    call_arity_relations: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    branch_underflow: Counter[str] = Counter()
    call_underflow: Counter[str] = Counter()
    return_states: Counter[str] = Counter()
    native_shapes: Counter[str] = Counter()
    call_sources: Counter[str] = Counter()
    mismatch_examples: list[dict[str, Any]] = []
    notable_modules: list[dict[str, Any]] = []
    hypothesis: dict[str, Any] | None = None
    functions_hitting_value_cap = 0
    functions_hitting_frame_cap = 0

    for summary in summaries:
        if hypothesis is None:
            hypothesis = summary.get("hypothesis")
        module_count += int(summary.get("module_count", 0) or 0)
        for key, value in summary.get("totals", {}).items():
            if key in {"max_observed_value_depth", "max_observed_operand_frame_depth"}:
                totals[key] = max(totals[key], int(value))
            else:
                totals[key] += int(value)
        functions_hitting_value_cap += int(summary.get("functions_hitting_value_cap", 0) or 0)
        functions_hitting_frame_cap += int(summary.get("functions_hitting_operand_frame_cap", 0) or 0)
        call_target_relations.update(summary.get("call_script_target_relations", {}))
        call_arity_relations.update(summary.get("call_script_arity_relations", {}))
        issue_counts.update(summary.get("issue_counts", {}))
        severity_counts.update(summary.get("severity_counts", {}))
        branch_underflow.update(summary.get("branch_underflow_evidence_top", {}))
        call_underflow.update(summary.get("call_underflow_evidence_top", {}))
        call_sources.update(summary.get("call_frame_source_shapes_top", {}))
        return_states.update(summary.get("return_states_top", {}))
        native_shapes.update(summary.get("native_call_shapes_top", {}))
        for item in summary.get("call_script_arity_difference_examples", []):
            if len(mismatch_examples) < 128:
                mismatch_examples.append(item)
        for item in summary.get("notable_modules", []):
            if len(notable_modules) < 64:
                notable_modules.append(item)

    return {
        "module_count": module_count,
        "hypothesis": hypothesis or {},
        "totals": dict(totals),
        "functions_hitting_value_cap": functions_hitting_value_cap,
        "functions_hitting_operand_frame_cap": functions_hitting_frame_cap,
        "call_script_target_relations": dict(call_target_relations),
        "call_script_arity_relations": dict(call_arity_relations),
        "call_script_arity_difference_count": call_arity_relations.get("definition_entry:encoded_count_differs_from_declared_arity", 0),
        "call_script_arity_difference_examples": mismatch_examples,
        "branch_underflow_evidence_top": dict(branch_underflow.most_common(64)),
        "call_underflow_evidence_top": dict(call_underflow.most_common(64)),
        "call_frame_source_shapes_top": dict(call_sources.most_common(64)),
        "return_states_top": dict(return_states.most_common(64)),
        "native_call_shapes_top": dict(native_shapes.most_common(64)),
        "issue_counts": dict(issue_counts),
        "severity_counts": dict(severity_counts),
        "notable_modules": notable_modules,
    }


def _summarize_one_path_worker(args: tuple[str, StackHypothesis]) -> dict[str, Any]:
    path, hypothesis = args
    return summarize_corpus_stack([Path(path)], hypothesis=hypothesis)


def summarize_corpus_stack_parallel(paths: Iterable[str | Path], *, hypothesis: StackHypothesis = StackHypothesis(), jobs: int = 4, chunk_size: int = 100) -> dict[str, Any]:
    """Process large corpora with fresh worker processes.

    The stack layer creates dense per-function state lattices.  Running hundreds
    of modules in one interpreter has poor memory locality on the full corpus.
    One-module workers plus fresh pools per chunk keep the production CLI
    deterministic and avoid stale abstract states.
    """

    from multiprocessing import Pool

    path_list = [Path(path) for path in paths]
    if not path_list:
        return summarize_corpus_stack([], hypothesis=hypothesis)
    if jobs <= 1 or len(path_list) == 1:
        return summarize_corpus_stack(path_list, hypothesis=hypothesis)

    chunk_size = max(1, int(chunk_size))
    merged_chunks: list[dict[str, Any]] = []
    for start in range(0, len(path_list), chunk_size):
        chunk = path_list[start:start + chunk_size]
        with Pool(processes=jobs, maxtasksperchild=1) as pool:
            summaries = list(pool.imap_unordered(_summarize_one_path_worker, [(str(path), hypothesis) for path in chunk]))
        merged_chunks.append(_merge_corpus_stack_summaries(summaries))
    return _merge_corpus_stack_summaries(merged_chunks)

def default_hypotheses(value_cap: int = 64, frame_cap: int = 32) -> list[StackHypothesis]:
    return [
        StackHypothesis(name="two_layer_branch_frame_clear", value_cap=value_cap, frame_cap=frame_cap, branch_frame_effect="clear"),
        StackHypothesis(name="two_layer_branch_frame_keep", value_cap=value_cap, frame_cap=frame_cap, branch_frame_effect="keep"),
        StackHypothesis(name="two_layer_branch_frame_pop01", value_cap=value_cap, frame_cap=frame_cap, branch_frame_effect="pop_candidates", branch_pop_candidates=(0, 1)),
        StackHypothesis(name="two_layer_no_call_returns", value_cap=value_cap, frame_cap=frame_cap, native_return_min=0, native_return_max=0, script_return_min=0, script_return_max=0, branch_frame_effect="clear"),
        StackHypothesis(name="two_layer_all_calls_return_one", value_cap=value_cap, frame_cap=frame_cap, native_return_min=1, native_return_max=1, script_return_min=1, script_return_max=2, branch_frame_effect="clear"),
    ]


def summarize_hypothesis_matrix(paths: Iterable[str | Path], *, value_cap: int = 64, frame_cap: int = 32) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for hypothesis in default_hypotheses(value_cap, frame_cap):
        summary = summarize_corpus_stack(paths, hypothesis=hypothesis)
        results.append({
            "hypothesis": hypothesis.to_dict(),
            "issue_counts": summary.get("issue_counts", {}),
            "severity_counts": summary.get("severity_counts", {}),
            "functions_hitting_value_cap": summary.get("functions_hitting_value_cap", 0),
            "functions_hitting_operand_frame_cap": summary.get("functions_hitting_operand_frame_cap", 0),
            "max_observed_value_depth": summary.get("totals", {}).get("max_observed_value_depth", 0),
            "max_observed_operand_frame_depth": summary.get("totals", {}).get("max_observed_operand_frame_depth", 0),
            "call_underflow_evidence_top": summary.get("call_underflow_evidence_top", {}),
            "branch_underflow_evidence_top": summary.get("branch_underflow_evidence_top", {}),
            "call_script_arity_difference_count": summary.get("call_script_arity_difference_count", 0),
            "notable_modules": summary.get("notable_modules", [])[:16],
        })
    return {"hypotheses": results}


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.mbc")))
        elif path.suffix.lower() == ".mbc":
            paths.append(path)
        else:
            raise FileNotFoundError(f"Unsupported input path: {raw}")
    return paths


def _parse_int_range(text: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not parts:
        return (0, 0)
    if len(parts) == 1:
        return (parts[0], parts[0])
    return (min(parts), max(parts))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate falsifiable VM stack/effect hypotheses over region CFG reports.")
    parser.add_argument("inputs", nargs="+", help=".mbc files or directories containing .mbc files")
    parser.add_argument("--json", dest="json_path", help="Write JSON summary to this path. Defaults to stdout.")
    parser.add_argument("--matrix", action="store_true", help="Run the default hypothesis matrix instead of one hypothesis.")
    parser.add_argument("--max-depth", type=int, default=None, help="Backward-compatible alias for --value-cap")
    parser.add_argument("--value-cap", type=int, default=64, help="Persistent value-stack depth cap")
    parser.add_argument("--frame-cap", type=int, default=32, help="Local operand-frame depth cap")
    parser.add_argument("--native-return", default="0,2", help="Native return arity range, e.g. 0,1 or 1")
    parser.add_argument("--script-return", default="0,2", help="Script return arity range, e.g. 0,1 or 1")
    parser.add_argument("--branch-frame-effect", default="clear", choices=BRANCH_FRAME_EFFECT_MODES)
    parser.add_argument("--branch-effect", dest="legacy_branch_effect", choices=("keep", "pop_candidates", "reset"), help="Deprecated alias: reset maps to branch-frame clear")
    parser.add_argument("--branch-pops", default="0,1", help="Comma-separated branch frame-pop candidates for pop_candidates mode")
    parser.add_argument("--start-with-declared-arity", action="store_true")
    parser.add_argument("--collect-call-source-shapes", action="store_true", help="Collect expensive call frame split provenance counters")
    parser.add_argument("--jobs", type=int, default=0, help="Worker processes for multi-module corpus runs; 0 uses up to 4 workers")
    args = parser.parse_args(argv)

    paths = _expand_inputs(args.inputs)
    value_cap = args.value_cap if args.max_depth is None else args.max_depth
    branch_frame_effect = args.branch_frame_effect
    if args.legacy_branch_effect:
        branch_frame_effect = "clear" if args.legacy_branch_effect == "reset" else args.legacy_branch_effect
    if args.matrix:
        payload = summarize_hypothesis_matrix(paths, value_cap=value_cap, frame_cap=args.frame_cap)
    else:
        branch_pops = tuple(sorted({int(part) for part in args.branch_pops.split(",") if part.strip()}))
        native_min, native_max = _parse_int_range(args.native_return)
        script_min, script_max = _parse_int_range(args.script_return)
        hypothesis = StackHypothesis(
            value_cap=value_cap,
            frame_cap=args.frame_cap,
            native_return_min=native_min,
            native_return_max=native_max,
            script_return_min=script_min,
            script_return_max=script_max,
            branch_frame_effect=branch_frame_effect,
            branch_pop_candidates=branch_pops,
            start_with_declared_arity=args.start_with_declared_arity,
            collect_call_source_shapes=args.collect_call_source_shapes,
        )
        jobs = args.jobs if args.jobs > 0 else (min(4, len(paths)) if len(paths) > 1 else 1)
        payload = summarize_corpus_stack_parallel(paths, hypothesis=hypothesis, jobs=jobs)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
