"""High level Lua reconstruction leveraging manual annotations and stack heuristics."""

from __future__ import annotations

import re

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .ir import IRBlock, IRInstruction, IRProgram
from .knowledge import KnowledgeBase
from .lua_ast import (
    Assignment,
    BlankLine,
    BlockStatement,
    BinaryExpr,
    CallExpr,
    CallStatement,
    CommentStatement,
    CommentedStatement,
    IfClause,
    IfStatement,
    LuaExpression,
    LiteralExpr,
    MethodCallExpr,
    MultiAssignment,
    NameExpr,
    LuaStatement,
    ReturnStatement,
    SwitchCase,
    SwitchStatement,
    TableExpr,
    TableField,
    WhileStatement,
    wrap_block,
)
from .lua_formatter import (
    CommentFormatter,
    EnumRegistry,
    HelperRegistry,
    HelperSignature,
    LuaRenderOptions,
    LuaWriter,
    MethodSignature,
    join_sections,
)
from .manual_semantics import InstructionSemantics
from .literal_sequences import (
    LiteralRun,
    LiteralRunTracker,
    LiteralRunReport,
    LiteralStatistics,
    build_literal_run_report,
    compute_literal_statistics,
)
from .lua_literals import LuaLiteral, LuaLiteralFormatter, escape_lua_string
from .vm_analysis import estimate_stack_io


@dataclass
class StackValue:
    """Represents a symbolic value stored on the evaluation stack."""

    expression: LuaExpression
    origin: Optional[int] = None
    comments: List[str] = field(default_factory=list)

    def add_comment(self, text: str) -> None:
        if not text:
            return
        if text not in self.comments:
            self.comments.append(text)

    def add_comments(self, comments: Iterable[str]) -> None:
        for comment in comments:
            self.add_comment(comment)

    def take_comments(self) -> List[str]:
        comments = list(self.comments)
        self.comments.clear()
        return comments


@dataclass
class InstructionTraceInfo:
    """Aggregated usage information for a single instruction."""

    offset: int
    mnemonic: str
    summary: str
    usages: List[Tuple[str, str]] = field(default_factory=list)

    def add_usage(self, role: str, comment: str) -> None:
        self.usages.append((role, comment))


@dataclass
class StackEvent:
    """Records a single stack mutation during reconstruction."""

    action: str
    value: str
    origin: Optional[int]
    comment: Optional[str] = None
    depth_before: int = 0
    depth_after: int = 0


class HighLevelStack:
    """Track symbolic stack values during reconstruction."""

    def __init__(self) -> None:
        self._values: List[StackValue] = []
        self._counter = 0
        self.warnings: List[str] = []
        self._events: List[StackEvent] = []
        self._depth = 0
        self._min_depth = 0
        self._max_depth = 0
        self._underflow_events = 0

    def new_symbol(self, prefix: str = "value") -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def push_literal(
        self,
        expression: LuaExpression,
        *,
        origin: Optional[int] = None,
    ) -> Tuple[List[LuaStatement], StackValue]:
        value = StackValue(expression=expression, origin=origin)
        self._values.append(value)
        self._record_push(expression.render(), origin)
        return [], value

    def push_expression(
        self,
        expression: LuaExpression,
        *,
        prefix: str = "tmp",
        make_local: bool = False,
        origin: Optional[int] = None,
    ) -> Tuple[List[LuaStatement], StackValue]:
        if make_local or not isinstance(expression, NameExpr):
            name = self.new_symbol(prefix)
            target = NameExpr(name)
            statement = Assignment([target], expression, is_local=True)
            value = StackValue(target, origin=origin)
            self._values.append(value)
            self._record_push(target.render(), origin)
            return [statement], value
        value = StackValue(expression, origin=origin)
        self._values.append(value)
        self._record_push(expression.render(), origin)
        return [], value

    def push_call_results(
        self,
        expression: CallExpr,
        outputs: int,
        prefix: str = "result",
        *,
        origin: Optional[int] = None,
    ) -> Tuple[List[LuaStatement], List[StackValue]]:
        if outputs <= 0:
            return [], []
        if outputs == 1:
            name = self.new_symbol(prefix)
            target = NameExpr(name)
            stmt = Assignment([target], expression, is_local=True)
            value = StackValue(target, origin=origin)
            self._values.append(value)
            self._record_push(target.render(), origin)
            return [stmt], [value]
        targets = [NameExpr(self.new_symbol(prefix)) for _ in range(outputs)]
        stmt = MultiAssignment(targets, [expression], is_local=True)
        values = [StackValue(target, origin=origin) for target in targets]
        self._values.extend(values)
        for target in targets:
            self._record_push(target.render(), origin)
        return [stmt], values

    def pop_single(self) -> StackValue:
        if self._values:
            value = self._values.pop()
            self._record_pop("pop", value.expression.render(), value.origin)
            return value
        placeholder = NameExpr(self.new_symbol("stack"))
        warning = f"underflow generated placeholder {placeholder.name}"
        self.warnings.append(warning)
        value = StackValue(placeholder)
        value.add_comment(warning)
        self._record_pop("pop", placeholder.render(), None, warning)
        return value

    def pop_many(self, count: int) -> List[StackValue]:
        items = [self.pop_single() for _ in range(count)]
        items.reverse()
        return items

    def pop_pair(self) -> Tuple[StackValue, StackValue]:
        lhs, rhs = self.pop_many(2)
        return lhs, rhs

    def flush(self) -> List[StackValue]:
        if not self._values:
            return []
        flushed: List[StackValue] = []
        while self._values:
            value = self._values.pop()
            self._record_pop("flush", value.expression.render(), value.origin)
            flushed.append(value)
        flushed.reverse()
        return flushed

    def annotate_top(self, comment: str) -> None:
        if not self._values:
            return
        self._values[-1].add_comment(comment)

    def events(self) -> Sequence[StackEvent]:
        return tuple(self._events)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def min_depth(self) -> int:
        return self._min_depth

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def underflow_events(self) -> int:
        return self._underflow_events

    def _log_event(
        self,
        action: str,
        value_repr: str,
        origin: Optional[int],
        comment: Optional[str] = None,
        *,
        depth_before: Optional[int] = None,
        depth_after: Optional[int] = None,
    ) -> None:
        before = self._depth if depth_before is None else depth_before
        after = self._depth if depth_after is None else depth_after
        self._events.append(
            StackEvent(
                action,
                value_repr,
                origin,
                comment,
                before,
                after,
            )
        )

    def _record_push(self, value_repr: str, origin: Optional[int]) -> None:
        before = self._depth
        self._depth += 1
        self._max_depth = max(self._max_depth, self._depth)
        self._log_event(
            "push",
            value_repr,
            origin,
            depth_before=before,
            depth_after=self._depth,
        )

    def _record_pop(
        self,
        action: str,
        value_repr: str,
        origin: Optional[int],
        comment: Optional[str] = None,
    ) -> None:
        before = self._depth
        self._depth -= 1
        self._min_depth = min(self._min_depth, self._depth)
        if self._depth < 0:
            self._underflow_events += 1
        self._log_event(
            action,
            value_repr,
            origin,
            comment,
            depth_before=before,
            depth_after=self._depth,
        )


class StringLiteralCollector:
    """Group consecutive string literal assignments into annotated sequences."""

    def __init__(self) -> None:
        self._pending: List[StackValue] = []
        self._fragments: List[str] = []

    def reset(self) -> None:
        self._pending.clear()
        self._fragments.clear()

    def enqueue(self, _offset: int, value: StackValue) -> None:
        literal_text = self._string_value(value.expression)
        if literal_text is None:
            self.flush()
            return
        self._pending.append(value)
        self._fragments.append(literal_text)

    def flush(self) -> None:
        if not self._pending:
            return
        combined = "".join(self._fragments)
        sequence_comment = (
            f"string literal sequence: {escape_lua_string(combined)}"
            f" (len={len(combined)})"
        )
        for idx, value in enumerate(self._pending):
            if idx == 0:
                value.add_comment(sequence_comment)
            else:
                fragment = escape_lua_string(self._fragments[idx])
                value.add_comment(f"string literal fragment {idx + 1}: {fragment}")
        self.reset()

    def finalize(self) -> None:
        self.flush()

    @staticmethod
    def _string_value(expression: LuaExpression) -> Optional[str]:
        if isinstance(expression, LiteralExpr):
            literal = expression.literal
            if literal is not None and literal.kind == "string":
                return str(literal.value)
        return None


@dataclass
class FunctionMetadata:
    """Descriptive statistics collected while reconstructing a function."""

    block_count: int
    instruction_count: int
    warnings: List[str] = field(default_factory=list)
    helper_calls: int = 0
    branch_count: int = 0
    literal_count: int = 0
    literal_runs: Sequence[LiteralRun] = field(default_factory=tuple)
    literal_stats: Optional[LiteralStatistics] = None
    literal_report: Optional[LiteralRunReport] = None
    value_comments: List[Tuple[str, Optional[int], str]] = field(default_factory=list)
    instruction_trace: Dict[int, InstructionTraceInfo] = field(default_factory=dict)
    stack_events: Sequence[StackEvent] = field(default_factory=tuple)
    helper_usage: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {"function": {}, "method": {}}
    )
    stack_depth_min: int = 0
    stack_depth_max: int = 0
    stack_depth_final: int = 0
    stack_underflows: int = 0

    def summary_lines(self) -> List[str]:
        lines = ["function summary:"]
        lines.append(f"- blocks: {self.block_count}")
        lines.append(f"- instructions: {self.instruction_count}")
        lines.append(f"- literal instructions: {self.literal_count}")
        lines.append(f"- helper invocations: {self.helper_calls}")
        lines.append(f"- branches: {self.branch_count}")
        if self.stack_depth_max:
            lines.append(f"- peak stack depth: {self.stack_depth_max}")
        if self.stack_depth_min < 0:
            lines.append(
                f"- deepest underflow depth: {self.stack_depth_min}"
                f" (events={self.stack_underflows})"
            )
        if self.stack_depth_final:
            lines.append(f"- final stack depth: {self.stack_depth_final}")
        string_runs = [run for run in self.literal_runs if run.kind == "string"]
        if string_runs:
            lines.append(f"- string literal sequences: {len(string_runs)}")
        numeric_runs = [run for run in self.literal_runs if run.kind == "number"]
        if numeric_runs:
            lines.append(f"- numeric literal runs: {len(numeric_runs)}")
        if self.literal_stats and self.literal_stats.kind_counts:
            breakdown = ", ".join(
                f"{kind}={count}" for kind, count in sorted(self.literal_stats.kind_counts.items())
            )
            lines.append(f"- literal run kinds: {breakdown}")
        return lines

    def warning_lines(self) -> List[str]:
        return [f"- {warning}" for warning in self.warnings]

    def literal_run_lines(
        self, *, limit: int = 8, preview: int = 72
    ) -> List[str]:
        if not self.literal_runs:
            return []
        lines: List[str] = []
        for run in list(self.literal_runs)[:limit]:
            summary = run.render_preview(preview)
            lines.append(
                "- 0x"
                f"{run.start_offset():06X}"
                f" kind={run.kind}"
                f" count={run.length()}: {summary}"
            )
        remaining = len(self.literal_runs) - limit
        if remaining > 0:
            lines.append(f"- ... ({remaining} additional runs)")
        return lines

    def literal_statistics_lines(self) -> List[str]:
        if not self.literal_stats:
            return []
        return self.literal_stats.summary_lines()

    def literal_report_lines(self, *, block_limit: int = 3) -> List[str]:
        if not self.literal_report or not self.literal_report.runs:
            return []
        summary = self.literal_report.summary_lines()
        blocks = self.literal_report.block_lines(limit=block_limit)
        return summary + blocks

    def value_comment_summary_lines(self, *, limit: int = 6) -> List[str]:
        if not self.value_comments:
            return []
        counter: Counter[str] = Counter()
        origins: Dict[str, List[str]] = {}
        for role, origin, comment in self.value_comments:
            key = comment
            counter[key] += 1
            if origin is not None:
                entry = f"{role} @0x{origin:06X}"
            else:
                entry = role
            origins.setdefault(key, []).append(entry)
        lines = ["value provenance summary:"]
        for comment, count in counter.most_common(limit):
            contexts = ", ".join(sorted(set(origins.get(comment, []))))
            if contexts:
                lines.append(f"- {comment} ×{count} ({contexts})")
            else:
                lines.append(f"- {comment} ×{count}")
        remaining = sum(counter.values()) - sum(count for _, count in counter.most_common(limit))
        if remaining > 0:
            lines.append(f"- ... {remaining} additional occurrences")
        return lines

    def instruction_trace_lines(self, *, limit: int = 12) -> List[str]:
        if not self.instruction_trace:
            return []
        lines = ["instruction usage trace:"]
        count = 0
        for offset in sorted(self.instruction_trace):
            info = self.instruction_trace[offset]
            header = f"- 0x{offset:06X}: {info.mnemonic}"
            if info.summary:
                header += f" — {info.summary}"
            lines.append(header)
            if info.usages:
                for role, comment in info.usages:
                    lines.append(f"    • {role} -> {comment}")
            else:
                lines.append("    • no recorded stack usage")
            count += 1
            if count >= limit:
                break
        remaining = len(self.instruction_trace) - count
        if remaining > 0:
            lines.append(f"- ... {remaining} additional instructions")
        return lines

    def stack_event_summary_lines(self, *, limit: int = 5) -> List[str]:
        if not self.stack_events:
            return []
        action_counter = Counter(event.action for event in self.stack_events)
        lines = ["stack event summary:"]
        for action in ("push", "pop", "flush"):
            if action_counter.get(action):
                lines.append(f"- {action} events: {action_counter[action]}")
        if self.stack_depth_max:
            lines.append(f"- peak stack depth: {self.stack_depth_max}")
        if self.stack_depth_min < 0:
            lines.append(
                f"- minimum stack depth: {self.stack_depth_min}"
                f" (underflows={self.stack_underflows})"
            )
        elif self.stack_depth_min > 0:
            lines.append(f"- minimum stack depth: {self.stack_depth_min}")
        if self.stack_depth_final:
            lines.append(f"- final stack depth: {self.stack_depth_final}")
        push_counter = Counter(
            event.value for event in self.stack_events if event.action == "push"
        )
        if push_counter:
            top_pushes = ", ".join(
                f"{value}×{count}"
                for value, count in push_counter.most_common(limit)
            )
            lines.append(f"- common push values: {top_pushes}")
            remaining = sum(push_counter.values()) - sum(
                count for _, count in push_counter.most_common(limit)
            )
            if remaining > 0:
                lines.append(f"- ... {remaining} additional push occurrences")
        noteworthy = [event for event in self.stack_events if event.comment]
        if noteworthy:
            lines.append("- noteworthy stack anomalies:")
            for event in noteworthy[:limit]:
                origin = f" @0x{event.origin:06X}" if event.origin is not None else ""
                depth = f" depth={event.depth_after}"
                lines.append(
                    f"  • {event.action}{origin}: {event.comment}"
                    f" ({event.value};{depth})"
                )
            extra = len(noteworthy) - min(len(noteworthy), limit)
            if extra > 0:
                lines.append(f"  • ... {extra} additional events with comments")
        return lines

    def helper_usage_lines(self, *, limit: int = 6) -> List[str]:
        function_usage = self.helper_usage.get("function", {})
        method_usage = self.helper_usage.get("method", {})
        if not function_usage and not method_usage:
            return []
        lines = ["helper usage summary:"]
        if function_usage:
            lines.append("- helper functions:")
            for name, count in sorted(
                function_usage.items(), key=lambda item: (-item[1], item[0])
            )[:limit]:
                lines.append(f"  • {name}: {count}")
            remaining = len(function_usage) - min(len(function_usage), limit)
            if remaining > 0:
                lines.append(f"  • ... {remaining} additional function helpers")
        if method_usage:
            lines.append("- struct methods:")
            for name, count in sorted(
                method_usage.items(), key=lambda item: (-item[1], item[0])
            )[:limit]:
                lines.append(f"  • {name}: {count}")
            remaining = len(method_usage) - min(len(method_usage), limit)
            if remaining > 0:
                lines.append(f"  • ... {remaining} additional methods")
        return lines


@dataclass
class HighLevelFunction:
    """Container describing a reconstructed Lua function."""

    name: str
    body: BlockStatement
    metadata: FunctionMetadata

    def render(self) -> str:
        writer = LuaWriter()
        summary = self.metadata.summary_lines()
        if summary:
            writer.write_comment_block(summary)
            writer.write_line("")
        literal_lines = self.metadata.literal_run_lines()
        if literal_lines:
            writer.write_comment_block(["literal runs:"] + literal_lines)
            writer.write_line("")
        report_lines = self.metadata.literal_report_lines()
        if report_lines:
            writer.write_comment_block(report_lines)
            writer.write_line("")
        stats_lines = self.metadata.literal_statistics_lines()
        if stats_lines:
            writer.write_comment_block(stats_lines)
            writer.write_line("")
        helper_lines = self.metadata.helper_usage_lines()
        if helper_lines:
            writer.write_comment_block(helper_lines)
            writer.write_line("")
        provenance_lines = self.metadata.value_comment_summary_lines()
        if provenance_lines:
            writer.write_comment_block(provenance_lines)
            writer.write_line("")
        trace_lines = self.metadata.instruction_trace_lines()
        if trace_lines:
            writer.write_comment_block(trace_lines)
            writer.write_line("")
        stack_lines = self.metadata.stack_event_summary_lines()
        if stack_lines:
            writer.write_comment_block(stack_lines)
            writer.write_line("")
        if self.metadata.warnings:
            writer.write_comment_block(
                ["stack reconstruction warnings:"] + self.metadata.warning_lines()
            )
            writer.write_line("")
        writer.write_line(f"function {self.name}()")
        with writer.indented():
            self.body.emit(writer)
        writer.write_line("end")
        return writer.render()


@dataclass
class Terminator:
    """Base terminator node summarising the end of a block."""


@dataclass
class ReturnTerminator(Terminator):
    values: List[LuaExpression]
    comment: Optional[str]


@dataclass
class JumpTerminator(Terminator):
    target: Optional[int]
    fallthrough: Optional[int]
    comment: Optional[str]


@dataclass
class BranchTerminator(Terminator):
    condition: LuaExpression
    true_target: Optional[int]
    false_target: Optional[int]
    fallthrough: Optional[int]
    comment: Optional[str]
    enum_namespace: Optional[str] = None
    enum_values: Dict[int, str] = field(default_factory=dict)


@dataclass
class BlockInfo:
    start: int
    statements: List[LuaStatement]
    terminator: Optional[Terminator]
    successors: List[int]


class BlockTranslator:
    """Translate IR blocks into :class:`BlockInfo` instances."""

    def __init__(self, reconstructor: "HighLevelReconstructor", program: IRProgram) -> None:
        self._reconstructor = reconstructor
        self._program = program
        self._stack = HighLevelStack()
        self._string_collector = StringLiteralCollector()
        self._literal_tracker = LiteralRunTracker()
        self.literal_count = 0
        self.helper_calls = 0
        self.branch_count = 0
        self.instruction_total = 0
        self.value_comment_log: List[Tuple[str, Optional[int], str]] = []
        self.trace_map: Dict[int, InstructionTraceInfo] = {}
        self.helper_usage_counter: Counter[Tuple[str, str]] = Counter()

    def translate(self) -> Dict[int, BlockInfo]:
        blocks: Dict[int, BlockInfo] = {}
        order = sorted(self._program.blocks)
        for idx, start in enumerate(order):
            block = self._program.blocks[start]
            next_offset = order[idx + 1] if idx + 1 < len(order) else None
            self._literal_tracker.start_block(block.start)
            blocks[start] = self._translate_block(block, next_offset)
            self.instruction_total += len(block.instructions)
        self._literal_tracker.finalize()
        return blocks

    # ------------------------------------------------------------------
    def _translate_block(self, block: IRBlock, fallthrough: Optional[int]) -> BlockInfo:
        statements: List[LuaStatement] = []
        terminator: Optional[Terminator] = None
        self._string_collector.reset()
        for index, instruction in enumerate(block.instructions):
            semantics = instruction.semantics
            self._reconstructor._register_enums(semantics)
            is_last = index == len(block.instructions) - 1
            self.trace_map.setdefault(
                instruction.offset,
                InstructionTraceInfo(
                    offset=instruction.offset,
                    mnemonic=semantics.mnemonic,
                    summary=semantics.summary or "",
                ),
            )
            if semantics.has_tag("literal"):
                operand_expr = self._reconstructor._operand_expression(
                    semantics, instruction.operand
                )
                literal_statements, value = self._reconstructor._translate_literal(
                    instruction,
                    semantics,
                    self,
                    operand_expr=operand_expr,
                )
                self._literal_tracker.observe(instruction.offset, value.expression)
                self._string_collector.enqueue(instruction.offset, value)
                statements.extend(literal_statements)
                continue
            else:
                self._literal_tracker.break_sequence()
            self._string_collector.flush()
            if semantics.has_tag("comparison"):
                statements.extend(
                    self._reconstructor._translate_comparison(instruction, semantics, self)
                )
                continue
            if semantics.control_flow == "branch" and is_last:
                terminator = self._reconstructor._translate_branch(
                    block.start,
                    instruction,
                    semantics,
                    self,
                    fallthrough,
                )
                continue
            if semantics.control_flow == "return" and is_last:
                terminator = self._reconstructor._translate_return(
                    instruction,
                    semantics,
                    self,
                )
                continue
            if semantics.has_tag("structure"):
                statements.extend(
                    self._reconstructor._translate_structure(instruction, semantics, self)
                )
                continue
            if semantics.has_tag("call"):
                statements.extend(
                    self._reconstructor._translate_call(instruction, semantics, self)
                )
                continue
            statements.extend(
                self._reconstructor._translate_generic(instruction, semantics, self)
            )
        self._string_collector.finalize()
        self._literal_tracker.break_sequence()
        if terminator is None:
            target = block.successors[0] if block.successors else fallthrough
            terminator = JumpTerminator(target=target, fallthrough=fallthrough, comment=None)
        return BlockInfo(
            start=block.start,
            statements=statements,
            terminator=terminator,
            successors=list(block.successors),
        )

    # ------------------------------------------------------------------
    @property
    def stack(self) -> HighLevelStack:
        return self._stack

    @property
    def program(self) -> IRProgram:
        return self._program

    @property
    def literal_runs(self) -> Sequence[LiteralRun]:
        return self._literal_tracker.runs()


class ControlFlowStructurer:
    """Best effort structuring of :class:`BlockInfo` sequences."""

    def __init__(self, blocks: Dict[int, BlockInfo]) -> None:
        self._blocks = blocks
        self._emitted: set[int] = set()
        self._predecessors: Dict[int, set[int]] = self._compute_predecessors(blocks)

    @staticmethod
    def _compute_predecessors(blocks: Dict[int, BlockInfo]) -> Dict[int, set[int]]:
        result: Dict[int, set[int]] = {start: set() for start in blocks}
        for start, info in blocks.items():
            for succ in info.successors:
                if succ in result:
                    result[succ].add(start)
        return result

    def structure(self, entry: int) -> BlockStatement:
        statements = self._structure_sequence(entry, stop=None)
        return wrap_block(statements)

    # ------------------------------------------------------------------
    def _structure_sequence(
        self, start: Optional[int], stop: Optional[int]
    ) -> List[LuaStatement]:
        current = start
        result: List[LuaStatement] = []
        while current is not None and current != stop and current in self._blocks:
            if current in self._emitted:
                # Create a comment to note repeated entry and avoid infinite loops.
                result.append(
                    CommentStatement(f"revisit block 0x{current:06X}")
                )
                break
            info = self._blocks[current]
            self._emitted.add(current)
            result.extend(info.statements)
            terminator = info.terminator
            if isinstance(terminator, ReturnTerminator):
                if terminator.comment:
                    result.append(CommentStatement(terminator.comment))
                result.append(ReturnStatement(terminator.values))
                current = None
                continue
            if isinstance(terminator, BranchTerminator):
                statement, next_block = self._handle_branch(current, terminator)
                result.append(statement)
                current = next_block
                continue
            if isinstance(terminator, JumpTerminator):
                if terminator.comment:
                    result.append(CommentStatement(terminator.comment))
                if terminator.target is None or terminator.target == terminator.fallthrough:
                    current = terminator.fallthrough
                else:
                    current = terminator.target
                continue
            break
        return result

    def _handle_branch(
        self, current: int, terminator: BranchTerminator
    ) -> Tuple[LuaStatement, Optional[int]]:
        condition = terminator.condition
        true_target = terminator.true_target
        false_target = terminator.false_target or terminator.fallthrough
        fallthrough = terminator.fallthrough

        # Loop heuristic: condition guarding a backward edge.
        if (
            true_target is not None
            and true_target in self._blocks
            and true_target <= current
            and false_target is not None
            and false_target != true_target
        ):
            body_statements = self._structure_sequence(false_target, stop=current)
            body = wrap_block(body_statements)
            if terminator.comment:
                body.statements.insert(0, CommentStatement(terminator.comment))
            return WhileStatement(condition, body), fallthrough

        if true_target is not None and false_target is None:
            then_body = wrap_block(self._structure_sequence(true_target, fallthrough))
            clauses = [IfClause(condition=condition, body=then_body)]
            if terminator.comment:
                then_body.statements.insert(0, CommentStatement(terminator.comment))
            return IfStatement(clauses), fallthrough

        if true_target is not None and false_target is not None:
            then_body = wrap_block(self._structure_sequence(true_target, fallthrough))
            else_body = wrap_block(self._structure_sequence(false_target, fallthrough))
            clauses = [IfClause(condition=condition, body=then_body), IfClause(None, else_body)]
            if terminator.comment:
                then_body.statements.insert(0, CommentStatement(terminator.comment))
            return IfStatement(clauses), fallthrough

        comment = terminator.comment or "unstructured branch"
        return CommentStatement(comment), fallthrough


class HighLevelReconstructor:
    """Best-effort reconstruction of high-level Lua from IR programs."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        *,
        options: Optional[LuaRenderOptions] = None,
    ) -> None:
        self.knowledge = knowledge
        self._literal_formatter = LuaLiteralFormatter()
        self._enum_registry = EnumRegistry()
        self._comment_formatter = CommentFormatter()
        self._helper_registry = HelperRegistry()
        self.options = options or LuaRenderOptions()
        self._last_summary: Optional[str] = None
        self._used_function_names: Set[str] = set()

    # ------------------------------------------------------------------
    def from_ir(self, program: IRProgram) -> HighLevelFunction:
        self._last_summary = None
        translator = BlockTranslator(self, program)
        blocks = translator.translate()
        entry = min(blocks)
        structurer = ControlFlowStructurer(blocks)
        body = structurer.structure(entry)
        literal_runs = sorted(
            translator.literal_runs, key=lambda run: (run.start_offset(), -run.length())
        )
        stats = compute_literal_statistics(literal_runs)
        report = (
            build_literal_run_report(literal_runs)
            if self.options.emit_literal_report
            else None
        )
        base_name = self._derive_function_name(program, literal_runs)
        function_name = self._unique_function_name(base_name)
        helper_usage: Dict[str, Dict[str, int]] = {"function": {}, "method": {}}
        for (name, kind), count in translator.helper_usage_counter.items():
            bucket = helper_usage.setdefault(kind, {})
            bucket[name] = count
        metadata = FunctionMetadata(
            block_count=len(blocks),
            instruction_count=translator.instruction_total,
            warnings=list(translator.stack.warnings),
            helper_calls=translator.helper_calls,
            branch_count=translator.branch_count,
            literal_count=translator.literal_count,
            literal_runs=tuple(literal_runs),
            literal_stats=stats,
            literal_report=report,
            value_comments=list(translator.value_comment_log),
            instruction_trace=dict(translator.trace_map),
            stack_events=translator.stack.events(),
            helper_usage=helper_usage,
            stack_depth_min=translator.stack.min_depth,
            stack_depth_max=translator.stack.max_depth,
            stack_depth_final=translator.stack.depth,
            stack_underflows=translator.stack.underflow_events,
        )
        return HighLevelFunction(name=function_name, body=body, metadata=metadata)

    # ------------------------------------------------------------------
    def render(self, functions: Sequence[HighLevelFunction]) -> str:
        sections: List[str] = []
        if self.options.emit_module_summary:
            summary_writer = LuaWriter()
            summary_writer.write_comment_block(self._module_summary_lines(functions))
            sections.append(summary_writer.render())
        if not self._enum_registry.is_empty():
            writer = LuaWriter()
            self._enum_registry.render(writer, options=self.options)
            sections.append(writer.render())
        if not self._helper_registry.is_empty():
            writer = LuaWriter()
            self._helper_registry.render(
                writer,
                self._comment_formatter,
                options=self.options,
            )
            sections.append(writer.render())
        for function in functions:
            sections.append(function.render().rstrip())
        return join_sections(sections)

    # ------------------------------------------------------------------
    def _register_enums(self, semantics: InstructionSemantics) -> None:
        if semantics.enum_namespace and semantics.enum_values:
            for value, label in semantics.enum_values.items():
                self._enum_registry.register(semantics.enum_namespace, value, label)

    # Translation helpers -------------------------------------------------
    def _translate_literal(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        *,
        operand_expr: Optional[LuaExpression] = None,
    ) -> Tuple[List[LuaStatement], StackValue]:
        operand = operand_expr or self._operand_expression(semantics, instruction.operand)
        statements, value = translator.stack.push_literal(
            operand, origin=instruction.offset
        )
        translator.literal_count += 1
        decorated = self._decorate_with_comment(statements, semantics, value=value)
        return decorated, value

    def _translate_comparison(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        lhs_value, rhs_value = translator.stack.pop_pair()
        operator = semantics.comparison_operator or "=="
        expression = BinaryExpr(lhs_value.expression, operator, rhs_value.expression)
        statements, value = translator.stack.push_expression(
            expression, prefix="cmp", make_local=True, origin=instruction.offset
        )
        prefix_comments = self._value_comment_lines(
            [lhs_value, rhs_value],
            prefix="cmp",
            collector=translator.value_comment_log,
            trace_map=translator.trace_map,
        )
        return self._decorate_with_comment(
            statements, semantics, value=value, prefix_comments=prefix_comments
        )

    def _translate_branch(
        self,
        block_start: int,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        fallthrough: Optional[int],
    ) -> BranchTerminator:
        condition_value = translator.stack.pop_single()
        condition = condition_value.expression
        # Determine targets based on IR metadata.
        true_target, false_target = self._select_branch_targets(
            block_start, translator, fallthrough
        )
        translator.branch_count += 1
        comment = self._comment_formatter.format_inline(semantics.summary or "")
        condition_comments = self._value_comment_lines(
            [condition_value],
            prefix="cond",
            collector=translator.value_comment_log,
            trace_map=translator.trace_map,
        )
        if condition_comments:
            suffix = " | ".join(condition_comments)
            comment = f"{comment} | {suffix}" if comment else suffix
        return BranchTerminator(
            condition=condition,
            true_target=true_target,
            false_target=false_target,
            fallthrough=fallthrough,
            comment=comment if comment else None,
            enum_namespace=semantics.enum_namespace,
            enum_values=dict(semantics.enum_values),
        )

    def _translate_return(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> ReturnTerminator:
        stack_values = translator.stack.flush()
        comment = self._comment_formatter.format_inline(semantics.summary or "")
        value_comments = self._value_comment_lines(
            stack_values,
            prefix="ret",
            collector=translator.value_comment_log,
            trace_map=translator.trace_map,
        )
        if value_comments:
            suffix = " | ".join(value_comments)
            comment = f"{comment} | {suffix}" if comment else suffix
        expressions = [value.expression for value in stack_values]
        return ReturnTerminator(values=expressions, comment=comment)

    def _translate_structure(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        arg_values = translator.stack.pop_many(inputs)
        call_args = [value.expression for value in arg_values]
        if semantics.uses_operand:
            call_args.append(self._operand_expression(semantics, instruction.operand))
        prefix_comments = self._value_comment_lines(
            arg_values,
            prefix="arg",
            collector=translator.value_comment_log,
            trace_map=translator.trace_map,
        )
        method = _snake_to_camel(semantics.mnemonic)
        target = NameExpr(semantics.struct_context or "struct")
        call_expr = MethodCallExpr(target, method, call_args)
        signature = MethodSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
            struct=semantics.struct_context or "struct",
            method=method,
        )
        self._helper_registry.register_method(signature)
        translator.helper_calls += 1
        translator.helper_usage_counter[(semantics.mnemonic, "method")] += 1
        if outputs > 0:
            statements, values = translator.stack.push_call_results(
                call_expr, outputs, prefix="struct", origin=instruction.offset
            )
            summary_text = semantics.summary or ""
            summary_lines = self._format_summary_lines(summary_text) if summary_text else []
            if summary_lines:
                for value in values:
                    value.add_comments(summary_lines)
        else:
            statements = [CallStatement(call_expr)]
        return self._decorate_with_comment(
            statements, semantics, prefix_comments=prefix_comments
        )

    def _translate_call(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        arg_values = translator.stack.pop_many(inputs)
        call_args = [value.expression for value in arg_values]
        if semantics.uses_operand:
            call_args.append(self._operand_expression(semantics, instruction.operand))
        prefix_comments = self._value_comment_lines(
            arg_values,
            prefix="arg",
            collector=translator.value_comment_log,
            trace_map=translator.trace_map,
        )
        call_expr = CallExpr(NameExpr(semantics.mnemonic), call_args)
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        translator.helper_calls += 1
        translator.helper_usage_counter[(semantics.mnemonic, "function")] += 1
        if outputs <= 0:
            statements = [CallStatement(call_expr)]
        else:
            statements, values = translator.stack.push_call_results(
                call_expr, outputs, origin=instruction.offset
            )
            summary_text = semantics.summary or ""
            summary_lines = self._format_summary_lines(summary_text) if summary_text else []
            if summary_lines:
                for value in values:
                    value.add_comments(summary_lines)
            value = values[0] if len(values) == 1 else None
            return self._decorate_with_comment(
                statements, semantics, value=value, prefix_comments=prefix_comments
            )
        return self._decorate_with_comment(
            statements, semantics, prefix_comments=prefix_comments
        )

    def _translate_generic(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        arg_values = translator.stack.pop_many(inputs)
        call_args = [value.expression for value in arg_values]
        if semantics.uses_operand:
            call_args.append(self._operand_expression(semantics, instruction.operand))
        prefix_comments = self._value_comment_lines(
            arg_values,
            prefix="arg",
            collector=translator.value_comment_log,
            trace_map=translator.trace_map,
        )
        call_expr = CallExpr(NameExpr(semantics.mnemonic), call_args)
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        translator.helper_calls += 1
        translator.helper_usage_counter[(semantics.mnemonic, "function")] += 1
        if outputs > 0:
            statements, values = translator.stack.push_call_results(
                call_expr, outputs, origin=instruction.offset
            )
            summary_text = semantics.summary or ""
            summary_lines = self._format_summary_lines(summary_text) if summary_text else []
            if summary_lines:
                for value in values:
                    value.add_comments(summary_lines)
            value = values[0] if len(values) == 1 else None
            return self._decorate_with_comment(
                statements, semantics, value=value, prefix_comments=prefix_comments
            )
        statements = [CallStatement(call_expr)]
        return self._decorate_with_comment(
            statements, semantics, prefix_comments=prefix_comments
        )

    def _select_branch_targets(
        self,
        block_start: int,
        translator: BlockTranslator,
        fallthrough: Optional[int],
    ) -> Tuple[Optional[int], Optional[int]]:
        block = translator.program.blocks[block_start]
        succ = list(block.successors)
        false_target = None
        true_target = None
        if fallthrough is not None and fallthrough in succ:
            false_target = fallthrough
            succ = [value for value in succ if value != fallthrough]
        if succ:
            true_target = succ[0]
        elif fallthrough is not None:
            true_target = fallthrough
        return true_target, false_target

    # ------------------------------------------------------------------
    def _operand_expression(
        self, semantics: InstructionSemantics, operand: int
    ) -> LuaExpression:
        if semantics.enum_values and operand in semantics.enum_values:
            label = semantics.enum_values[operand]
            if semantics.enum_namespace:
                return NameExpr(f"{semantics.enum_namespace}.{label}")
            return NameExpr(label)
        literal = self._literal_formatter.format_operand(operand)
        return LiteralExpr(literal)

    # ------------------------------------------------------------------
    def _decorate_with_comment(
        self,
        statements: List[LuaStatement],
        semantics: InstructionSemantics,
        *,
        value: Optional[StackValue] = None,
        prefix_comments: Sequence[str] = (),
    ) -> List[LuaStatement]:
        summary = semantics.summary or ""
        comment_lines: List[str] = list(prefix_comments)
        formatted_summary: List[str] = []
        if summary:
            formatted_summary = self._format_summary_lines(summary)
            if value is not None:
                value.add_comments(formatted_summary)
            if self._should_emit_comment(summary):
                comment_lines = formatted_summary + comment_lines
        else:
            self._last_summary = None
        if not comment_lines:
            return statements
        if not statements:
            if value is not None and formatted_summary and not prefix_comments:
                return statements
            return [CommentStatement(line) for line in comment_lines] + statements
        first, *rest = statements
        commented = CommentedStatement(comment_lines, first)
        return [commented] + rest

    def _format_summary_lines(self, summary: str) -> List[str]:
        inline = self._comment_formatter.format_inline(summary)
        if inline:
            return [inline]
        return self._comment_formatter.wrap(summary)

    def _value_comment_lines(
        self,
        values: Sequence[StackValue],
        *,
        prefix: str,
        collector: Optional[List[Tuple[str, Optional[int], str]]] = None,
        trace_map: Optional[Dict[int, InstructionTraceInfo]] = None,
    ) -> List[str]:
        lines: List[str] = []
        for index, value in enumerate(values, 1):
            comments = value.take_comments()
            if not comments:
                continue
            suffix = f" @0x{value.origin:06X}" if value.origin is not None else ""
            label = f"{prefix}{index}{suffix}"
            for comment in comments:
                lines.append(f"{label}: {comment}")
                if collector is not None:
                    collector.append((f"{prefix}{index}", value.origin, comment))
                if trace_map is not None and value.origin is not None:
                    info = trace_map.get(value.origin)
                    if info is None:
                        info = InstructionTraceInfo(
                            offset=value.origin,
                            mnemonic="unknown",
                            summary="",
                        )
                        trace_map[value.origin] = info
                    info.add_usage(f"{prefix}{index}", comment)
        return lines

    def _should_emit_comment(self, summary: str) -> bool:
        if not summary:
            self._last_summary = None
            return False
        if self.options.deduplicate_comments and summary == self._last_summary:
            return False
        self._last_summary = summary
        return True

    def _module_summary_lines(
        self, functions: Sequence[HighLevelFunction]
    ) -> List[str]:
        lines = ["module summary:"]
        lines.append(f"- functions: {len(functions)}")
        helper_functions = self._helper_registry.function_count()
        helper_methods = self._helper_registry.method_count()
        lines.append(f"- helper functions: {helper_functions}")
        lines.append(f"- struct helpers: {helper_methods}")
        literal_total = sum(func.metadata.literal_count for func in functions)
        lines.append(f"- literal instructions: {literal_total}")
        branch_total = sum(func.metadata.branch_count for func in functions)
        lines.append(f"- branch instructions: {branch_total}")
        if not self._enum_registry.is_empty():
            lines.append(
                "- enum namespaces: "
                f"{self._enum_registry.namespace_count()} "
                f"({self._enum_registry.total_values()} values)"
            )
        warnings = sum(len(func.metadata.warnings) for func in functions)
        lines.append(f"- stack warnings: {warnings}")
        string_total = sum(
            sum(1 for run in func.metadata.literal_runs if run.kind == "string")
            for func in functions
        )
        if string_total:
            lines.append(f"- string literal sequences: {string_total}")
        numeric_total = sum(
            sum(1 for run in func.metadata.literal_runs if run.kind == "number")
            for func in functions
        )
        if numeric_total:
            lines.append(f"- numeric literal runs: {numeric_total}")
        if self.options.emit_literal_report:
            all_runs: List[LiteralRun] = []
            for func in functions:
                all_runs.extend(func.metadata.literal_runs)
            if all_runs:
                report = build_literal_run_report(all_runs)
                lines.append(f"- literal run blocks: {len(report.block_summaries)}")
                tokens = report.top_tokens(limit=3)
                if tokens:
                    token_line = ", ".join(f"{token}:{count}" for token, count in tokens)
                    lines.append(f"- literal tokens: {token_line}")
                numbers = report.top_numbers(limit=3)
                if numbers:
                    number_line = ", ".join(f"{value}:{count}" for value, count in numbers)
                    lines.append(f"- literal numbers: {number_line}")
                previews = report.longest_previews(limit=2)
                if previews:
                    lines.append(
                        "- notable literal runs: " + "; ".join(previews)
                    )
        return lines

    def _derive_function_name(
        self,
        program: IRProgram,
        sequences: Sequence[LiteralRun],
    ) -> str:
        candidate = self._select_string_name(program, sequences)
        if candidate:
            return candidate
        return f"segment_{program.segment_index:03d}"

    def _select_string_name(
        self,
        program: IRProgram,
        sequences: Sequence[LiteralRun],
    ) -> Optional[str]:
        if not sequences:
            return None
        entry_offset = min(program.blocks) if program.blocks else 0
        raw_candidates: List[Tuple[LiteralRun, str, int]] = []
        frequency: Counter[str] = Counter()
        for sequence in sequences:
            if sequence.kind != "string":
                continue
            text = (sequence.combined_string() or "").strip()
            if text and not any(ch.isspace() for ch in text):
                sanitized = _sanitize_identifier(text)
                if sanitized and sanitized.lower() not in _STRING_NAME_STOPWORDS:
                    key = sanitized.lower()
                    raw_candidates.append((sequence, sanitized, sequence.start_offset()))
                    frequency[key] += 1
            for token, token_offset in _identifier_tokens(sequence):
                sanitized = _sanitize_identifier(token)
                if not sanitized:
                    continue
                lowered = sanitized.lower()
                if lowered in _STRING_NAME_STOPWORDS:
                    continue
                raw_candidates.append((sequence, sanitized, token_offset))
                frequency[lowered] += 1
        if not raw_candidates:
            return None
        candidates: List[Tuple[Tuple[int, int, int, int, int, str], str]] = []
        seen: set[str] = set()
        for sequence, sanitized, offset in raw_candidates:
            lowered = sanitized.lower()
            if sanitized in seen:
                continue
            count = frequency[lowered]
            score = self._score_name_candidate(
                sequence,
                sanitized,
                entry_offset,
                count=count,
                candidate_offset=offset,
            )
            candidates.append((score, sanitized))
            seen.add(sanitized)
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        selected = candidates[0][1]
        if selected.lower() in _LUA_KEYWORDS:
            selected = f"{selected}_fn"
        return selected

    def _score_name_candidate(
        self,
        sequence: LiteralRun,
        sanitized: str,
        entry_offset: int,
        *,
        count: int = 1,
        candidate_offset: Optional[int] = None,
    ) -> Tuple[int, int, int, int, int, str]:
        offset = candidate_offset if candidate_offset is not None else sequence.start_offset()
        distance = max(0, offset - entry_offset)
        underscore_penalty = 0 if "_" in sanitized else 1
        lower = sum(1 for ch in sanitized if ch.islower())
        upper = sum(1 for ch in sanitized if ch.isupper())
        if lower and upper:
            case_penalty = 0
        elif lower:
            case_penalty = 1
        elif upper:
            case_penalty = 2
        else:
            case_penalty = 3
        digit_penalty = sum(1 for ch in sanitized if ch.isdigit())
        length_penalty = len(sanitized)
        tie_breaker = sanitized.lower()
        return (
            -count,
            underscore_penalty,
            distance,
            case_penalty,
            digit_penalty,
            length_penalty,
            tie_breaker,
        )

    def _unique_function_name(self, base: str) -> str:
        candidate = base
        index = 1
        while candidate in self._used_function_names:
            candidate = f"{base}_{index}"
            index += 1
        self._used_function_names.add(candidate)
        return candidate

_LUA_KEYWORDS = {
    "and",
    "break",
    "do",
    "else",
    "elseif",
    "end",
    "false",
    "for",
    "function",
    "goto",
    "if",
    "in",
    "local",
    "nil",
    "not",
    "or",
    "repeat",
    "return",
    "then",
    "true",
    "until",
    "while",
}

_STRING_NAME_STOPWORDS = {"usage", "warning"}


_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _identifier_tokens(
    sequence: LiteralRun,
) -> Iterable[Tuple[str, int]]:
    if sequence.kind != "string":
        return []
    text = sequence.combined_string() or ""
    if not text:
        return []
    offsets = sequence.offsets
    limit = len(offsets) - 1
    results: List[Tuple[str, int]] = []
    for match in _IDENTIFIER_PATTERN.finditer(text):
        start = match.start()
        # Align tokens to instruction boundaries to avoid partial fragments.
        if start % 2 != 0:
            continue
        chunk_index = start // 2
        if chunk_index > limit:
            chunk_index = limit
        token = match.group(0)
        underscore_index = token.find("_")
        include_full = True
        if underscore_index != -1 and any(
            ch.isupper() for ch in token[underscore_index + 1 :]
        ):
            include_full = False
        if include_full:
            results.append((token, offsets[chunk_index]))
        for rel_start, sub in _split_identifier_subtokens(token):
            if rel_start == 0 and include_full:
                continue
            absolute = start + rel_start
            if absolute % 2 != 0:
                continue
            sub_index = absolute // 2
            if sub_index > limit:
                sub_index = limit
            results.append((sub, offsets[sub_index]))
    return results


def _split_identifier_subtokens(token: str) -> List[Tuple[int, str]]:
    parts: List[Tuple[int, str]] = []
    start = 0
    for index in range(1, len(token)):
        if token[index].isupper() and token[index - 1].islower():
            if index - start >= 3:
                parts.append((start, token[start:index]))
            start = index
    if len(token) - start >= 3:
        parts.append((start, token[start:]))
    return parts


def _sanitize_identifier(text: str) -> Optional[str]:
    if not text:
        return None
    pieces: List[str] = []
    for char in text:
        if char.isalnum() or char == "_":
            pieces.append(char)
        else:
            pieces.append("_")
    candidate = "".join(pieces)
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if not candidate:
        return None
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    return candidate


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
