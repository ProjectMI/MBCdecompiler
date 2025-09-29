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
from .vm_analysis import (
    VMBlockTrace,
    VMOperation,
    VMProgramTrace,
    VMStackValue,
    VirtualMachineAnalyzer,
)


@dataclass
class StackEntry:
    value: VMStackValue
    expression: LuaExpression


class VMValueExpressionRegistry:
    """Map VM stack values to reusable Lua expressions."""

    def __init__(self) -> None:
        self._expressions: Dict[str, LuaExpression] = {}

    def ensure(self, value: VMStackValue) -> NameExpr:
        existing = self._expressions.get(value.name)
        if isinstance(existing, NameExpr):
            return existing
        symbol = _sanitize_stack_name(value.name)
        expr = NameExpr(symbol)
        self._expressions[value.name] = expr
        return expr

    def register(self, value: VMStackValue, expression: LuaExpression) -> LuaExpression:
        self._expressions[value.name] = expression
        return expression

    def fetch(self, value: VMStackValue) -> Optional[LuaExpression]:
        return self._expressions.get(value.name)


def _sanitize_stack_name(name: str) -> str:
    cleaned = [ch if ch.isalnum() or ch == '_' else '_' for ch in name]
    if not cleaned:
        return "value"
    identifier = "".join(cleaned)
    identifier = identifier.strip("_") or "value"
    if identifier[0].isdigit():
        identifier = f"value_{identifier}"
    return identifier


class HighLevelStack:
    """Track symbolic stack values during reconstruction."""

    def __init__(self, registry: VMValueExpressionRegistry) -> None:
        self._registry = registry
        self._entries: List[StackEntry] = []
        self._counter = 0
        self.warnings: List[str] = []

    def seed(self, values: Sequence[VMStackValue]) -> None:
        self._entries = [StackEntry(value, self._registry.ensure(value)) for value in values]

    def new_symbol(self, prefix: str = "value") -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def pop_for_operation(self, operation: VMOperation) -> List[LuaExpression]:
        popped: List[StackEntry] = []
        for vm_value in reversed(operation.inputs):
            entry = self._pop_entry(vm_value)
            popped.append(entry)
        popped.reverse()
        return [entry.expression for entry in popped]

    def push_for_operation(
        self, operation: VMOperation, expressions: Sequence[LuaExpression]
    ) -> List[StackEntry]:
        entries: List[StackEntry] = []
        for vm_value, expression in zip(operation.outputs, expressions):
            self._registry.register(vm_value, expression)
            entry = StackEntry(vm_value, expression)
            self._entries.append(entry)
            entries.append(entry)
        return entries

    def flush(self) -> List[LuaExpression]:
        expressions = [entry.expression for entry in self._entries]
        self._entries.clear()
        return expressions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pop_entry(self, vm_value: VMStackValue) -> StackEntry:
        if not self._entries:
            placeholder = NameExpr(self.new_symbol("stack"))
            self.warnings.append(
                f"underflow generated placeholder {placeholder.name} for {vm_value.name}"
            )
            self._registry.register(vm_value, placeholder)
            return StackEntry(vm_value, placeholder)

        entry = self._entries.pop()
        if entry.value.name == vm_value.name:
            return entry

        for idx in range(len(self._entries) - 1, -1, -1):
            candidate = self._entries[idx]
            if candidate.value.name == vm_value.name:
                self._entries.pop(idx)
                self.warnings.append(
                    f"stack reorder to satisfy value {vm_value.name}"
                )
                return candidate

        placeholder = NameExpr(self.new_symbol("stack"))
        self.warnings.append(
            f"missing stack value {vm_value.name}, created {placeholder.name}"
        )
        self._registry.register(vm_value, placeholder)
        return StackEntry(vm_value, placeholder)


class StringLiteralCollector:
    """Group consecutive string literal assignments into annotated sequences."""

    def __init__(self) -> None:
        self._pending: List[List[LuaStatement]] = []
        self._fragments: List[str] = []
        self._offsets: List[int] = []

    def reset(self) -> None:
        self._pending.clear()
        self._fragments.clear()
        self._offsets.clear()

    def enqueue(
        self,
        offset: int,
        statements: List[LuaStatement],
        expression: LuaExpression,
    ) -> List[LuaStatement]:
        literal_text = self._string_value(expression)
        if literal_text is None:
            flushed = self.flush()
            return flushed + statements
        self._pending.append(statements)
        self._fragments.append(literal_text)
        self._offsets.append(offset)
        return []

    def flush(self) -> List[LuaStatement]:
        if not self._pending:
            return []
        combined = "".join(self._fragments)
        comment = CommentStatement(
            f"string literal sequence: {escape_lua_string(combined)}"
            f" (len={len(combined)})"
        )
        result: List[LuaStatement] = [comment]
        for group in self._pending:
            result.extend(group)
        self.reset()
        return result

    def finalize(self) -> List[LuaStatement]:
        return self.flush()

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

    def summary_lines(self) -> List[str]:
        lines = ["function summary:"]
        lines.append(f"- blocks: {self.block_count}")
        lines.append(f"- instructions: {self.instruction_count}")
        lines.append(f"- literal instructions: {self.literal_count}")
        lines.append(f"- helper invocations: {self.helper_calls}")
        lines.append(f"- branches: {self.branch_count}")
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

    def __init__(
        self,
        reconstructor: "HighLevelReconstructor",
        program: IRProgram,
        vm_trace: VMProgramTrace,
    ) -> None:
        self._reconstructor = reconstructor
        self._program = program
        self._vm_trace = vm_trace
        self._value_registry = VMValueExpressionRegistry()
        self._current_stack: Optional[HighLevelStack] = None
        self._stack_warnings: List[str] = []
        self._string_collector = StringLiteralCollector()
        self._literal_tracker = LiteralRunTracker()
        self.literal_count = 0
        self.helper_calls = 0
        self.branch_count = 0
        self.instruction_total = 0

    def translate(self) -> Dict[int, BlockInfo]:
        blocks: Dict[int, BlockInfo] = {}
        order = sorted(self._program.blocks)
        for idx, start in enumerate(order):
            block = self._program.blocks[start]
            next_offset = order[idx + 1] if idx + 1 < len(order) else None
            trace = self._vm_trace.blocks.get(start)
            stack = HighLevelStack(self._value_registry)
            if trace is not None:
                stack.seed(trace.entry_stack)
            self._current_stack = stack
            self._literal_tracker.start_block(block.start)
            blocks[start] = self._translate_block(block, trace, next_offset)
            self.instruction_total += len(block.instructions)
            self._stack_warnings.extend(stack.warnings)
        self._literal_tracker.finalize()
        self._current_stack = None
        return blocks

    # ------------------------------------------------------------------
    def _translate_block(
        self, block: IRBlock, trace: Optional[VMBlockTrace], fallthrough: Optional[int]
    ) -> BlockInfo:
        statements: List[LuaStatement] = []
        terminator: Optional[Terminator] = None
        self._string_collector.reset()
        for index, instruction in enumerate(block.instructions):
            semantics = instruction.semantics
            operation = trace.instructions[index].operation if trace else None
            self._reconstructor._register_enums(semantics)
            is_last = index == len(block.instructions) - 1
            if semantics.has_tag("literal"):
                operand_expr = self._reconstructor._operand_expression(
                    semantics, instruction.operand
                )
                literal_statements, operand_expr = self._reconstructor._translate_literal(
                    instruction,
                    semantics,
                    self,
                    operation,
                    operand_expr=operand_expr,
                )
                self._literal_tracker.observe(instruction.offset, operand_expr)
                queued = self._string_collector.enqueue(
                    instruction.offset, literal_statements, operand_expr
                )
                statements.extend(queued)
                continue
            else:
                self._literal_tracker.break_sequence()
            statements.extend(self._string_collector.flush())
            if semantics.has_tag("comparison"):
                statements.extend(
                    self._reconstructor._translate_comparison(
                        instruction, semantics, self, operation
                    )
                )
                continue
            if semantics.control_flow == "branch" and is_last:
                terminator = self._reconstructor._translate_branch(
                    block.start,
                    instruction,
                    semantics,
                    self,
                    operation,
                    fallthrough,
                )
                continue
            if semantics.control_flow == "return" and is_last:
                terminator = self._reconstructor._translate_return(
                    instruction,
                    semantics,
                    self,
                    operation,
                )
                continue
            if semantics.has_tag("structure"):
                statements.extend(
                    self._reconstructor._translate_structure(
                        instruction, semantics, self, operation
                    )
                )
                continue
            if semantics.has_tag("call"):
                statements.extend(
                    self._reconstructor._translate_call(
                        instruction, semantics, self, operation
                    )
                )
                continue
            statements.extend(
                self._reconstructor._translate_generic(
                    instruction, semantics, self, operation
                )
            )
        statements.extend(self._string_collector.finalize())
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
        if self._current_stack is None:
            raise RuntimeError("stack not initialised")
        return self._current_stack

    @property
    def stack_warnings(self) -> Sequence[str]:
        return tuple(self._stack_warnings)

    def pop_inputs(self, operation: Optional[VMOperation]) -> List[LuaExpression]:
        if operation is None:
            return []
        return self.stack.pop_for_operation(operation)

    def assign_outputs(
        self,
        operation: Optional[VMOperation],
        expressions: Sequence[LuaExpression],
        *,
        make_local: bool = True,
    ) -> Tuple[List[LuaStatement], List[LuaExpression]]:
        if operation is None or not operation.outputs:
            return [], []

        targets = [self._value_registry.ensure(value) for value in operation.outputs]
        statements: List[LuaStatement] = []

        if len(targets) == 1:
            target = targets[0]
            expr = expressions[0]
            if isinstance(expr, NameExpr) and expr.name == target.name:
                final_exprs = [expr]
            else:
                statements.append(Assignment([target], expr, is_local=make_local))
                final_exprs = [target]
        else:
            statements.append(
                MultiAssignment(targets, list(expressions), is_local=make_local)
            )
            final_exprs = list(targets)

        self.stack.push_for_operation(operation, final_exprs)
        return statements, final_exprs

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
        analyzer = VirtualMachineAnalyzer()
        vm_trace = analyzer.trace_program(program)
        translator = BlockTranslator(self, program, vm_trace)
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
        metadata = FunctionMetadata(
            block_count=len(blocks),
            instruction_count=translator.instruction_total,
            warnings=list(translator.stack_warnings),
            helper_calls=translator.helper_calls,
            branch_count=translator.branch_count,
            literal_count=translator.literal_count,
            literal_runs=tuple(literal_runs),
            literal_stats=stats,
            literal_report=report,
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
        operation: Optional[VMOperation],
        *,
        operand_expr: Optional[LuaExpression] = None,
    ) -> Tuple[List[LuaStatement], LuaExpression]:
        operand = operand_expr or self._operand_expression(semantics, instruction.operand)
        translator.literal_count += 1
        statements, outputs = translator.assign_outputs(operation, [operand])
        decorated = self._decorate_with_comment(statements, semantics)
        return decorated, operand

    def _translate_comparison(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        operation: Optional[VMOperation],
    ) -> List[LuaStatement]:
        inputs = translator.pop_inputs(operation)
        if len(inputs) >= 2:
            lhs, rhs = inputs[0], inputs[1]
        elif len(inputs) == 1:
            lhs, rhs = inputs[0], NameExpr(translator.stack.new_symbol("stack"))
        else:
            lhs = NameExpr(translator.stack.new_symbol("stack"))
            rhs = NameExpr(translator.stack.new_symbol("stack"))
        operator = semantics.comparison_operator or "=="
        expression = BinaryExpr(lhs, operator, rhs)
        statements, _ = translator.assign_outputs(operation, [expression], make_local=True)
        return self._decorate_with_comment(statements, semantics)

    def _translate_branch(
        self,
        block_start: int,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        operation: Optional[VMOperation],
        fallthrough: Optional[int],
    ) -> BranchTerminator:
        inputs = translator.pop_inputs(operation)
        if inputs:
            condition = inputs[-1]
        else:
            condition = NameExpr(translator.stack.new_symbol("cond"))
        # Determine targets based on IR metadata.
        true_target, false_target = self._select_branch_targets(
            block_start, translator, fallthrough
        )
        translator.branch_count += 1
        comment = self._comment_formatter.format_inline(semantics.summary or "")
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
        operation: Optional[VMOperation],
    ) -> ReturnTerminator:
        values = translator.stack.flush()
        comment = self._comment_formatter.format_inline(semantics.summary or "")
        return ReturnTerminator(values=values, comment=comment)

    def _translate_structure(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        operation: Optional[VMOperation],
    ) -> List[LuaStatement]:
        args = translator.pop_inputs(operation)
        if semantics.uses_operand:
            args.append(self._operand_expression(semantics, instruction.operand))
        method = _snake_to_camel(semantics.mnemonic)
        target = NameExpr(semantics.struct_context or "struct")
        call_expr = MethodCallExpr(target, method, args)
        input_count = (
            len(operation.inputs) if operation is not None else semantics.stack_inputs or 0
        )
        output_count = (
            len(operation.outputs) if operation is not None else semantics.stack_outputs or 0
        )
        signature = MethodSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=input_count,
            outputs=output_count,
            uses_operand=semantics.uses_operand,
            struct=semantics.struct_context or "struct",
            method=method,
        )
        self._helper_registry.register_method(signature)
        translator.helper_calls += 1
        if operation and operation.outputs:
            statements, _ = translator.assign_outputs(operation, [call_expr])
        else:
            statements = [CallStatement(call_expr)]
        return self._decorate_with_comment(statements, semantics)

    def _translate_call(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        operation: Optional[VMOperation],
    ) -> List[LuaStatement]:
        args = translator.pop_inputs(operation)
        if semantics.uses_operand:
            args.append(self._operand_expression(semantics, instruction.operand))
        call_expr = CallExpr(NameExpr(semantics.mnemonic), args)
        input_count = (
            len(operation.inputs) if operation is not None else semantics.stack_inputs or 0
        )
        output_count = (
            len(operation.outputs) if operation is not None else semantics.stack_outputs or 0
        )
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=input_count,
            outputs=output_count,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        translator.helper_calls += 1
        if operation and operation.outputs:
            statements, _ = translator.assign_outputs(operation, [call_expr])
        else:
            statements = [CallStatement(call_expr)]
        return self._decorate_with_comment(statements, semantics)

    def _translate_generic(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        operation: Optional[VMOperation],
    ) -> List[LuaStatement]:
        args = translator.pop_inputs(operation)
        if semantics.uses_operand:
            args.append(self._operand_expression(semantics, instruction.operand))
        call_expr = CallExpr(NameExpr(semantics.mnemonic), args)
        input_count = (
            len(operation.inputs) if operation is not None else semantics.stack_inputs or 0
        )
        output_count = (
            len(operation.outputs) if operation is not None else semantics.stack_outputs or 0
        )
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=input_count,
            outputs=output_count,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        translator.helper_calls += 1
        if operation and operation.outputs:
            statements, _ = translator.assign_outputs(operation, [call_expr])
        else:
            statements = [CallStatement(call_expr)]
        return self._decorate_with_comment(statements, semantics)

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
        self, statements: List[LuaStatement], semantics: InstructionSemantics
    ) -> List[LuaStatement]:
        summary = semantics.summary or ""
        if not self._should_emit_comment(summary):
            return statements
        comment = self._comment_formatter.format_inline(summary)
        if not comment:
            wrapped = self._comment_formatter.wrap(summary)
            return [CommentStatement(line) for line in wrapped] + statements
        return [CommentStatement(comment)] + statements

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
