"""High level Lua reconstruction leveraging manual annotations and stack heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
from .inline_strings import (
    InlineStringAccumulator,
    InlineStringCollector,
    InlineStringChunk,
    is_inline_semantics,
    render_inline_tables,
)
from .manual_semantics import InstructionSemantics
from .vm_analysis import LuaLiteralFormatter, estimate_stack_io


@dataclass
class StackValue:
    """Representation of a value currently residing on the reconstruction stack."""

    name: str
    expression: LuaExpression
    source: Optional[LuaExpression] = None
    py_value: object | None = None
    origin: str = "value"

    def as_expression(self) -> LuaExpression:
        return self.expression

    def debug_label(self) -> str:
        label = self.expression.render()
        if self.py_value is not None:
            label += f"={self.py_value!r}"
        return label


class HighLevelStack:
    """Track symbolic stack values during reconstruction."""

    def __init__(self) -> None:
        self._values: List[StackValue] = []
        self._counter = 0
        self._known_values: Dict[str, object | None] = {}
        self.warnings: List[str] = []

    def new_symbol(self, prefix: str = "value") -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def push_literal(
        self, expression: LuaExpression
    ) -> Tuple[List[LuaStatement], StackValue]:
        name = self.new_symbol("literal")
        target = NameExpr(name)
        statement = Assignment([target], expression, is_local=True)
        value = StackValue(
            name=name,
            expression=target,
            source=expression,
            py_value=self._infer_py_value(expression),
            origin="literal",
        )
        self._values.append(value)
        self._known_values[name] = value.py_value
        return [statement], value

    def push_expression(
        self,
        expression: LuaExpression,
        *,
        prefix: str = "tmp",
        make_local: bool = False,
    ) -> Tuple[List[LuaStatement], StackValue]:
        if make_local or not isinstance(expression, NameExpr):
            name = self.new_symbol(prefix)
            target = NameExpr(name)
            statement = Assignment([target], expression, is_local=True)
            value = StackValue(
                name=name,
                expression=target,
                source=expression,
                py_value=self._infer_py_value(expression),
                origin=prefix,
            )
            self._values.append(value)
            self._known_values[name] = value.py_value
            return [statement], value
        py_value = self._known_values.get(expression.name)
        value = StackValue(
            name=expression.name,
            expression=expression,
            source=None,
            py_value=py_value,
            origin="alias",
        )
        self._values.append(value)
        return [], value

    def push_call_results(
        self, expression: CallExpr, outputs: int, prefix: str = "result"
    ) -> Tuple[List[LuaStatement], List[StackValue]]:
        if outputs <= 0:
            return [], []
        if outputs == 1:
            name = self.new_symbol(prefix)
            target = NameExpr(name)
            stmt = Assignment([target], expression, is_local=True)
            value = StackValue(
                name=name,
                expression=target,
                source=expression,
                py_value=None,
                origin=prefix,
            )
            self._values.append(value)
            self._known_values[name] = value.py_value
            return [stmt], [value]
        targets = [NameExpr(self.new_symbol(prefix)) for _ in range(outputs)]
        stmt = MultiAssignment(targets, [expression], is_local=True)
        values: List[StackValue] = []
        for target in targets:
            value = StackValue(
                name=target.name,
                expression=target,
                source=expression,
                py_value=None,
                origin=prefix,
            )
            self._values.append(value)
            self._known_values[target.name] = value.py_value
            values.append(value)
        return [stmt], values

    def pop_single(self) -> StackValue:
        if self._values:
            return self._values.pop()
        name = self.new_symbol("stack")
        placeholder = StackValue(name=name, expression=NameExpr(name), origin="placeholder")
        self.warnings.append(f"underflow generated placeholder {placeholder.name}")
        self._known_values.setdefault(name, None)
        return placeholder

    def pop_many(self, count: int) -> List[StackValue]:
        items = [self.pop_single() for _ in range(count)]
        items.reverse()
        return items

    def pop_pair(self) -> Tuple[StackValue, StackValue]:
        lhs, rhs = self.pop_many(2)
        return lhs, rhs

    def flush(self) -> List[StackValue]:
        values = list(self._values)
        self._values.clear()
        return values

    def _infer_py_value(self, expression: LuaExpression) -> object | None:
        if isinstance(expression, LiteralExpr):
            return expression.py_value
        if isinstance(expression, NameExpr):
            return self._known_values.get(expression.name)
        return None

    def describe(self) -> List[str]:
        lines: List[str] = []
        for index, value in enumerate(reversed(self._values)):
            label = value.debug_label()
            lines.append(f"[{index}] {label}")
        return lines

    def snapshot(self) -> List[Dict[str, object]]:
        return [
            {
                "name": value.name,
                "expression": value.expression.render(),
                "origin": value.origin,
                "py_value": value.py_value,
            }
            for value in self._values
        ]


@dataclass
class FunctionMetadata:
    """Descriptive statistics collected while reconstructing a function."""

    block_count: int
    instruction_count: int
    warnings: List[str] = field(default_factory=list)
    helper_calls: int = 0
    branch_count: int = 0
    literal_count: int = 0
    inline_chunk_count: int = 0
    inline_byte_count: int = 0

    def summary_lines(self) -> List[str]:
        lines = ["function summary:"]
        lines.append(f"- blocks: {self.block_count}")
        lines.append(f"- instructions: {self.instruction_count}")
        lines.append(f"- literal instructions: {self.literal_count}")
        lines.append(f"- helper invocations: {self.helper_calls}")
        lines.append(f"- branches: {self.branch_count}")
        lines.append(f"- inline chunks: {self.inline_chunk_count}")
        lines.append(f"- inline bytes: {self.inline_byte_count}")
        return lines

    def warning_lines(self) -> List[str]:
        return [f"- {warning}" for warning in self.warnings]

    def to_dict(self) -> Dict[str, object]:
        return {
            "blocks": self.block_count,
            "instructions": self.instruction_count,
            "warnings": list(self.warnings),
            "helper_calls": self.helper_calls,
            "branches": self.branch_count,
            "literals": self.literal_count,
            "inline_chunks": self.inline_chunk_count,
            "inline_bytes": self.inline_byte_count,
        }


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

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "metadata": self.metadata.to_dict()}


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
        self.literal_count = 0
        self.helper_calls = 0
        self.branch_count = 0
        self.instruction_total = 0
        self.inline_chunk_count = 0
        self.inline_byte_count = 0

    def translate(self) -> Dict[int, BlockInfo]:
        blocks: Dict[int, BlockInfo] = {}
        order = sorted(self._program.blocks)
        for idx, start in enumerate(order):
            block = self._program.blocks[start]
            next_offset = order[idx + 1] if idx + 1 < len(order) else None
            blocks[start] = self._translate_block(block, next_offset)
            self.instruction_total += len(block.instructions)
        return blocks

    # ------------------------------------------------------------------
    def _translate_block(self, block: IRBlock, fallthrough: Optional[int]) -> BlockInfo:
        statements: List[LuaStatement] = []
        terminator: Optional[Terminator] = None
        inline_accumulator = InlineStringAccumulator()
        for index, instruction in enumerate(block.instructions):
            semantics = instruction.semantics
            self._reconstructor._register_enums(semantics)
            is_last = index == len(block.instructions) - 1
            if is_inline_semantics(semantics):
                inline_accumulator.feed(instruction)
                continue
            if inline_accumulator.has_data():
                self._flush_inline(block.start, inline_accumulator, statements)
            if semantics.has_tag("literal"):
                statements.extend(
                    self._reconstructor._translate_literal(instruction, semantics, self)
                )
                continue
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
        if terminator is None:
            target = block.successors[0] if block.successors else fallthrough
            terminator = JumpTerminator(target=target, fallthrough=fallthrough, comment=None)
        if inline_accumulator.has_data():
            self._flush_inline(block.start, inline_accumulator, statements)
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

    def _flush_inline(
        self,
        block_start: int,
        accumulator: InlineStringAccumulator,
        statements: List[LuaStatement],
    ) -> None:
        chunk = accumulator.finish(self._program.segment_index, block_start)
        self.inline_chunk_count += 1
        self.inline_byte_count += len(chunk.data)
        statements.extend(self._reconstructor._inline_chunk_statements(chunk))
        self._reconstructor._record_inline_chunk(chunk)


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
        self._inline_strings = InlineStringCollector()

    # ------------------------------------------------------------------
    def from_ir(self, program: IRProgram) -> HighLevelFunction:
        self._last_summary = None
        translator = BlockTranslator(self, program)
        blocks = translator.translate()
        entry = min(blocks)
        structurer = ControlFlowStructurer(blocks)
        body = structurer.structure(entry)
        function_name = f"segment_{program.segment_index:03d}"
        metadata = FunctionMetadata(
            block_count=len(blocks),
            instruction_count=translator.instruction_total,
            warnings=list(translator.stack.warnings),
            helper_calls=translator.helper_calls,
            branch_count=translator.branch_count,
            literal_count=translator.literal_count,
            inline_chunk_count=translator.inline_chunk_count,
            inline_byte_count=translator.inline_byte_count,
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
        if not self._inline_strings.is_empty():
            sections.append(render_inline_tables(self._inline_strings).rstrip())
        for function in functions:
            sections.append(function.render().rstrip())
        return join_sections(sections)

    def build_report(self, functions: Sequence[HighLevelFunction]) -> Dict[str, object]:
        inline_report = self._inline_strings.build_report()
        return {
            "module": self._module_summary_lines(functions),
            "inline": inline_report.to_dict(),
            "inline_samples": self._inline_summary(),
            "functions": [function.to_dict() for function in functions],
        }

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
    ) -> List[LuaStatement]:
        operand = self._operand_expression(semantics, instruction.operand)
        statements, _ = translator.stack.push_literal(operand)
        translator.literal_count += 1
        return self._decorate_with_comment(statements, semantics)

    def _translate_comparison(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        lhs, rhs = translator.stack.pop_pair()
        lhs_expr = lhs.as_expression()
        rhs_expr = rhs.as_expression()
        operator = semantics.comparison_operator or "=="
        expression = BinaryExpr(lhs_expr, operator, rhs_expr)
        statements, _ = translator.stack.push_expression(expression, prefix="cmp", make_local=True)
        return self._decorate_with_comment(statements, semantics)

    def _translate_branch(
        self,
        block_start: int,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        fallthrough: Optional[int],
    ) -> BranchTerminator:
        condition = translator.stack.pop_single().as_expression()
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
    ) -> ReturnTerminator:
        values = [value.as_expression() for value in translator.stack.flush()]
        comment = self._comment_formatter.format_inline(semantics.summary or "")
        return ReturnTerminator(values=values, comment=comment)

    def _translate_structure(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        args = [value.as_expression() for value in translator.stack.pop_many(inputs)]
        if semantics.uses_operand:
            args.append(self._operand_expression(semantics, instruction.operand))
        method = _snake_to_camel(semantics.mnemonic)
        target = NameExpr(semantics.struct_context or "struct")
        call_expr = MethodCallExpr(target, method, args)
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
        if outputs > 0:
            statements, _ = translator.stack.push_call_results(call_expr, outputs, prefix="struct")
        else:
            statements = [CallStatement(call_expr)]
        return self._decorate_with_comment(statements, semantics)

    def _translate_call(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        args = [value.as_expression() for value in translator.stack.pop_many(inputs)]
        if semantics.uses_operand:
            args.append(self._operand_expression(semantics, instruction.operand))
        call_expr = CallExpr(NameExpr(semantics.mnemonic), args)
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        translator.helper_calls += 1
        if outputs <= 0:
            statements = [CallStatement(call_expr)]
        else:
            statements, _ = translator.stack.push_call_results(call_expr, outputs)
        return self._decorate_with_comment(statements, semantics)

    def _translate_generic(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        args = [value.as_expression() for value in translator.stack.pop_many(inputs)]
        if semantics.uses_operand:
            args.append(self._operand_expression(semantics, instruction.operand))
        call_expr = CallExpr(NameExpr(semantics.mnemonic), args)
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        translator.helper_calls += 1
        if outputs > 0:
            statements, _ = translator.stack.push_call_results(call_expr, outputs)
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
        literal, py_value = self._literal_formatter.format_operand(operand)
        return LiteralExpr(literal, py_value)

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
        report = self._inline_strings.build_report()
        lines.append(
            f"- inline string data: {report.entry_count} entries ({report.total_bytes} bytes)"
        )
        lines.append(f"- inline segments: {report.segment_count}")
        if report.entry_count:
            lines.append(
                f"- average inline chunk: {report.average_length:.1f} bytes"
            )
            longest = report.longest_summary()
            if longest:
                lines.append(f"- largest inline chunk: {longest}")
            for sample in self._inline_summary()[:3]:
                lines.append(f"- inline sample: {sample}")
        return lines

    def _record_inline_chunk(self, chunk: InlineStringChunk) -> None:
        self._inline_strings.add(chunk)

    def _inline_chunk_statements(self, chunk: InlineStringChunk) -> List[LuaStatement]:
        if not getattr(self.options, "emit_inline_comments", True):
            return []
        preview_limit = getattr(self.options, "inline_preview_limit", 72)
        threshold = getattr(self.options, "inline_text_threshold", 0.65)
        ratio = chunk.printable_ratio()
        preview = chunk.preview(limit=preview_limit)
        if chunk.start_offset == chunk.end_offset:
            offset_desc = f"0x{chunk.start_offset:06X}"
        else:
            offset_desc = f"0x{chunk.start_offset:06X}..0x{chunk.end_offset:06X}"
        descriptor = f"inline chunk {offset_desc} ({chunk.length} bytes)"
        lines: List[str] = []
        if preview and preview != "<empty>":
            descriptor += f" => {preview}"
        else:
            descriptor += f" [printable {ratio:.2f}]"
        lines.append(descriptor)
        if ratio < threshold:
            hex_sample = chunk.data[:16].hex()
            if len(chunk.data) > 16:
                hex_sample += "…"
            lines.append(f"inline hex sample: {hex_sample}")
        return [CommentStatement(line) for line in lines]

    def _inline_summary(self) -> List[str]:
        threshold = getattr(self.options, "inline_text_threshold", 0.65)
        preview_limit = getattr(self.options, "inline_preview_limit", 72)
        lines: List[str] = []
        for sequence in self._inline_strings.iter_sequences():
            ratio = sequence.printable_ratio()
            quality = "text" if ratio >= threshold else f"binary ({ratio:.2f})"
            preview = sequence.preview(limit=preview_limit)
            lines.append(
                (
                    f"segment {sequence.segment_index:03d} "
                    f"block 0x{sequence.start_block:06X} "
                    f"len={sequence.total_length} bytes => {quality}: {preview}"
                )
            )
        return lines

def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
