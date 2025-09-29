"""High level Lua reconstruction leveraging manual annotations and stack heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ir import IRBlock, IRInstruction, IRProgram
from .knowledge import KnowledgeBase
from .lua_ast import (
    Assignment,
    BlankLine,
    BlockStatement,
    BinaryExpr,
    BreakStatement,
    CallExpr,
    CallStatement,
    CommentStatement,
    GenericForStatement,
    IfClause,
    IfStatement,
    LuaExpression,
    LiteralExpr,
    MethodCallExpr,
    MultiAssignment,
    NumericForStatement,
    NameExpr,
    LuaStatement,
    RepeatStatement,
    ReturnStatement,
    SwitchCase,
    SwitchStatement,
    TableExpr,
    TableField,
    UnaryExpr,
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
from .naming import NameAllocator, derive_stack_symbol_name, sanitize_identifier
from .vm_analysis import LuaLiteralFormatter, estimate_stack_io


@dataclass
class StackVariable:
    """Book-keeping entry describing a symbolic stack value."""

    name: str
    origin: str
    source: Optional[LuaExpression] = None
    comment: Optional[str] = None
    semantics: Optional[InstructionSemantics] = None

    @property
    def name_expr(self) -> NameExpr:
        return NameExpr(self.name)


@dataclass
class StackParameter:
    """Description of a function parameter inferred from stack usage."""

    name: str
    index: int
    origin: str = "inferred"


class HighLevelStack:
    """Track symbolic stack values during reconstruction."""

    def __init__(self) -> None:
        self._values: List[StackVariable] = []
        self._allocator = NameAllocator()
        self._parameters: List[StackParameter] = []
        self.warnings: List[str] = []
        self._registry: Dict[str, StackVariable] = {}

    # ------------------------------------------------------------------
    def push_literal(
        self,
        expression: LuaExpression,
        *,
        semantics: Optional[InstructionSemantics] = None,
    ) -> Tuple[List[LuaStatement], NameExpr]:
        base = self._literal_base(expression)
        variable = self._create_variable(
            base,
            origin="literal",
            source=expression,
            semantics=semantics,
        )
        statement = Assignment([variable.name_expr], expression, is_local=True)
        self._values.append(variable)
        return [statement], variable.name_expr

    def push_expression(
        self,
        expression: LuaExpression,
        *,
        prefix: str = "tmp",
        make_local: bool = False,
        base_name: Optional[str] = None,
        origin: str = "expression",
        semantics: Optional[InstructionSemantics] = None,
    ) -> Tuple[List[LuaStatement], NameExpr]:
        if not make_local and isinstance(expression, NameExpr):
            alias = StackVariable(name=expression.name, origin="alias")
            self._values.append(alias)
            return [], expression
        base = base_name or prefix
        variable = self._create_variable(
            base,
            origin=origin,
            source=expression,
            semantics=semantics,
        )
        statement = Assignment([variable.name_expr], expression, is_local=True)
        self._values.append(variable)
        return [statement], variable.name_expr

    def push_call_results(
        self,
        expression: CallExpr,
        outputs: int,
        *,
        base_name: Optional[str] = None,
        origin: str = "call",
        semantics: Optional[InstructionSemantics] = None,
    ) -> Tuple[List[LuaStatement], List[NameExpr]]:
        if outputs <= 0:
            return [], []
        base = base_name or "result"
        if outputs == 1:
            variable = self._create_variable(
                base,
                origin=origin,
                source=expression,
                semantics=semantics,
            )
            statement = Assignment([variable.name_expr], expression, is_local=True)
            self._values.append(variable)
            return [statement], [variable.name_expr]
        variables = [
            self._create_variable(
                base,
                origin=origin,
                source=expression,
                semantics=semantics,
            )
            for _ in range(outputs)
        ]
        targets = [variable.name_expr for variable in variables]
        statement = MultiAssignment(targets, [expression], is_local=True)
        self._values.extend(variables)
        return [statement], targets

    def pop_single(self) -> NameExpr:
        if self._values:
            variable = self._values.pop()
            return variable.name_expr
        parameter = self._declare_parameter()
        return NameExpr(parameter.name)

    def pop_many(self, count: int) -> List[NameExpr]:
        items = [self.pop_single() for _ in range(count)]
        items.reverse()
        return items

    def pop_pair(self) -> Tuple[NameExpr, NameExpr]:
        lhs, rhs = self.pop_many(2)
        return lhs, rhs

    def flush(self) -> List[NameExpr]:
        values = [variable.name_expr for variable in self._values]
        self._values.clear()
        return values

    # ------------------------------------------------------------------
    def parameters(self) -> Sequence[StackParameter]:
        return list(self._parameters)

    def variable_metadata(self) -> Dict[str, StackVariable]:
        return dict(self._registry)

    # Internal helpers -------------------------------------------------
    def _create_variable(
        self,
        base: str,
        *,
        origin: str,
        source: Optional[LuaExpression] = None,
        semantics: Optional[InstructionSemantics] = None,
    ) -> StackVariable:
        name = self._allocator.allocate(base)
        variable = StackVariable(
            name=name,
            origin=origin,
            source=source,
            semantics=semantics,
        )
        self._registry[name] = variable
        return variable

    def _declare_parameter(self) -> StackParameter:
        index = len(self._parameters)
        name = f"arg_{index}"
        parameter = StackParameter(name=name, index=index)
        self._parameters.append(parameter)
        self.warnings.append(f"inferred function argument {name}")
        return parameter

    def _literal_base(self, expression: LuaExpression) -> str:
        if isinstance(expression, LiteralExpr):
            value = expression.value
            if value.startswith('"') and value.endswith('"'):
                inner = _decode_lua_string(value[1:-1])
                if inner:
                    snippet = inner[:8]
                    hint = sanitize_identifier(snippet, "literal")
                    if hint and hint != "literal":
                        return f"str_{hint}" if not hint.startswith("literal") else hint
        return "literal"


def _decode_lua_string(text: str) -> str:
    """Best-effort unescape of a Lua string literal body."""

    result: List[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            index += 1
            escape = text[index]
            if escape == "n":
                result.append("\n")
            elif escape == "r":
                result.append("\r")
            elif escape == "t":
                result.append("\t")
            elif escape == "\\":
                result.append("\\")
            elif escape == '"':
                result.append('"')
            else:
                result.append(escape)
            index += 1
            continue
        result.append(char)
        index += 1
    return "".join(result)


@dataclass
class FunctionMetadata:
    """Descriptive statistics collected while reconstructing a function."""

    block_count: int
    instruction_count: int
    warnings: List[str] = field(default_factory=list)
    helper_calls: int = 0
    branch_count: int = 0
    literal_count: int = 0
    parameter_count: int = 0
    variable_count: int = 0
    string_constants: Dict[str, str] = field(default_factory=dict)
    parameter_usage: Dict[str, int] = field(default_factory=dict)
    helper_breakdown: Dict[str, int] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        lines = ["function summary:"]
        lines.append(f"- blocks: {self.block_count}")
        lines.append(f"- instructions: {self.instruction_count}")
        lines.append(f"- parameters: {self.parameter_count}")
        lines.append(f"- literal instructions: {self.literal_count}")
        lines.append(f"- helper invocations: {self.helper_calls}")
        lines.append(f"- branches: {self.branch_count}")
        lines.append(f"- locals: {self.variable_count}")
        if self.string_constants:
            lines.append(f"- string literals: {len(self.string_constants)}")
        if self.parameter_usage:
            total_reads = sum(self.parameter_usage.values())
            lines.append(f"- parameter reads: {total_reads}")
        return lines

    def warning_lines(self) -> List[str]:
        return [f"- {warning}" for warning in self.warnings]

    def string_literal_lines(self) -> List[str]:
        lines = ["string literals:"]
        for name in sorted(self.string_constants):
            value = self.string_constants[name]
            lines.append(f"- {name}: {value}")
        return lines

    def parameter_usage_lines(self) -> List[str]:
        lines = ["parameter usage:"]
        for name in sorted(self.parameter_usage):
            count = self.parameter_usage[name]
            lines.append(f"- {name}: {count}")
        return lines

    def helper_lines(self) -> List[str]:
        lines = ["helper usage:"]
        for name in sorted(self.helper_breakdown):
            count = self.helper_breakdown[name]
            lines.append(f"- {name}: {count}")
        return lines


@dataclass
class HighLevelFunction:
    """Container describing a reconstructed Lua function."""

    name: str
    parameters: Sequence[str]
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
        if self.metadata.string_constants:
            writer.write_comment_block(self.metadata.string_literal_lines())
            writer.write_line("")
        if self.metadata.parameter_usage:
            writer.write_comment_block(self.metadata.parameter_usage_lines())
            writer.write_line("")
        if self.metadata.helper_breakdown:
            writer.write_comment_block(self.metadata.helper_lines())
            writer.write_line("")
        signature = ", ".join(self.parameters)
        writer.write_line(f"function {self.name}({signature})")
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
            blocks[start] = self._translate_block(block, next_offset)
            self.instruction_total += len(block.instructions)
        return blocks

    # ------------------------------------------------------------------
    def _translate_block(self, block: IRBlock, fallthrough: Optional[int]) -> BlockInfo:
        statements: List[LuaStatement] = []
        terminator: Optional[Terminator] = None
        for index, instruction in enumerate(block.instructions):
            semantics = instruction.semantics
            self._reconstructor._register_enums(semantics)
            is_last = index == len(block.instructions) - 1
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

    # ------------------------------------------------------------------
    def from_ir(self, program: IRProgram) -> HighLevelFunction:
        self._last_summary = None
        translator = BlockTranslator(self, program)
        blocks = translator.translate()
        entry = min(blocks)
        structurer = ControlFlowStructurer(blocks)
        body = structurer.structure(entry)
        function_name = f"segment_{program.segment_index:03d}"
        parameters = [parameter.name for parameter in translator.stack.parameters()]
        variable_metadata = translator.stack.variable_metadata()
        renamer = VariableRenamer(variable_metadata)
        renamer.reserve_names(parameters)
        mapping = renamer.build_mapping()
        warnings = list(translator.stack.warnings)
        if mapping:
            renamer.apply(body, mapping)
            parameters = renamer.rename_parameters(parameters, mapping)
            warnings = renamer.rename_warnings(warnings, mapping)
        usage_analyzer = VariableUsageAnalyzer(parameters)
        usage_report = usage_analyzer.inspect(body)
        helper_invocations = sum(usage_report.helper_calls.values())
        metadata = FunctionMetadata(
            block_count=len(blocks),
            instruction_count=translator.instruction_total,
            warnings=warnings,
            helper_calls=helper_invocations or translator.helper_calls,
            branch_count=translator.branch_count,
            literal_count=translator.literal_count,
            parameter_count=len(parameters),
        )
        metadata.variable_count = usage_report.local_count()
        metadata.string_constants = dict(usage_report.string_literals)
        metadata.parameter_usage = dict(usage_report.parameter_usage)
        metadata.helper_breakdown = dict(usage_report.helper_calls)
        return HighLevelFunction(
            name=function_name,
            parameters=parameters,
            body=body,
            metadata=metadata,
        )

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
    ) -> List[LuaStatement]:
        operand = self._operand_expression(semantics, instruction.operand)
        statements, _ = translator.stack.push_literal(operand, semantics=semantics)
        translator.literal_count += 1
        return self._decorate_with_comment(statements, semantics)

    def _translate_comparison(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        lhs, rhs = translator.stack.pop_pair()
        operator = semantics.comparison_operator or "=="
        expression = BinaryExpr(lhs, operator, rhs)
        base = derive_stack_symbol_name(semantics, fallback="cmp")
        statements, _ = translator.stack.push_expression(
            expression,
            prefix=base,
            base_name=base,
            make_local=True,
            origin="comparison",
        )
        return self._decorate_with_comment(statements, semantics)

    def _translate_branch(
        self,
        block_start: int,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        fallthrough: Optional[int],
    ) -> BranchTerminator:
        condition = translator.stack.pop_single()
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
        values = translator.stack.flush()
        comment = self._comment_formatter.format_inline(semantics.summary or "")
        return ReturnTerminator(values=values, comment=comment)

    def _translate_structure(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        args = translator.stack.pop_many(inputs)
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
            base = derive_stack_symbol_name(semantics, fallback="struct")
            statements, _ = translator.stack.push_call_results(
                call_expr,
                outputs,
                base_name=base,
                origin="structure",
                semantics=semantics,
            )
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
        args = translator.stack.pop_many(inputs)
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
            base = derive_stack_symbol_name(semantics, fallback="result")
            statements, _ = translator.stack.push_call_results(
                call_expr,
                outputs,
                base_name=base,
                semantics=semantics,
            )
        return self._decorate_with_comment(statements, semantics)

    def _translate_generic(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        args = translator.stack.pop_many(inputs)
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
            base = derive_stack_symbol_name(semantics, fallback="result")
            statements, _ = translator.stack.push_call_results(
                call_expr,
                outputs,
                base_name=base,
                semantics=semantics,
            )
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
        parameter_total = sum(func.metadata.parameter_count for func in functions)
        lines.append(f"- inferred parameters: {parameter_total}")
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
        return lines


class VariableRenamer:
    """Best-effort variable renaming based on reconstruction metadata."""

    def __init__(self, variables: Mapping[str, StackVariable]) -> None:
        self._variables = variables
        self._used: set[str] = set()

    def reserve_names(self, names: Iterable[str]) -> None:
        for name in names:
            if not name:
                continue
            self._used.add(sanitize_identifier(name, name))

    def build_mapping(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for name in sorted(self._variables):
            variable = self._variables[name]
            suggestion = self._suggest_name(variable)
            new_name = self._make_unique(suggestion, current=name)
            if new_name != name:
                mapping[name] = new_name
                variable.name = new_name
            self._used.add(new_name)
        return mapping

    def _make_unique(self, base: str, *, current: str) -> str:
        cleaned = sanitize_identifier(base, current or "value")
        candidate = cleaned
        if candidate == current and candidate not in self._used:
            return candidate
        suffix = 1
        while candidate in self._used and candidate != current:
            candidate = f"{cleaned}_{suffix}"
            suffix += 1
        return candidate

    def _suggest_name(self, variable: StackVariable) -> str:
        semantics = variable.semantics
        if semantics and semantics.enum_namespace:
            base = f"{semantics.enum_namespace}_value"
            return sanitize_identifier(base, "value")
        if variable.origin == "literal" and isinstance(variable.source, LiteralExpr):
            text = variable.source.value
            if text.startswith('"') and text.endswith('"'):
                inner = _decode_lua_string(text[1:-1])
                if inner:
                    snippet = sanitize_identifier(inner[:12], "text")
                    return f"text_{snippet}" if snippet else "text_literal"
            cleaned = sanitize_identifier(text, "literal")
            return f"literal_{cleaned}" if cleaned else "literal"
        if semantics:
            if semantics.manual_name:
                return sanitize_identifier(semantics.manual_name, "value")
            if semantics.mnemonic:
                return sanitize_identifier(semantics.mnemonic, "value")
        if variable.origin:
            return sanitize_identifier(variable.origin, "value")
        return sanitize_identifier(variable.name, "value")

    def apply(self, block: BlockStatement, mapping: Mapping[str, str]) -> None:
        if not mapping:
            return
        self._rename_block(block, mapping)

    def rename_parameters(
        self, parameters: Sequence[str], mapping: Mapping[str, str]
    ) -> List[str]:
        return [mapping.get(name, name) for name in parameters]

    def rename_warnings(
        self, warnings: Sequence[str], mapping: Mapping[str, str]
    ) -> List[str]:
        renamed: List[str] = []
        for warning in warnings:
            updated = warning
            for old, new in mapping.items():
                if old in updated:
                    updated = updated.replace(old, new)
            renamed.append(updated)
        return renamed

    def _rename_block(
        self, block: BlockStatement, mapping: Mapping[str, str]
    ) -> None:
        for statement in block.statements:
            self._rename_statement(statement, mapping)

    def _rename_statement(
        self, statement: LuaStatement, mapping: Mapping[str, str]
    ) -> None:
        if isinstance(statement, Assignment):
            for target in statement.targets:
                self._rename_expression(target, mapping)
            self._rename_expression(statement.value, mapping)
        elif isinstance(statement, MultiAssignment):
            for target in statement.targets:
                self._rename_expression(target, mapping)
            for value in statement.values:
                self._rename_expression(value, mapping)
        elif isinstance(statement, CallStatement):
            self._rename_expression(statement.expression, mapping)
        elif isinstance(statement, ReturnStatement):
            for value in statement.values:
                self._rename_expression(value, mapping)
        elif isinstance(statement, IfStatement):
            for clause in statement.clauses:
                if clause.condition is not None:
                    self._rename_expression(clause.condition, mapping)
                self._rename_block(clause.body, mapping)
        elif isinstance(statement, WhileStatement):
            self._rename_expression(statement.condition, mapping)
            self._rename_block(statement.body, mapping)
        elif isinstance(statement, RepeatStatement):
            self._rename_block(statement.body, mapping)
            self._rename_expression(statement.condition, mapping)
        elif isinstance(statement, NumericForStatement):
            self._rename_expression(statement.variable, mapping)
            self._rename_expression(statement.start, mapping)
            self._rename_expression(statement.stop, mapping)
            if statement.step is not None:
                self._rename_expression(statement.step, mapping)
            self._rename_block(statement.body, mapping)
        elif isinstance(statement, GenericForStatement):
            for var in statement.variables:
                self._rename_expression(var, mapping)
            for expr in statement.iterator:
                self._rename_expression(expr, mapping)
            self._rename_block(statement.body, mapping)
        elif isinstance(statement, SwitchStatement):
            self._rename_expression(statement.expression, mapping)
            for case in statement.cases:
                for value in case.values:
                    self._rename_expression(value, mapping)
                self._rename_block(case.body, mapping)
            if statement.default is not None:
                self._rename_block(statement.default, mapping)
        elif isinstance(statement, BlockStatement):
            self._rename_block(statement, mapping)
        else:
            # Comment, blank lines and break statements require no updates.
            return

    def _rename_expression(
        self, expression: LuaExpression, mapping: Mapping[str, str]
    ) -> None:
        if isinstance(expression, NameExpr):
            if expression.name in mapping:
                expression.name = mapping[expression.name]
        elif isinstance(expression, BinaryExpr):
            self._rename_expression(expression.left, mapping)
            self._rename_expression(expression.right, mapping)
        elif isinstance(expression, UnaryExpr):
            self._rename_expression(expression.operand, mapping)
        elif isinstance(expression, CallExpr):
            self._rename_expression(expression.callee, mapping)
            for arg in expression.arguments:
                self._rename_expression(arg, mapping)
        elif isinstance(expression, MethodCallExpr):
            self._rename_expression(expression.target, mapping)
            for arg in expression.arguments:
                self._rename_expression(arg, mapping)
        elif isinstance(expression, TableExpr):
            for field in expression.fields:
                self._rename_table_field(field, mapping)

    def _rename_table_field(
        self, field: TableField, mapping: Mapping[str, str]
    ) -> None:
        if field.key is not None:
            self._rename_expression(field.key, mapping)
        self._rename_expression(field.value, mapping)


@dataclass
class UsageReport:
    local_names: set[str] = field(default_factory=set)
    parameter_usage: Dict[str, int] = field(default_factory=dict)
    string_literals: Dict[str, str] = field(default_factory=dict)
    helper_calls: Dict[str, int] = field(default_factory=dict)

    def local_count(self) -> int:
        return len(self.local_names)


class VariableUsageAnalyzer:
    """Inspect the emitted AST and gather usage statistics."""

    def __init__(
        self,
        parameters: Sequence[str],
        *,
        helper_names: Optional[Iterable[str]] = None,
    ) -> None:
        self._parameters = set(parameters)
        self._helper_names = set(helper_names or []) if helper_names is not None else None

    def inspect(self, block: BlockStatement) -> UsageReport:
        report = UsageReport()
        self._visit_block(block, report)
        return report

    # ------------------------------------------------------------------
    def _visit_block(self, block: BlockStatement, report: UsageReport) -> None:
        for statement in block.statements:
            self._visit_statement(statement, report)

    def _visit_statement(self, statement: LuaStatement, report: UsageReport) -> None:
        if isinstance(statement, Assignment):
            for target in statement.targets:
                if isinstance(target, NameExpr):
                    if (
                        statement.is_local
                        and target.name not in self._parameters
                    ):
                        report.local_names.add(target.name)
                        self._record_literal(target.name, statement.value, report)
            self._visit_expression(statement.value, report)
        elif isinstance(statement, MultiAssignment):
            self._handle_multi_assignment(statement, report)
        elif isinstance(statement, CallStatement):
            self._visit_expression(statement.expression, report)
        elif isinstance(statement, ReturnStatement):
            for value in statement.values:
                self._visit_expression(value, report)
        elif isinstance(statement, IfStatement):
            for clause in statement.clauses:
                if clause.condition is not None:
                    self._visit_expression(clause.condition, report)
                self._visit_block(clause.body, report)
        elif isinstance(statement, WhileStatement):
            self._visit_expression(statement.condition, report)
            self._visit_block(statement.body, report)
        elif isinstance(statement, RepeatStatement):
            self._visit_block(statement.body, report)
            self._visit_expression(statement.condition, report)
        elif isinstance(statement, NumericForStatement):
            if isinstance(statement.variable, NameExpr):
                report.local_names.add(statement.variable.name)
            self._visit_expression(statement.start, report)
            self._visit_expression(statement.stop, report)
            if statement.step is not None:
                self._visit_expression(statement.step, report)
            self._visit_block(statement.body, report)
        elif isinstance(statement, GenericForStatement):
            for variable in statement.variables:
                if isinstance(variable, NameExpr):
                    report.local_names.add(variable.name)
            for iterator in statement.iterator:
                self._visit_expression(iterator, report)
            self._visit_block(statement.body, report)
        elif isinstance(statement, SwitchStatement):
            self._visit_expression(statement.expression, report)
            for case in statement.cases:
                for value in case.values:
                    self._visit_expression(value, report)
                self._visit_block(case.body, report)
            if statement.default is not None:
                self._visit_block(statement.default, report)
        elif isinstance(statement, BlockStatement):
            self._visit_block(statement, report)

    def _handle_multi_assignment(
        self, statement: MultiAssignment, report: UsageReport
    ) -> None:
        values = list(statement.values)
        for index, target in enumerate(statement.targets):
            if isinstance(target, NameExpr):
                if statement.is_local and target.name not in self._parameters:
                    report.local_names.add(target.name)
                if index < len(values):
                    self._record_literal(target.name, values[index], report)
        for value in values:
            self._visit_expression(value, report)

    def _visit_expression(self, expression: LuaExpression, report: UsageReport) -> None:
        if isinstance(expression, NameExpr):
            self._record_name_usage(expression.name, report)
        elif isinstance(expression, BinaryExpr):
            self._visit_expression(expression.left, report)
            self._visit_expression(expression.right, report)
        elif isinstance(expression, UnaryExpr):
            self._visit_expression(expression.operand, report)
        elif isinstance(expression, CallExpr):
            self._record_call(expression, report)
            self._visit_expression(expression.callee, report)
            for argument in expression.arguments:
                self._visit_expression(argument, report)
        elif isinstance(expression, MethodCallExpr):
            self._record_method_call(expression, report)
            self._visit_expression(expression.target, report)
            for argument in expression.arguments:
                self._visit_expression(argument, report)
        elif isinstance(expression, TableExpr):
            for field in expression.fields:
                if field.key is not None:
                    self._visit_expression(field.key, report)
                self._visit_expression(field.value, report)

    def _record_literal(
        self, name: str, value: LuaExpression, report: UsageReport
    ) -> None:
        if name in self._parameters:
            return
        if isinstance(value, LiteralExpr):
            literal = value.value
            if literal.startswith('"') and literal.endswith('"'):
                report.string_literals.setdefault(name, literal)

    def _record_name_usage(self, name: str, report: UsageReport) -> None:
        if name in self._parameters:
            report.parameter_usage[name] = report.parameter_usage.get(name, 0) + 1

    def _record_call(self, call: CallExpr, report: UsageReport) -> None:
        if isinstance(call.callee, NameExpr):
            helper = call.callee.name
            if self._helper_names is None or helper in self._helper_names:
                report.helper_calls[helper] = report.helper_calls.get(helper, 0) + 1

    def _record_method_call(
        self, call: MethodCallExpr, report: UsageReport
    ) -> None:
        target = self._expression_label(call.target)
        helper = f"{target}:{call.method}" if target else call.method
        if self._helper_names is None or helper in self._helper_names:
            report.helper_calls[helper] = report.helper_calls.get(helper, 0) + 1

    def _expression_label(self, expression: LuaExpression) -> str:
        if isinstance(expression, NameExpr):
            return expression.name
        if isinstance(expression, MethodCallExpr):
            return f"{self._expression_label(expression.target)}:{expression.method}"
        return ""

def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
