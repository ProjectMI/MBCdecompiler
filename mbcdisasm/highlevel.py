"""High level Lua reconstruction leveraging manual annotations and stack heuristics."""

from __future__ import annotations

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
from .lua_literals import LuaLiteral, LuaLiteralFormatter, escape_lua_string
from .string_analysis import (
    NameCandidate,
    StringSequenceProfile,
    chunk_name_hints,
    extract_identifier_tokens,
    sanitize_identifier,
    score_candidate,
    sequence_base_candidates,
)
from .vm_analysis import estimate_stack_io


class HighLevelStack:
    """Track symbolic stack values during reconstruction."""

    def __init__(self) -> None:
        self._values: List[NameExpr] = []
        self._counter = 0
        self._used: Set[str] = set()
        self.warnings: List[str] = []

    def _ensure_unique(self, candidate: str) -> str:
        if candidate not in self._used:
            self._used.add(candidate)
            return candidate
        index = 1
        base = candidate
        while True:
            candidate = f"{base}_{index}"
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate
            index += 1

    def new_symbol(self, prefix: str = "value", *, hint: Optional[str] = None) -> str:
        if hint:
            return self._ensure_unique(hint)
        while True:
            name = f"{prefix}_{self._counter}"
            self._counter += 1
            if name not in self._used:
                self._used.add(name)
                return name

    def rename(self, target: NameExpr, candidate: str) -> str:
        if not candidate:
            return target.name
        if candidate == target.name:
            self._used.add(candidate)
            return candidate
        normalized = candidate
        if normalized in self._used:
            index = 1
            base = normalized
            while True:
                normalized = f"{base}_{index}"
                if normalized not in self._used:
                    break
                index += 1
        if target.name in self._used:
            self._used.discard(target.name)
        target.name = normalized
        self._used.add(normalized)
        return normalized

    def push_literal(
        self, expression: LuaExpression, *, prefix: Optional[str] = None
    ) -> Tuple[List[LuaStatement], NameExpr]:
        name = self.new_symbol(prefix or "literal")
        target = NameExpr(name)
        statement = Assignment([target], expression, is_local=True)
        self._values.append(target)
        return [statement], target

    def push_expression(
        self,
        expression: LuaExpression,
        *,
        prefix: str = "tmp",
        make_local: bool = False,
    ) -> Tuple[List[LuaStatement], NameExpr]:
        if make_local:
            name = self.new_symbol(prefix)
            target = NameExpr(name)
            statement = Assignment([target], expression, is_local=True)
            self._values.append(target)
            return [statement], target
        if isinstance(expression, NameExpr):
            self._used.add(expression.name)
            self._values.append(expression)
            return [], expression
        name = self.new_symbol(prefix)
        target = NameExpr(name)
        statement = Assignment([target], expression, is_local=True)
        self._values.append(target)
        return [statement], target

    def push_call_results(
        self, expression: CallExpr, outputs: int, prefix: str = "result"
    ) -> Tuple[List[LuaStatement], List[NameExpr]]:
        if outputs <= 0:
            return [], []
        if outputs == 1:
            name = self.new_symbol(prefix)
            target = NameExpr(name)
            stmt = Assignment([target], expression, is_local=True)
            self._values.append(target)
            return [stmt], [target]
        targets = [NameExpr(self.new_symbol(prefix)) for _ in range(outputs)]
        stmt = MultiAssignment(targets, [expression], is_local=True)
        for target in targets:
            self._values.append(target)
        return [stmt], targets

    def pop_single(self) -> NameExpr:
        if self._values:
            return self._values.pop()
        placeholder = NameExpr(self.new_symbol("stack"))
        self.warnings.append(
            f"underflow generated placeholder {placeholder.name}"
        )
        return placeholder

    def pop_many(self, count: int) -> List[NameExpr]:
        items = [self.pop_single() for _ in range(count)]
        items.reverse()
        return items

    def pop_pair(self) -> Tuple[NameExpr, NameExpr]:
        lhs, rhs = self.pop_many(2)
        return lhs, rhs

    def flush(self) -> List[NameExpr]:
        values = list(self._values)
        self._values.clear()
        return values


@dataclass(frozen=True)
class StringLiteralChunk:
    offset: int
    text: str
    name: str
    renamed: bool


@dataclass(frozen=True)
class StringLiteralSequence:
    """Metadata describing a detected string literal run inside a block."""

    profile: StringSequenceProfile
    chunks: Tuple[StringLiteralChunk, ...]

    @property
    def text(self) -> str:
        return self.profile.text

    @property
    def offsets(self) -> Tuple[int, ...]:
        return self.profile.offsets

    @property
    def start_offset(self) -> int:
        return self.profile.start_offset

    @property
    def end_offset(self) -> int:
        return self.profile.end_offset

    def chunk_count(self) -> int:
        return len(self.chunks)

    def length(self) -> int:
        return len(self.profile.text)

    def preview(self, limit: int = 80) -> str:
        return self.profile.preview(limit)

    def chunk_names(self) -> Tuple[str, ...]:
        return tuple(chunk.name for chunk in self.chunks)

    def renamed_names(self) -> Tuple[str, ...]:
        return tuple(chunk.name for chunk in self.chunks if chunk.renamed)

    def render_comment(self, *, preview: int = 96) -> str:
        length = len(self.profile.text)
        pieces = [f"string literal sequence len={length}", f"chunks={self.chunk_count()}"]
        renamed = self.renamed_names()
        if renamed:
            summary = list(renamed[:3])
            if len(renamed) > 3:
                summary.append("...")
            pieces.append(f"locals={', '.join(summary)}")
        preview_text = escape_lua_string(self.preview(preview))
        return f"{' '.join(pieces)}: {preview_text}"


@dataclass
class _PendingChunk:
    offset: int
    statements: List[LuaStatement]
    text: str
    target: NameExpr


class StringLiteralCollector:
    """Group consecutive string literal assignments into annotated sequences."""

    def __init__(self, stack: HighLevelStack, *, stopwords: Sequence[str]) -> None:
        self._stack = stack
        self._stopwords = tuple(stopwords)
        self._pending: List[_PendingChunk] = []
        self._sequences: List[StringLiteralSequence] = []

    def reset(self) -> None:
        self._pending.clear()

    def enqueue(
        self,
        offset: int,
        statements: List[LuaStatement],
        expression: LuaExpression,
        target: NameExpr,
    ) -> List[LuaStatement]:
        literal_text = self._string_value(expression)
        if literal_text is None:
            flushed = self.flush()
            return flushed + statements
        self._pending.append(
            _PendingChunk(offset=offset, statements=statements, text=literal_text, target=target)
        )
        return []

    def flush(self) -> List[LuaStatement]:
        if not self._pending:
            return []
        offsets = [chunk.offset for chunk in self._pending]
        fragments = [chunk.text for chunk in self._pending]
        combined = "".join(fragments)
        profile = StringSequenceProfile(
            text=combined,
            offsets=tuple(offsets),
            tokens=tuple(extract_identifier_tokens(combined, offsets)),
        )
        hints = chunk_name_hints(profile, fragments, stopwords=self._stopwords)
        chunks: List[StringLiteralChunk] = []
        statements: List[LuaStatement] = []
        for index, chunk in enumerate(self._pending):
            original_name = chunk.target.name
            hint = hints[index] if index < len(hints) else None
            new_name = original_name
            renamed = False
            if hint:
                candidate = hint
                if candidate.lower() in _LUA_KEYWORDS:
                    candidate = f"{candidate}_str"
                new_name = self._stack.rename(chunk.target, candidate)
                renamed = new_name != original_name
            chunks.append(
                StringLiteralChunk(
                    offset=chunk.offset,
                    text=chunk.text,
                    name=new_name,
                    renamed=renamed,
                )
            )
            statements.extend(chunk.statements)
        sequence = StringLiteralSequence(
            profile=profile,
            chunks=tuple(chunks),
        )
        comment = sequence.render_comment()
        self._sequences.append(sequence)
        self.reset()
        return [CommentStatement(comment), *statements]

    def finalize(self) -> List[LuaStatement]:
        return self.flush()

    def drain_sequences(self) -> List[StringLiteralSequence]:
        if not self._sequences:
            return []
        sequences = list(self._sequences)
        self._sequences.clear()
        return sequences

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
    string_sequences: Sequence[StringLiteralSequence] = field(default_factory=tuple)

    def summary_lines(self) -> List[str]:
        lines = ["function summary:"]
        lines.append(f"- blocks: {self.block_count}")
        lines.append(f"- instructions: {self.instruction_count}")
        lines.append(f"- literal instructions: {self.literal_count}")
        lines.append(f"- helper invocations: {self.helper_calls}")
        lines.append(f"- branches: {self.branch_count}")
        if self.string_sequences:
            lines.append(
                f"- string literal sequences: {len(self.string_sequences)}"
            )
        return lines

    def warning_lines(self) -> List[str]:
        return [f"- {warning}" for warning in self.warnings]

    def string_lines(
        self, *, limit: int = 8, preview: int = 72
    ) -> List[str]:
        if not self.string_sequences:
            return []
        lines: List[str] = []
        for sequence in list(self.string_sequences)[:limit]:
            comment = sequence.render_comment(preview=preview)
            lines.append(f"- {comment}")
        remaining = len(self.string_sequences) - limit
        if remaining > 0:
            lines.append(f"- ... ({remaining} additional sequences)")
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
        string_lines = self.metadata.string_lines()
        if string_lines:
            writer.write_comment_block(["string literal sequences:"] + string_lines)
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
        self._string_collector = StringLiteralCollector(
            self._stack, stopwords=_STRING_NAME_STOPWORDS
        )
        self._collected_sequences: List[StringLiteralSequence] = []
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
        self._string_collector.reset()
        for index, instruction in enumerate(block.instructions):
            semantics = instruction.semantics
            self._reconstructor._register_enums(semantics)
            is_last = index == len(block.instructions) - 1
            if semantics.has_tag("literal"):
                operand_expr = self._reconstructor._operand_expression(
                    semantics, instruction.operand
                )
                (
                    literal_statements,
                    operand_expr,
                    target,
                ) = self._reconstructor._translate_literal(
                    instruction,
                    semantics,
                    self,
                    operand_expr=operand_expr,
                )
                queued = self._string_collector.enqueue(
                    instruction.offset, literal_statements, operand_expr, target
                )
                statements.extend(queued)
                continue
            statements.extend(self._string_collector.flush())
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
        statements.extend(self._string_collector.finalize())
        self._collected_sequences.extend(self._string_collector.drain_sequences())
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
    def string_sequences(self) -> List[StringLiteralSequence]:
        return list(self._collected_sequences)


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
        string_sequences = sorted(
            translator.string_sequences, key=lambda seq: (seq.start_offset, seq.chunk_count())
        )
        base_name = self._derive_function_name(program, string_sequences)
        function_name = self._unique_function_name(base_name)
        metadata = FunctionMetadata(
            block_count=len(blocks),
            instruction_count=translator.instruction_total,
            warnings=list(translator.stack.warnings),
            helper_calls=translator.helper_calls,
            branch_count=translator.branch_count,
            literal_count=translator.literal_count,
            string_sequences=tuple(string_sequences),
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
    ) -> Tuple[List[LuaStatement], LuaExpression, NameExpr]:
        operand = operand_expr or self._operand_expression(semantics, instruction.operand)
        prefix: Optional[str] = None
        if isinstance(operand, LiteralExpr):
            if operand.kind == "string":
                prefix = "string"
            elif operand.kind == "number":
                prefix = "literal"
        elif isinstance(operand, NameExpr):
            prefix = "enum"
        statements, target = translator.stack.push_literal(operand, prefix=prefix)
        translator.literal_count += 1
        decorated = self._decorate_with_comment(statements, semantics)
        return decorated, operand, target

    def _translate_comparison(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        lhs, rhs = translator.stack.pop_pair()
        operator = semantics.comparison_operator or "=="
        expression = BinaryExpr(lhs, operator, rhs)
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
            statements, _ = translator.stack.push_call_results(call_expr, outputs)
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
        string_total = sum(len(func.metadata.string_sequences) for func in functions)
        if string_total:
            lines.append(f"- string literal sequences: {string_total}")
        return lines

    def _derive_function_name(
        self,
        program: IRProgram,
        sequences: Sequence[StringLiteralSequence],
    ) -> str:
        candidate = self._select_string_name(program, sequences)
        if candidate:
            return candidate
        return f"segment_{program.segment_index:03d}"

    def _select_string_name(
        self,
        program: IRProgram,
        sequences: Sequence[StringLiteralSequence],
    ) -> Optional[str]:
        if not sequences:
            return None
        entry_offset = min(program.blocks) if program.blocks else 0
        aggregated: List[NameCandidate] = []
        frequency: Counter[str] = Counter()
        for sequence in sequences:
            profile = sequence.profile
            seq_candidates = sequence_base_candidates(
                profile.text,
                profile.offsets,
                stopwords=_STRING_NAME_STOPWORDS,
            )
            if not seq_candidates:
                stripped = profile.text.strip()
                if stripped and " " not in stripped:
                    sanitized = sanitize_identifier(stripped)
                    if sanitized and sanitized.lower() not in _STRING_NAME_STOPWORDS:
                        seq_candidates = [
                            NameCandidate(
                                identifier=sanitized,
                                normalized=sanitized.lower(),
                                score=score_candidate(
                                    sanitized,
                                    entry_offset=entry_offset,
                                    candidate_offset=sequence.start_offset,
                                    count=1,
                                ),
                                source_offset=sequence.start_offset,
                                chunk_index=0,
                            )
                        ]
            for candidate in seq_candidates:
                frequency[candidate.normalized] += 1
                aggregated.append(candidate)
        if not aggregated:
            return None
        best_name: Optional[str] = None
        best_score: Optional[Tuple[int, int, int, int, int, int, str]] = None
        seen: set[str] = set()
        for candidate in aggregated:
            identifier = candidate.identifier
            if identifier in seen:
                continue
            count = frequency[candidate.normalized]
            score = score_candidate(
                identifier,
                entry_offset=entry_offset,
                candidate_offset=candidate.source_offset,
                count=count,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_name = identifier
            seen.add(identifier)
        if best_name is None:
            return None
        if best_name.lower() in _LUA_KEYWORDS:
            best_name = f"{best_name}_fn"
        return best_name

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


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
