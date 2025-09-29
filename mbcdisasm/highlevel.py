"""High level Lua reconstruction leveraging manual annotations and stack heuristics."""

from __future__ import annotations

import re

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    StringClassifier,
    StringInsight,
    summarise_insights,
    token_histogram,
)
from .vm_analysis import estimate_stack_io


class HighLevelStack:
    """Track symbolic stack values during reconstruction."""

    def __init__(self) -> None:
        self._values: List[NameExpr] = []
        self._counter = 0
        self._used_names: Set[str] = set()
        self.warnings: List[str] = []
        self._string_registry: Dict[str, Tuple[StringLiteralSequence, int]] = {}
        self._sequence_consumers: Dict[int, List[str]] = {}

    def new_symbol(self, prefix: str = "value") -> str:
        candidate = self._reserve_candidate(prefix)
        self._used_names.add(candidate)
        return candidate

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
            self._register_existing_name(expression.name)
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

    # ------------------------------------------------------------------
    def rename_symbol(self, target: NameExpr, suggested: str) -> Optional[str]:
        """Rename *target* to a readable identifier derived from ``suggested``.

        The method returns the new symbol name when renaming succeeds.  When
        ``suggested`` cannot be converted into a valid Lua identifier the
        original name is preserved and ``None`` is returned.
        """

        candidate = _sanitize_identifier(suggested)
        if not candidate:
            return None
        candidate = _to_snake_case(candidate)
        if not candidate:
            return None
        if candidate in _LUA_KEYWORDS:
            candidate = f"{candidate}_value"
        unique = self._ensure_unique(candidate)
        if unique == target.name:
            return unique
        if target.name in self._used_names:
            self._used_names.remove(target.name)
        self._used_names.add(unique)
        target.name = unique
        return unique

    # ------------------------------------------------------------------
    def _ensure_unique(self, base: str) -> str:
        if base not in self._used_names:
            return base
        index = 1
        while True:
            candidate = f"{base}_{index}"
            if candidate not in self._used_names:
                return candidate
            index += 1

    def _register_existing_name(self, name: str) -> None:
        if not name:
            return
        self._used_names.add(name)

    def _reserve_candidate(self, prefix: str) -> str:
        while True:
            name = f"{prefix}_{self._counter}"
            self._counter += 1
            if name not in self._used_names:
                return name

    # ------------------------------------------------------------------
    def register_string_sequence(self, sequence: StringLiteralSequence) -> None:
        self._sequence_consumers.setdefault(id(sequence), [])
        for index, name in enumerate(sequence.symbol_names):
            self._string_registry[name] = (sequence, index)

    def match_sequence_suffix(
        self, arguments: Sequence[LuaExpression]
    ) -> Optional[StringLiteralSequence]:
        candidate: Optional[StringLiteralSequence] = None
        collected: List[int] = []
        seen: Set[int] = set()
        started = False
        for expression in reversed(arguments):
            if isinstance(expression, NameExpr):
                entry = self._string_registry.get(expression.name)
                if entry is None:
                    if started:
                        break
                    return None
                sequence, index = entry
                if candidate is None:
                    candidate = sequence
                elif candidate is not sequence:
                    break
                if index in seen:
                    break
                seen.add(index)
                collected.append(index)
                started = True
            else:
                if started:
                    break
                continue
        if candidate is None:
            return None
        expected = list(range(len(candidate.symbol_names)))
        if not expected:
            return None
        if sorted(collected) != expected:
            return None
        return candidate

    def note_sequence_consumer(
        self, sequence: StringLiteralSequence, consumer: str
    ) -> None:
        consumer = consumer.strip()
        if not consumer:
            return
        self._sequence_consumers.setdefault(id(sequence), []).append(consumer)

    def sequence_consumers(self, sequence: StringLiteralSequence) -> Tuple[str, ...]:
        return tuple(self._sequence_consumers.get(id(sequence), ()))

    def rename_call_results(
        self,
        targets: Sequence[NameExpr],
        sequence: Optional[StringLiteralSequence],
        semantics: InstructionSemantics,
    ) -> None:
        if not targets:
            return
        if sequence is None:
            return
        base_name = sequence.base_name
        if not base_name and sequence.insight and sequence.insight.tokens:
            candidate = _sanitize_identifier(sequence.insight.tokens[0])
            base_name = _to_snake_case(candidate) if candidate else None
        if not base_name:
            return
        base_name = _to_snake_case(base_name) or base_name
        if len(targets) == 1:
            suffix = self._result_suffix(semantics)
            candidate = f"{base_name}_{suffix}" if suffix else base_name
            self.rename_symbol(targets[0], candidate)
        else:
            for index, target in enumerate(targets):
                candidate = f"{base_name}_result_{index}"
                self.rename_symbol(target, candidate)

    def _result_suffix(self, semantics: InstructionSemantics) -> str:
        manual = (semantics.manual_name or semantics.mnemonic or "").lower()
        if any(keyword in manual for keyword in ("count", "number", "total")):
            return "count"
        if any(keyword in manual for keyword in ("flag", "status", "state")):
            return "status"
        if any(keyword in manual for keyword in ("id", "identifier")):
            return "id"
        if any(keyword in manual for keyword in ("name", "label", "string")):
            return "value"
        return "result"


@dataclass(frozen=True)
class StringLiteralSequence:
    """Metadata describing a detected string literal run inside a block."""

    text: str
    offsets: Tuple[int, ...]
    symbol_names: Tuple[str, ...] = field(default_factory=tuple)
    base_name: Optional[str] = None
    insight: Optional[StringInsight] = None
    comment: Optional[CommentStatement] = None
    consumers: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def start_offset(self) -> int:
        return self.offsets[0]

    @property
    def end_offset(self) -> int:
        return self.offsets[-1]

    def chunk_count(self) -> int:
        return len(self.offsets)

    def length(self) -> int:
        return len(self.text)

    def preview(self, limit: int = 80) -> str:
        if len(self.text) <= limit:
            return self.text
        if limit <= 3:
            return "..."
        return self.text[: limit - 3] + "..."

    def annotation_strings(self) -> List[str]:
        annotations: List[str] = []
        if self.base_name:
            annotations.append(f"name={self.base_name}")
        if self.symbol_names:
            annotations.append("locals=" + ", ".join(self.symbol_names))
        if self.insight:
            annotations.append(f"type={self.insight.classification}")
            if self.insight.tokens:
                sample = ", ".join(self.insight.tokens[:4])
                annotations.append(f"tokens=[{sample}]")
            if self.insight.entropy:
                annotations.append(f"entropy={self.insight.entropy:.2f}")
            if self.insight.case_style and self.insight.case_style != "neutral":
                annotations.append(f"case={self.insight.case_style}")
            if self.insight.token_density:
                annotations.append(f"density={self.insight.token_density:.3f}")
            if self.insight.printable_ratio < 1.0:
                annotations.append(f"printable={self.insight.printable_ratio:.2f}")
        if self.consumers:
            preview = ", ".join(self.consumers[:3])
            if len(self.consumers) > 3:
                preview += ", …"
            annotations.append(f"used_by=[{preview}]")
        return annotations

    def refresh_comment(self) -> None:
        if not self.comment:
            return
        preview = escape_lua_string(self.preview())
        annotations = self.annotation_strings()
        suffix = f" ({'; '.join(annotations)})" if annotations else ""
        self.comment.text = (
            f"string literal sequence: {preview} (len={self.length()}){suffix}"
        )

    def with_consumers(self, consumers: Sequence[str]) -> "StringLiteralSequence":
        if tuple(consumers) == self.consumers:
            return self
        updated = replace(self, consumers=tuple(consumers))
        updated.refresh_comment()
        return updated


class StringLiteralCollector:
    """Group consecutive string literal assignments into annotated sequences."""

    @dataclass
    class _Chunk:
        offset: int
        text: str
        statements: List[LuaStatement]
        target: NameExpr

    def __init__(
        self,
        stack: HighLevelStack,
        classifier: Optional[StringClassifier] = None,
    ) -> None:
        self._stack = stack
        self._pending: List[StringLiteralCollector._Chunk] = []
        self._sequences: List[StringLiteralSequence] = []
        self._classifier = classifier or StringClassifier()

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
            StringLiteralCollector._Chunk(
                offset=offset, text=literal_text, statements=statements, target=target
            )
        )
        return []

    def flush(self) -> List[LuaStatement]:
        if not self._pending:
            return []
        combined = "".join(chunk.text for chunk in self._pending)
        offsets = tuple(chunk.offset for chunk in self._pending)
        insight = self._classifier.classify(combined)
        base_name = self._derive_base_name(combined, insight)
        symbol_names = self._assign_names(base_name, insight)
        comment = CommentStatement("")
        sequence = StringLiteralSequence(
            text=combined,
            offsets=offsets,
            symbol_names=symbol_names,
            base_name=base_name,
            insight=insight,
            comment=comment,
        )
        sequence.refresh_comment()
        self._sequences.append(sequence)
        self._stack.register_string_sequence(sequence)
        result: List[LuaStatement] = [comment]
        for chunk in self._pending:
            result.extend(chunk.statements)
        self.reset()
        return result

    def finalize(self) -> List[LuaStatement]:
        return self.flush()

    def drain_sequences(self) -> List[StringLiteralSequence]:
        if not self._sequences:
            return []
        sequences = list(self._sequences)
        self._sequences.clear()
        enriched: List[StringLiteralSequence] = []
        for sequence in sequences:
            consumers = self._stack.sequence_consumers(sequence)
            if consumers:
                sequence = sequence.with_consumers(consumers)
            else:
                sequence.refresh_comment()
            enriched.append(sequence)
        return enriched

    def _assign_names(
        self, base_name: Optional[str], insight: Optional[StringInsight]
    ) -> Tuple[str, ...]:
        if not self._pending:
            return tuple()
        assigned: List[str] = []
        tokens = list(insight.tokens) if insight else []
        if base_name:
            for index, chunk in enumerate(self._pending):
                suffix = "" if len(self._pending) == 1 else f"_{index}"
                candidate = f"{base_name}{suffix}"
                renamed = self._stack.rename_symbol(chunk.target, candidate)
                assigned.append(renamed or chunk.target.name)
        else:
            for index, chunk in enumerate(self._pending):
                fallback = self._select_fallback_name(tokens, index, chunk)
                renamed = self._stack.rename_symbol(chunk.target, fallback)
                assigned.append(renamed or chunk.target.name)
        return tuple(assigned)

    def _derive_base_name(
        self, text: str, insight: Optional[StringInsight]
    ) -> Optional[str]:
        if not text:
            return None
        trimmed = text.strip()
        candidates: List[str] = []
        if insight and insight.tokens:
            for token in insight.tokens:
                sanitized = _sanitize_identifier(token)
                if sanitized and len(sanitized) >= 3:
                    candidates.append(sanitized)
        if trimmed:
            sanitized = _sanitize_identifier(trimmed)
            if sanitized and len(sanitized) >= 3:
                candidates.append(sanitized)
        if not insight or not insight.tokens:
            for match in _IDENTIFIER_PATTERN.finditer(text):
                token = match.group(0)
                sanitized = _sanitize_identifier(token)
                if not sanitized:
                    continue
                if len(sanitized) < 3:
                    continue
                candidates.append(sanitized)
        if not candidates:
            return None
        unique: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            lowered = candidate.lower()
            if lowered in seen or lowered in _STRING_NAME_STOPWORDS:
                continue
            seen.add(lowered)
            unique.append(candidate)
        if not unique:
            return None
        unique.sort(key=self._score_candidate)
        best = _to_snake_case(unique[0])
        if not best:
            return None
        if best in _LUA_KEYWORDS:
            best = f"{best}_str"
        if len(best) > 48:
            best = best[:48].rstrip("_")
        return best

    def _select_fallback_name(
        self, tokens: Sequence[str], index: int, chunk: "StringLiteralCollector._Chunk"
    ) -> str:
        if tokens:
            token = tokens[min(index, len(tokens) - 1)]
            sanitized = _sanitize_identifier(token) or chunk.target.name
            sanitized = _to_snake_case(sanitized) or sanitized
            return sanitized
        fallback = _sanitize_identifier(chunk.text.strip()) or chunk.target.name
        fallback = _to_snake_case(fallback) or fallback
        return fallback

    @staticmethod
    def _score_candidate(candidate: str) -> Tuple[int, int, int]:
        length_penalty = -len(candidate)
        digit_penalty = sum(ch.isdigit() for ch in candidate)
        underscore_penalty = candidate.count("_")
        return (underscore_penalty, digit_penalty, length_penalty)

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
            text = escape_lua_string(sequence.preview(preview))
            detail = (
                "- 0x"
                f"{sequence.start_offset:06X}"
                f" len={sequence.length()}"
                f" chunks={sequence.chunk_count()}: {text}"
            )
            annotations = []
            for item in sequence.annotation_strings():
                if item.startswith("locals="):
                    payload = item[len("locals="):]
                    annotations.append(f"symbols=[{payload}]")
                else:
                    annotations.append(item)
            if annotations:
                detail += " (" + ", ".join(annotations) + ")"
            lines.append(detail)
        remaining = len(self.string_sequences) - limit
        if remaining > 0:
            lines.append(f"- ... ({remaining} additional sequences)")
        return lines

    def classification_lines(self) -> List[str]:
        insights = [
            sequence.insight for sequence in self.string_sequences if sequence.insight
        ]
        if not insights:
            return []
        _, histogram = summarise_insights(insights)
        if not histogram:
            return []
        lines = ["string classification summary:"]
        for name, count in histogram:
            lines.append(f"- {name}: {count}")
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
        classification_lines = self.metadata.classification_lines()
        if classification_lines:
            writer.write_comment_block(classification_lines)
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
        self._string_classifier = StringClassifier()
        self._string_collector = StringLiteralCollector(
            self._stack, classifier=self._string_classifier
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
                literal_statements, operand_expr, target = self._reconstructor._translate_literal(
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
        if self.options.emit_string_catalog:
            catalog_lines = self._string_catalog_lines(functions)
            if catalog_lines:
                writer = LuaWriter()
                writer.write_comment_block(catalog_lines)
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
        sequence = translator.stack.match_sequence_suffix(args)
        if sequence:
            translator.stack.note_sequence_consumer(
                sequence, self._format_sequence_consumer(instruction, semantics)
            )
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
            statements, targets = translator.stack.push_call_results(
                call_expr, outputs, prefix="struct"
            )
            translator.stack.rename_call_results(targets, sequence, semantics)
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
        sequence = translator.stack.match_sequence_suffix(args)
        if sequence:
            translator.stack.note_sequence_consumer(
                sequence, self._format_sequence_consumer(instruction, semantics)
            )
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
            statements, targets = translator.stack.push_call_results(call_expr, outputs)
            translator.stack.rename_call_results(targets, sequence, semantics)
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
        sequence = translator.stack.match_sequence_suffix(args)
        if sequence:
            translator.stack.note_sequence_consumer(
                sequence, self._format_sequence_consumer(instruction, semantics)
            )
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
            statements, targets = translator.stack.push_call_results(call_expr, outputs)
            translator.stack.rename_call_results(targets, sequence, semantics)
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

    def _format_sequence_consumer(
        self, instruction: IRInstruction, semantics: InstructionSemantics
    ) -> str:
        label = (semantics.manual_name or semantics.mnemonic or "call").strip()
        if not label:
            label = semantics.mnemonic or "call"
        return f"{label}@0x{instruction.offset:06X}"

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
            insight_values = [
                sequence.insight
                for func in functions
                for sequence in func.metadata.string_sequences
                if sequence.insight is not None
            ]
            if insight_values:
                _, histogram = summarise_insights([insight for insight in insight_values if insight])
                if histogram:
                    preview = ", ".join(
                        f"{name}={count}" for name, count in histogram[:4]
                    )
                    lines.append(f"- string categories: {preview}")
        return lines

    def string_insight_report(
        self, functions: Sequence[HighLevelFunction]
    ) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        insights: List[StringInsight] = []
        for function in functions:
            for sequence in function.metadata.string_sequences:
                entry: Dict[str, Any] = {
                    "function": function.name,
                    "offset": sequence.start_offset,
                    "end_offset": sequence.end_offset,
                    "length": sequence.length(),
                    "chunks": sequence.chunk_count(),
                    "base_name": sequence.base_name,
                    "symbol_names": list(sequence.symbol_names),
                    "consumers": list(sequence.consumers),
                    "text": sequence.text,
                }
                if sequence.insight:
                    insights.append(sequence.insight)
                    entry.update(
                        {
                            "classification": sequence.insight.classification,
                            "tokens": list(sequence.insight.tokens),
                            "entropy": sequence.insight.entropy,
                            "case_style": sequence.insight.case_style,
                            "printable_ratio": sequence.insight.printable_ratio,
                            "token_density": sequence.insight.token_density,
                            "hints": list(sequence.insight.hints),
                        }
                    )
                entries.append(entry)
        total, histogram = summarise_insights(insights)
        entropy_total = 0.0
        for insight in insights:
            entropy_total += insight.entropy
        average_entropy = round(entropy_total / total, 3) if total else 0.0
        top_tokens = token_histogram(insights, limit=10)
        return {
            "functions": [function.name for function in functions],
            "entry_count": len(entries),
            "classifications": histogram,
            "summary": {
                "total_sequences": total,
                "average_entropy": average_entropy,
                "top_tokens": top_tokens,
            },
            "entries": entries,
        }

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
        raw_candidates: List[Tuple[StringLiteralSequence, str, int]] = []
        frequency: Counter[str] = Counter()
        for sequence in sequences:
            if sequence.base_name:
                lowered = sequence.base_name.lower()
                raw_candidates.append((sequence, sequence.base_name, sequence.start_offset))
                frequency[lowered] += 1
            text = sequence.text.strip()
            if text and not any(ch.isspace() for ch in text):
                sanitized = _sanitize_identifier(text)
                if sanitized and sanitized.lower() not in _STRING_NAME_STOPWORDS:
                    key = sanitized.lower()
                    raw_candidates.append((sequence, sanitized, sequence.start_offset))
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
                insight=sequence.insight,
            )
            candidates.append((score, sanitized))
            seen.add(sanitized)
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        selected = candidates[0][1]
        snake = _to_snake_case(selected)
        if snake:
            selected = snake
        if selected.lower() in _LUA_KEYWORDS:
            selected = f"{selected}_fn"
        return selected

    def _score_name_candidate(
        self,
        sequence: StringLiteralSequence,
        sanitized: str,
        entry_offset: int,
        *,
        count: int = 1,
        candidate_offset: Optional[int] = None,
        insight: Optional[StringInsight] = None,
    ) -> Tuple[int, int, int, int, int, int, str]:
        offset = candidate_offset if candidate_offset is not None else sequence.start_offset
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
        classification_penalty = 0
        if insight:
            if insight.classification == "identifier":
                classification_penalty = -1
            elif insight.classification in {"keyword", "numeric"}:
                classification_penalty = -1
            elif insight.classification in {"sentence", "dialogue"}:
                classification_penalty = 1
        tie_breaker = sanitized.lower()
        return (
            -count,
            underscore_penalty,
            distance,
            case_penalty,
            digit_penalty,
            length_penalty,
            classification_penalty,
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

    def _string_catalog_lines(
        self, functions: Sequence[HighLevelFunction]
    ) -> List[str]:
        catalog: Dict[str, List[Tuple[HighLevelFunction, StringLiteralSequence]]] = {}
        for function in functions:
            for sequence in function.metadata.string_sequences:
                key = self._catalog_key(sequence)
                catalog.setdefault(key, []).append((function, sequence))
        if not catalog:
            return []
        lines: List[str] = ["string catalog:"]
        for key in sorted(catalog):
            group = catalog[key]
            representative = group[0][1]
            insight = representative.insight
            classification = insight.classification if insight else "unknown"
            tokens = ", ".join(insight.tokens[:4]) if insight and insight.tokens else ""
            entropy = insight.entropy if insight else 0.0
            case_style = insight.case_style if insight else "neutral"
            density = insight.token_density if insight else 0.0
            printable_ratio = insight.printable_ratio if insight else 1.0
            preview = escape_lua_string(representative.preview(64))
            header = (
                f"- {key}: len={representative.length()} "
                f"chunks={representative.chunk_count()} "
                f"type={classification} entropy={entropy:.2f} preview={preview}"
            )
            if tokens:
                header += f" tokens=[{tokens}]"
            if case_style and case_style != "neutral":
                header += f" case={case_style}"
            if density:
                header += f" density={density:.3f}"
            if printable_ratio < 1.0:
                header += f" printable={printable_ratio:.2f}"
            if representative.consumers:
                consumer_preview = ", ".join(representative.consumers[:3])
                if len(representative.consumers) > 3:
                    consumer_preview += ", …"
                header += f" used_by=[{consumer_preview}]"
            lines.append(header)
            for function, sequence in group:
                symbol_hint = ", ".join(sequence.symbol_names) if sequence.symbol_names else ""
                hint = f" symbols=[{symbol_hint}]" if symbol_hint else ""
                lines.append(
                    "  * "
                    f"{function.name} @0x{sequence.start_offset:06X}"
                    f" -> 0x{sequence.end_offset:06X}" + hint
                )
        return lines

    def _catalog_key(self, sequence: StringLiteralSequence) -> str:
        if sequence.base_name:
            return sequence.base_name
        if sequence.symbol_names:
            return sequence.symbol_names[0]
        return f"literal_{sequence.start_offset:06X}"

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
    sequence: StringLiteralSequence,
) -> Iterable[Tuple[str, int]]:
    text = sequence.text
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


def _to_snake_case(name: str) -> str:
    if not name:
        return ""
    result: List[str] = []
    previous_lower = False
    for char in name:
        if char.isalnum() or char == "_":
            if char.isupper() and previous_lower:
                result.append("_")
            result.append(char.lower())
            previous_lower = char.islower() or char.isdigit()
        else:
            if result and result[-1] != "_":
                result.append("_")
            previous_lower = False
    sanitized = "".join(result).strip("_")
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized
