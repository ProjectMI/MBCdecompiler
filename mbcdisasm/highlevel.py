"""High-level Lua reconstruction leveraging manual annotations and stack heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .ir import IRBlock, IRInstruction, IRProgram
from .knowledge import KnowledgeBase
from .manual_semantics import InstructionSemantics
from .vm_analysis import LuaLiteralFormatter, estimate_stack_io
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


class HighLevelStack:
    """Track symbolic stack values during reconstruction."""

    def __init__(self) -> None:
        self._values: List[str] = []
        self._counter = 0
        self.warnings: List[str] = []

    def new_symbol(self, prefix: str = "value") -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def push_literal(self, expression: str) -> Tuple[str, str]:
        name = self.new_symbol("literal")
        self._values.append(name)
        return f"local {name} = {expression}", name

    def push_result(self, expression: str, prefix: str = "tmp") -> Tuple[str, str]:
        name = self.new_symbol(prefix)
        self._values.append(name)
        return f"local {name} = {expression}", name

    def push_existing(self, name: str) -> None:
        self._values.append(name)

    def allocate_results(self, count: int, prefix: str = "result") -> List[str]:
        names = []
        for _ in range(count):
            name = self.new_symbol(prefix)
            self._values.append(name)
            names.append(name)
        return names

    def pop_single(self) -> str:
        if self._values:
            return self._values.pop()
        placeholder = self.new_symbol("stack")
        self.warnings.append(f"underflow generated placeholder {placeholder}")
        return placeholder

    def pop_many(self, count: int) -> List[str]:
        items = [self.pop_single() for _ in range(count)]
        items.reverse()
        return items

    def pop_pair(self) -> Tuple[str, str]:
        lhs, rhs = self.pop_many(2)
        return lhs, rhs

    def flush(self) -> List[str]:
        values = list(self._values)
        self._values.clear()
        return values


@dataclass
class FunctionMetadata:
    """Descriptive statistics collected while reconstructing a function."""

    block_count: int
    instruction_count: int
    warnings: List[str] = field(default_factory=list)
    helper_calls: int = 0
    branch_count: int = 0
    literal_count: int = 0

    def summary_lines(self) -> List[str]:
        lines = ["function summary:"]
        lines.append(f"- blocks: {self.block_count}")
        lines.append(f"- instructions: {self.instruction_count}")
        lines.append(f"- literal instructions: {self.literal_count}")
        lines.append(f"- helper invocations: {self.helper_calls}")
        lines.append(f"- branches: {self.branch_count}")
        return lines

    def warning_lines(self) -> List[str]:
        return [f"- {warning}" for warning in self.warnings]


@dataclass
class HighLevelFunction:
    """Container describing a reconstructed Lua function."""

    name: str
    statements: List[str]
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
            for stmt in self.statements:
                if not stmt:
                    writer.write_line("")
                    continue
                if stmt.startswith("::") and stmt.endswith("::"):
                    writer.write_label(stmt[2:-2])
                else:
                    writer.write_line(stmt)
        writer.write_line("end")
        return writer.render()


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

    def from_ir(self, program: IRProgram) -> HighLevelFunction:
        stack = HighLevelStack()
        self._last_summary = None
        statements: List[str] = []
        block_order = sorted(program.blocks)
        instruction_total = 0
        stats: Dict[str, int] = {"literals": 0, "helpers": 0, "branches": 0}
        for index, start in enumerate(block_order):
            block = program.blocks[start]
            instruction_total += len(block.instructions)
            next_offset = program.blocks[block_order[index + 1]].start if index + 1 < len(block_order) else None
            if statements and statements[-1] != "":
                statements.append("")
            statements.extend(
                self._emit_block(block, stack, stats=stats, next_offset=next_offset)
            )
        function_name = f"segment_{program.segment_index:03d}"
        metadata = FunctionMetadata(
            block_count=len(block_order),
            instruction_count=instruction_total,
            warnings=list(stack.warnings),
            helper_calls=stats["helpers"],
            branch_count=stats["branches"],
            literal_count=stats["literals"],
        )
        return HighLevelFunction(
            name=function_name,
            statements=statements,
            metadata=metadata,
        )

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

    def _emit_block(
        self,
        block: IRBlock,
        stack: HighLevelStack,
        *,
        stats: Dict[str, int],
        next_offset: Optional[int],
    ) -> List[str]:
        lines: List[str] = [f"::block_{block.start:06X}::"]
        for idx, instruction in enumerate(block.instructions):
            semantics = instruction.semantics
            self._register_enums(semantics)
            is_last = idx == len(block.instructions) - 1
            successors = block.successors if is_last else []
            fallthrough = next_offset if is_last else None
            lines.extend(
                self._emit_instruction(
                    block.start,
                    instruction,
                    semantics,
                    stack,
                    stats,
                    successors=successors,
                    fallthrough=fallthrough,
                )
            )
        return lines

    def _register_enums(self, semantics: InstructionSemantics) -> None:
        if semantics.enum_namespace and semantics.enum_values:
            for value, label in semantics.enum_values.items():
                self._enum_registry.register(semantics.enum_namespace, value, label)

    def _emit_instruction(
        self,
        block_start: int,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        stack: HighLevelStack,
        stats: Dict[str, int],
        *,
        successors: Sequence[int],
        fallthrough: Optional[int],
    ) -> List[str]:
        if semantics.has_tag("literal"):
            return self._emit_literal(instruction, semantics, stack, stats)
        if semantics.has_tag("comparison"):
            return self._emit_comparison(instruction, semantics, stack)
        if semantics.control_flow == "branch":
            return self._emit_branch(block_start, instruction, semantics, stack, stats, successors, fallthrough)
        if semantics.control_flow == "return":
            return self._emit_return(instruction, semantics, stack)
        if semantics.has_tag("structure"):
            return self._emit_structure(instruction, semantics, stack, stats)
        if semantics.has_tag("call"):
            return self._emit_call(instruction, semantics, stack, stats)
        return self._emit_generic(instruction, semantics, stack, stats)

    def _emit_literal(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        stack: HighLevelStack,
        stats: Dict[str, int],
    ) -> List[str]:
        operand = self._format_operand(semantics, instruction.operand)
        line, _ = stack.push_literal(operand)
        stats["literals"] += 1
        return self._decorate_with_comment(line, semantics)

    def _emit_comparison(
        self, instruction: IRInstruction, semantics: InstructionSemantics, stack: HighLevelStack
    ) -> List[str]:
        lhs, rhs = stack.pop_pair()
        operator = semantics.comparison_operator or "=="
        expression = f"({lhs} {operator} {rhs})"
        line, name = stack.push_result(expression, prefix="cmp")
        return self._decorate_with_comment(line, semantics)

    def _emit_branch(
        self,
        block_start: int,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        stack: HighLevelStack,
        stats: Dict[str, int],
        successors: Sequence[int],
        fallthrough: Optional[int],
    ) -> List[str]:
        condition = stack.pop_single()
        true_target, false_target = self._select_branch_targets(block_start, successors, fallthrough)
        comment_lines = self._comment_lines(semantics)
        stats["branches"] += 1
        if true_target is not None and true_target <= block_start:
            lines = list(comment_lines)
            lines.append(f"while {condition} do")
            body = f"  goto {self._format_block_label(true_target)}"
            lines.append(body)
            lines.append("end")
            if false_target is not None and false_target != true_target:
                lines.append(f"-- fallthrough to {self._format_block_label(false_target)}")
            return lines

        lines = list(comment_lines)
        header = f"if {condition} then"
        if comment_lines:
            lines.append(header)
        else:
            inline = self._comment_formatter.format_inline(semantics.summary or "")
            if inline and inline == (semantics.summary or "") and len(header) + len(inline) + 4 <= 100:
                header += f"  -- {inline}"
            lines.append(header)
        if true_target is not None:
            lines.append(f"  goto {self._format_block_label(true_target)}")
        else:
            lines.append("  -- missing branch target")
        if false_target is not None:
            lines.append("else")
            lines.append(f"  goto {self._format_block_label(false_target)}")
        lines.append("end")
        return lines

    def _emit_return(
        self, instruction: IRInstruction, semantics: InstructionSemantics, stack: HighLevelStack
    ) -> List[str]:
        values = stack.flush()
        if values:
            line = "return " + ", ".join(values)
        else:
            line = "return"
        return self._decorate_with_comment(line, semantics)

    def _emit_structure(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        stack: HighLevelStack,
        stats: Dict[str, int],
    ) -> List[str]:
        inputs, outputs = estimate_stack_io(semantics)
        args = stack.pop_many(inputs)
        if semantics.uses_operand:
            args.append(self._format_operand(semantics, instruction.operand))
        method = _snake_to_camel(semantics.mnemonic)
        target = semantics.struct_context or "struct"
        call = f"{target}:{method}({', '.join(args)})" if args else f"{target}:{method}()"
        signature = MethodSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
            struct=target,
            method=method,
        )
        self._helper_registry.register_method(signature)
        stats["helpers"] += 1
        if outputs > 0:
            line, _ = stack.push_result(call, prefix="struct")
        else:
            line = call
        return self._decorate_with_comment(line, semantics)

    def _emit_call(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        stack: HighLevelStack,
        stats: Dict[str, int],
    ) -> List[str]:
        inputs, outputs = estimate_stack_io(semantics)
        args = stack.pop_many(inputs)
        if semantics.uses_operand:
            args.append(self._format_operand(semantics, instruction.operand))
        call = f"{semantics.mnemonic}({', '.join(args)})" if args else f"{semantics.mnemonic}()"
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        stats["helpers"] += 1
        if outputs <= 0:
            return self._decorate_with_comment(call, semantics)
        if outputs == 1:
            line, name = stack.push_result(call, prefix="result")
            return self._decorate_with_comment(line, semantics)
        names = stack.allocate_results(outputs, prefix="result")
        assignment = f"local {', '.join(names)} = {call}"
        return self._decorate_with_comment(assignment, semantics)

    def _emit_generic(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        stack: HighLevelStack,
        stats: Dict[str, int],
    ) -> List[str]:
        inputs, outputs = estimate_stack_io(semantics)
        args = stack.pop_many(inputs)
        if semantics.uses_operand:
            args.append(self._format_operand(semantics, instruction.operand))
        invocation = f"{semantics.mnemonic}({', '.join(args)})" if args else f"{semantics.mnemonic}()"
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        stats["helpers"] += 1
        if outputs > 0:
            line, _ = stack.push_result(invocation)
        else:
            line = invocation
        return self._decorate_with_comment(line, semantics)

    def _select_branch_targets(
        self,
        block_start: int,
        successors: Sequence[int],
        fallthrough: Optional[int],
    ) -> Tuple[Optional[int], Optional[int]]:
        succ = list(successors)
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

    def _format_operand(self, semantics: InstructionSemantics, operand: int) -> str:
        if semantics.enum_values and operand in semantics.enum_values:
            label = semantics.enum_values[operand]
            if semantics.enum_namespace:
                return f"{semantics.enum_namespace}.{label}"
            return label
        return self._literal_formatter.format_operand(operand)

    @staticmethod
    def _format_block_label(offset: int) -> str:
        return f"block_{offset:06X}"

    def _comment_lines(self, semantics: InstructionSemantics) -> List[str]:
        summary = semantics.summary or ""
        if not self._should_emit_comment(summary):
            return []
        return [f"-- {line}" for line in self._comment_formatter.wrap(summary)]

    def _decorate_with_comment(
        self, line: str, semantics: InstructionSemantics
    ) -> List[str]:
        summary = semantics.summary or ""
        if not self._should_emit_comment(summary):
            return [line]
        inline = self._comment_formatter.format_inline(summary)
        if (
            inline
            and inline == summary
            and len(line) + len(inline) + 4 <= self.options.max_inline_comment
        ):
            return [f"{line}  -- {inline}"]
        lines = [f"-- {wrapped}" for wrapped in self._comment_formatter.wrap(summary)]
        lines.append(line)
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
        return lines

def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
