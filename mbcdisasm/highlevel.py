"""High-level Lua reconstruction leveraging manual annotations and stack heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ir import IRBlock, IRInstruction, IRProgram
from .knowledge import KnowledgeBase
from .manual_semantics import InstructionSemantics
from .vm_analysis import LuaLiteralFormatter, estimate_stack_io


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
class HighLevelFunction:
    """Container describing a reconstructed Lua function."""

    name: str
    statements: List[str]

    def render(self) -> str:
        lines = [f"function {self.name}()"]
        for stmt in self.statements:
            if stmt:
                lines.append("  " + stmt)
            else:
                lines.append("")
        lines.append("end")
        return "\n".join(lines) + "\n"


class HighLevelReconstructor:
    """Best-effort reconstruction of high-level Lua from IR programs."""

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge
        self._literal_formatter = LuaLiteralFormatter()
        self._enum_registry: Dict[str, Dict[int, str]] = {}

    def from_ir(self, program: IRProgram) -> HighLevelFunction:
        stack = HighLevelStack()
        statements: List[str] = []
        block_order = sorted(program.blocks)
        for index, start in enumerate(block_order):
            block = program.blocks[start]
            statements.append(f"-- begin block 0x{start:06X}")
            next_offset = program.blocks[block_order[index + 1]].start if index + 1 < len(block_order) else None
            statements.extend(
                self._emit_block(block, stack, next_offset=next_offset)
            )
            statements.append(f"-- end block 0x{start:06X}")
            statements.append("")
        function_name = f"segment_{program.segment_index:03d}"
        return HighLevelFunction(name=function_name, statements=statements[:-1])

    def render(self, functions: Sequence[HighLevelFunction]) -> str:
        lines: List[str] = []
        for namespace, values in sorted(self._enum_registry.items()):
            lines.append(f"local {namespace} = {{")
            for value, label in sorted(values.items()):
                lines.append(f"  [{value}] = \"{label}\",")
            lines.append("}\n")
        for function in functions:
            lines.append(function.render().rstrip())
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _emit_block(
        self,
        block: IRBlock,
        stack: HighLevelStack,
        *,
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
                    successors=successors,
                    fallthrough=fallthrough,
                )
            )
        return lines

    def _register_enums(self, semantics: InstructionSemantics) -> None:
        if semantics.enum_namespace and semantics.enum_values:
            registry = self._enum_registry.setdefault(semantics.enum_namespace, {})
            registry.update(semantics.enum_values)

    def _emit_instruction(
        self,
        block_start: int,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        stack: HighLevelStack,
        *,
        successors: Sequence[int],
        fallthrough: Optional[int],
    ) -> List[str]:
        if semantics.has_tag("literal"):
            return [self._emit_literal(instruction, semantics, stack)]
        if semantics.has_tag("comparison"):
            return [self._emit_comparison(instruction, semantics, stack)]
        if semantics.control_flow == "branch":
            return self._emit_branch(block_start, instruction, semantics, stack, successors, fallthrough)
        if semantics.control_flow == "return":
            return [self._emit_return(instruction, semantics, stack)]
        if semantics.has_tag("structure"):
            return [self._emit_structure(instruction, semantics, stack)]
        if semantics.has_tag("call"):
            return self._emit_call(instruction, semantics, stack)
        return [self._emit_generic(instruction, semantics, stack)]

    def _emit_literal(
        self, instruction: IRInstruction, semantics: InstructionSemantics, stack: HighLevelStack
    ) -> str:
        operand = self._format_operand(semantics, instruction.operand)
        line, _ = stack.push_literal(operand)
        comment = self._format_comment(semantics)
        if comment:
            line += f"  -- {comment}"
        return line

    def _emit_comparison(
        self, instruction: IRInstruction, semantics: InstructionSemantics, stack: HighLevelStack
    ) -> str:
        lhs, rhs = stack.pop_pair()
        operator = semantics.comparison_operator or "=="
        expression = f"({lhs} {operator} {rhs})"
        line, name = stack.push_result(expression, prefix="cmp")
        comment = self._format_comment(semantics)
        if comment:
            line += f"  -- {comment}"
        return line

    def _emit_branch(
        self,
        block_start: int,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        stack: HighLevelStack,
        successors: Sequence[int],
        fallthrough: Optional[int],
    ) -> List[str]:
        condition = stack.pop_single()
        true_target, false_target = self._select_branch_targets(block_start, successors, fallthrough)
        comment = self._format_comment(semantics)
        if true_target is not None and true_target <= block_start:
            lines = [f"while {condition} do"]
            body = f"  goto {self._format_block_label(true_target)}"
            if comment:
                body += f"  -- {comment}"
            lines.append(body)
            lines.append("end")
            if false_target is not None and false_target != true_target:
                lines.append(f"-- fallthrough to {self._format_block_label(false_target)}")
            return lines

        header = f"if {condition} then"
        if comment:
            header += f"  -- {comment}"
        lines = [header]
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
    ) -> str:
        values = stack.flush()
        comment = self._format_comment(semantics)
        if values:
            line = "return " + ", ".join(values)
        else:
            line = "return"
        if comment:
            line += f"  -- {comment}"
        return line

    def _emit_structure(
        self, instruction: IRInstruction, semantics: InstructionSemantics, stack: HighLevelStack
    ) -> str:
        inputs, outputs = estimate_stack_io(semantics)
        args = stack.pop_many(inputs)
        if semantics.uses_operand:
            args.append(self._format_operand(semantics, instruction.operand))
        method = _snake_to_camel(semantics.mnemonic)
        target = semantics.struct_context or "struct"
        call = f"{target}:{method}({', '.join(args)})" if args else f"{target}:{method}()"
        if outputs > 0:
            line, _ = stack.push_result(call, prefix="struct")
        else:
            line = call
        comment = self._format_comment(semantics)
        if comment:
            line += f"  -- {comment}"
        return line

    def _emit_call(
        self, instruction: IRInstruction, semantics: InstructionSemantics, stack: HighLevelStack
    ) -> List[str]:
        inputs, outputs = estimate_stack_io(semantics)
        args = stack.pop_many(inputs)
        if semantics.uses_operand:
            args.append(self._format_operand(semantics, instruction.operand))
        call = f"{semantics.mnemonic}({', '.join(args)})" if args else f"{semantics.mnemonic}()"
        comment = self._format_comment(semantics)
        if outputs <= 0:
            line = call
            if comment:
                line += f"  -- {comment}"
            return [line]
        if outputs == 1:
            line, name = stack.push_result(call, prefix="result")
            if comment:
                line += f"  -- {comment}"
            return [line]
        names = []
        assignments = []
        for index in range(outputs):
            line, name = stack.push_result("", prefix=f"result{index}")
            assignments.append(name)
            names.append(name)
        stack.push_existing(names[-1])
        assignment = f"local {', '.join(names)} = {call}"
        if comment:
            assignment += f"  -- {comment}"
        return [assignment]

    def _emit_generic(
        self, instruction: IRInstruction, semantics: InstructionSemantics, stack: HighLevelStack
    ) -> str:
        inputs, outputs = estimate_stack_io(semantics)
        args = stack.pop_many(inputs)
        if semantics.uses_operand:
            args.append(self._format_operand(semantics, instruction.operand))
        invocation = f"{semantics.mnemonic}({', '.join(args)})" if args else f"{semantics.mnemonic}()"
        if outputs > 0:
            line, _ = stack.push_result(invocation)
        else:
            line = invocation
        comment = self._format_comment(semantics)
        if comment:
            line += f"  -- {comment}"
        return line

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

    @staticmethod
    def _format_comment(semantics: InstructionSemantics) -> str:
        return semantics.summary or ""

def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
