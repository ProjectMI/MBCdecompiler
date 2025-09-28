"""High-level Lua reconstruction leveraging manual annotations and stack heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .ast import LuaLiteralFormatter
from .ir import IRBlock, IRInstruction, IRProgram
from .knowledge import KnowledgeBase


@dataclass
class InstructionSemantics:
    """Normalised manual context describing an instruction."""

    key: str
    mnemonic: str
    summary: Optional[str]
    control_flow: Optional[str]
    stack_delta: Optional[float]
    tags: set[str] = field(default_factory=set)
    enum_values: Dict[int, str] = field(default_factory=dict)
    enum_namespace: Optional[str] = None
    struct_context: Optional[str] = None
    comparison_operator: Optional[str] = None
    stack_inputs: Optional[int] = None
    stack_outputs: Optional[int] = None
    uses_operand: bool = True


class ManualSemanticsDatabase:
    """Adapter that exposes rich manual annotations for reconstruction."""

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge
        self._cache: Dict[str, InstructionSemantics] = {}
        self.enum_registry: Dict[str, Dict[int, str]] = {}

    def semantics_for(self, instruction: IRInstruction) -> InstructionSemantics:
        cached = self._cache.get(instruction.key)
        if cached is not None:
            return cached

        metadata = self.knowledge.instruction_metadata(instruction.key)
        manual = self.knowledge.manual_annotation(instruction.key)

        semantics = InstructionSemantics(
            key=instruction.key,
            mnemonic=metadata.mnemonic,
            summary=metadata.summary,
            control_flow=metadata.control_flow,
            stack_delta=metadata.stack_delta,
        )

        raw_tags = manual.get("tags") if manual else None
        if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes)):
            semantics.tags.update(str(tag).lower() for tag in raw_tags)
        category = manual.get("category") if manual else None
        if category:
            semantics.tags.add(str(category).lower())

        name_lower = semantics.mnemonic.lower()
        summary_lower = metadata.summary.lower() if metadata.summary else ""

        if name_lower.startswith("push_literal"):
            semantics.tags.add("literal")
        if "compare" in name_lower or "cmp" in name_lower or "compare" in summary_lower:
            semantics.tags.add("comparison")
        if semantics.control_flow == "branch" or "branch" in name_lower:
            semantics.tags.add("branch")
        if semantics.control_flow == "return":
            semantics.tags.add("return")
        if semantics.control_flow == "call":
            semantics.tags.add("call")
        if "structured" in name_lower or manual.get("structure") or manual.get("struct"):
            semantics.tags.add("structure")
        if "loop" in name_lower:
            semantics.tags.add("loop")

        semantics.struct_context = _string_field(manual, "struct") or _string_field(
            manual, "structure"
        )
        semantics.enum_namespace = _string_field(manual, "enum_namespace") or _string_field(
            manual, "enum_type"
        )
        semantics.stack_inputs = _int_field(manual, "stack_inputs")
        semantics.stack_outputs = _int_field(manual, "stack_outputs")

        operator_hint = _string_field(manual, "comparison_operator")
        if operator_hint:
            semantics.comparison_operator = operator_hint
        else:
            semantics.comparison_operator = self._deduce_operator(name_lower)

        enum_payload = _mapping_field(manual, "enum_values") or _mapping_field(
            manual, "enums"
        )
        if enum_payload:
            parsed = {
                _parse_int(key): str(value)
                for key, value in enum_payload.items()
                if _parse_int(key) is not None
            }
            semantics.enum_values = parsed
            namespace = semantics.enum_namespace or _sanitize_namespace(semantics.mnemonic)
            registry = self.enum_registry.setdefault(namespace, {})
            registry.update(parsed)
            semantics.enum_namespace = namespace

        operand_usage = manual.get("uses_operand") if manual else None
        if operand_usage is not None:
            semantics.uses_operand = bool(operand_usage)

        self._cache[instruction.key] = semantics
        return semantics

    @staticmethod
    def _deduce_operator(name_lower: str) -> Optional[str]:
        candidates = [
            ("not_equal", "~="),
            ("not_eq", "~="),
            ("ne", "~="),
            ("less_equal", "<="),
            ("le", "<="),
            ("greater_equal", ">="),
            ("ge", ">="),
            ("greater", ">"),
            ("gt", ">"),
            ("less", "<"),
            ("lt", "<"),
            ("equal", "=="),
            ("eq", "=="),
        ]
        for needle, operator in candidates:
            if needle in name_lower:
                return operator
        return None


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
        self._semantics = ManualSemanticsDatabase(knowledge)
        self._literal_formatter = LuaLiteralFormatter()

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
        for namespace, values in sorted(self._semantics.enum_registry.items()):
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
            semantics = self._semantics.semantics_for(instruction)
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
        if "literal" in semantics.tags:
            return [self._emit_literal(instruction, semantics, stack)]
        if "comparison" in semantics.tags:
            return [self._emit_comparison(instruction, semantics, stack)]
        if semantics.control_flow == "branch":
            return self._emit_branch(block_start, instruction, semantics, stack, successors, fallthrough)
        if semantics.control_flow == "return":
            return [self._emit_return(instruction, semantics, stack)]
        if "structure" in semantics.tags:
            return [self._emit_structure(instruction, semantics, stack)]
        if "call" in semantics.tags:
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
        inputs, outputs = self._estimate_stack_io(semantics)
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
        inputs, outputs = self._estimate_stack_io(semantics)
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
        inputs, outputs = self._estimate_stack_io(semantics)
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

    def _estimate_stack_io(self, semantics: InstructionSemantics) -> Tuple[int, int]:
        inputs = semantics.stack_inputs if semantics.stack_inputs is not None else 0
        outputs = semantics.stack_outputs if semantics.stack_outputs is not None else 0
        delta = semantics.stack_delta
        if "comparison" in semantics.tags:
            inputs = max(inputs, 2)
            outputs = max(outputs, 1)
        elif semantics.control_flow == "branch":
            inputs = max(inputs, 1)
        elif semantics.control_flow == "return":
            inputs = max(inputs, 1)
        else:
            if delta is not None:
                rounded = int(round(delta))
                if rounded < 0:
                    inputs = max(inputs, abs(rounded))
                elif rounded > 0:
                    outputs = max(outputs, rounded)
        return inputs, outputs

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


def _parse_int(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def _string_field(data: Optional[Mapping[str, object]], key: str) -> Optional[str]:
    if not data:
        return None
    value = data.get(key)
    if value is None:
        return None
    return str(value)


def _int_field(data: Optional[Mapping[str, object]], key: str) -> Optional[int]:
    if not data:
        return None
    value = data.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(str(value), 0)
        except ValueError:
            return None


def _mapping_field(data: Optional[Mapping[str, object]], key: str) -> Optional[Mapping[str, object]]:
    if not data:
        return None
    value = data.get(key)
    if isinstance(value, Mapping):
        return value
    return None


def _sanitize_namespace(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "N_" + cleaned
    return cleaned or "Enum"


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
