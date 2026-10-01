"""Generate native module classes from recovered, typed source functions.

The backend consumes source_ir.Function, never bytecode instructions. Module
storage is native C++ members; shared algorithms are ordinary C++ templates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
import struct
from typing import Any

from .native_families import Families, Family, Implementation, function_signature
from .native_project import DYNAMIC, VOID, Field, Module, cpp_type, identifier, infer_expression
from .source_ir import Block, Expression, Function, Statement


def quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=True)


def escaped(data: bytes) -> str:
    parts = []
    for value in data:
        if value == 34:
            parts.append('\\"')
        elif value == 92:
            parts.append('\\\\')
        elif 32 <= value < 127:
            parts.append(chr(value))
        else:
            parts.append(f'\\{value:03o}')
    return '"' + ''.join(parts) + '"'


def escaped_chunks(data: bytes, limit: int = 2048) -> list[tuple[int, bytes]]:
    chunks: list[tuple[int, bytes]] = []
    start = 0
    encoded = 0
    for index, value in enumerate(data):
        width = 2 if value in (34, 92) else 1 if 32 <= value < 127 else 4
        if index > start and encoded + width > limit:
            chunks.append((start, data[start:index]))
            start = index
            encoded = 0
        encoded += width
    if start < len(data):
        chunks.append((start, data[start:]))
    return chunks


def initial_data_definition(ctype: str, name: str, data: bytes) -> str:
    literal = escaped(data)
    if len(literal) <= 4096:
        return f"inline const {ctype} {name} = initialMember<{ctype}>({literal});"

    nonzero = [(index, value) for index, value in enumerate(data) if value]
    sparse_cost = len(nonzero) * 32
    if sparse_cost < len(literal) // 2:
        lines = [f"inline const {ctype} {name} = []", "{", f"    {ctype} value{{}};",
                 "    auto *bytes = reinterpret_cast<std::uint8_t *>(std::addressof(value));"]
        lines.extend(f"    bytes[{index}] = {value};" for index, value in nonzero)
    else:
        lines = [f"inline const {ctype} {name} = []", "{", f"    {ctype} value{{}};",
                 "    auto *bytes = reinterpret_cast<std::uint8_t *>(std::addressof(value));"]
        for offset, chunk in escaped_chunks(data):
            lines.append(f"    std::memcpy(bytes + {offset}, {escaped(chunk)}, {len(chunk)});")
    lines += ["    return value;", "}();"]
    return "\n".join(lines)


def result_type(function: Function) -> str:
    value = cpp_type(function.return_type)
    return f"Task<{value}>" if function.asynchronous else value


def parameters(function: Function, *, named: bool = True) -> str:
    return ", ".join(cpp_type(arg["type"]) + (f" parameter{index + 1}" if named else "") for index, arg in enumerate(function.arguments))


def coerce(value: str, target: int | None, actual: int | None, *, parameter: bool = False) -> str:
    if target == actual and target is not None:
        return value
    operation = "argumentValue" if parameter else "storedValue"
    return f"{operation}<{cpp_type(target)}>({value})"


@dataclass
class Helper:
    name: str
    example: Implementation
    instances: list[Implementation]
    literal_parameters: list[list[int]] = field(default_factory=list)
    literal_types: list[int] = field(default_factory=list)


class Renderer:
    def __init__(self, generator: CppGenerator, helper: Helper):
        self.generator = generator
        self.helper = helper
        self.implementation = helper.example
        self.module = self.implementation.module
        self.function = self.implementation.function
        self.locals = {name: f"value{index + 1}" for index, name in enumerate(self.function.variables)}
        self.literal_names = {}
        for index, positions in enumerate(helper.literal_parameters):
            for position in positions:
                self.literal_names[id(self.implementation.pattern.literals[position])] = f"setting{index + 1}"
        self.used_engine = False
        self.sequence = 0

    def typ(self, node: Expression) -> int | None:
        return infer_expression(node, self.function, self.module, self.generator.public_types)

    def binding(self, node: Expression):
        if node.kind != "name":
            raise ValueError(f"Storage is not a named native member: {node}")
        found = self.module.binding(self.function, str(node.value))
        if not found:
            raise ValueError(f"Unresolved native member {node.value} in {self.module.name}.{self.function.name}")
        return found

    def member_expression(self, member: Field) -> str:
        return f"self.{member.name}"

    def member_read(self, member: Field, offset: int, typ: int | None, width: int | None = None) -> str:
        name = self.member_expression(member)
        physical = 0 if width == 1 else typ
        if member.scalar and offset == 0 and member.type_id == physical:
            value = name
        elif member.type_id == physical and physical is not None and offset % member.element_size == 0:
            value = f"{name}[{offset // member.element_size}]"
        else:
            value = f"memberView<{cpp_type(physical)}>({name}, {offset})"
        if width == 1 or physical == 0:
            return f"std::int32_t({value})"
        return value

    def member_write(self, member: Field, offset: int, typ: int | None, value: str) -> str:
        name = self.member_expression(member)
        if member.scalar and offset == 0 and member.type_id == typ:
            return "" if name == value else f"{name} = {value};"
        if member.type_id == typ and typ is not None and offset % member.element_size == 0:
            target = f"{name}[{offset // member.element_size}]"
            return "" if target == value else f"{target} = {value};"
        return f"setMemberView({name}, {offset}, {value});"

    def native_address(self, member: Field, offset: int = 0, width: int | None = None) -> str:
        width = member.size - offset if width is None else width
        tail = "" if offset == 0 and width == member.size else f", {offset}, {width}"
        return f"self.address({self.member_expression(member)}{tail})"

    def access(self, node: Expression) -> str:
        if node.kind == "located":
            return self.expr(node.children[0])
        if node.kind == "storage":
            member, binding = self.binding(node.children[0])
            return self.native_address(member, binding["offset"] - member.offset, node.value[1])
        self.used_engine = True
        if node.kind == "element":
            _, stride, count, absolute, checked, span, _ = node.value
            base = self.expr(node.children[0])
            index = self.expr(node.children[1])
            base = f"pointer({base})"
            index = index if self.typ(node.children[1]) == 16 else f"integer({index})"
            return f"engine.element({base}, {index}, {stride}, {count}, {str(absolute).lower()}, {span}, {str(absolute or checked).lower()})"
        if node.kind == "field":
            _, offset, width = node.value
            base = self.expr(node.children[0])
            base = f"pointer({base})"
            return f"engine.field({base}, {offset}, {width})"
        if node.kind == "indirect":
            return f"engine.indirect(payload({self.expr(node.children[0])}))"
        raise ValueError(f"Unrecognized native address: {node}")

    @staticmethod
    def access_width(node: Expression) -> int:
        kind, value = (node.value if node.kind == "located" else (node.kind, node.value))
        return value[-1] if kind in {"storage", "element", "field", "indirect"} else 4

    def load(self, node: Expression) -> str:
        access = node.children[0]
        typ = self.typ(node)
        width = self.access_width(access)
        if node.value:
            address = self.access(access)
            return f"storedValue<{cpp_type(typ)}>({address})" if typ not in {48, None, DYNAMIC} else address
        if access.kind == "storage":
            member, binding = self.binding(access.children[0])
            return self.member_read(member, binding["offset"] - member.offset, typ, width)
        self.used_engine = True
        physical = 0 if width == 1 else typ
        value = f"engine.read<{cpp_type(physical)}>({self.access(access)})"
        return f"std::int32_t({value})" if width == 1 else value

    def comparison(self, node: Expression, invert: bool = False) -> str:
        left, right = node.children
        typ = 32 if self.typ(left) == 32 else 16
        conversion = "realBits" if typ == 32 else "integerBits"
        operands = [self.expr(child) if self.typ(child) == typ else f"{conversion}({self.expr(child)})" for child in (left, right)]
        operation = str(node.value)
        if invert:
            operation = {"==": "!=", "!=": "==", "<": ">=", ">": "<=", "<=": ">", ">=": "<"}[operation]
            # Negating an ordered float comparison must retain unordered NaNs.
            if typ == 32 and node.value not in {"==", "!="}:
                return f"!({operands[0]} {node.value} {operands[1]})"
        return f"{operands[0]} {operation} {operands[1]}"

    def condition(self, node: Expression, branch_when: bool) -> str:
        if node.kind == "binary" and node.value in {"==", "!=", "<", ">", "<=", ">="}:
            return self.comparison(node, not branch_when)
        value = self.expr(node)
        if self.typ(node) != 16:
            value = f"integer({value})"
        return f"{value} {'!=' if branch_when else '=='} 0"

    def constant_integer(self, node: Expression) -> int | None:
        if id(node) in self.literal_names or self.typ(node) not in {0, 16}:
            return None
        if node.kind == "number":
            text = str(node.value)
            return int(text, 0) if text.lower().startswith("0x") else int(text)
        if node.kind == "prefix" and node.value == "-":
            value = self.constant_integer(node.children[0])
            return None if value is None else ((-value + 2147483648) % 4294967296) - 2147483648
        if node.kind != "binary" or node.value not in {"+", "-", "*", "/", "%"}:
            return None
        left, right = (self.constant_integer(child) for child in node.children)
        if left is None or right is None or (node.value in {"/", "%"} and not right):
            return None
        if node.value in {"/", "%"}:
            quotient = abs(left) // abs(right)
            quotient = -quotient if (left < 0) != (right < 0) else quotient
            value = quotient if node.value == "/" else left - quotient * right
        else:
            value = left + right if node.value == "+" else left - right if node.value == "-" else left * right
        return ((value + 2147483648) % 4294967296) - 2147483648

    def expr(self, node: Expression) -> str:
        if id(node) in self.literal_names:
            return self.literal_names[id(node)]
        kind = node.kind
        if kind in {"binary", "prefix"}:
            constant = self.constant_integer(node)
            if constant is not None:
                return "(-2147483647 - 1)" if constant == -2147483648 else str(constant)
        if kind == "name":
            name = str(node.value)
            if name in self.locals:
                return self.locals[name]
            if name in {"nan", "inf"}:
                return "std::numeric_limits<float>::" + ("quiet_NaN()" if name == "nan" else "infinity()")
            member, binding = self.binding(node)
            offset = binding["offset"] - member.offset
            role = binding.get("role")
            typ = self.typ(node)
            if role in {"span", "const_span", "literal", "array"}:
                width = binding.get("length") or member.size - offset
                if role == "array":
                    width *= binding.get("stride") or member.element_size
                return f"referenceCast<{cpp_type(typ)}>({self.native_address(member, offset, width)})"
            return self.member_read(member, offset, typ)
        if kind == "number":
            text = str(node.value)
            if self.typ(node) == 32:
                value = float(text)
                if math.isfinite(value):
                    result = repr(value)
                    return result + ("f" if "." in result or "e" in result.lower() else ".0f")
                return "std::numeric_limits<float>::quiet_NaN()" if math.isnan(value) else "std::numeric_limits<float>::infinity()"
            value = int(text, 0) if text.lower().startswith("0x") else int(text)
            if -2147483648 <= value <= 2147483647:
                return "(-2147483647 - 1)" if value == -2147483648 else str(value)
            return f"integerBits(std::uint32_t{{{value & 0xffffffff}}})"
        if kind == "stored":
            value = self.expr(node.children[0])
            if node.value == 1:
                return f"storedByte({value})"
            return coerce(value, self.typ(node), self.typ(node.children[0]))
        if kind == "present":
            return f"std::int32_t({self.expr(node.children[0])}.present)"
        if kind == "load":
            return self.load(node)
        if kind in {"storage", "element", "field", "indirect", "located"}:
            return self.access(node)
        if kind == "address":
            return coerce(self.access(node.children[0]), self.typ(node), 48)
        if kind == "by_reference":
            return f"byReference({self.expr(node.children[0])}, {self.access(node.children[1])})"
        if kind == "member":
            match = re.fullmatch(r"span_([\da-fA-F]+)_(\d+)", str(node.value))
            if match:
                self.used_engine = True
                offset, width = int(match[1], 16), int(match[2])
                value = f"engine.field(pointer({self.expr(node.children[0])}), {offset}, {width})"
                return f"storedValue<{cpp_type(self.typ(node))}>({value})"
            raise ValueError(f"Unresolved recovered record member: {node}")
        if kind == "binary":
            left, right = node.children
            lhs, rhs = self.expr(left), self.expr(right)
            operation = str(node.value)
            if operation in {"+", "-", "*", "/", "%"}:
                typ = self.typ(node)
                if typ in {1, 2, 17, 18, 33, 34, 48, 49} and operation in {"+", "-"}:
                    return f"shifted({lhs}, {rhs if operation == '+' else f'negate32({rhs})'})"
                if typ == 32 and operation != "%":
                    return f"SferaNumeric::real32(double(realBits({lhs})) {operation} double(realBits({rhs})))"
                name = {"+": "add32", "-": "subtract32", "*": "multiply32", "/": "divide32", "%": "remainder32"}[operation]
                return f"{name}({lhs}, {rhs})"
            if operation in {"==", "!=", "<", "<=", ">", ">="}:
                return f"std::int32_t({self.comparison(node)})"
            if operation in {"&&", "||"}:
                return f"std::int32_t(truth({lhs}) {operation} truth({rhs}))"
            if operation in {"&", "|", "^", "<<", ">>"}:
                return f"integerBits(word({lhs}) {operation} word({rhs}))"
            raise ValueError(f"Unsupported source binary operator {operation}")
        if kind == "prefix":
            child = self.expr(node.children[0])
            if node.value == "-":
                return f"SferaNumeric::real32(-double(realBits({child})))" if self.typ(node) == 32 else f"negate32({child})"
            if node.value == "+":
                return child
            if node.value == "!":
                return f"std::int32_t(!truth({child}))"
            if node.value == "~":
                return f"integerBits(~word({child}))"
            if node.value == "&" and node.children[0].kind == "name":
                member, binding = self.binding(node.children[0])
                offset = binding["offset"] - member.offset
                width = binding.get("length") or member.size - offset
                return f"storedValue<{cpp_type(self.typ(node))}>({self.native_address(member, offset, width)})"
            raise ValueError(f"Unresolved source unary operator: {node}")
        if kind == "call":
            name = str(node.children[0].value)
            arguments = [self.expr(child) for child in node.children[1:]]
            helpers = {"int": "integer", "int8": "storedByte", "float": "real", "word_as_int": "integerBits",
                       "stored_byte": "storedByte", "real_from_integer_bits": "real", "integer_from_real_bits": "integer"}
            if name in helpers:
                if name == "real_from_integer_bits":
                    arguments = [f"integerBits({arguments[0]})"]
                elif name == "integer_from_real_bits":
                    arguments = [f"realBits({arguments[0]})"]
                return f"{helpers[name]}({', '.join(arguments)})"
            if name == "detached_address":
                return f"storedValue<{cpp_type(self.typ(node))}>(Address{{UINT32_MAX, 1, 1}})"
            raise ValueError(f"Unresolved source helper {name}")
        raise ValueError(f"Unsupported source expression {node}")

    def store(self, access: Expression, value: str, typ: int | None, width: int, actual: int | None = None) -> str:
        physical = 0 if width == 1 else 32 if typ == 32 else 16 if typ is not None and typ >= 0 and typ % 16 == 0 else typ
        converted = coerce(value, physical, actual)
        if access.kind == "storage":
            member, binding = self.binding(access.children[0])
            return self.member_write(member, binding["offset"] - member.offset, physical, converted)
        self.used_engine = True
        return f"engine.write({self.access(access)}, {converted}, {width});"

    def update(self, statement: Statement) -> list[str]:
        node = statement.expression
        assert node is not None
        operation, amount, width = node.value
        access, old = node.children
        typ = self.typ(node)
        pointer_update = "assign_u16" in operation
        subtract = "dec" in operation or "sub" in operation
        postfix = operation.startswith("post") or operation in {"add_assign_u16", "sub_assign_u16"}
        previous = self.expr(old)
        if pointer_update:
            changed = f"shifted(previous, {'-' if subtract else ''}{amount})"
        elif typ == 32:
            changed = f"SferaNumeric::real32(double(realBits(previous)) {'-' if subtract else '+'} 1.0)"
        else:
            changed = f"{'subtract32' if subtract else 'add32'}(previous, 1)"
        # Direct native members do not need an artificial temporary scope.
        # Keep the old sequence for computed addresses and conversion-sensitive
        # results, where evaluation order or the pre-update value is observable.
        if access.kind == "storage" and not statement.result:
            stored = changed.replace("previous", f"({previous})")
            if typ == 32 and operation == "post_dec":
                stored = f"SferaNumeric::real32(double({stored}) - 1.0)"
            return [self.store(access, stored, 16 if pointer_update else typ,
                               4 if pointer_update else width)]
        result = ["{", f"    const auto previous = {previous};", f"    const auto updated = {changed};"]
        stored = "updated"
        if typ == 32 and operation == "post_dec":
            stored = "SferaNumeric::real32(double(updated) - 1.0)"
        result.append("    " + self.store(access, stored, 16 if pointer_update else typ, 4 if pointer_update else width))
        if statement.result:
            returned = "updated" if typ == 32 and operation == "post_dec" else "previous" if postfix else "updated"
            result.append(f"    {self.locals[statement.result]} = {coerce(returned, self.function.variables[statement.result], typ)};")
        result.append("}")
        return result

    def call(self, statement: Statement) -> str:
        node = statement.expression
        assert node is not None
        args = [self.expr(child) for child in node.children]
        if statement.kind == "builtin":
            self.used_engine = True
            command = statement.metadata["subopcode"]
            if command in {20, 21}:
                return f"co_await engine.call(self, {str(command == 21).lower()}{', ' if args else ''}{', '.join(args)})"
            name = self.generator.builtins.get(command)
            if name is None:
                raise ValueError(f"No native engine command for builtin {command}")
            result = cpp_type(self.function.variables[statement.result]) if statement.result else "void"
            return f"engine.invoke<{result}>(Builtin::{name}{', ' if args else ''}{', '.join(args)})"
        target = self.module.by_name.get(str(node.value))
        if target is not None:
            capacity = abs(target.parameter_capacity)
            if len(args) > capacity or (target.parameter_capacity >= 0 and len(args) != capacity):
                return f"invalidArguments<{cpp_type(target.return_type)}>()"
            while len(args) < len(target.arguments):
                args.append("0")
            converted = []
            for index, argument in enumerate(target.arguments):
                actual = self.typ(node.children[index]) if index < len(node.children) else 16
                converted.append(coerce(args[index], argument["type"], actual, parameter=True))
            name = self.generator.families.names[target.name, function_signature(target)]
            result = f"self.{name}({', '.join(converted)})"
            return f"co_await std::move({result}).in(self, {quoted(target.name)})" if target.asynchronous else result
        raise ValueError(f"Unresolved generation-time call: {node.value}")

    def statement(self, statement: Statement) -> list[str]:
        kind = statement.kind
        if kind in {"call", "builtin"}:
            value = self.call(statement)
            if statement.result:
                if kind == "builtin" and statement.metadata["subopcode"] in {20, 21}:
                    value = f"storedValue<{cpp_type(self.function.variables[statement.result])}>({value})"
                return [f"{self.locals[statement.result]} = {value};"]
            return [value + ";"]
        if kind in {"read", "let", "phi", "address", "checked"}:
            assert statement.expression is not None
            value = self.access(statement.expression) if kind == "address" else self.expr(statement.expression)
            if statement.result:
                actual = 48 if kind == "address" else self.typ(statement.expression)
                value = coerce(value, self.function.variables[statement.result], actual)
                return [f"{self.locals[statement.result]} = {value};"]
            return [f"(void){value};"]
        if kind == "increment":
            return self.update(statement)
        if kind == "store":
            node = statement.expression
            assert node is not None
            if node.kind != "assignment":
                raise ValueError(f"Unlowered source assignment {node}")
            typ, width = node.value
            return [self.store(node.children[0], self.expr(node.children[1]), typ, width, self.typ(node.children[1]))]
        if kind == "return":
            keyword = "co_return" if self.function.asynchronous else "return"
            if statement.expression is None:
                if self.function.return_type == DYNAMIC:
                    return [keyword + " Value{};"]
                if self.function.return_type != VOID:
                    raise ValueError(f"Non-void source function has a bare return: {self.module.name}.{self.function.name}")
                return [keyword + ";"]
            value = coerce(self.expr(statement.expression), self.function.return_type, self.typ(statement.expression))
            return [f"{keyword} {value};"]
        if kind == "yield":
            return ["co_await Suspend{};"]
        if kind == "fault":
            return ['throw std::runtime_error(' + quoted(statement.metadata['message']) + ');']
        if kind == "finish":
            return ["co_await Finish{};"]
        if kind == "halt":
            self.used_engine = True
            return ["engine.halt();", "co_await Finish{};"]
        if kind == "program":
            action = {"program_restart": "Start", "program_restart_child": "StartChild", "program_reset_alt_pc": "Stop", "program_stop": "Pause", "program_activate": "Resume"}.get(statement.metadata["operation"])
            if action is None:
                raise ValueError(f"Unknown source program operation {statement.metadata}")
            index = statement.metadata["program_index"]
            target = self.module.script.programs[index].name
            return [f"self.control({quoted(target)}, ProgramAction::{action});"]
        raise ValueError(f"Unsupported source statement {kind}")

    def body(self) -> list[str]:
        flow = StructuredFlow(self)
        body = flow.render()
        compact = []
        index = 0
        while index < len(body):
            line = body[index]
            count = 1
            if line.strip() == "co_await Suspend{};":
                while index + count < len(body) and body[index + count] == line:
                    count += 1
            if count >= 3:
                indent = line[:-len(line.lstrip())] if line != line.lstrip() else ""
                compact += [indent + f"for (unsigned pause = 0; pause < {count}; ++pause)", indent + "{", indent + "    co_await Suspend{};", indent + "}"]
            else:
                compact.extend(body[index:index + count])
            index += count
        body = compact
        prologue = [f"FunctionScope context(self, {quoted(self.function.name)});"]
        if self.used_engine or flow.has_loop:
            prologue.append("Host &engine = self.host();")
        for index, argument in enumerate(self.function.arguments):
            member = self.module.layout.at(argument["data_offset"])
            prologue.append(self.member_write(member, argument["data_offset"] - member.offset, argument["type"], f"parameter{index + 1}"))
        for name, typ in self.function.variables.items():
            prologue.append(f"{cpp_type(typ)} {self.locals[name]};")
        if prologue and body:
            prologue.append("")
        return prologue + body


class StructuredFlow:
    """Render natural loops and conditionals, keeping labels for irregular edges.

    Labels are source-level control flow, never VM instruction positions.
    Synthetic postdominator boundaries are analysis-only; irregular edges retain
    their explicit source destinations.
    """
    def __init__(self, renderer: Renderer):
        self.renderer = renderer
        self.function = renderer.function
        self.blocks = {block.index: block for block in self.function.blocks}
        self.order = {block.index: index + 1 for index, block in enumerate(self.function.blocks)}
        self.emitted: set[int] = set()
        self.lines: list[str] = []
        self.indent = 0
        self.has_loop = False
        self.loops, self.postdominators = self.analyze()

    def analyze(self):
        nodes = set(self.blocks)
        entry = self.function.blocks[0].index
        predecessors = {index: set() for index in nodes}
        for block in self.blocks.values():
            for successor in block.successors:
                if successor not in nodes:
                    raise ValueError(f"Native CFG edge leaves its recovered function: {successor}")
                predecessors[successor].add(block.index)
        dominators = {index: ({entry} if index == entry else set(nodes)) for index in nodes}
        changed = True
        while changed:
            changed = False
            for index in nodes - {entry}:
                incoming = predecessors[index]
                value = {index} | (set.intersection(*(dominators[item] for item in incoming)) if incoming else set())
                if value != dominators[index]:
                    dominators[index] = value
                    changed = True
        loops: dict[int, set[int]] = {}
        for block in self.blocks.values():
            for successor in block.successors:
                if successor not in dominators[block.index]:
                    continue
                body = {successor, block.index}
                work = [block.index] if block.index != successor else []
                while work:
                    current = work.pop()
                    for previous in predecessors[current]:
                        if previous not in body:
                            body.add(previous)
                            work.append(previous)
                loops.setdefault(successor, set()).update(body)
        # Closed loops have no path to the function exit. Without an analysis
        # boundary their nodes falsely postdominate each other, so a branch join
        # can skip the loop latch (including its Yield). Model an exit at the
        # header of each non-terminating loop for postdominator analysis only.
        # No synthetic edge is emitted into the generated function.
        end = -1
        successors = {index: list(block.successors) or [end] for index, block in self.blocks.items()}

        def reaches_boundary(boundaries):
            reached = set(boundaries)
            pending = list(reached)
            while pending:
                for previous in predecessors[pending.pop()]:
                    if previous not in reached:
                        reached.add(previous)
                        pending.append(previous)
            return reached

        exits = {index for index in nodes if not self.blocks[index].successors}
        terminating = reaches_boundary(exits)
        boundaries = set(loops) - terminating
        # An irreducible closed component may have no natural-loop header.
        # Treat its remaining nodes conservatively instead of inventing a join.
        boundaries |= nodes - reaches_boundary(exits | boundaries)
        for index in boundaries:
            successors[index].append(end)
        post = {index: set(nodes) | {end} for index in nodes}
        post[end] = {end}
        changed = True
        while changed:
            changed = False
            for index in reversed(list(self.blocks)):
                value = {index} | set.intersection(*(post[item] for item in successors[index]))
                if value != post[index]:
                    post[index] = value
                    changed = True
        immediate = {}
        for index in nodes:
            candidates = post[index] - {index, end}
            immediate[index] = max(candidates, key=lambda item: len(post[item])) if candidates else None
        return loops, immediate

    def line(self, text: str = ""):
        self.lines.append("    " * self.indent + text if text else "")

    def label(self, index: int) -> str:
        return f"branch{self.order[index]}"

    def transfer(self, target: int, stop: set[int], active: list[tuple[int, set[int], int | None]]) -> bool:
        if active:
            header, members, follow = active[-1]
            if target == header:
                self.line("engine.checkpoint();")
                self.line("continue;")
                return True
            if target not in members:
                if target == follow:
                    self.line("break;")
                else:
                    self.line(f"goto {self.label(target)};")
                return True
        if target in stop:
            return True
        if target in self.emitted:
            self.line(f"goto {self.label(target)};")
            return True
        return False

    def walk(self, start: int | None, stop: set[int], active: list[tuple[int, set[int], int | None]]):
        current = start
        while current is not None and current not in stop:
            if current in self.emitted:
                self.transfer(current, stop, active)
                return
            if current in self.loops and all(header != current for header, _, _ in active):
                members = self.loops[current]
                follow = self.postdominators[current]
                if follow in members:
                    exits = {target for index in members for target in self.blocks[index].successors if target not in members}
                    follow = next(iter(exits)) if len(exits) == 1 else None
                self.has_loop = True
                self.line("while (true)")
                self.line("{")
                self.indent += 1
                self.line("engine.checkpoint();")
                self.walk(current, stop, active + [(current, members, follow)])
                self.indent -= 1
                self.line("}")
                if follow is None or self.transfer(follow, stop, active):
                    return
                current = follow
                continue
            self.emitted.add(current)
            block = self.blocks[current]
            self.line(self.label(current) + ":")
            for statement in block.statements:
                for text in self.renderer.statement(statement):
                    self.line(text)
            if not block.successors:
                return
            if len(block.successors) == 1:
                target = block.successors[0]
                if self.transfer(target, stop, active):
                    return
                current = target
                continue
            if len(block.successors) != 2 or block.condition is None:
                raise ValueError(f"Unstructured source branch {block}")
            first, second = block.successors
            join = self.postdominators[current]
            if join == current:
                join = None
            test = self.renderer.condition(block.condition, block.branch_when)
            self.line(f"if ({test})")
            self.line("{")
            self.indent += 1
            boundary = stop | ({join} if join is not None else set())
            if not self.transfer(first, boundary, active):
                self.walk(first, boundary, active)
            self.indent -= 1
            self.line("}")
            self.line("else")
            self.line("{")
            self.indent += 1
            if not self.transfer(second, boundary, active):
                self.walk(second, boundary, active)
            self.indent -= 1
            self.line("}")
            if join is None or join in stop:
                return
            if self.transfer(join, stop, active):
                return
            current = join

    def render(self) -> list[str]:
        if not self.function.blocks:
            raise ValueError("Native source function has no body")
        self.walk(self.function.blocks[0].index, set(), [])
        for block in self.function.blocks:
            if block.index not in self.emitted:
                # Irreducible targets retain explicit source labels. They are
                # reached only through the preserved goto edge.
                self.walk(block.index, set(), [])
        labels = {match.group(1) for text in self.lines for match in [re.search(r"\bgoto (branch\d+);", text)] if match}
        self.lines = [text for text in self.lines if not re.fullmatch(r"\s*branch\d+:", text) or text.strip()[:-1] in labels]
        self.lines = [text + " ;" if re.fullmatch(r"\s*branch\d+:", text) else text for text in self.lines]
        return compact_control_flow(self.lines)


@dataclass
class ControlGroup:
    header: str
    body: list
    otherwise: list | None = None


def compact_control_flow(lines: list[str]) -> list[str]:
    """Remove redundant else nesting without altering source evaluation order."""
    source = [line.strip() for line in lines]

    def parse(position: int) -> tuple[list, int]:
        result = []
        while position < len(source) and source[position] != "}":
            text = source[position]
            if text == "{" or (text.startswith(("if (", "while (")) and position + 1 < len(source) and source[position + 1] == "{"):
                body, position = parse(position + (1 if text == "{" else 2))
                if position >= len(source) or source[position] != "}":
                    raise ValueError("Unbalanced native control-flow group")
                position += 1
                other = None
                if position < len(source) and source[position] == "else":
                    if position + 1 == len(source) or source[position + 1] != "{":
                        raise ValueError("Malformed native else group")
                    other, position = parse(position + 2)
                    position += 1
                result.append(ControlGroup("" if text == "{" else text, body, other))
            else:
                result.append(text)
                position += 1
        return result, position

    def terminal(nodes: list) -> bool:
        if not nodes:
            return False
        last = nodes[-1]
        if isinstance(last, str):
            return last.startswith(("return", "co_return", "throw ", "goto ")) or last in {"break;", "continue;"}
        if last.header.startswith("if ("):
            return terminal(last.body) and last.otherwise is not None and terminal(last.otherwise)
        return not last.header and terminal(last.body)

    def weight(nodes: list) -> int:
        return sum(1 if isinstance(node, str) else 1 + weight(node.body) + weight(node.otherwise or []) for node in nodes)

    def invert(header: str) -> str:
        condition = header[4:-1]
        return f"if (!({condition}))"

    def simplify(nodes: list) -> list:
        output = []
        for node in nodes:
            if isinstance(node, str):
                output.append(node)
                continue
            node.body = simplify(node.body)
            if node.otherwise is not None:
                node.otherwise = simplify(node.otherwise)
            if not node.header.startswith("if (") or node.otherwise is None:
                output.append(node)
            elif not node.body and not node.otherwise:
                output.append(f"(void)({node.header[4:-1]});")
            elif not node.otherwise:
                node.otherwise = None
                output.append(node)
            elif not node.body:
                node.header, node.body, node.otherwise = invert(node.header), node.otherwise, None
                output.append(node)
            elif terminal(node.otherwise) and (not terminal(node.body) or weight(node.otherwise) < weight(node.body)):
                continuation = node.body
                node.header, node.body, node.otherwise = invert(node.header), node.otherwise, None
                output.append(node)
                output.extend(continuation)
            elif terminal(node.body):
                continuation, node.otherwise = node.otherwise, None
                output.append(node)
                output.extend(continuation)
            else:
                output.append(node)
        return output

    output = []
    def emit(nodes: list, depth: int) -> None:
        prefix = "    " * depth
        for node in nodes:
            if isinstance(node, str):
                output.append(prefix + node)
                continue
            if node.header:
                output.append(prefix + node.header)
            output.append(prefix + "{")
            emit(node.body, depth + 1)
            output.append(prefix + "}")
            if node.otherwise is not None:
                output.extend((prefix + "else", prefix + "{"))
                emit(node.otherwise, depth + 1)
                output.append(prefix + "}")
    tree, end = parse(0)
    if end != len(source):
        raise ValueError("Trailing native control-flow brace")
    emit(simplify(tree), 0)
    return output


class CppGenerator:
    def __init__(self, families: Families, public_types: dict[str, int | None], client_root: Path):
        self.families = families
        self.modules = families.modules
        self.public_types = public_types
        command_source = (client_root / "core/public/script/MbcCommands.h").read_text()
        command_source = command_source.split("enum class SferaMbcRuntimeBuiltin", 1)[1].split("};", 1)[0]
        self.builtins = {int(value): name for name, value in re.findall(r"\b(\w+)\s*=\s*(\d+)\s*,?", command_source)}
        # ffSYS movement selectors 230/231 are engine System subcommands, never
        # standalone bytecode operations. Their AST still carries System id 103.
        self.helpers = self.factor_helpers()
        self.helper_for = {instance.exact: helper for helper in self.helpers for instance in helper.instances}
        self.nodes = list(families.nodes())
        self.node_defaults: dict[int, dict[int, bytes]] = {}
        self.data_pools: dict[tuple[str, bytes], str] = {}
        self.metadata_pools: dict[tuple[str, tuple], str] = {}
        self.pool_definitions: list[str] = []
        self.header_extra: list[str] = []
        self.statistics: dict[str, Any] = {}

    def helper_shape(self, implementation: Implementation) -> bytes:
        return implementation.shape

    def factor_helpers(self) -> list[Helper]:
        groups: dict[bytes, list[Implementation]] = defaultdict(list)
        for implementation in self.families.implementations.values():
            groups[self.helper_shape(implementation)].append(implementation)
        helpers = []
        names: Counter[str] = Counter()
        for instances in groups.values():
            example = instances[0]
            columns = []
            for index in range(len(example.pattern.literals)):
                column = tuple(item.pattern.literals[index].key() for item in instances)
                if len(set(column)) > 1:
                    columns.append((index, column))
            merged: dict[tuple, list[int]] = {}
            for index, column in columns:
                merged.setdefault(column, []).append(index)
            split = False
            # A shared expression object may occur more than once. It must not
            # acquire inconsistent configuration parameters after factoring.
            mappings = {}
            for group_index, positions in enumerate(merged.values()):
                for position in positions:
                    identity = id(example.pattern.literals[position])
                    if identity in mappings and mappings[identity] != group_index:
                        split = True
                    mappings[identity] = group_index
            for group in ([ [item] for item in instances ] if split else [instances]):
                chosen = group[0]
                stem = identifier(chosen.function.name)
                names[stem] += 1
                name = stem if names[stem] == 1 else f"{stem}Variant{names[stem]}"
                positions = [] if split else list(merged.values())
                types = [infer_expression(chosen.pattern.literals[item[0]], chosen.function, chosen.module, self.public_types) for item in positions]
                helpers.append(Helper(name, chosen, group, positions, types))
        return helpers

    def helper_parameters(self, helper: Helper) -> str:
        values = ["Self &self"]
        arguments = parameters(helper.example.function)
        if arguments:
            values.append(arguments)
        values.extend(f"{cpp_type(typ)} setting{index + 1}" for index, typ in enumerate(helper.literal_types))
        return ", ".join(values)

    def helper_arguments(self, helper: Helper, instance: Implementation) -> str:
        values = ["*this"]
        values.extend(f"parameter{index + 1}" for index in range(len(instance.function.arguments)))
        render = Renderer(self, Helper("literal", instance, [instance]))
        values.extend(render.expr(instance.pattern.literals[positions[0]]) for positions in helper.literal_parameters)
        return ", ".join(values)

    def sanitized_initial(self, module: Module, member: Field) -> bytes:
        value = bytearray(member.initial)
        for offset in module.layout.metadata.data_relocations:
            if member.offset <= offset < member.offset + member.size:
                value[offset - member.offset:offset - member.offset + 4] = b"\0" * 4
        return bytes(value)

    def initial_expression(self, member: Field, data: bytes) -> str:
        if not any(data):
            return "{}"
        if member.scalar:
            if member.type_id == 16:
                return str(struct.unpack("<i", data)[0])
            if member.type_id == 0:
                return str(struct.unpack("<b", data)[0])
            if member.type_id == 32:
                return f"std::bit_cast<float>(std::uint32_t{{{struct.unpack('<I', data)[0]}}})"
            return "{" + ", ".join(map(str, struct.unpack("<III", data))) + "}"
        key = member.ctype, data
        name = self.data_pools.get(key)
        if name is None:
            name = f"initialData{len(self.data_pools) + 1}"
            self.data_pools[key] = name
            # Trailing zero storage is default-initialized by the small native
            # array constructor; no flat process image is embedded.
            last = max(index for index, value in enumerate(data) if value) + 1
            self.header_extra.append(initial_data_definition(member.ctype, name, data[:last]))
        return name

    def defaults(self):
        for node in self.nodes:
            defaults = {}
            for identity, member in node.fields.items():
                choices = Counter()
                for index in node.members:
                    module = self.modules[index]
                    candidate = next((item for item in module.layout.fields if item.canonical == identity), None)
                    if candidate is not None:
                        choices[self.sanitized_initial(module, candidate)] += 1
                defaults[identity] = choices.most_common(1)[0][0]
            self.node_defaults[node.index] = defaults

    def initialize(self, node: Family, module: Module) -> list[str]:
        available = {}
        for ancestor in reversed(list(node.ancestors())):
            available.update(self.node_defaults[ancestor.index])
        body = []
        for member in module.layout.fields:
            initial = self.sanitized_initial(module, member)
            if initial != available[member.canonical]:
                value = self.initial_expression(member, initial)
                body.append(f"self.{member.name} = {value};")
        for offset in module.layout.metadata.data_relocations:
            destination = module.layout.at(offset, 4)
            target = struct.unpack_from("<I", module.script.data, offset)[0]
            if target == len(module.script.data):
                # One-past-the-end initial pointers retain a non-dereferenceable
                # native boundary without recreating contiguous module memory.
                source = module.layout.fields[-1]
                displacement = source.size
            else:
                source = module.layout.at(target)
                displacement = target - source.offset
            address = f"self.address(self.{source.name}, {displacement}, 0).base"
            body.append(f"setMemberView(self.{destination.name}, {offset - destination.offset}, {address});")
        return body

    def metadata_pool(self, typ: str, rows: list[str], key: tuple | None = None) -> str:
        if not rows:
            return "{}"
        identity = typ, key if key is not None else tuple(rows)
        name = self.metadata_pools.get(identity)
        if name is None:
            name = f"{identifier(typ).lower()}s{len(self.metadata_pools) + 1}"
            self.metadata_pools[identity] = name
            self.pool_definitions += [f"static constexpr std::array<{typ}, {len(rows)}> {name}{{{{", *["    " + row + "," for row in rows], "}};", ""]
        return name

