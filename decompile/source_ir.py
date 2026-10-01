"""Typed source representation built by the existing symbolic decompiler.

The native backend consumes this representation. No instruction decoder, VM
register array or bytecode dispatcher is emitted into the C++ output.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re
from typing import Any

from .vm_ast import StackAstBuilder, VMSlot, VMStackMachine
from .calls import builtin_effect
from mbc_format.common import storage_size_for_type, dereferenced_type

VOID = -2
DYNAMIC = -1


class MissingReservedResult(Exception):
    """A statically reached operand underflow after an implemented no-op."""


@dataclass(frozen=True, slots=True)
class Expression:
    kind: str
    value: Any = None
    children: tuple[Expression, ...] = ()
    type_id: int | None = None

    def key(self):
        return self.kind, self.value, self.type_id, tuple(child.key() for child in self.children)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(slots=True)
class Statement:
    kind: str
    expression: Expression | None = None
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Block:
    index: int
    statements: list[Statement] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)
    condition: Expression | None = None
    branch_when: bool = False


@dataclass(slots=True)
class Function:
    name: str
    entry: int
    arguments: list[dict[str, Any]]
    parameter_capacity: int
    allow_reentry: bool = False
    return_type: int | None = DYNAMIC
    asynchronous: bool = False
    blocks: list[Block] = field(default_factory=list)
    variables: dict[str, int | None] = field(default_factory=dict)
    bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    instructions: list[Any] = field(default_factory=list)
    program: Any = None
    diagnostics: list[str] = field(default_factory=list)
    ignored_return: bool = False
    optional_return: bool = False


_TOKEN = re.compile(r'\s*(0[xX][\da-fA-F]+|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|==|!=|<=|>=|&&|\|\||<<|>>|.)')
_PRECEDENCE = {'||': 1, '&&': 2, '|': 3, '^': 4, '&': 5, '==': 6, '!=': 6,
               '<': 7, '>': 7, '<=': 7, '>=': 7, '<<': 8, '>>': 8, '+': 9, '-': 9,
               '*': 10, '/': 10, '%': 10}


def parse_expression(text: str, values: dict[str, Expression], type_id: int | None = None) -> Expression:
    tokens = _TOKEN.findall(text.strip())
    cursor = 0

    def parse(minimum=0):
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError(f'Empty recovered expression: {text}')
        token = tokens[cursor]
        cursor += 1
        if token == '(':
            node = parse()
            if tokens[cursor] != ')':
                raise ValueError(f'Unbalanced recovered expression: {text}')
            cursor += 1
        elif token in {'+', '-', '!', '~', '&', '*'}:
            child = parse(11)
            typ = 16 if token == '!' else child.type_id
            node = Expression('prefix', token, (child,), typ)
        elif token[0].isdigit() or token.startswith('.'):
            typ = 32 if not token.lower().startswith('0x') and any(c in token for c in '.eE') else 16
            node = Expression('number', token, (), typ)
        else:
            node = values.get(token, Expression('name', token, (), type_id))
        while cursor < len(tokens):
            token = tokens[cursor]
            if token == '(':
                cursor += 1
                args = []
                if tokens[cursor] != ')':
                    while True:
                        args.append(parse())
                        if tokens[cursor] != ',':
                            break
                        cursor += 1
                if tokens[cursor] != ')':
                    raise ValueError(f'Malformed recovered call: {text}')
                cursor += 1
                typ = 32 if node.value in {'float', 'real_from_integer_bits'} else 16
                node = Expression('call', None, (node, *args), typ)
            elif token in _PRECEDENCE and _PRECEDENCE[token] >= minimum:
                cursor += 1
                right = parse(_PRECEDENCE[token] + 1)
                typ = 16 if _PRECEDENCE[token] <= 8 else node.type_id
                node = Expression('binary', token, (node, right), typ)
            else:
                break
        return node

    result = parse()
    if cursor != len(tokens):
        raise ValueError(f'Unsupported recovered expression: {text}; suffix {tokens[cursor:]}')
    if type_id is not None and type_id != result.type_id:
        result = Expression(result.kind, result.value, result.children, type_id)
    return result


class TypedSourceBuilder(StackAstBuilder):
    """Retain typed nodes and effects from StackAstBuilder's symbolic stack."""

    def __init__(self, module, function: Function, return_types: dict):
        super().__init__(module.script, function.program, linker=module.linker)
        self.module = module
        self.function = function
        self.return_types = return_types
        self.nodes: dict[str, Expression] = {}
        self.source: list[Statement] = []
        self.offset = function.entry
        self.sequence = 0
        self.variant = 0
        self.optional_result = None
        self.condition: Expression | None = None
        self.return_values: list[int | None] = []
        for index, argument in enumerate(function.arguments):
            self.memory.bind_argument(index=index, type_id=argument['type'], data_offset=argument['data_offset'])

    def value(self, slot: VMSlot) -> Expression:
        if 'node' in slot.metadata:
            node = slot.metadata['node']
            if node.type_id == slot.type_id:
                return node
            return Expression(node.kind, node.value, node.children, slot.type_id)
        return parse_expression(slot.expr, self.nodes, slot.type_id)

    def slot(self, node: Expression, access: Expression | None = None, width: int = 4) -> VMSlot:
        self.sequence += 1
        name = f'n{self.offset}_{self.sequence}_{self.variant}'
        self.nodes[name] = node
        return VMSlot(name, type_id=node.type_id, storage_size=width,
                      metadata={'node': node, 'access': access}, is_lvalue=access is not None)

    def temporary(self, node: Expression, kind='let', metadata=None) -> Expression:
        self.sequence += 1
        name = f'v{self.offset}_{self.sequence}_{self.variant}'
        self.function.variables[name] = node.type_id
        self.source.append(Statement(kind, node, name, metadata or {}))
        return Expression('name', name, (), node.type_id)

    def binding(self, offset: int, typ: int | None, width: int, role='data', stride=None) -> Expression:
        name = f'data_{offset}_{typ}_{width}_{role}'
        binding = {'offset': offset, 'type': typ, 'length': width, 'role': role}
        if stride is not None:
            binding['stride'] = stride
        self.function.bindings[name] = binding
        return Expression('name', name, (), typ)

    def storage(self, offset: int, typ: int | None, width: int) -> Expression:
        return Expression('storage', (typ, width), (self.binding(offset, typ, width),), 48)

    def load(self, access: Expression, typ: int | None, width: int, *, address_value=False):
        # pushReference/Dereference promote a loaded byte to the integer stack
        # type while retaining its one-byte lvalue width for stores and updates.
        if typ == 0 and not address_value:
            typ = 16
        node = Expression('load', address_value, (access,), typ)
        value = self.temporary(node, 'read')
        self.vm.push(self.slot(value, access, width))

    def _pop_value(self, ins, name):
        if not self.vm.stack:
            for statement in reversed(self.source):
                if statement.kind == 'builtin':
                    opcode = statement.metadata['subopcode']
                    if opcode in {122, 130, 133}:
                        raise MissingReservedResult(
                            f'{self.module.name}.{self.function.name}: missing operand at {ins.offset} '
                            f'after reserved no-result builtin {opcode}')
                    break
            self.function.diagnostics.append(f'{self.function.name}: stack underflow at {ins.offset} ({name})')
            return self.slot(Expression('number', '0', (), DYNAMIC))
        return self.vm.stack.pop()

    def _push_decoded_value(self, ins):
        op = ins.operands
        typ = op.get('type', 1 if ins.mnemonic == 'push_inline_span' else 16)
        if ins.mnemonic == 'push_data_ref':
            width = storage_size_for_type(typ)
            self.load(self.storage(op['data_offset'], typ, width), typ, width)
            return
        if ins.mnemonic in {'push_inline_span', 'push_typed_span_ref', 'push_inline_typed_span'}:
            access = self.storage(op['data_offset'], typ, op['length'])
            node = Expression('address', None, (access,), typ)
            self.vm.push(self.slot(node, access if ins.mnemonic == 'push_typed_span_ref' else None, 12))
            return
        super()._push_decoded_value(ins)
        old = self.vm.stack.pop()
        node = parse_expression(old.expr, self.nodes, old.type_id)
        self.vm.push(self.slot(node))

    def _push_index_or_slice(self, ins):
        op = ins.operands
        typ = op.get('type')
        name = ins.mnemonic
        width = 1 if typ == 0 else 4 if typ in {16, 32} else 12
        address_value = typ == 48
        if name == 'array_index_abs':
            index = self.value(self._pop_value(ins, 'index'))
            size = abs(op['count']) * op['element_size']
            base = Expression('address', None, (self.storage(op['base'], typ, size),), 48)
            access = Expression('element', (typ, op['element_size'], op['count'], True, True, op.get('span', op.get('length', width)), width), (base, index), 48)
            address_value = typ == 48 or (typ not in {0, 16, 32} and op['count'] >= 0)
        elif name in {'array2_index', 'array2_index_checked'}:
            base = self.value(self._pop_value(ins, 'base'))
            index = self.value(self._pop_value(ins, 'index'))
            checked = name == 'array2_index_checked'
            count = op.get('count', 0)
            access = Expression('element', (typ, op['element_size'], count, False, checked, 0, width), (base, index), 48)
            address_value = typ == 48 or (typ not in {0, 16, 32} and checked and count >= 0)
        else:
            base = self.value(self._pop_value(ins, 'base'))
            width = op.get('length', width)
            access = Expression('field', (typ, op['offset'], width), (base,), 48)
            if name == 'slice_offset_span':
                self.vm.push(self.slot(Expression('address', None, (access,), typ)))
                return
        self.load(access, typ, width, address_value=address_value)

    def _apply_call(self, ins, kind, name, args, effect):
        raise RuntimeError('Typed call effects must pass through emit_call')

    def emit_call(self, ins, kind, name, args, typ, pushes):
        nodes = []
        for argument in args:
            node = self.value(argument)
            access = argument.metadata.get('access')
            if kind == 'builtin' and access is not None:
                node = Expression('by_reference', None, (node, access), node.type_id)
            nodes.append(node)
        metadata = {'subopcode': ins.operands['subopcode']} if kind == 'builtin' else {}
        expression = Expression('invocation', name, tuple(nodes), typ)
        if pushes:
            result = self.temporary(expression, kind, metadata)
            self.vm.push(self.slot(result))
        else:
            self.source.append(Statement(kind, expression, metadata=metadata))

    def _handle_function_call(self, ins):
        target = ins.operands['target']
        name = self.module.call_names.get(target) or self.linker.callable_name_for_offset(target)
        if not name:
            raise ValueError(f'Unresolved local function at {target} in {self.module.name}')
        count = self._consume_arg_count() or 0
        args = self._pop_arg_slots(ins, count)
        typ = self.return_types.get((self.module.name, target), self.return_types.get(name, DYNAMIC))
        self.emit_call(ins, 'call', name, args, typ, typ != VOID)
        if typ != VOID and self.return_types.get(('optional', self.module.name, target), self.return_types.get(('optional', name), False)):
            self.optional_result = self.value(self.vm.stack[-1])

    def _visit(self, ins):
        self.offset = ins.offset
        self.sequence = 0
        name = ins.mnemonic
        op = ins.operands
        if op.get('subopcode') is not None:
            count = self._consume_arg_count() or 0
            args = self._pop_arg_slots(ins, count)
            effect = builtin_effect(op['subopcode'], argc=count)
            # Use the original engine selector arguments; pretty-printer aliases
            # are not a separate ABI and must not change the native command id.
            from .calls import SELECTOR_BUILTINS, selector_from_args, specialize_builtin_call
            arguments = [self.value(arg) for arg in args]
            if op['subopcode'] in SELECTOR_BUILTINS and selector_from_args(arguments) is None:
                raise ValueError(f'Unresolved engine selector in {self.module.name}.{self.function.name} at {ins.offset}')
            _, _, specialized = specialize_builtin_call(op['subopcode'], arguments, effect)
            typ = specialized.return_type_id if specialized.returns_value else VOID
            self.emit_call(ins, 'builtin', specialized.name, args, typ, specialized.returns_value)
            return
        if name == 'store':
            value = self._pop_value(ins, 'value')
            target = self._pop_value(ins, 'target')
            access = target.metadata.get('access')
            if access is None:
                access = Expression('indirect', (target.type_id, target.storage_size), (Expression('number', '4294967295', (), 16),), 48)
            expression = Expression('assignment', (target.type_id, target.storage_size), (access, self.value(value)))
            self.source.append(Statement('store', expression))
            result_type = 16 if target.storage_size == 1 else target.type_id
            returned = Expression('stored', target.storage_size, (self.value(value),), result_type)
            self.vm.push(self.slot(returned, access, target.storage_size))
            return
        if name in {'pre_inc', 'post_inc', 'pre_dec', 'post_dec'} or name.endswith('assign_u16'):
            target = self._pop_value(ins, 'target')
            access = target.metadata.get('access')
            if access is None:
                raise ValueError(f'Update without storage in {self.module.name}.{self.function.name} at {ins.offset}')
            node = Expression('update', (name, op.get('value', 1), target.storage_size), (access, self.value(target)), target.type_id)
            result = self.temporary(node, 'increment')
            self.vm.push(self.slot(result, access, target.storage_size))
            return
        if name == 'address_of':
            source = self._pop_value(ins, 'value')
            access = source.metadata.get('access')
            if access is None:
                node = Expression('call', None, (Expression('name', 'detached_address'),), (source.type_id or 0) + 1)
            else:
                node = Expression('address', None, (access,), (source.type_id or 0) + 1)
            self.vm.push(self.slot(node))
            return
        if name == 'deref':
            source = self._pop_value(ins, 'pointer')
            typ = dereferenced_type(source.type_id)
            width = 1 if source.type_id == 1 else storage_size_for_type(typ)
            access = Expression('indirect', (typ, width), (self.value(source),), 48)
            self.load(access, typ, width, address_value=typ == 48)
            return
        if name in {'return', 'return_local'}:
            value = self.value(self.vm.stack[-1]) if self.vm.stack and not self.function.ignored_return else None
            self.return_values.append(value.type_id if value else VOID)
            self.source.append(Statement('return', value))
            return
        if name in {'jfalse_rel16', 'jfalse_rel32', 'logical_or_rel16', 'logical_and_rel16'}:
            self.condition = self.value(self._pop_value(ins, 'condition'))
            return
        if name in {'jmp_rel16', 'jmp_rel32'}:
            return
        if name == 'yield_program':
            self.source.append(Statement('yield'))
            self.vm.suspend()
            self.pending_arg_count = None
            return
        if name == 'end_program':
            self.source.append(Statement('finish'))
            return
        if name == 'halt_interpreter':
            self.source.append(Statement('halt'))
            return
        if name in {'program_restart', 'program_restart_child', 'program_activate', 'program_reset_alt_pc', 'program_stop'}:
            self.source.append(Statement('program', metadata={'operation': name, 'program_index': op['program_index']}))
            return
        before = list(self.vm.stack)
        super()._visit(ins)
        if len(self.statements):
            unknown = [statement for statement in self.statements if statement.kind not in {'meta', 'decl'}]
            if unknown:
                raise ValueError(f'Unrepresented source statement in {self.module.name}.{self.function.name}: {unknown[0].text}')
            self.statements.clear()
        # Base symbolic arithmetic is reused, while the tree retains the exact
        # type chosen by the left operand of the recovered runtime operation.
        for index, slot in enumerate(self.vm.stack):
            unchanged = slot.expr in self.nodes and self.nodes[slot.expr] == slot.metadata.get('node')
            if not unchanged:
                if name in {'add', 'sub', 'mul', 'div', 'mod'} and len(before) >= 2 and index == len(self.vm.stack) - 1:
                    slot.type_id = before[-2].type_id
                node = parse_expression(slot.expr, self.nodes, slot.type_id)
                self.vm.stack[index] = self.slot(node, slot.metadata.get('access'), slot.storage_size)



def merge_type(types):
    found = set(types)
    if len(found) == 1:
        return next(iter(found))
    if found <= {0, 16}:
        return 16
    return DYNAMIC


def recover_function(module, function: Function, return_types: dict) -> Function:
    """Recover typed source CFG, specializing only genuinely different stack shapes.

    Unequal operand heights are not an error by themselves: a procedure can
    leave an optional value, and a later reset can discard it. Those paths stay
    distinct until their shapes converge. No execution stack is emitted.
    """
    instructions = function.instructions
    if not instructions:
        function.return_type = VOID
        function.optional_return = False
        function.blocks = [Block(0, [Statement('return')])]
        return function
    by_offset = {ins.offset: ins for ins in instructions}
    jumps = {'jmp_rel16', 'jmp_rel32', 'jfalse_rel16', 'jfalse_rel32', 'logical_or_rel16', 'logical_and_rel16'}
    terminals = {'return', 'return_local', 'end_program', 'halt_interpreter'}
    optional_calls = set()
    for ins in instructions:
        if ins.mnemonic == 'call_rel32':
            target = ins.operands['target']
            name = module.call_names.get(target)
            if return_types.get(('optional', module.name, target), return_types.get(('optional', name), False)):
                optional_calls.add(ins.offset)
    leaders = {instructions[0].offset}
    for ins in instructions:
        if ins.mnemonic in jumps and ins.operands['target'] in by_offset:
            leaders.add(ins.operands['target'])
        if ins.mnemonic in jumps | terminals | {'yield_program'} or ins.offset in optional_calls:
            following = ins.offset + ins.length
            if following in by_offset:
                leaders.add(following)
    blocks = {}
    for ins in instructions:
        if ins.offset in leaders:
            current = ins.offset
            blocks[current] = []
        blocks[current].append(ins)

    def edges(block):
        last = block[-1]
        following = last.offset + last.length
        if last.mnemonic in terminals:
            return []
        if last.mnemonic.startswith('jmp_'):
            return [(last.operands['target'], True)]
        if last.mnemonic in jumps:
            return [(last.operands['target'], True), (following, False)]
        if last.offset in optional_calls:
            return [(following, True), (following, False)]
        return [(following, False)] if following in blocks else []

    def discards_input(start, seen=None):
        seen = set() if seen is None else seen
        if start in seen or start not in blocks:
            return False
        seen.add(start)
        for instruction in blocks[start]:
            operation = instruction.mnemonic
            if operation in {'stack_frame_reset', 'yield_program', 'end_program', 'halt_interpreter'} or (operation == 'return_local' and function.ignored_return):
                return True
            if operation.startswith('jmp_'):
                return discards_input(instruction.operands['target'], seen)
            if operation.startswith('program_') and operation != 'program_prologue':
                continue
            if operation == 'call_rel32':
                callee = module.by_entry.get(instruction.operands['target'])
                if callee and not callee.arguments:
                    continue
            return False
        return False

    function.variables.clear()
    function.bindings.clear()
    function.diagnostics.clear()
    builder = TypedSourceBuilder(module, function, return_types)
    states = {}
    identities = {}
    locations = {}
    work = deque()
    outputs = {}

    def intern(start, shape, frames, argc):
        if len(shape) > 256 or len(frames) > 50:
            raise ValueError(f'Unbounded source evaluation stack: {module.name}.{function.name}')
        key = start, tuple((typ, width, access is not None) for typ, access, width in shape), frames, argc
        if key not in identities:
            index = len(identities)
            identities[key] = index
            locations[index] = start
            states[index] = shape, frames, argc
            work.append(index)
            return index
        index = identities[key]
        old = states[index][0]
        merged = []
        for position, (previous, incoming) in enumerate(zip(old, shape)):
            typ = merge_type((previous[0], incoming[0]))
            access = previous[1]
            width = previous[2]
            if width != incoming[2]:
                function.diagnostics.append(f'Incompatible storage widths at {start}: {width}/{incoming[2]}')
            if access != incoming[1]:
                if access is not None and incoming[1] is not None:
                    name = f'a{index}_{position}'
                    function.variables[name] = 48
                    kind, value = (access.value if access.kind == 'located' else (access.kind, access.value))
                    access = Expression('located', (kind, value), (Expression('name', name, (), 48),), 48)
                else:
                    access = None
            merged.append((typ, access, width))
        new = tuple(merged), frames, argc
        if new != states[index]:
            states[index] = new
            work.append(index)
        return index

    intern(instructions[0].offset, (), (0,), None)
    attempts = 0
    while work:
        index = work.popleft()
        start = locations[index]
        attempts += 1
        if attempts > max(300, len(blocks) * 100):
            raise ValueError(f'Non-converging source CFG: {module.name}.{function.name}')
        shape, frames, argc = states[index]
        builder.variant = index
        builder.vm = VMStackMachine(memory=builder.memory)
        builder.vm.frame_bases = list(frames)
        for position, (typ, access, width) in enumerate(shape):
            name = f'p{index}_{position}'
            function.variables[name] = typ
            builder.vm.stack.append(builder.slot(Expression('name', name, (), typ), access, width))
        builder.pending_arg_count = argc
        builder.source = []
        builder.condition = None
        builder.optional_result = None
        failed = False
        try:
            for ins in blocks[start]:
                builder._visit(ins)
        except MissingReservedResult as error:
            builder.source.append(Statement('fault', metadata={'message': str(error)}))
            builder.condition = None
            failed = True
        outgoing = []
        for target, taken in (() if failed else edges(blocks[start])):
            if target not in blocks:
                raise ValueError(f'Control flow leaves recovered source function: {module.name}.{function.name} -> {target}')
            slots = list(builder.vm.stack)
            if builder.optional_result is not None and not taken:
                slots.pop()
            if taken and blocks[start][-1].mnemonic in {'logical_or_rel16', 'logical_and_rel16'}:
                slots.append(builder.slot(builder.condition))
            if discards_input(target):
                slots = slots[:builder.vm.frame_bases[-1]]
            next_shape = tuple((slot.type_id, slot.metadata.get('access'), slot.storage_size) for slot in slots)
            next_index = intern(target, next_shape, tuple(builder.vm.frame_bases), builder.pending_arg_count)
            outgoing.append((next_index, slots))
        condition = Expression('present', None, (builder.optional_result,), 16) if builder.optional_result is not None else builder.condition
        positive = builder.optional_result is not None or blocks[start][-1].mnemonic == 'logical_or_rel16'
        outputs[index] = list(builder.source), outgoing, condition, positive
    result = []
    edge_blocks = []
    returns = []
    edge_index = len(states)
    for index in sorted(outputs):
        body, outgoing, condition, positive = outputs[index]
        successors = []
        for target, slots in outgoing:
            transfers = []
            copies = []
            for position, slot in enumerate(slots):
                typ, access, width = states[target][0][position]
                name = f'p{target}_{position}'
                function.variables[name] = typ
                copy = f'e{index}_{target}_{position}'
                function.variables[copy] = typ
                transfers.append(Statement('phi', builder.value(slot), copy))
                copies.append(Statement('phi', Expression('name', copy, (), typ), name))
                if access is not None and access.kind == 'located' and access.children[0].value == f'a{target}_{position}':
                    source = slot.metadata.get('access')
                    if source is not None:
                        address_copy = copy + '_address'
                        function.variables[address_copy] = 48
                        transfers.append(Statement('address', source, address_copy))
                        copies.append(Statement('phi', Expression('name', address_copy, (), 48), f'a{target}_{position}'))
            if transfers:
                successors.append(edge_index)
                edge_blocks.append(Block(edge_index, transfers + copies, [target]))
                edge_index += 1
            else:
                successors.append(target)
        result.append(Block(index, body, successors, condition, positive))
        for statement in body:
            if statement.kind == 'return':
                returns.append(statement.expression.type_id if statement.expression else VOID)
    result.extend(edge_blocks)
    function.blocks = result
    function.return_type = merge_type(returns) if returns else VOID
    function.optional_return = VOID in returns and any(typ != VOID for typ in returns)
    function.asynchronous = any(statement.kind in {'yield', 'finish', 'halt'} or (statement.kind == 'builtin' and statement.metadata.get('subopcode') in {20, 21})
                                for block in result for statement in block.statements)
    return function
