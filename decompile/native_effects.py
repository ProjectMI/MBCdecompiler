"""Apply the recovered client's engine-call argument contract to source IR."""
from collections import Counter
from dataclasses import replace
from .source_optimize import _names, _rewrite, optimize_function

# These two engine operations write the source slot of a supplied argument.
# Other operations consume the value (including pointer values), not that slot.
SOURCE_WRITING_BUILTINS = frozenset({33, 121})


def simplify_engine_arguments(function):
    count = 0
    for block in function.blocks:
        for statement in block.statements:
            if statement.kind not in {'call', 'builtin'}:
                continue
            if statement.kind == 'builtin' and statement.metadata['subopcode'] in SOURCE_WRITING_BUILTINS:
                continue
            node = statement.expression
            children = []
            for child in node.children:
                if child.kind == 'by_reference':
                    child = child.children[0]
                    count += 1
                children.append(child)
            statement.expression = replace(node, children=tuple(children))
    return count


def inline_stable_fields(function):
    uses = Counter()
    for block in function.blocks:
        uses.update(_names(block.condition))
        for statement in block.statements:
            uses.update(_names(statement.expression))
    count = 0
    for block in function.blocks:
        kept = []
        pending = {}
        for statement in block.statements:
            node = statement.expression
            if statement.kind == 'read' and statement.result and uses[statement.result] == 1 and node.kind == 'load' and node.children[0].kind == 'storage':
                pending[statement.result] = node
                kept.append(statement)
                continue
            referenced = set(_names(node))
            replaced = referenced & pending.keys()
            if replaced:
                statement.expression = _rewrite(node, {name: pending.pop(name) for name in replaced})
                kept = [previous for previous in kept if previous.result not in replaced]
                count += len(replaced)
            kept.append(statement)
            if statement.kind in {'store', 'increment', 'call', 'builtin', 'yield', 'program', 'halt', 'finish'}:
                pending.clear()
        replaced = set(_names(block.condition)) & pending.keys()
        if replaced:
            block.condition = _rewrite(block.condition, {name: pending[name] for name in replaced})
            kept = [statement for statement in kept if statement.result not in replaced]
            count += len(replaced)
        block.statements = kept
    return count


def simplify_project(modules):
    aliases = snapshots = 0
    for module in modules:
        for function in module.functions:
            aliases += simplify_engine_arguments(function)
            snapshots += optimize_function(function)
            snapshots += inline_stable_fields(function)
            snapshots += optimize_function(function)
    return aliases, snapshots


def engine_memory_accesses(command, arguments, constant):
    """Known byte extents consumed by the restored client, not pointer types.

    A float pointer passed to ObjectRotation addresses a Vec3F (12 bytes).
    A source slot supplied to Receive is a separate, by-reference destination.
    Unknown/dynamic lengths retain their declared array extent and runtime checks.
    Entries are (argument index, byte length, writes source slot instead of value).
    """
    count = len(arguments)
    if command in {100, 101, 102} and count >= 2:
        yield 1, 12, False
    elif command == 81 and count == 2:
        yield 0, 12, False
        yield 1, 12, False
    elif command == 131 and count >= 3 and constant(arguments[0]) == 0:
        yield 1, 4, False
        yield 2, 12, False
    elif command == 83 and count == 5:
        for index, width in ((1, 4), (2, 12), (3, 4)):
            yield index, width, False
    elif command in {112, 113} and count:
        yield 0, 12, False
    elif command == 121 and count == 1:
        yield 0, 12, True
    elif command == 33:
        for index, argument in enumerate(arguments[1:], 1):
            if argument.kind == 'by_reference' and argument.type_id in {0, 16, 32, 48}:
                # Byte lvalues are promoted to Integer on the executable VM stack.
                yield index, 4, True
    elif 148 <= command <= 152 and count:
        yield 0, min(command - 147, 4), False
    elif 154 <= command <= 158 and count >= 2:
        width = min(command - 153, 4)
        yield 0, width, False
        yield 1, 1 if width == 1 else 4, False
    elif command in {77, 78, 79, 96, 147}:
        length_index = 4 if command == 77 else 2
        if count > length_index:
            width = constant(arguments[length_index])
            if width is not None and width > 0:
                for index in ((1, 3) if command == 77 else (0,) if command == 79 else (0, 1)):
                    yield index, width, False
    elif command == 160 and count >= 2:
        length = constant(arguments[1])
        if length is not None and length > 0:
            yield 0, length * 4, False


class MemoryOrigins:
    """Trace source addresses without replacing loads with their future values.

    This is a may-alias analysis used only to group native storage. It never
    rewrites execution, removes bounds checks or manufactures a runtime pointer.
    """
    def __init__(self, function):
        from collections import defaultdict
        self.function = function
        self.definitions = defaultdict(list)
        self.stores = defaultdict(list)
        self.parameters = {arg['data_offset']: index for index, arg in enumerate(function.arguments)}
        for block in function.blocks:
            for statement in block.statements:
                if statement.result:
                    self.definitions[statement.result].append(statement)
                node = statement.expression
                if statement.kind == 'store' and node.children[0].kind == 'storage':
                    binding = function.bindings[node.children[0].children[0].value]
                    self.stores[binding['offset']].append(node.children[1])
        self.memo = {}
        self.cycles = 0

    def constant(self, node, visiting=frozenset()):
        if node.kind == 'by_reference':
            return self.constant(node.children[0], visiting)
        if node.kind == 'number' and node.type_id in {0, 16}:
            text = str(node.value)
            return int(text, 0) if text.lower().startswith('0x') else int(text)
        if node.kind == 'name' and node.value not in visiting:
            definitions = self.definitions.get(node.value, ())
            values = {self.constant(item.expression, visiting | {node.value}) for item in definitions}
            if len(values) == 1:
                return values.pop()
        if node.kind == 'prefix' and node.value in {'+', '-'}:
            value = self.constant(node.children[0], visiting)
            return None if value is None else -value if node.value == '-' else value
        if node.kind == 'binary' and node.value in {'+', '-', '*'}:
            left, right = (self.constant(child, visiting) for child in node.children)
            if left is not None and right is not None:
                return left + right if node.value == '+' else left - right if node.value == '-' else left * right
        return None

    @staticmethod
    def shift(origins, low, high=None):
        high = low if high is None else high
        return {(kind, identity, begin + low, end + high) for kind, identity, begin, end in origins}

    def origins(self, node, address=False, visiting=frozenset()):
        if node is None:
            return set()
        key = id(node), address
        if key in visiting:
            self.cycles += 1
            return set()
        if key in self.memo:
            return self.memo[key]
        cycles = self.cycles
        result = self._origins(node, address, visiting | {key})
        if self.cycles == cycles:
            self.memo[key] = result
        return result

    def _origins(self, node, address, visiting):
        children = node.children
        kind = node.kind
        if kind == 'name':
            result = set()
            for statement in self.definitions.get(node.value, ()):
                result.update(self.origins(statement.expression, statement.kind == 'address', visiting))
            return result
        if kind == 'by_reference':
            return self.origins(children[1] if address else children[0], address, visiting)
        if kind == 'address':
            return self.origins(children[0], True, visiting)
        if kind == 'load':
            return self.origins(children[0], bool(node.value), visiting)
        if kind in {'stored', 'cast'}:
            return self.origins(children[0], address, visiting)
        if kind == 'storage':
            binding = self.function.bindings[children[0].value]
            offset = binding['offset']
            if address:
                return {('data', offset, 0, 0)}
            result = set()
            if offset in self.parameters and node.value[0] not in {0, 16, 32}:
                result.add(('argument', self.parameters[offset], 0, 0))
            store_key = ('stored', offset)
            if node.value[0] in {0, 16, 32}:
                return result
            if store_key in visiting:
                self.cycles += 1
                return result
            for source in self.stores.get(offset, ()):
                result.update(self.origins(source, False, visiting | {store_key}))
            return result
        if kind == 'located' and address:
            return self.origins(children[0], False, visiting)
        if kind == 'indirect' and address:
            return self.origins(children[0], False, visiting)
        if kind == 'field' and address:
            return self.shift(self.origins(children[0], False, visiting), node.value[1])
        if kind == 'element' and address:
            _, stride, count, absolute, checked, _, _ = node.value
            bases = self.origins(children[0], False, visiting)
            index = self.constant(children[1])
            if index is not None:
                if absolute or checked:
                    size = abs(count)
                    if index < 0 or index >= size:
                        index = (0 if index < 0 else size - 1) if absolute else size - 1
                return self.shift(bases, index * stride)
            if (absolute or checked) and count:
                return self.shift(bases, 0, (abs(count) - 1) * stride)
            return set()
        if kind == 'binary' and node.value in {'+', '-'}:
            displacement = self.constant(children[1])
            if displacement is not None:
                return self.shift(self.origins(children[0], False, visiting), displacement if node.value == '+' else -displacement)
        return set()


def resolve_memory_extents(modules):
    """Preserve object identity for accesses crossing syntactic scalar slots.

    Propagate fixed byte requirements through source function arguments before
    splitting process data into private members. Results describe only original
    data extents: no zero padding, independent alias copies or catch-all buffer.
    """
    from collections import defaultdict
    analyses = {}
    summaries = defaultdict(dict)
    calls = []
    owners = {}
    public = defaultdict(list)
    for module in modules:
        module.memory_extents.clear()
        for function in module.functions:
            owners[id(function)] = module
        for symbol in module.script.functions:
            if not symbol.is_import and symbol.code_offset in module.by_entry:
                public[symbol.name].append(module.by_entry[symbol.code_offset])
    extents = defaultdict(set)

    def require(function, origins, width, reason):
        if width <= 0:
            return False
        changed = False
        for kind, identity, begin, end in origins:
            if kind == 'data':
                first, last = min(0, begin), max(1, end + width)
                extents[id(owners[id(function)])].add((identity + first, last - first, (function.name, *reason)))
            else:
                previous = summaries[id(function)].get(identity)
                value = (begin, end + width)
                if previous:
                    value = (min(previous[0], value[0]), max(previous[1], value[1]))
                if previous != value:
                    summaries[id(function)][identity] = value
                    changed = True
        return changed

    for module in modules:
        for function in module.functions:
            analysis = analyses[id(function)] = MemoryOrigins(function)
            for block in function.blocks:
                for statement in block.statements:
                    node = statement.expression
                    if node is None:
                        continue
                    if statement.kind == 'builtin':
                        for index, width, source in engine_memory_accesses(statement.metadata['subopcode'], node.children, analysis.constant):
                            require(function, analysis.origins(node.children[index], source), width, ('engine', statement.metadata['subopcode'], index))
                    elif statement.kind == 'call':
                        targets = [module.by_name[node.value]] if node.value in module.by_name else public.get(node.value, ())
                        calls.append((function, node, targets))
                    elif statement.kind == 'store':
                        require(function, analysis.origins(node.children[0], True), node.value[1], ('store',))
                    for child in node.walk():
                        if child.kind == 'load' and not child.value:
                            access = child.children[0]
                            value = access.value[1] if access.kind == 'located' else access.value
                            require(function, analysis.origins(access, True), value[-1], ('load',))
    for _ in range(64):
        changed = False
        for function, node, targets in calls:
            analysis = analyses[id(function)]
            for target in targets:
                for index, (begin, end) in tuple(summaries[id(target)].items()):
                    if index < len(node.children):
                        origins = analysis.shift(analysis.origins(node.children[index]), begin)
                        changed |= require(function, origins, end - begin, ('call', node.value, index))
        if not changed:
            break
    else:
        raise ValueError('Source memory extents do not converge through recursive calls')
    from .native_project import build_layout
    for module in modules:
        build_layout(module)
        required = []
        for offset, width, reason in sorted(extents[id(module)]):
            try:
                module.layout.at(offset, width)
            except ValueError:
                required.append((offset, width, reason))
        module.memory_extents = required
