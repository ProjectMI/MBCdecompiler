"""Deduplicate recovered source bodies and place shared members once.

Equality includes scalar types, storage aliasing, control-flow edges, call
identities and effects. Only literal values are configurable between variants.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib

from .native_project import identifier
from .source_ir import Expression


def function_signature(function):
    return function.return_type, function.asynchronous, tuple(argument['type'] for argument in function.arguments), function.parameter_capacity


@dataclass
class Pattern:
    literals: list[Expression]
    members: list
    structure: tuple


def pattern(module, function, *, local_members=False, name=None, resolve_call=None):
    literals = []
    members = []
    member_indices = {}
    variables = {}
    blocks = {block.index: index for index, block in enumerate(function.blocks)}

    def member(offset):
        item = module.layout.at(offset)
        if item.canonical not in member_indices:
            member_indices[item.canonical] = len(members)
            members.append(item)
        return member_indices[item.canonical] if local_members else item.canonical, offset - item.offset, item.ctype

    def node(expression):
        if expression is None:
            return None
        if expression.kind == 'number':
            literals.append(expression)
            return 'literal', expression.type_id
        if expression.kind == 'name':
            name = expression.value
            if name in function.bindings:
                binding = function.bindings[name]
                access = tuple((key, binding[key]) for key in ('type', 'role', 'length', 'stride') if key in binding)
                shape = ('storage_name', member(binding['offset']), expression.type_id)
                return (*shape, access) if local_members else shape
            if name in function.variables:
                return 'variable', variables.setdefault(name, len(variables)), function.variables[name]
        value = resolve_call(expression.value) if resolve_call is not None and expression.kind == 'invocation' else expression.value
        if expression.kind == 'invocation' and expression.value in module.by_name:
            value = value, function_signature(module.by_name[expression.value])
        return expression.kind, value, expression.type_id, tuple(node(child) for child in expression.children)

    arguments = tuple((argument['type'], member(argument['data_offset'])) for argument in function.arguments)
    body = []
    for block in function.blocks:
        statements = []
        for statement in block.statements:
            output = variables.setdefault(statement.result, len(variables)) if statement.result else None
            metadata = dict(statement.metadata)
            if statement.kind == 'program':
                metadata['program_index'] = module.script.programs[metadata['program_index']].name
            statements.append((statement.kind, output, node(statement.expression), tuple(sorted(metadata.items()))))
        body.append((tuple(statements), tuple(blocks[target] for target in block.successors), node(block.condition), block.branch_when))
    return Pattern(literals, members, (name or function.name, function_signature(function), arguments, tuple(body)))


@dataclass
class Implementation:
    module: object
    function: object
    pattern: Pattern
    exact: bytes
    shape: bytes
    cpp_name: str


@dataclass
class Family:
    index: int
    name: str
    members: frozenset[int]
    parent: Family | None = None
    children: list[Family] = field(default_factory=list)
    methods: set[bytes] = field(default_factory=set)
    fields: dict = field(default_factory=dict)
    required: dict = field(default_factory=dict)
    leaf: bool = False

    def ancestors(self):
        node = self
        while node is not None:
            yield node
            node = node.parent


class Families:
    """Factor source algorithms independently of the module class hierarchy.

    Every source function retains its own binding. The emitter resolves local
    callees for each owner and specializes divergent targets at compile time.
    """
    def __init__(self, modules):
        self.modules = modules
        self.implementations = {}
        self.by_function = {}
        self.names = {}
        signatures = defaultdict(Counter)
        for module in modules:
            for function in module.functions:
                signatures[function.name][function_signature(function)] += 1
        for name, choices in signatures.items():
            for index, (signature, _) in enumerate(choices.most_common()):
                self.names[name, signature] = identifier(name) + (str(index + 1) if index else '')

        records = []
        indices = {}
        for index, module in enumerate(modules):
            for function in module.functions:
                source = pattern(module, function)
                shape = hashlib.sha256(repr(source.structure).encode()).digest()
                exact = hashlib.sha256(shape + repr([node.key() for node in source.literals]).encode()).digest()
                indices[index, function.entry] = len(records)
                records.append((index, function, source, shape, exact))
        # Refine only the compile-time call configuration. The source algorithm
        # remains shared even when a callee selects different literal settings.
        callees = []
        for module_index, function, _, _, _ in records:
            module = modules[module_index]
            callees.append(tuple(indices[module_index, module.by_name[statement.expression.value].entry]
                                 for block in function.blocks for statement in block.statements
                                 if statement.kind == 'call' and statement.expression.value in module.by_name))
        labels, identities = [], {}
        for _, _, _, _, exact in records:
            labels.append(identities.setdefault(exact, len(identities)))
        while True:
            refined, identities = [], {}
            for index, targets in enumerate(callees):
                key = labels[index], tuple(labels[target] for target in targets)
                refined.append(identities.setdefault(key, len(identities)))
            if len(identities) == len(set(labels)):
                labels = refined
                break
            labels = refined
        self.call_profiles = {(module_index, function.entry): tuple(labels[target] for target in callees[index])
                              for index, (module_index, function, _, _, _) in enumerate(records)}
        original_methods = [set() for _ in modules]
        for index, _, _, _, exact in records:
            original_methods[index].add(exact)
        # Retain whole owning fields, never slices. Metadata roots include all
        # relocations (also those in otherwise unused data), engine extents and
        # the position vector. The source layout remains intact for resolution.
        self.required_fields = [set() for _ in modules]
        for module_index, _, source, _, _ in records:
            self.required_fields[module_index].update(member.canonical for member in source.members)
        for module_index, module in enumerate(modules):
            required = self.required_fields[module_index]
            metadata = module.layout.metadata
            if metadata.position_offset:
                required.add(module.layout.at(metadata.position_offset, 12).canonical)
            for offset, width, _ in module.memory_extents:
                required.add(module.layout.at(offset, width).canonical)
            for offset in metadata.data_relocations:
                required.add(module.layout.at(offset, 4).canonical)
                target = int.from_bytes(module.script.data[offset:offset + 4], 'little')
                member = module.layout.fields[-1] if target == len(module.script.data) else module.layout.at(target)
                required.add(member.canonical)
        self.storage_groups = self._group_storage(original_methods)
        self.module_methods = [set() for _ in modules]
        for index, (module_index, function, source, original_shape, _) in enumerate(records):
            module = modules[module_index]
            shape = original_shape
            exact = hashlib.sha256(shape + repr([node.key() for node in source.literals]).encode()).digest()
            instance = Implementation(module, function, source, exact, shape, self.names[function.name, function_signature(function)])
            self.implementations.setdefault(exact, instance)
            self.by_function[module_index, function.entry] = instance
            self.module_methods[module_index].add(exact)
        self._nodes = [Family(index, module.cclass, frozenset({index}), methods=self.module_methods[index],
                              fields={member.canonical: member for member in module.layout.fields
                                      if member.canonical in self.required_fields[index]}, leaf=True)
                       for index, module in enumerate(modules)]
        self.leaves = {node.index: node for node in self._nodes}
        self.roots = self._nodes

    def _group_storage(self, methods):
        # This analysis tree is never emitted as C++ inheritance. It clusters
        # frequently co-used fields without allocating a universal script image.
        parents, memberships, leaves = [], [], {}
        def partition(members, parent, inherited):
            common = set.intersection(*(methods[index] for index in members)) - inherited
            if len(members) == 1 or common:
                current = len(parents)
                parents.append(parent)
                memberships.append(members)
                parent = current
                inherited = inherited | common
            if len(members) == 1:
                leaves[next(iter(members))] = parent
                return
            frequency = Counter(key for index in members for key in methods[index] - inherited)
            choices = [(count, key) for key, count in frequency.items() if 1 < count < len(members)]
            if not choices:
                for index in sorted(members):
                    partition(frozenset({index}), parent, inherited)
                return
            _, chosen = max(choices)
            group = frozenset(index for index in members if chosen in methods[index])
            partition(group, parent, inherited)
            partition(members - group, parent, inherited)
        partition(frozenset(range(len(self.modules))), None, set())
        paths = {}
        for index, leaf in leaves.items():
            path = []
            while leaf is not None:
                path.append(leaf)
                leaf = parents[leaf]
            paths[index] = path
        uses = defaultdict(set)
        fields = {}
        for index, module in enumerate(self.modules):
            for member in module.layout.fields:
                if member.canonical not in self.required_fields[index]:
                    continue
                uses[member.canonical].add(index)
                fields.setdefault(member.canonical, member)
        grouped = defaultdict(list)
        for identity, owners in uses.items():
            common = set.intersection(*(set(paths[index]) for index in owners))
            # Unrelated trees keep this field explicit rather than duplicating
            # a state type or inventing a common base with unrelated data.
            owner = next((node for node in paths[min(owners)] if node in common), None)
            group = memberships[owner] if owner is not None else frozenset(owners)
            grouped[group].append(fields[identity])
        return grouped

    def nodes(self):
        return iter(self._nodes)

    @staticmethod
    def lca(nodes):
        nodes = list(nodes)
        if not nodes:
            return None
        return nodes[0] if all(node is nodes[0] for node in nodes) else None
