"""Generation-time resolution of module calls and actual dynamic entry points."""
from collections import defaultdict
from dataclasses import dataclass

from .native_families import function_signature
from .native_project import identifier


@dataclass
class EntryCase:
    name: str
    module: object
    function: object
    owner: object
    external: bool


@dataclass
class ImportCall:
    name: str
    arguments: tuple
    result: int | None
    asynchronous: bool
    cpp_name: str
    providers: list


class Bindings:
    def __init__(self, families, public_types):
        self.families = families
        self.public_types = public_types
        self.modules = families.modules
        self.function_keys = {}
        for index, module in enumerate(self.modules):
            for key in families.module_methods[index]:
                item = families.implementations[key]
                self.function_keys[index, item.function.name, function_signature(item.function)] = key
        self.owners = {}
        for index, module in enumerate(self.modules):
            for function in module.functions:
                key = self.function_keys[index, function.name, function_signature(function)]
                owner = next(node for node in families.leaves[index].ancestors() if key in node.methods)
                self.owners[index, function.entry] = owner
        self.entries = defaultdict(list)
        names = set()
        grouped = defaultdict(list)
        self.public = defaultdict(list)
        for index, module in enumerate(self.modules):
            exposed = {}
            for program in module.script.programs:
                function = module.by_entry[program.start]
                exposed[program.name] = function, False
                cleanup = module.by_entry.get(program.end) if program.end != program.start else None
                if cleanup:
                    exposed.setdefault(cleanup.name, (cleanup, False))
            for symbol in module.script.functions:
                names.add(symbol.name)
                if not symbol.is_import and symbol.code_offset in module.by_entry:
                    function = module.by_entry[symbol.code_offset]
                    exposed[symbol.name] = function, True
                    self.public[symbol.name].append((index, module, function, self.owners[index, function.entry]))
            for name, (function, external) in exposed.items():
                names.add(name)
                names.add(function.name)
                owner = self.owners[index, function.entry]
                group = name, function.name, function_signature(function), function.allow_reentry, external, owner.index
                grouped[group].append((index, module, function, owner))
        self.names = [''] + sorted(names)
        self.ids = {name: index for index, name in enumerate(self.names)}
        used = set()
        self.enum = {'': 'None'}
        for name in self.names[1:]:
            stem = identifier(name)
            candidate = stem
            suffix = 2
            while candidate in used or candidate == 'None':
                candidate = stem + '_' + str(suffix)
                suffix += 1
            used.add(candidate)
            self.enum[name] = candidate
        for group, records in grouped.items():
            name, _, _, _, external, _ = group
            eligible = {row[0] for row in records}
            representative = records[0]
            for node in self.cover(representative[3], eligible):
                self.entries[node.index].append(EntryCase(name, representative[1], representative[2], node, external))
        self.imports = {}
        for module in self.modules:
            for function in module.functions:
                for block in function.blocks:
                    for statement in block.statements:
                        if statement.kind != 'call' or statement.expression.value in module.by_name:
                            continue
                        name = statement.expression.value
                        arguments = tuple(node.type_id for node in statement.expression.children)
                        key = name, arguments
                        if key in self.imports:
                            continue
                        providers = self.public.get(name, [])
                        actual = defaultdict(list)
                        for index, candidate, target, owner in providers:
                            actual[owner.index, target.name, function_signature(target)].append((index, candidate, target, owner))
                        choices = []
                        for rows in actual.values():
                            eligible = {row[0] for row in rows}
                            sample = rows[0]
                            for node in self.cover(sample[3], eligible):
                                choices.append((node, sample[1], sample[2]))
                        asynchronous = any(target.asynchronous for _, _, target in choices)
                        cpp_name = 'call_' + identifier(name) + '_' + str(sum(item.name == name for item in self.imports.values()) + 1)
                        self.imports[key] = ImportCall(name, arguments, public_types.get(name, -1), asynchronous, cpp_name, choices)

    def cover(self, node, eligible):
        if node.members <= eligible:
            return [node]
        result = []
        for child in node.children:
            if child.members & eligible:
                result.extend(self.cover(child, eligible))
        return result

    def entry(self, name):
        return 'Entry::' + self.enum[name]
