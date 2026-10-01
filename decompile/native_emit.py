"""Emit native algorithms, owned module data and directly bound entry objects.

Local source calls are resolved before factoring. Shared data uses composition;
there is no generated inheritance tree, per-function forwarding method or global
module catalogue. Scheduler recipes retain only source scheduling information.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import struct
import hashlib
import re
import zlib

from .native_bindings import Bindings
from .native_outline import outline_bodies
from .native_loops import roll_constant_sequences
from .native_cpp import CppGenerator, Renderer, Helper, coerce, parameters, quoted, result_type
from .native_families import function_signature, pattern
from .native_project import VOID, cpp_type


@dataclass
class Storage:
    index: int
    fields: list
    owners: frozenset
    name: str
    variable: str
    shared: bool

    @property
    def ctype(self):
        return self.name if self.shared else self.fields[0].ctype


class NativeRenderer(Renderer):
    def member_expression(self, member):
        if self.helper.name in self.generator.reference_helpers:
            index = next(index for index, field in enumerate(self.helper.example.pattern.members)
                         if field.canonical == member.canonical)
            return f'field{index + 1}'
        if self.generator.object_context(self.helper):
            return self.generator.object_member(member)
        unit = self.generator.storage_for[member.canonical]
        return f'{unit.variable}.{member.name}' if unit.shared else unit.variable

    def call(self, statement):
        node = statement.expression
        if statement.kind != 'call':
            return super().call(statement)
        target = self.module.by_name.get(node.value)
        if target is not None:
            args = [self.expr(child) for child in node.children]
            count, capacity = len(args), abs(target.parameter_capacity)
            if count > capacity or (target.parameter_capacity >= 0 and count != capacity):
                return f'invalidArguments<{cpp_type(target.return_type)}>()'
            converted = [coerce(args[index] if index < count else '0', argument['type'],
                                self.typ(node.children[index]) if index < count else 16, parameter=True)
                         for index, argument in enumerate(target.arguments)]
            call = self.generator.local_call(self.helper, target, converted)
            return f'co_await std::move({call}).in(self, {quoted(target.name)})' if target.asynchronous else call
        plan = self.generator.bindings.imports[node.value, tuple(child.type_id for child in node.children)]
        args = ', '.join(self.expr(child) for child in node.children)
        call = f'ScriptHelpers::{plan.cpp_name}(self{", " if args else ""}{args})'
        return 'co_await ' + call if plan.asynchronous else call


class SequencePool:
    """Share repeated metadata spans, keeping ordering and duplicate entries."""
    def __init__(self, typ, name):
        self.typ, self.name = typ, name
        self.rows = []
        self.chunks = {}
        self.segments = []
        self.recipes = {}

    def intern(self, rows):
        if not rows:
            return '{}'
        chunks = []
        current = []
        rolling = 0
        for row in rows:
            current.append(row)
            rolling = ((rolling << 1) + zlib.crc32(row.encode())) & 0xffffffff
            if len(current) >= 24 or (len(current) >= 6 and rolling & 7 == 0):
                chunks.append(tuple(current))
                current = []
                rolling = 0
        if current:
            chunks.append(tuple(current))
        recipe = []
        for chunk in chunks:
            interval = self.chunks.get(chunk)
            if interval is None:
                interval = len(self.rows), len(chunk)
                self.rows.extend(chunk)
                self.chunks[chunk] = interval
            recipe.append(interval)
        key = tuple(recipe)
        if key not in self.recipes:
            self.recipes[key] = len(self.segments), len(recipe)
            self.segments.extend(recipe)
        offset, count = self.recipes[key]
        return f'{{{self.name}Segments.data() + {offset}, {count}}}'

    def emit(self):
        if not self.rows:
            return []
        result = [f'static constexpr std::array<{self.typ}, {len(self.rows)}> {self.name}Data{{{{']
        result += ['    ' + row + ',' for row in self.rows]
        result += ['}};', '', f'static constexpr std::array<std::span<const {self.typ}>, {len(self.segments)}> {self.name}Segments{{{{']
        result += [f'    {{{self.name}Data.data() + {offset}, {count}}},' for offset, count in self.segments]
        return result + ['}};', '']


class NativeGenerator(CppGenerator):
    def __init__(self, families, public_types, client_root):
        super().__init__(families, public_types, client_root)
        self.module_indices = {module.name: index for index, module in enumerate(self.modules)}
        # These algorithms differ only in the identity of their owned data.
        # Explicit field references preserve aliases and avoid one function body
        # (or one owner-template instantiation) per otherwise equivalent layout.
        self.reference_helpers = {
            helper.name for helper in self.helpers
            if len({tuple(field.canonical for field in instance.pattern.members)
                    for instance in helper.instances}) > 1
        }
        self.field_templates = {
            helper.name for helper in self.helpers if helper.name in self.reference_helpers
            and any(statement.kind == 'call' and statement.expression.value in helper.example.module.by_name
                    for block in helper.example.function.blocks for statement in block.statements)
        }
        self.bindings = Bindings(families, public_types)
        usage = defaultdict(set)
        for (index, _), instance in families.by_function.items():
            usage[self.helper_for[instance.exact].name].add(index)
        self.inline_owners = {helper.name: next(iter(usage[helper.name])) for helper in self.helpers
                              if len(usage[helper.name]) == 1 and len(helper.instances) == 1}
        self.initializer_fields = {}
        self.program_pool = SequencePool('Program', 'program')
        self.name_pool = SequencePool('Entry', 'functionName')
        self.storage, self.storage_for = self.factor_storage()
        self.storage_defaults = {}
        for unit in self.storage:
            defaults = {}
            for field in unit.fields:
                choices = Counter(self.sanitized_initial(self.modules[index], candidate)
                                  for index in sorted(unit.owners) for candidate in self.modules[index].layout.fields
                                  if candidate.canonical == field.canonical)
                defaults[field.canonical] = choices.most_common(1)[0][0]
            self.storage_defaults[unit.index] = defaults
        self.requirements = {
            helper.name: set() if helper.name in self.reference_helpers else
            {self.storage_for[field.canonical].index for field in helper.example.pattern.members}
            for helper in self.helpers
        }
        dependencies = defaultdict(set)
        self.local_variants = defaultdict(lambda: defaultdict(dict))
        for (index, _), instance in families.by_function.items():
            helper = self.helper_for[instance.exact]
            module = self.modules[index]
            for block in instance.function.blocks:
                for statement in block.statements:
                    if statement.kind != 'call' or statement.expression.value not in module.by_name:
                        continue
                    name = statement.expression.value
                    target = self.instance(module, module.by_name[name])
                    self.local_variants[helper.name][name][index] = target
                    dependencies[helper.name].add(self.helper_for[target.exact].name)
        changed = True
        while changed:
            before = len(self.inline_owners)
            for name, targets in dependencies.items():
                if name not in self.inline_owners:
                    for target in targets:
                        self.inline_owners.pop(target, None)
            changed = before != len(self.inline_owners)
        self.representatives = defaultdict(dict)
        for (index, _), instance in families.by_function.items():
            self.representatives[self.helper_for[instance.exact].name][index] = instance
        self.profiles = {}
        self.variant_helpers = set()
        for helper in self.helpers:
            configurations, profiles = {}, {}
            for index in sorted(usage[helper.name]):
                instance = self.representatives[helper.name][index]
                profile = families.call_profiles[index, instance.function.entry]
                profiles[index] = configurations.setdefault(profile, len(configurations))
            self.profiles[helper.name] = profiles
            if len(configurations) > 1 and helper.name not in self.inline_owners:
                self.variant_helpers.add(helper.name)
        # Track the actual transitive state of each compile-time configuration,
        # rather than the union of unrelated module implementations.
        needed = {}
        children = {}
        for key, instance in families.by_function.items():
            helper = self.helper_for[instance.exact]
            configuration = helper.name, self.profiles[helper.name][key[0]]
            needed.setdefault(configuration, set(self.requirements[helper.name]))
            targets = children.setdefault(configuration, set())
            for block in instance.function.blocks:
                for statement in block.statements:
                    if statement.kind != 'call' or statement.expression.value not in instance.module.by_name:
                        continue
                    target = self.instance(instance.module, instance.module.by_name[statement.expression.value])
                    callee = self.helper_for[target.exact]
                    if callee.name in self.reference_helpers:
                        needed[configuration].update(self.storage_for[field.canonical].index
                                                     for field in target.pattern.members)
                    targets.add((callee.name, self.profiles[callee.name][key[0]]))
        changed = True
        while changed:
            changed = False
            for configuration, targets in children.items():
                before = len(needed[configuration])
                for target in targets:
                    needed[configuration].update(needed[target])
                changed |= len(needed[configuration]) != before
        self.view_units = {}
        self.own_units = {}
        self.template_helpers = set()
        for helper in self.helpers:
            own = self.requirements[helper.name]
            self.own_units[helper.name] = sorted(own)
            choices = set(self.profiles[helper.name].values())
            for profile in choices:
                units = needed[helper.name, profile]
                self.view_units[helper.name, profile] = sorted(own) + sorted(units - own)
            all_units = set().union(*(needed[helper.name, profile] for profile in choices))
            self.requirements[helper.name] = all_units
            if helper.name not in self.inline_owners and any(
                    needed[helper.name, profile] != all_units for profile in choices):
                self.template_helpers.add(helper.name)
        self.template_helpers.update(self.field_templates)
        self.variant_helpers.difference_update(self.template_helpers)
        views = set()
        for index, cases in self.bindings.entries.items():
            for case in cases:
                instance = self.instance(case.module, case.function)
                helper = self.helper_for[instance.exact]
                if helper.name in self.template_helpers:
                    views.add(self.tuple_type(helper, index))
        self.state_views = {value: f'StateView{index + 1}' for index, value in enumerate(sorted(views))}

    def helper_shape(self, implementation):
        function, module = implementation.function, implementation.module
        source = pattern(module, function, local_members=True)
        return hashlib.sha256(b'field-references:' + repr(source.structure).encode()).digest()

    def tuple_type(self, helper, index):
        fields = ([member.ctype + ' &' for member in self.representatives[helper.name][index].pattern.members]
                  if helper.name in self.field_templates else [])
        fields.extend(unit.ctype + ' &' for unit in self.selected_units(helper, index))
        return 'std::tuple<' + ', '.join(fields) + '>'

    def profile(self, instance):
        helper = self.helper_for[instance.exact]
        return self.profiles[helper.name][self.module_indices[instance.module.name]]

    def selected_units(self, helper, index):
        identities = (self.view_units[helper.name, self.profiles[helper.name][index]]
                      if helper.name in self.template_helpers else sorted(self.requirements[helper.name]))
        return [self.storage[identity] for identity in identities]

    def unit_expression(self, helper, index, unit):
        if helper.name in self.inline_owners:
            return self.object_unit(unit)
        if helper.name in self.template_helpers:
            order = self.view_units[helper.name, self.profiles[helper.name][index]]
            offset = len(helper.example.pattern.members) if helper.name in self.field_templates else 0
            return f'std::get<{offset + order.index(unit.index)}>(state)'
        return unit.variable

    def object_context(self, helper):
        return helper.name in self.inline_owners

    def instance(self, module, function):
        return self.families.by_function[self.module_indices[module.name], function.entry]

    def factor_storage(self):
        groups = self.families.storage_groups
        storage, by_field = [], {}
        shared_count = 0
        for owners, members in sorted(groups.items(), key=lambda item: (-len(item[0]), min(field.canonical for field in item[1]))):
            members.sort(key=lambda field: field.canonical)
            shared = len(owners) > 1 and len(members) > 2
            if shared:
                shared_count += 1
                units = [Storage(len(storage), members, owners, f'ScriptState{shared_count}', f'state{shared_count}', True)]
            else:
                units = [Storage(len(storage) + offset, [member], owners, '', member.name, False)
                         for offset, member in enumerate(members)]
            for unit in units:
                storage.append(unit)
                for member in unit.fields:
                    by_field[member.canonical] = unit
        return storage, by_field

    def units_for(self, helper):
        return [self.storage[index] for index in sorted(self.requirements[helper.name])]

    @staticmethod
    def object_unit(unit, owner='self'):
        return f'{owner}.{unit.variable}_' if unit.shared else f'{owner}.{unit.variable}'

    def object_member(self, member, owner='self'):
        unit = self.storage_for[member.canonical]
        base = self.object_unit(unit, owner)
        return f'{base}.{member.name}' if unit.shared else base

    def helper_parameters(self, helper):
        values = ['Module &self']
        if helper.name in self.reference_helpers and helper.name not in self.field_templates:
            values.extend(f'{member.ctype} &field{index + 1}'
                          for index, member in enumerate(helper.example.pattern.members))
        if helper.name in self.template_helpers:
            values.append('State state')
        if helper.name not in self.template_helpers:
            values.extend(f'{unit.ctype} &{unit.variable}' for unit in self.units_for(helper))
        if parameters(helper.example.function):
            values.append(parameters(helper.example.function))
        values.extend(f'{cpp_type(typ)} setting{index + 1}' for index, typ in enumerate(helper.literal_types))
        return ', '.join(values)

    def literal_arguments(self, helper, instance):
        renderer = Renderer(self, Helper('literal', instance, [instance]))
        return [renderer.expr(instance.pattern.literals[positions[0]]) for positions in helper.literal_parameters]

    def call_expression(self, instance, caller, arguments):
        helper = self.helper_for[instance.exact]
        index = self.module_indices[instance.module.name]
        if helper.name in self.inline_owners:
            if caller.name not in self.inline_owners:
                raise ValueError(f'Shared helper calls module-specific method {helper.name}')
            return f'self.{instance.cpp_name}({", ".join(arguments)})'
        if helper.name in self.reference_helpers:
            fields = []
            for member in instance.pattern.members:
                unit = self.storage_for[member.canonical]
                source = self.unit_expression(caller, index, unit)
                fields.append(f'{source}.{member.name}' if unit.shared else source)
            if helper.name in self.field_templates:
                fields.extend(self.unit_expression(caller, index, unit) for unit in self.selected_units(helper, index))
        else:
            fields = [self.unit_expression(caller, index, unit) for unit in self.selected_units(helper, index)]
        values = ['self']
        if helper.name in self.template_helpers:
            values.append(f'std::tie({", ".join(fields)})')
        else:
            values.extend(fields)
        values.extend(arguments)
        values.extend(self.literal_arguments(helper, instance))
        specialization = f'<{self.profile(instance)}>' if helper.name in self.variant_helpers | self.template_helpers else ''
        return f'ScriptHelpers::{helper.name}{specialization}({", ".join(values)})'

    def local_call(self, helper, target, arguments):
        variants = self.local_variants[helper.name][target.name]
        expressions = defaultdict(list)
        for index, instance in variants.items():
            expression = self.call_expression(instance, helper, arguments)
            expressions[expression].append(index)
        resolved = {}
        for expression, owners in expressions.items():
            for index in owners:
                profile = self.profiles[helper.name][index]
                if profile in resolved and resolved[profile] != expression:
                    raise ValueError(f'Conflicting callee configuration in {helper.name}.{target.name}')
                resolved[profile] = expression
        if len(expressions) == 1:
            return next(iter(expressions))
        if helper.name not in self.template_helpers | self.variant_helpers:
            raise ValueError(f'Unresolved local specialization: {helper.name}.{target.name}')
        choices = sorted(expressions.items(), key=lambda item: (len(item[1]), item[1]))
        lines = [f'[&]() -> {result_type(target)} {{']
        for expression, owners in choices[:-1]:
            profiles = sorted({self.profiles[helper.name][index] for index in owners})
            condition = ' || '.join(f'Variant == {index}' for index in profiles)
            lines += [f'    if constexpr ({condition})', f'        return {expression};', '    else']
        lines += [f'        return {choices[-1][0]};', '}()']
        return '\n'.join(lines)

    def settings_bits(self, helper, instance):
        values = []
        for typ, positions in zip(helper.literal_types, helper.literal_parameters):
            text = str(instance.pattern.literals[positions[0]].value)
            if typ == 32:
                value = struct.unpack('<I', struct.pack('<f', float(text)))[0]
            elif typ in {0, 16}:
                value = (int(text, 16) if text.lower().startswith('0x') else int(text)) & 0xffffffff
            else:
                raise ValueError(f'Non-numeric helper setting: {helper.name}: {typ}')
            values.append(f'{value}u')
        return values

    def binding_line(self, case, owner_type='Self'):
        instance = self.instance(case.module, case.function)
        helper = self.helper_for[instance.exact]
        if helper.name in self.inline_owners:
            fields = []
            function = f'&{owner_type}::{instance.cpp_name}'
        else:
            index = self.module_indices[instance.module.name]
            units = self.selected_units(helper, index)
            fields = ([self.object_member(member) for member in instance.pattern.members]
                      if helper.name in self.reference_helpers else [self.object_unit(unit) for unit in units])
            if helper.name in self.field_templates:
                fields.extend(self.object_unit(unit) for unit in units)
            if helper.name in self.template_helpers:
                specialization = f'<{self.profile(instance)}, {self.state_views[self.tuple_type(helper, index)]}>'
            else:
                specialization = f'<{self.profile(instance)}>' if helper.name in self.variant_helpers else ''
            function = f'&ScriptHelpers::{helper.name}{specialization}'
        state = f'std::tie({", ".join(fields)})' if fields else 'std::tuple{}'
        settings = self.settings_bits(helper, instance)
        arguments = ['self', self.bindings.entry(case.name), function]
        optional = [state, str(case.external).lower(), str(case.function.parameter_capacity),
                    str(case.function.allow_reentry).lower(), quoted(case.function.name)]
        defaults = ['std::tuple{}', 'false', str(len(case.function.arguments)), 'false', quoted(case.name)]
        if settings:
            optional.append(f'std::array<std::uint32_t, {len(settings)}>' + '{' + ', '.join(settings) + '}')
            defaults.append('{}')
        while optional and optional[-1] == defaults[len(optional) - 1]:
            optional.pop()
        return f'bindNative({", ".join(arguments + optional)});'

    @staticmethod
    def normalize_settings(lines):
        constants = []
        def replace(match):
            values = match[2].split(', ')
            constants.extend(values)
            return match[1] + ', '.join('$constant' for _ in values) + '}'
        pattern = r'(std::array<std::uint32_t, \d+>\{)([0-9u, ]+)\}'
        return tuple(re.sub(pattern, replace, line) for line in lines), tuple(constants)

    @staticmethod
    def substitute_settings(lines, values):
        positions = iter(values)
        return [re.sub(r'\$constant', lambda _: next(positions), line) for line in lines]

    def initialization_block(self, name, lines, variants):
        substitutions, live, identities = [], [], {}
        for position, column in enumerate(zip(*variants)):
            if len(set(column)) == 1:
                substitutions.append(column[0])
            else:
                if column not in identities:
                    identities[column] = f'constant{len(live) + 1}'
                    live.append(position)
                substitutions.append(identities[column])
        body = self.substitute_settings(lines, substitutions)
        # Most construction blocks only use known state objects, not the module
        # class. Pass those objects directly instead of instantiating Self again
        # for every owner. Member-function pointers still require a typed owner.
        dependent = any(re.search(r'\bSelf\b', line) for line in body)
        fields = []
        if not dependent:
            units = {self.object_unit(unit): unit for unit in self.storage}
            for line in body:
                for match in re.finditer(r'\bself\.(?:state\d+_|member\d+_)\b', line):
                    unit = units[match[0]]
                    if unit not in fields:
                        fields.append(unit)
            replacements = {self.object_unit(unit): unit.variable for unit in fields}
            body = [re.sub(r'\bself\.(?:state\d+_|member\d+_)\b',
                           lambda match: replacements[match[0]], line) for line in body]
        self.initializer_fields[name] = fields
        template = 'template <class Self> ' if dependent else ''
        parameters = ', '.join(['Self &self' if dependent else 'Module &self'] +
                               [f'{unit.ctype} &{unit.variable}' for unit in fields] +
                               [f'std::uint32_t constant{number + 1}' for number in range(len(live))])
        declaration = f'    {template}static void {name}({parameters});'
        definition = [f'{template}void ScriptHelpers::{name}({parameters})', '{',
                      *['    ' + line for line in body], '}', '']
        return declaration, definition, live

    def initialization_call(self, name, live, values):
        fields = ''.join(', ' + self.object_unit(unit) for unit in self.initializer_fields[name])
        settings = ''.join(', ' + values[position] for position in live)
        return f'ScriptHelpers::{name}(self{fields}{settings});'

    def factor_bindings(self):
        rows = defaultdict(dict)
        for index in range(len(self.modules)):
            for case in self.bindings.entries[index]:
                normalized, constants = self.normalize_settings([self.binding_line(case)])
                rows[normalized[0]][index] = constants
        groups = defaultdict(list)
        for line, owners in rows.items():
            groups[frozenset(owners)].append(line)
        declarations, definitions, result = [], [], defaultdict(list)
        count = 0
        for owners, lines in sorted(groups.items(), key=lambda item: (-len(item[0]), min(item[0]))):
            lines.sort()
            ordered = sorted(owners)
            variants = [tuple(value for line in lines for value in rows[line][index]) for index in ordered]
            if len(owners) > 1 and len(lines) >= 2:
                count += 1
                name = f'bindEntries{count}'
                declaration, definition, live = self.initialization_block(name, lines, variants)
                declarations.append(declaration)
                definitions.extend(definition)
                for index, values in zip(ordered, variants):
                    result[index].append(self.initialization_call(name, live, values))
            else:
                for index, values in zip(ordered, variants):
                    result[index].extend(self.substitute_settings(lines, values))
        return declarations, definitions, result, count

    def factor_initializers(self, initializers):
        """Share flat construction blocks; shared blocks never call each other."""
        recipes, variants = [], defaultdict(list)
        for lines in initializers:
            chunks, current, constants, rolling = [], [], [], 0
            for line in lines:
                # Keep already shared entry initialization at the module level.
                if line.startswith('ScriptHelpers::bindEntries'):
                    if current:
                        chunk, values = tuple(current), tuple(constants)
                        chunks.append((chunk, values))
                        variants[chunk].append(values)
                    chunks.append(((line,), ()))
                    variants[(line,)].append(())
                    current, constants, rolling = [], [], 0
                    continue
                normalized, values = self.normalize_settings([line])
                current.extend(normalized)
                constants.extend(values)
                rolling = ((rolling << 1) + zlib.crc32(normalized[0].encode())) & 0xffffffff
                if len(current) >= 20 or (len(current) >= 4 and rolling & 7 == 0):
                    chunk, values = tuple(current), tuple(constants)
                    chunks.append((chunk, values))
                    variants[chunk].append(values)
                    current, constants, rolling = [], [], 0
            if current:
                chunk, values = tuple(current), tuple(constants)
                chunks.append((chunk, values))
                variants[chunk].append(values)
            recipes.append(chunks)
        declarations, definitions, rewritten = [], [], []
        names = {}
        for index, chunks in enumerate(recipes):
            lines = []
            for chunk, values in chunks:
                if len(variants[chunk]) > 1 and len(chunk) >= 3:
                    if chunk not in names:
                        name = f'initializePart{len(names) + 1}'
                        declaration, definition, live = self.initialization_block(name, chunk, variants[chunk])
                        declarations.append(declaration)
                        definitions.extend(definition)
                        names[chunk] = name, live
                    name, live = names[chunk]
                    lines.append(self.initialization_call(name, live, values))
                else:
                    lines.extend(re.sub(r'\bSelf\b', self.modules[index].cclass, line) for line in self.substitute_settings(chunk, values))
            rewritten.append(lines)
        return declarations, definitions, rewritten, len(names)

    def import_declaration(self, plan):
        args = ', '.join(f'{cpp_type(typ)} argument{index + 1}' for index, typ in enumerate(plan.arguments))
        result = f'Task<{cpp_type(plan.result)}>' if plan.asynchronous else cpp_type(plan.result)
        return f'{result} {plan.cpp_name}(Module &self{", " if args else ""}{args})'

    def emit_import(self, plan):
        signature = self.import_declaration(plan).replace(plan.cpp_name + '(', 'ScriptHelpers::' + plan.cpp_name + '(', 1)
        lines = [signature, '{']
        if not plan.providers:
            return lines + ['    throw std::runtime_error(' + quoted('No native implementation of script function ' + plan.name) + ');', '}', '']
        key = self.bindings.entry(plan.name)
        lines += [f'    auto &target = self.host().selectModule(self, provides<{key}>);',
                  f'    const auto method = target.entry({key}, true);']
        values = ', '.join(f'Argument(argument{index + 1})' for index in range(len(plan.arguments)))
        if plan.asynchronous:
            call = f'co_await method.start(std::vector<Argument>{{{values}}})'
            keyword = 'co_return'
        else:
            lines.append(f'    const std::array<Argument, {len(plan.arguments)}> arguments{{{values}}};')
            call = 'method.invoke(arguments)'
            keyword = 'return'
        if plan.result == VOID:
            lines.extend(['    ' + call + ';', f'    {keyword};'])
        else:
            lines.append(f'    {keyword} storedValue<{cpp_type(plan.result)}>({call});')
        return lines + ['}', '']

    def module_metadata(self, module):
        entry = self.bindings.entry
        programs = []
        for program in module.script.programs:
            cleanup = module.by_entry.get(program.end) if program.end != program.start else None
            programs.append('{' + f'{entry(program.name)}, {program.state}, {program.queue_id}, {entry(cleanup.name if cleanup else "")}' + '}')
        recipe = self.program_pool.intern(programs)
        names = self.name_pool.intern([entry(symbol.name) for symbol in module.script.functions])
        data, cursor, regions = module.layout.metadata.regions, 0, []
        while cursor < len(data):
            if data[cursor] != 127:
                cursor += 1
                continue
            if cursor + 5 > len(data):
                raise ValueError('Truncated native region definition')
            index, flags, program = data[cursor + 1], struct.unpack_from('b', data, cursor + 2)[0], struct.unpack_from('<H', data, cursor + 3)[0]
            cursor += 5
            formats = []
            while cursor < len(data) and data[cursor] != 127:
                if len(formats) < 28:
                    formats.append(str(struct.unpack_from('b', data, cursor)[0]))
                cursor += 1
            if index >= 62:
                continue
            format_pool = self.metadata_pool('std::int8_t', formats)
            target = module.script.programs[program].name if program < len(module.script.programs) else ''
            regions.append('{' + f'{index}, {flags}, {entry(target)}, {format_pool}' + '}')
        region_pool = self.metadata_pool('RegionDefinition', regions)
        lines = [f'self.bindPrograms({recipe}, {names}, {region_pool});']
        for index, value in enumerate(module.layout.metadata.function_map):
            if value < len(module.script.functions):
                symbol = module.script.functions[value]
                target = entry(symbol.name) if not symbol.is_import else 'Entry::None'
                lines.append(f'self.bindCallback({index}, {target});')
        return lines

    def fields_for(self, module):
        required = self.families.required_fields[self.module_indices[module.name]]
        return (member for member in module.layout.fields if member.canonical in required)

    def initialize_module(self, module):
        body = []
        for member in self.fields_for(module):
            unit = self.storage_for[member.canonical]
            initial = self.sanitized_initial(module, member)
            default = self.storage_defaults[unit.index][member.canonical] if unit.shared else bytes(member.size)
            if initial != default:
                value = self.initial_expression(member, initial)
                body.append(f'{self.object_member(member)} = {value};')
        for offset in module.layout.metadata.data_relocations:
            destination = module.layout.at(offset, 4)
            target = struct.unpack_from('<I', module.script.data, offset)[0]
            if target == len(module.script.data):
                source, displacement = module.layout.fields[-1], module.layout.fields[-1].size
            else:
                source = module.layout.at(target)
                displacement = target - source.offset
            address = f'self.address({self.object_member(source)}, {displacement}, 0).base'
            body.append(f'setMemberView({self.object_member(destination)}, {offset - destination.offset}, {address});')
        return body

    def emit_names(self):
        lines = [f'static constexpr std::array<std::string_view, {len(self.bindings.names)}> entryNames{{{{']
        lines += ['    ' + quoted(name) + ',' for name in self.bindings.names]
        lines += ['}};', '', 'std::string_view entryName(Entry entry) noexcept', '{',
                  '    const auto index = std::size_t(entry);', '    return index < entryNames.size() ? entryNames[index] : std::string_view{};', '}', '',
                  'Entry entryNamed(std::string_view name) noexcept', '{',
                  '    const auto found = std::lower_bound(entryNames.begin(), entryNames.end(), name);',
                  '    return found != entryNames.end() && *found == name ? Entry(found - entryNames.begin()) : Entry::None;', '}', '',
                  'std::span<const std::string_view> moduleNames() noexcept', '{',
                  f'    static constexpr std::array<std::string_view, {len(self.modules)}> names{{{{']
        lines += ['        ' + quoted(module.name) + ',' for module in self.modules]
        lines += ['    }};', '    return names;', '}', '', 'std::uint32_t moduleTag(std::string_view name) noexcept', '{',
                  '    const auto names = moduleNames();',
                  f'    static constexpr std::array<std::uint32_t, {len(self.modules)}> tags{{{{']
        lines += ['        ' + str(module.script.header.module_tag) + 'u,' for module in self.modules]
        lines += ['    }};', '    for (std::size_t index = 0; index < names.size(); ++index)',
                  '        if (moduleNameEqual(names[index], name))', '            return tags[index];', '    return UINT32_MAX;', '}', '',
                  'std::string_view moduleName(std::uint32_t tag) noexcept', '{', '    switch (tag)', '    {']
        seen = set()
        for module in self.modules:
            tag = module.script.header.module_tag
            if tag not in seen:
                lines.append(f'    case {tag}u: return {quoted(module.name)};')
                seen.add(tag)
        lines += ['    default: return {};', '    }', '}', '',
                  'std::shared_ptr<Module> createModule(Host &host, std::string_view name)', '{']
        # Names, not tags, are the factory identity. Source tags need not be unique.
        for module in self.modules:
            lines.append(f'    if (moduleNameEqual(name, {quoted(module.name)})) return std::make_shared<{module.cclass}>(host);')
        return lines + ['    throw std::invalid_argument("Unknown native script module: " + std::string(name));', '}', '']

    def simplify_module_receivers(self, source):
        """Member definitions use their own fields without a redundant self alias."""
        owners = '|'.join(re.escape(module.cclass) for module in self.modules)
        signature = re.compile(r'\b(?:' + owners + r')::\w+\(')
        token = re.compile(r'"(?:[^"\\]|\\.)*"|//[^\n]*|\(\*this\)\.|\bself\.|\bself\b')
        output, active, removed = [], False, 0
        for line in '\n'.join(source).split('\n'):
            if signature.search(line) and not line.startswith((' ', '\t')):
                active = True
            if active:
                if line.strip() == 'auto &self = *this;':
                    removed += 1
                    continue
                line = token.sub(lambda match: '' if match[0] in {'self.', '(*this).'} else
                                 '*this' if match[0] == 'self' else match[0], line)
                if line == '}':
                    active = False
            output.append(line)
        self.removed_receiver_aliases = removed
        return output

    def inline_initializers(self, source):
        """Construct single-use constants at their destination, not via a pool."""
        token = re.compile(r'"(?:[^"\\]|\\.)*"|//[^\n]*|\binitialData\d+\b')
        text = '\n'.join(source)
        uses = Counter(match[0] for match in token.finditer(text) if match[0].startswith('initialData'))
        retained, replacements = [], {}
        for definition in self.header_extra:
            match = re.fullmatch(r'inline const .+ (initialData\d+) = (initialMember<[^\n]+);', definition)
            if match and uses[match[1]] == 1:
                replacements[match[1]] = match[2]
            else:
                retained.append(definition.replace('inline const ', 'static const ', 1))
        self.inline_initial_data_count = len(replacements)
        source = token.sub(lambda match: replacements.get(match[0], match[0]), text).split('\n')
        return retained, source

    def write(self, destination: Path):
        destination.mkdir(parents=True, exist_ok=True)
        header = ['#pragma once', '', '#include "script/MbcNative.h"', '', 'namespace SphereScripts', '{', 'class ScriptHelpers;', '',
                  'enum class Entry : std::uint16_t', '{']
        header += ['    ' + self.bindings.enum[name] + ',' for name in self.bindings.names]
        header += ['};', '']
        constructors = []
        for unit in self.storage:
            if not unit.shared:
                continue
            initializers = []
            for member in unit.fields:
                data = self.storage_defaults[unit.index][member.canonical]
                if any(data):
                    initializers.append(f'{member.name}({self.initial_expression(member, data)})')
            header += [f'struct {unit.name}', '{']
            if initializers:
                header.append(f'    {unit.name}();')
                constructors += [f'{unit.name}::{unit.name}() :', '    ' + ',\n    '.join(initializers), '{', '}', '']
            header += [f'    {member.ctype} {member.name}{{}};' for member in unit.fields]
            header += ['};', '']
        for index, module in enumerate(self.modules):
            header += [f'class {module.cclass} final : public Module', '{', '    friend class ScriptHelpers;', '  public:',
                       f'    explicit {module.cclass}(Host &host);', '    void initializeMembers() override;']
            if module.layout.metadata.position_offset:
                header.append('    Address position() override;')
            header += ['', '  private:']
            for helper in self.helpers:
                if self.inline_owners.get(helper.name) == index:
                    function = helper.example.function
                    header.append(f'    {result_type(function)} {helper.example.cpp_name}({parameters(function)});')
            units = {self.storage_for[field.canonical].index for field in self.fields_for(module)}
            for identity in sorted(units):
                unit = self.storage[identity]
                header.append(f'    {unit.ctype} {unit.variable}{"_" if unit.shared else ""}{{}};')
            header += ['};', '']
        header += ['std::shared_ptr<Module> createModule(Host &host, std::string_view name);',
                   'std::span<const std::string_view> moduleNames() noexcept;',
                   'std::string_view moduleName(std::uint32_t tag) noexcept;',
                   'std::uint32_t moduleTag(std::string_view name) noexcept;', '}', '']
        declarations, bodies = [], []
        for helper in self.helpers:
            function = helper.example.function
            lines = NativeRenderer(self, helper).body()
            if helper.name in self.template_helpers:
                if helper.name in self.field_templates:
                    aliases = [f'auto &field{index + 1} = std::get<{index}>(state);'
                               for index in range(len(helper.example.pattern.members))]
                else:
                    aliases = [f'auto &{self.storage[identity].variable} = std::get<{index}>(state);'
                               for index, identity in enumerate(self.own_units[helper.name])]
                lines[1:1] = aliases
            if helper.name in self.inline_owners:
                owner = self.modules[self.inline_owners[helper.name]].cclass
                signature = f'{result_type(function)} {owner}::{helper.example.cpp_name}({parameters(function)})'
                lines.insert(0, 'auto &self = *this;')
            else:
                signature = f'{result_type(function)} {helper.name}({self.helper_parameters(helper)})'
                template = ('template <std::size_t Variant, class State> ' if helper.name in self.template_helpers else
                            'template <std::size_t Variant> ' if helper.name in self.variant_helpers else '')
                declarations.append('    ' + template + 'static ' + signature + ';')
                signature = template + signature.replace(helper.name + '(', 'ScriptHelpers::' + helper.name + '(', 1)
            bodies += [signature, '{', *['    ' + line if line else '' for line in lines], '}', '']
        for plan in self.bindings.imports.values():
            declarations.append('    static ' + self.import_declaration(plan) + ';')
            bodies += self.emit_import(plan)
        bodies, recovered_loops, loop_bytes = roll_constant_sequences(bodies)
        for _ in range(3):
            outlined, bodies = outline_bodies(self, bodies)
            declarations.extend(outlined)
            if not outlined:
                break
        binding_declarations, binding_bodies, binding_lines, binding_groups = self.factor_bindings()
        module_initializers = [self.initialize_module(module) + binding_lines[index] + self.module_metadata(module)
                               for index, module in enumerate(self.modules)]
        initial_declarations, initial_bodies, module_initializers, initial_groups = self.factor_initializers(module_initializers)
        declarations.extend(binding_declarations + initial_declarations)
        initializers = []
        for index, module in enumerate(self.modules):
            initializers += [f'{module.cclass}::{module.cclass}(Host &host) : Module(host, {quoted(module.name)}, {module.script.header.module_tag}u)', '{', '}', '',
                             f'void {module.cclass}::initializeMembers()', '{', '    auto &self = *this;']
            initializers += ['    ' + line for line in module_initializers[index]] + ['}', '']
            if module.layout.metadata.position_offset:
                offset = module.layout.metadata.position_offset
                member = module.layout.at(offset, 12)
                initializers += [f'Address {module.cclass}::position()', '{',
                                 f'    return address({self.object_member(member, "(*this)")}, {offset - member.offset}, 12);', '}', '']
        # Large constant arrays never enter the public header.
        contents = constructors + [f'using {name} = {typ};' for typ, name in self.state_views.items()] + ['']
        contents += ['class ScriptHelpers', '{', '  public:', *declarations, '};', ''] + bodies
        contents += self.program_pool.emit() + self.name_pool.emit() + self.pool_definitions
        contents += binding_bodies + initial_bodies + initializers + self.emit_names()
        contents = self.simplify_module_receivers(contents)
        initial_data, contents = self.inline_initializers(contents)
        source = ['#include "script/GeneratedScripts.h"', '', 'namespace SphereScripts', '{', '']
        source += initial_data + [''] + contents + ['}', '']
        (destination / 'GeneratedScripts.h').write_text('\n'.join(header), encoding='utf-8')
        (destination / 'GeneratedScripts.cpp').write_text('\n'.join(source), encoding='utf-8')
        stats = {'modules': len(self.modules), 'functions': sum(len(module.functions) for module in self.modules),
                 'retained_owned_fields': sum(map(len, self.families.required_fields)),
                 'source_layout_fields': sum(len(module.layout.fields) for module in self.modules),
                 'module_classes': len(self.modules), 'shared_state_records': sum(unit.shared for unit in self.storage),
                 'inheritance_depth': 1, 'unique_algorithms': len(self.helpers), 'field_reference_algorithms': len(self.reference_helpers), 'field_templates': len(self.field_templates), 'function_forwarders': 0, 'module_specific_methods': len(self.inline_owners), 'shared_template_algorithms': len(self.template_helpers) + len(self.variant_helpers),
                 'owner_templates': 0, 'state_view_aliases': len(self.state_views), 'state_view_templates': len(self.template_helpers), 'configuration_templates': len(self.variant_helpers),
                 'inlined_initial_values': self.inline_initial_data_count, 'removed_receiver_aliases': self.removed_receiver_aliases, 'concrete_initializer_groups': sum(bool(fields) for fields in self.initializer_fields.values()),
                 'shared_binding_groups': binding_groups, 'shared_initializer_groups': initial_groups, 'static_import_contracts': len(self.bindings.imports),
                 'program_recipe_rows': len(self.program_pool.rows), 'programs': sum(len(module.script.programs) for module in self.modules),
                 'header_bytes': (destination / 'GeneratedScripts.h').stat().st_size,
                 'source_bytes': (destination / 'GeneratedScripts.cpp').stat().st_size, 'binary_resource_bytes': 0,
                 'recovered_constant_loops': recovered_loops, 'constant_loop_bytes_saved': loop_bytes}
        stats.update(getattr(self, 'outline_stats', {}))
        return stats
