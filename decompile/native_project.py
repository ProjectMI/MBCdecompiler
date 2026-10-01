"""Whole-project native source model, ownership and type propagation."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re

from mbc_format.loader import MbcLoader, MbcProgram
from mbc_format.bytecode import MbcDecoder, MbcControlFlow
from mbc_format.metadata import ModuleMetadata
from .linker import MbcStaticLinker
from .decompiler import discover_local_helpers
from .source_ir import DYNAMIC, VOID, Expression, Function, recover_function, merge_type

_TYPES = {0: 'std::int8_t', 1: 'String', 2: 'StringRef', 16: 'std::int32_t', 17: 'IntRef',
          18: 'IntRefRef', 32: 'float', 33: 'FloatRef', 34: 'FloatRefRef', 48: 'Address', 49: 'AddressRef', VOID: 'void'}


def cpp_type(typ):
    return _TYPES.get(typ, 'Value')


def identifier(name):
    value = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    if not value or value[0].isdigit():
        value = 'script_' + value
    return value


def class_name(name):
    return 'Mbc' + ''.join(part[:1].upper() + part[1:] for part in name.split('_') if part)


def infer_expression(node, function, module, public_types):
    if node.kind == 'name' and node.value in function.variables:
        return function.variables[node.value]
    return node.type_id


@dataclass
class Field:
    offset: int
    size: int
    type_id: int | None
    name: str
    initial: bytes
    canonical: int = -1
    roles: tuple = ()

    @property
    def element_size(self):
        return 1 if self.type_id in {0, None} else 4 if self.type_id in {16, 32} else 12

    @property
    def scalar(self):
        return self.type_id is not None and self.size == self.element_size

    @property
    def ctype(self):
        typ = cpp_type(self.type_id) if self.type_id is not None else 'std::uint8_t'
        return typ if self.scalar else f'std::array<{typ}, {self.size // self.element_size}>'


@dataclass
class Layout:
    fields: list[Field]
    metadata: ModuleMetadata

    def at(self, offset, width=1):
        for member in self.fields:
            if member.offset <= offset and offset + width <= member.offset + member.size:
                return member
        raise ValueError(f'Unmapped native data access: {offset}+{width}')


@dataclass
class Module:
    name: str
    script: object
    linker: object
    functions: list[Function] = field(default_factory=list)
    by_entry: dict = field(default_factory=dict)
    by_name: dict = field(default_factory=dict)
    call_names: dict = field(default_factory=dict)
    layout: Layout | None = None
    cclass: str = ''
    memory_extents: list = field(default_factory=list)

    def binding(self, function, name):
        binding = function.bindings.get(name)
        if binding is None:
            return None
        return self.layout.at(binding['offset']), binding


def load_module(path):
    script = MbcLoader.load(path)
    linker = MbcStaticLinker(script)
    decoder = MbcDecoder(script, linker=linker, cache_decodes=True)
    flow = MbcControlFlow(script, decoder=decoder)
    helpers = discover_local_helpers(script, flow, linker)
    module = Module(path.stem, script, linker, cclass=class_name(path.stem))
    entries = {program.start: program for program in script.programs}
    for helper in helpers.helpers.values():
        entries.setdefault(helper.offset, helper.program)
    # The second table address is a stop/cleanup entry, not a source extent.
    for program in script.programs:
        if program.end != program.start and 0 <= program.end < len(script.code) and decoder.decode_at(program.end).mnemonic != 'end_program':
            entries.setdefault(program.end, MbcProgram(program.index, 'cleanup_' + program.name,
                                                       program.end, len(script.code) - 1, 0, 0, 0))
    for offset, program in entries.items():
        first = decoder.decode_at(offset, program)
        args = list(first.operands.get('descriptors', [])) if first.mnemonic == 'program_prologue' else []
        capacity = int(first.operands.get('signed_count', 0)) if args else 0
        symbol = linker.internal_at(offset)
        function = Function(program.name, offset, args, capacity,
                            bool(symbol.flags_or_module) if symbol else False, program=program)
        module.functions.append(function)
        module.by_entry[offset] = function
        module.by_name[function.name] = function
        module.call_names[offset] = function.name
    for symbol in script.functions:
        module.call_names.setdefault(symbol.code_offset, symbol.name)
        if symbol.code_offset in module.by_entry and not symbol.is_import:
            module.by_name[symbol.name] = module.by_entry[symbol.code_offset]
    stops = {function.entry for function in module.functions if not function.name.startswith("cleanup_")}
    for function in module.functions:
        # Native source recovery deliberately reuses the decompiler CFG decoder.
        synthetic = MbcProgram(function.program.index, function.name, function.entry,
                               len(script.code) - 1, function.program.state_raw, function.program.queue_id, 0)
        function.instructions = flow.decode_program(synthetic, follow_local_calls=False, stop_offsets=stops)
    # Display-only decode annotations dwarf the executable AST. The native
    # frontend keeps operands, types and control-flow information only.
    annotations = {'handler', 'handler_ea', 'semantic', 'function_symbols', 'function_signature', 'target_signature', 'builtin_semantic', 'builtin_confidence'}
    for function in module.functions:
        for instruction in function.instructions:
            for key in annotations:
                instruction.operands.pop(key, None)
    local_callers = defaultdict(list)
    for caller in module.functions:
        for index, instruction in enumerate(caller.instructions):
            if instruction.mnemonic == 'call_rel32':
                following = caller.instructions[index + 1:] if index + 1 < len(caller.instructions) else []
                local_callers[instruction.operands['target']].append(following)
    def ignored_tail(tail):
        if not tail:
            return False
        available = {instruction.offset: instruction for caller in module.functions for instruction in caller.instructions}
        visited = set()
        current = tail[0].offset
        while current in available and current not in visited:
            visited.add(current)
            instruction = available[current]
            if instruction.mnemonic in {'stack_frame_reset', 'end_program', 'yield_program'}:
                return True
            if instruction.mnemonic.startswith('jmp_'):
                current = instruction.operands['target']
                continue
            if instruction.mnemonic == 'call_rel32':
                target = module.by_entry.get(instruction.operands['target'])
                if target and not target.arguments:
                    current += instruction.length
                    continue
            return False
        return False
    for function in module.functions:
        callers = local_callers[function.entry]
        if function.name.startswith('local_') and callers and all(ignored_tail(tail) for tail in callers):
            function.ignored_return = True
    return module


def build_layout(module):
    metadata = ModuleMetadata.parse(module.script.metadata, code_size=len(module.script.code), data_size=len(module.script.data))
    uses = []
    for function in module.functions:
        for index, argument in enumerate(function.arguments):
            typ = argument['type']
            width = 1 if typ == 0 else 4 if typ in {16, 32} else 12
            uses.append((argument['data_offset'], width, typ, (function.name, 'argument', index)))
        for index, binding in enumerate(function.bindings.values()):
            uses.append((binding['offset'], binding['length'], binding['type'], (function.name, 'member', index)))
    for offset, width, reason in module.memory_extents:
        uses.append((offset, width, None, ('extent', *reason)))
    if metadata.position_offset:
        uses.append((metadata.position_offset, 12, None, ('position',)))
    for index, offset in enumerate(metadata.data_relocations):
        uses.append((offset, 4, None, ('reference', index)))
    intervals = []
    for offset, width, typ, role in sorted(uses, key=lambda item: (item[0], item[1])):
        if width <= 0:
            continue
        if offset < 0 or offset + width > len(module.script.data):
            raise ValueError(f'Data extent outside {module.name}: {offset}+{width}')
        if intervals and offset < intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], offset + width)
            intervals[-1][2].append((offset, width, typ, role))
        else:
            intervals.append([offset, offset + width, [(offset, width, typ, role)]])
    members = []
    cursor = 0
    for start, end, entries in intervals:
        if start > cursor:
            members.append(Field(cursor, start - cursor, None, '', module.script.data[cursor:start], roles=(('unobserved', cursor),)))
        types = {typ for _, _, typ, role in entries if role[0] != 'extent'}
        typ = next(iter(types)) if len(types) == 1 else None
        unit = 1 if typ in {0, None} else 4 if typ in {16, 32} else 12
        if (end - start) % unit or any((offset - start) % unit for offset, _, _, _ in entries):
            typ = None
        roles = tuple(sorted({(role, offset - start, width) for offset, width, _, role in entries}, key=repr))
        members.append(Field(start, end - start, typ, '', module.script.data[start:end], roles=roles))
        cursor = end
    if cursor < len(module.script.data):
        members.append(Field(cursor, len(module.script.data) - cursor, None, '', module.script.data[cursor:], roles=(('tail',),)))
    module.layout = Layout(members, metadata)


def canonicalize_fields(modules):
    identities = {}
    for module in modules:
        build_layout(module)
        for member in module.layout.fields:
            key = member.ctype, member.roles
            if key not in identities:
                identities[key] = len(identities)
            member.canonical = identities[key]
            member.name = f'member{member.canonical + 1}_'
    return identities


def recover_project(paths, progress=None, *, strict=True):
    modules = []
    for index, path in enumerate(paths):
        modules.append(load_module(path))
        if progress and (index + 1) % 32 == 0:
            progress(f'Decoded {index + 1}/{len(paths)} modules')
    public = defaultdict(list)
    functions = {}
    for module in modules:
        for function in module.functions:
            functions[module.name, function.entry] = (module, function)
        for symbol in module.script.functions:
            if not symbol.is_import and symbol.code_offset in module.by_entry:
                public[symbol.name].append((module.name, symbol.code_offset))
    types = {key: DYNAMIC for key in functions}
    optional = {key: False for key in functions}
    changed = set(functions)
    for iteration in range(12):
        for name, providers in public.items():
            types[name] = merge_type(types[key] for key in providers)
            types['optional', name] = any(optional[key] for key in providers) or (any(types[key] == VOID for key in providers) and any(types[key] != VOID for key in providers))
        updates = {}
        optional_updates = {}
        for index, key in enumerate(sorted(changed)):
            module, function = functions[key]
            recover_function(module, function, types)
            if function.return_type != types[key] or function.optional_return != optional[key]:
                updates[key] = function.return_type
                optional_updates[key] = function.optional_return
        if progress:
            progress(f'Type pass {iteration + 1}: {len(changed)} functions, {len(updates)} changed returns')
        if not updates:
            break
        types.update(updates)
        optional.update(optional_updates)
        types.update({('optional', *key): value for key, value in optional_updates.items()})
        # Dependency-sensitive invalidation avoids repeatedly lifting unrelated bodies.
        affected_names = {name for name, providers in public.items() if any(key in updates for key in providers)}
        changed = set()
        for key, (module, function) in functions.items():
            for ins in function.instructions:
                if ins.mnemonic != 'call_rel32':
                    continue
                target = ins.operands['target']
                if (module.name, target) in updates or module.call_names.get(target) in affected_names:
                    changed.add(key)
                    break
        if not changed:
            break
    else:
        raise ValueError('Whole-project source return types did not converge')
    diagnostics = [(module.name, function.name, function.diagnostics) for module in modules for function in module.functions if function.diagnostics]
    if diagnostics and strict:
        sample = '; '.join(f'{module}.{name}: {messages[0]}' for module, name, messages in diagnostics[:8])
        raise ValueError(f'{len(diagnostics)} functions have unresolved source stack diagnostics: {sample}')
    for module in modules:
        for function in module.functions:
            for block in function.blocks:
                for statement in block.statements:
                    if statement.kind == 'builtin' and statement.metadata['subopcode'] == 61:
                        typ = statement.expression.children[0].type_id if statement.expression.children else None
                        if typ is None or typ < 0:
                            raise ValueError(f'Unresolved Text argument type in {module.name}.{function.name}')
    # Suspension is transitive through ordinary source calls and imported providers.
    changed = True
    while changed:
        changed = False
        async_public = {name for name, providers in public.items() if any(functions[key][1].asynchronous for key in providers)}
        for module in modules:
            for function in module.functions:
                if function.asynchronous:
                    continue
                for block in function.blocks:
                    for statement in block.statements:
                        if statement.kind != 'call':
                            continue
                        name = statement.expression.value
                        target = module.by_name.get(name)
                        if (target and target.asynchronous) or (target is None and name in async_public):
                            function.asynchronous = True
                            changed = True
                            break
    from .native_effects import resolve_memory_extents
    resolve_memory_extents(modules)
    canonicalize_fields(modules)
    return modules, {name: types[name] for name in public}
