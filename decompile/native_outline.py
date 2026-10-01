"""Outline repeated structured source regions without instantiating module copies.

Fragments have one entry and one fall-through exit. Named module fields and live
locals are explicit typed references. Branch exits and labels are never moved.
"""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re

_TOKEN = re.compile(r'\bScriptHelpers::\w+<[^<>;\n]*>|"(?:[^"\\]|\\.)*"|\b(?:(?:self(?:\.state\d+_)?|state\d+)\.)?(member\d+_)\b|\b(?:self\.)?(state\d+_?)\b|\b(?:value|parameter|setting)\d+\b|(?<![\w.])(?:\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+|\d+)[fFuU]?(?![\w.])')
_FORBIDDEN = re.compile(r'\bstate\b|\brepeat\d+\b|\b(?:return|co_return|goto|break|continue|FunctionScope)\b|^\s*\w+:|(?:^|\s)(?:Host|auto)\s*&\s*(?:self|engine)\b')
_DECLARATION = re.compile(r'^\s*(?:std::\w+|Value|Address|String|IntRef|StringRef|FloatRef|AddressRef)\s+value\d+;\s*$')


@dataclass
class Body:
    signature: str
    lines: list
    owner: object
    variables: dict
    start: int
    end: int


@dataclass
class Candidate:
    body: int
    start: int
    end: int
    text: str
    arguments: list
    types: tuple
    calls: frozenset
    asynchronous: bool


def _brace_delta(line):
    code = re.sub(r'"(?:[^"\\]|\\.)*"', '', line)
    return code.count('{') - code.count('}')


def _normalize(lines, variables, members, storage=None):
    storage = storage or {}
    names, arguments, types = {}, [], []
    valid = True
    def token(match):
        nonlocal valid
        text = match[0]
        if text.startswith(('"', 'ScriptHelpers::')):
            return text
        numeric = text[0].isdigit()
        if numeric:
            if any(char in text.lower() for char in '.e'):
                typ = 'float' if text.lower().endswith('f') else 'double'
            else:
                typ = 'std::uint32_t' if text.lower().endswith('u') or int(text) > 2147483647 else 'std::int32_t'
            actual = text
            text = '#literal' + str(len(arguments))
        elif match[1]:
            member = members.get(match[1])
            if member is None:
                valid = False
                return text
            typ = member.ctype
            array = re.fullmatch(r'std::array<(.+), \d+>', typ)
            typ = f'std::span<{array[1]}>' if array else typ + ' &'
            actual = f'std::span({text})' if array else text
        elif match[2]:
            unit = storage.get(match[2].rstrip('_'))
            if unit is None:
                valid = False
                return text
            typ, actual = unit.ctype + ' &', text
        else:
            if text not in variables:
                valid = False
                return text
            typ = variables[text] + ' &'
            actual = text
        if text not in names:
            names[text] = f'part{len(names) + 1}'
            arguments.append(actual)
            types.append(typ)
        return names[text]
    indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    normalized = _TOKEN.sub(token, '\n'.join(line[indent:] for line in lines))
    if re.search(r'\bstate\d+\b', normalized):
        valid = False
    if any(word in normalized and f'const auto {word} =' not in normalized for word in ('previous', 'updated')):
        valid = False
    return (normalized, arguments, tuple(types)) if valid and sum(typ.endswith('&') or typ.startswith('std::span') for typ in types) <= 20 else None


def outline_bodies(generator, source):
    nodes = {node.name: node for node in generator.nodes}
    from types import SimpleNamespace
    native_owner = SimpleNamespace(name='Module')
    storage = {unit.variable: unit for unit in getattr(generator, 'storage', []) if unit.shared}
    members = {member.name: member for node in generator.nodes for member in node.fields.values()}
    bodies = []
    cursor = 0
    while cursor + 1 < len(source):
        line = source[cursor]
        if source[cursor + 1] != '{' or '::' not in line:
            cursor += 1
            continue
        end = cursor + 2
        while end < len(source) and source[end] != '}':
            end += 1
        lines = source[cursor + 2:end]
        if not any('FunctionScope context(' in line for line in lines):
            cursor = end + 1
            continue
        match = re.search(r'(\w+)::(\w+)\(', line)
        if not match:
            cursor = end + 1
            continue
        owner = nodes.get(match[1])
        if match[1] == 'ScriptHelpers':
            context = re.search(r'\((\w+) &self', line)
            owner = (native_owner if context and context[1] in {'Module', 'Self'} else nodes.get(context[1]) if context else None)
        if owner is None:
            cursor = end + 1
            continue
        variables = {}
        for typ, name in re.findall(r'([\w:]+)\s+((?:parameter|setting)\d+)', line):
            variables[name] = typ
        for body_line in lines:
            declaration = re.fullmatch(r'\s*([\w:]+)\s+(value\d+);', body_line)
            if declaration and declaration[1] not in {'return', 'co_return', 'throw'}:
                variables[declaration[2]] = declaration[1]
        bodies.append(Body(line, lines, owner, variables, cursor, end + 1))
        cursor = end + 1
    groups = defaultdict(list)
    for body_index, body in enumerate(bodies):
        lines = body.lines
        depths = [0]
        for line in lines:
            depths.append(depths[-1] + _brace_delta(line))
        blocked = [_FORBIDDEN.search(line) or _DECLARATION.match(line) for line in lines]
        for start, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or blocked[start] or stripped.startswith(('}', 'else', '{', '//')):
                continue
            base = depths[start]
            units = 0
            for end in range(start + 1, min(len(lines), start + 100) + 1):
                if blocked[end - 1] or depths[end] < base:
                    break
                if depths[end] != base or not (lines[end - 1].rstrip().endswith(';') or lines[end - 1].strip() == '}'):
                    continue
                if end < len(lines) and lines[end].lstrip().startswith('else'):
                    continue
                units += 1
                if units not in {1, 4, 8, 16}:
                    continue
                if sum(map(len, lines[start:end])) < 700:
                    continue
                normalized = _normalize(lines[start:end], body.variables, members, storage)
                if normalized is None:
                    continue
                text, arguments, types = normalized
                calls = frozenset(re.findall(r'\bself\.(\w+)(?:<[^>]*>)?\(', text)) - {'host', 'address', 'reference', 'control'}
                asynchronous = 'co_await' in text
                key = hashlib.sha256(repr((text, types, asynchronous)).encode()).digest()
                groups[key].append(Candidate(body_index, start, end, text, arguments, types, calls, asynchronous))
    def score(items):
        item = items[0]
        parameters = [typ for typ in item.types if typ.endswith('&') or typ.startswith('std::span')]
        declaration = sum(map(len, parameters)) + 30 * len(parameters) + 240
        call = 70 + 20 * len(parameters)
        return (len(items) - 1) * len(item.text) - len(items) * call - declaration
    groups = [items for items in groups.values() if len(items) > 1 and score(items) > 800]
    groups.sort(key=lambda items: (-score(items), items[0].body, items[0].start, -items[0].end))
    occupied = [set() for _ in bodies]
    changes = defaultdict(list)
    declarations, definitions = [], []
    removed = 0
    outlined = 0
    for items in groups:
        eligible = []
        seen = defaultdict(set)
        for item in items:
            interval = set(range(item.start, item.end))
            if interval & occupied[item.body] or interval & seen[item.body]:
                continue
            seen[item.body].update(interval)
            eligible.append(item)
        if len(eligible) < 2 or score(eligible) <= 800:
            continue
        common = generator.families.lca(bodies[item.body].owner for item in eligible)
        required = eligible[0].calls
        if required:
            if common is None:
                continue
            visible = {generator.families.implementations[key].cpp_name for ancestor in common.ancestors() for key in ancestor.methods}
            visible.update(name for ancestor in common.ancestors() for name in ancestor.required)
            if not required <= visible:
                continue
            owner = common.name
        else:
            owner = 'Module'
        example = eligible[0]
        fixed = {}
        for index, typ in enumerate(example.types):
            if not typ.endswith('&') and not typ.startswith('std::span'):
                choices = {item.arguments[index] for item in eligible}
                if len(choices) == 1:
                    fixed[index] = next(iter(choices))
        live_parameters = [index for index in range(len(example.types)) if index not in fixed]
        if len(live_parameters) > 32:
            continue
        outlined += 1
        name = f'sharedPart{outlined + getattr(generator, "outline_count", 0)}'
        body_text = re.sub(r'\bpart(\d+)\b', lambda match: fixed.get(int(match[1]) - 1, match[0]), example.text)
        parameters = ', '.join(f'{example.types[index]} part{index + 1}' for index in live_parameters)
        result = 'Task<void>' if example.asynchronous else 'void'
        dependent = bool(re.search(r'\bSelf\b', body_text)) or any(re.search(r'\bScriptHelpers::' + re.escape(callee) + r'\(', body_text) for callee in (getattr(generator, 'template_helpers', set()) | getattr(generator, 'template_regions', set())))
        if dependent:
            generator.template_regions = getattr(generator, 'template_regions', set()) | {name}
        template = 'template <class Self> ' if dependent else ''
        owner = 'Self' if dependent else owner
        prototype = f'{result} {name}({owner} &self' + (', ' + parameters if parameters else '') + ')'
        declarations.append('    ' + template + 'static ' + prototype + ';')
        definition = [template + prototype.replace(name + '(', 'ScriptHelpers::' + name + '(', 1), '{']
        if re.search(r'\bengine\b', example.text):
            definition.append('    Host &engine = self.host();')
        definition += ['    ' + line if line else '' for line in body_text.splitlines()]
        if example.asynchronous:
            definition.append('    co_return;')
        definition += ['}', '']
        definitions.extend(definition)
        for item in eligible:
            body = bodies[item.body]
            indent = body.lines[item.start][:-len(body.lines[item.start].lstrip())] if body.lines[item.start] != body.lines[item.start].lstrip() else ''
            call = ('co_await ' if item.asynchronous else '') + f'ScriptHelpers::{name}(self'
            if live_parameters:
                call += ', ' + ', '.join(item.arguments[index] for index in live_parameters)
            call = indent + call + ');'
            occupied[item.body].update(range(item.start, item.end))
            changes[item.body].append((item.start, item.end, call))
            removed += sum(len(line) for line in body.lines[item.start:item.end]) - len(call)
    replacements = {}
    for body_index, changes_for_body in changes.items():
        body = bodies[body_index]
        lines = body.lines[:]
        for start, end, call in sorted(changes_for_body, reverse=True):
            lines[start:end] = [call]
        replacements[body.start] = (body.end, [body.signature, '{', *lines, '}'])
    rewritten = []
    cursor = 0
    while cursor < len(source):
        if cursor in replacements:
            cursor, lines = replacements[cursor]
            rewritten.extend(lines)
        else:
            rewritten.append(source[cursor])
            cursor += 1
    generator.outline_count = outlined + getattr(generator, 'outline_count', 0)
    previous = getattr(generator, 'outline_stats', {})
    generator.outline_stats = {'shared_source_regions': generator.outline_count,
                               'region_calls': previous.get('region_calls', 0) + sum(map(len, changes.values())),
                               'region_source_bytes_saved': previous.get('region_source_bytes_saved', 0) + removed - sum(map(len, definitions))}
    return declarations, definitions + rewritten
