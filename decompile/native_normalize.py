"""Normalize source-local identities without collapsing per-instance storage."""
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib

from .native_families import function_signature, pattern
from .native_project import build_layout, identifier


def normalize_helpers(modules):
    helpers = {(module.name, function.entry): (module, function) for module in modules
               for function in module.functions if function.name.startswith('local_')}
    if not helpers:
        return
    labels = {key: 0 for key in helpers}
    for _ in range(32):
        shapes = {}
        for key, (module, function) in helpers.items():
            def resolve(name):
                target = module.by_name.get(name)
                if target is not None and (module.name, target.entry) in helpers:
                    return 'helper', labels[module.name, target.entry], function_signature(target)
                return name
            source = pattern(module, function, local_members=True, name='helper', resolve_call=resolve)
            shapes[key] = hashlib.sha256(repr((labels[key], source.structure)).encode()).digest()
        choices = {shape: index for index, shape in enumerate(sorted(set(shapes.values())))}
        updated = {key: choices[value] for key, value in shapes.items()}
        stable = len(set(updated.values())) == len(set(labels.values()))
        labels = updated
        if stable:
            break
    # Multiple equivalent-looking locals in one module must not acquire the
    # same member name: their storage and source callers can still differ.
    renames = {}
    for module in modules:
        occurrences = Counter()
        for function in sorted(module.functions, key=lambda item: item.entry):
            key = module.name, function.entry
            if key not in labels:
                continue
            group = labels[key]
            occurrences[group] += 1
            renames[module.name, function.name] = f'helper{group + 1}_{occurrences[group]}'
    def rewrite(module, node):
        if node is None:
            return None
        value = renames.get((module.name, node.value), node.value) if node.kind == 'invocation' else node.value
        children = tuple(rewrite(module, child) for child in node.children)
        return node if value == node.value and children == node.children else replace(node, value=value, children=children)
    for module in modules:
        for function in module.functions:
            for block in function.blocks:
                block.condition = rewrite(module, block.condition)
                for statement in block.statements:
                    statement.expression = rewrite(module, statement.expression)
            old = function.name
            new = renames.get((module.name, old))
            if new:
                function.name = new
                module.by_name.pop(old, None)
                module.by_name[new] = function
                module.call_names[function.entry] = new


def normalize_storage(modules):
    # Give every physical member a separate initial identity. Unions are then
    # driven by equivalent source uses, with an injectivity check per module.
    members = []
    owners = []
    for module_index, module in enumerate(modules):
        build_layout(module)
        for member in module.layout.fields:
            member.canonical = len(members)
            members.append(member)
            owners.append(module_index)
    parents = list(range(len(members)))
    occupied = [{owner} for owner in owners]
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    def join(first, second):
        a, b = root(first), root(second)
        if a == b:
            return True
        if members[a].ctype != members[b].ctype or occupied[a] & occupied[b]:
            return False
        if len(occupied[a]) < len(occupied[b]):
            a, b = b, a
        parents[b] = a
        occupied[a].update(occupied[b])
        occupied[b].clear()
        return True
    buckets = defaultdict(list)
    for module in modules:
        for function in module.functions:
            normalized = pattern(module, function, local_members=True)
            digest = hashlib.sha256(repr(normalized.structure).encode()).digest()
            buckets[digest].append(normalized.members)
    for bucket in sorted(buckets.values(), key=lambda items: (-len(items), -len(items[0]))):
        reference = bucket[0]
        for candidates in bucket[1:]:
            for first, second in zip(reference, candidates):
                join(first.canonical, second.canonical)
    # Fields not yet aligned by a shared method can share a declaration when
    # all semantic roles, width and type agree. They still live per instance.
    roles = {}
    for member in members:
        key = member.ctype, member.roles
        previous = roles.setdefault(key, member.canonical)
        join(previous, member.canonical)
    compact = {}
    for member in members:
        identity = root(member.canonical)
        member.canonical = compact.setdefault(identity, len(compact))
        member.name = f'member{member.canonical + 1}_'
    return len(compact)


def normalize_project(modules):
    def unique_literals(node):
        if node is None:
            return None
        return replace(node, children=tuple(unique_literals(child) for child in node.children))
    for module in modules:
        for function in module.functions:
            for block in function.blocks:
                block.condition = unique_literals(block.condition)
                for statement in block.statements:
                    statement.expression = unique_literals(statement.expression)
    normalize_helpers(modules)
    return normalize_storage(modules)
