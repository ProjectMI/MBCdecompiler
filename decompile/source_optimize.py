"""Conservative source simplification before grouping native implementations.

Definitions are folded only when their inputs cannot be changed by another
path. Reads keep their execution position unless consumed immediately.
"""
from collections import Counter, defaultdict
from dataclasses import replace

from .source_ir import Expression


def _names(node):
    if node is not None:
        for item in node.walk():
            if item.kind == 'name':
                yield item.value


def _rewrite(node, replacements):
    if node is None:
        return None
    if node.kind == 'name' and node.value in replacements:
        return replacements[node.value]
    children = tuple(_rewrite(child, replacements) for child in node.children)
    return node if children == node.children else replace(node, children=children)


def _pure(node):
    if node is None:
        return True
    for item in node.walk():
        if item.kind in {'load', 'invocation', 'by_reference', 'element', 'field', 'indirect', 'address'}:
            return False
        if item.kind == 'binary' and item.value in {'/', '%'}:
            return False
    return True


def optimize_function(function):
    total_removed = 0
    for _ in range(64):
        definitions = defaultdict(list)
        uses = Counter()
        for block in function.blocks:
            uses.update(_names(block.condition))
            for statement in block.statements:
                if statement.result:
                    definitions[statement.result].append((block, statement))
                uses.update(_names(statement.expression))
        replacements = {}
        removable = set()
        for name, locations in definitions.items():
            if len(locations) != 1:
                continue
            block, statement = locations[0]
            node = statement.expression
            if statement.kind not in {'phi', 'let', 'read'} or node is None:
                continue
            if not uses[name]:
                # A statically bounded field read cannot fail. An indirect read
                # is retained because its bounds error is observable.
                bounded = statement.kind == 'read' and node.kind == 'load' and node.children[0].kind == 'storage'
                if _pure(node) or bounded:
                    removable.add(id(statement))
                continue
            referenced = list(_names(node))
            stable = all(len(definitions.get(source, ())) <= 1 for source in referenced)
            if uses[name] == 1 and stable and _pure(node) and name not in referenced:
                replacements[name] = node
                removable.add(id(statement))
        resolved = {}
        visiting = set()
        cyclic = set()
        def resolve(name):
            if name in resolved:
                return resolved[name]
            if name in visiting:
                cyclic.update(visiting)
                return Expression('name', name, (), function.variables.get(name))
            visiting.add(name)
            node = replacements[name]
            dependencies = {source: resolve(source) for source in set(_names(node)) if source in replacements}
            visiting.remove(name)
            resolved[name] = _rewrite(node, dependencies)
            return resolved[name]
        for name in replacements:
            resolve(name)
        for name in cyclic:
            removable.discard(id(definitions[name][0][1]))
            resolved.pop(name, None)
        replacements = resolved
        if not removable:
            break
        total_removed += len(removable)
        for block in function.blocks:
            block.condition = _rewrite(block.condition, replacements)
            retained = []
            for statement in block.statements:
                if id(statement) in removable:
                    continue
                statement.expression = _rewrite(statement.expression, replacements)
                retained.append(statement)
            block.statements = retained
    # Inline a snapshot only into its immediate consumer. This preserves reads
    # across writes, builtin calls and suspension points.
    global_uses = Counter()
    for block in function.blocks:
        global_uses.update(_names(block.condition))
        for statement in block.statements:
            global_uses.update(_names(statement.expression))
    for block in function.blocks:
        index = len(block.statements) - 1
        while index >= 0:
            statement = block.statements[index]
            if statement.kind != 'read' or not statement.result:
                index -= 1
                continue
            name = statement.result
            consumer = block.statements[index + 1] if index + 1 < len(block.statements) else None
            target = consumer.expression if consumer else block.condition
            if global_uses[name] == 1 and name in set(_names(target)):
                # Expressions passed by reference retain their storage access;
                # only the value snapshot itself is substituted.
                replacement = _rewrite(target, {name: statement.expression})
                if consumer:
                    consumer.expression = replacement
                else:
                    block.condition = replacement
                block.statements.pop(index)
                total_removed += 1
            index -= 1
    uses = Counter()
    for block in function.blocks:
        uses.update(_names(block.condition))
        for statement in block.statements:
            uses.update(_names(statement.expression))
    for block in function.blocks:
        for statement in block.statements:
            if statement.kind == 'builtin' and statement.result and not uses[statement.result]:
                statement.result = None
                total_removed += 1
    active = set()
    for block in function.blocks:
        active.update(_names(block.condition))
        for statement in block.statements:
            active.update(_names(statement.expression))
            if statement.result:
                active.add(statement.result)
    function.variables = {name: typ for name, typ in function.variables.items() if name in active}
    function.bindings = {name: data for name, data in function.bindings.items() if name in active}
    return total_removed


def optimize_project(modules):
    return sum(optimize_function(function) for module in modules for function in module.functions)
