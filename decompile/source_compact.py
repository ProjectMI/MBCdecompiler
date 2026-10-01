"""Compact native locals using CFG liveness; preserve memory and suspension effects."""
from collections import defaultdict
from .source_ir import Expression
from .source_optimize import _names, _rewrite, _pure


def compact_control(function):
    blocks = {block.index: block for block in function.blocks}
    if not blocks:
        return
    first = function.blocks[0].index
    changed = True
    while changed:
        changed = False
        incoming = defaultdict(list)
        for block in blocks.values():
            for target in block.successors:
                incoming[target].append(block.index)
        for index, block in list(blocks.items()):
            if index not in blocks:
                continue
            if not block.statements and block.condition is None and len(block.successors) == 1 and block.successors[0] != index and index != first:
                target = block.successors[0]
                for parent in incoming[index]:
                    if parent in blocks:
                        blocks[parent].successors = tuple(target if child == index else child for child in blocks[parent].successors)
                del blocks[index]
                changed = True
                break
            if block.condition is None and len(block.successors) == 1:
                target = block.successors[0]
                if target != first and target != index and len(incoming[target]) == 1:
                    following = blocks.pop(target)
                    block.statements.extend(following.statements)
                    block.condition = following.condition
                    block.successors = following.successors
                    block.branch_when = following.branch_when
                    changed = True
                    break
    function.blocks = list(blocks.values())


def coalesce_locals(function):
    variables = function.variables
    if not variables:
        return 0
    order = {}
    use, definitions = {}, {}
    for block in function.blocks:
        seen, needed = set(), set()
        for statement in block.statements:
            for name in _names(statement.expression):
                if name in variables:
                    order.setdefault(name, len(order))
                    if name not in seen:
                        needed.add(name)
            if statement.result in variables:
                order.setdefault(statement.result, len(order))
                seen.add(statement.result)
        for name in _names(block.condition):
            if name in variables:
                order.setdefault(name, len(order))
                if name not in seen:
                    needed.add(name)
        use[block.index], definitions[block.index] = needed, seen
    live_in = {block.index: set() for block in function.blocks}
    live_out = {block.index: set() for block in function.blocks}
    changed = True
    while changed:
        changed = False
        for block in reversed(function.blocks):
            after = set().union(*(live_in[target] for target in block.successors))
            before = use[block.index] | (after - definitions[block.index])
            if before != live_in[block.index] or after != live_out[block.index]:
                live_in[block.index], live_out[block.index] = before, after
                changed = True
    graph = {name: set() for name in order}
    copies = defaultdict(set)
    for block in function.blocks:
        live = live_out[block.index] | (set(_names(block.condition)) & variables.keys())
        for statement in reversed(block.statements):
            target = statement.result
            if target in graph:
                for other in live:
                    if target != other and variables[target] == variables[other]:
                        graph[target].add(other)
                        graph[other].add(target)
                live.discard(target)
                node = statement.expression
                if statement.kind in {'phi', 'let'} and node is not None and node.kind == 'name' and node.value in graph and variables[target] == variables[node.value]:
                    copies[target].add(node.value)
                    copies[node.value].add(target)
            live.update(set(_names(statement.expression)) & variables.keys())
    assigned = {}
    for name in sorted(graph, key=lambda name: (-len(graph[name]), order[name])):
        forbidden = {assigned[other] for other in graph[name] if other in assigned}
        preferred = sorted({assigned[other] for other in copies[name] if other in assigned} - forbidden)
        color = preferred[0] if preferred else 0
        while color in forbidden:
            color += 1
        assigned[name] = color
    slots = {}
    replacement = {}
    types = {}
    for name in sorted(graph, key=order.get):
        key = variables[name], assigned[name]
        target = slots.setdefault(key, 'temporary' + str(len(slots) + 1))
        replacement[name] = Expression('name', target, (), variables[name])
        types[target] = variables[name]
    removed = len(variables) - len(types)
    function.variables = types
    for block in function.blocks:
        block.condition = _rewrite(block.condition, replacement)
        kept = []
        for statement in block.statements:
            statement.expression = _rewrite(statement.expression, replacement)
            if statement.result in replacement:
                statement.result = replacement[statement.result].value
            node = statement.expression
            if statement.kind in {'phi', 'let'} and node is not None:
                if node.kind == 'name' and node.value == statement.result:
                    continue
                if not statement.result and _pure(node):
                    continue
            kept.append(statement)
        block.statements = kept
    return removed


def compact_project(modules):
    count = 0
    for module in modules:
        for function in module.functions:
            count += coalesce_locals(function)
            compact_control(function)
    return count
