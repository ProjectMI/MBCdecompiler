"""Recover finite loops from consecutive, affine constant-index source operations.

Only identical statement sequences with integer literals forming arithmetic
progressions are folded. Control transfers, declarations and external scopes
are excluded. Effects retain their exact order, including reads and yields.
"""
import re

_NUMBER = re.compile(r'"(?:[^"\\]|\\.)*"|(?<![\w.])\d+[uU]?(?![\w.])')
_FORBIDDEN = re.compile(r'\b(?:return|co_return|goto|break|continue|case|for|while|do|FunctionScope|checkpoint|template|constexpr|static)\b|\b(?:const|auto|std::\w+)\s+\w+|^\S')


def _pattern(lines):
    literals = []
    def replace(match):
        text = match[0]
        if text.startswith('"'):
            return text
        unsigned = text.lower().endswith('u')
        value = int(text.rstrip('uU'))
        if value > (4294967295 if unsigned else 2147483647):
            return text
        literals.append((value, unsigned))
        return f'constant{len(literals)}_' + ('u' if unsigned else 'i')
    return _NUMBER.sub(replace, '\n'.join(lines)), literals


def roll_constant_sequences(lines):
    output = []
    position = loops = saved = 0
    while position < len(lines):
        best = None
        for width in range(1, 9):
            unit = lines[position:position + width]
            if unit and unit[0].lstrip().startswith(('}', 'else', '{')):
                continue
            if len(unit) < width or any(not line.strip() or _FORBIDDEN.search(line) for line in unit):
                break
            code = re.sub(r'"(?:[^"\\]|\\.)*"', '', '\n'.join(unit))
            depth = 0
            bad_scope = False
            for brace in re.findall(r'[{}]', code):
                depth += 1 if brace == '{' else -1
                bad_scope |= depth < 0
            if depth or bad_scope or re.search(r'<\s*\d', code) or not unit[-1].rstrip().endswith((';', '}')):
                continue
            pattern, first = _pattern(unit)
            if not first:
                continue
            rows = [first]
            end = position + width
            while end + width <= len(lines) and len(rows) < 256:
                candidate, values = _pattern(lines[end:end + width])
                if candidate != pattern:
                    break
                if len(rows) >= 2 and any(value[0] != first[index][0] + len(rows) * (rows[1][index][0] - first[index][0]) for index, value in enumerate(values)):
                    break
                rows.append(values)
                end += width
            if len(rows) < 3:
                continue
            varying = [index for index, item in enumerate(first) if rows[1][index][0] != item[0]]
            if not varying:
                continue
            index_name = f'repeat{loops + 1}'
            replacements = {}
            for index, (value, unsigned) in enumerate(first):
                suffix = 'u' if unsigned else ''
                key = f'constant{index + 1}_' + ('u' if unsigned else 'i')
                stride = rows[1][index][0] - value
                if not stride:
                    replacements[key] = str(value) + suffix
                else:
                    term = index_name if stride == 1 else f'{stride} * {index_name}'
                    expression = term if value == 0 else f'{value} + {term}'
                    replacements[key] = f'std::uint32_t({value}ll + {stride}ll * {index_name})' if unsigned else f'({expression})'
            folded = re.sub(r'constant\d+_[ui]\b', lambda match: replacements[match[0]], pattern)
            indent = unit[0][:len(unit[0]) - len(unit[0].lstrip())]
            replacement = [f'{indent}for (std::int32_t {index_name} = 0; {index_name} < {len(rows)}; ++{index_name})', indent + '{']
            replacement += ['    ' + line for line in folded.splitlines()]
            replacement.append(indent + '}')
            benefit = sum(map(len, lines[position:end])) - sum(map(len, replacement))
            if benefit > 120 and (best is None or benefit > best[0]):
                best = benefit, end, replacement
        if best:
            benefit, position, replacement = best
            output.extend(replacement)
            loops += 1
            saved += benefit
        else:
            output.append(lines[position])
            position += 1
    return output, loops, saved
