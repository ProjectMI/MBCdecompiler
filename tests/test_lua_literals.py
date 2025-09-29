from mbcdisasm.lua_literals import LuaLiteralFormatter, merge_adjacent_strings


def test_literal_formatter_classifies_numbers() -> None:
    formatter = LuaLiteralFormatter()
    literal = formatter.format_operand(5)
    assert literal.kind == "number"
    assert literal.render() == "5"


def test_literal_formatter_classifies_strings() -> None:
    formatter = LuaLiteralFormatter()
    literal = formatter.format_operand(0x2165)  # 'e!'
    assert literal.kind == "string"
    assert literal.render() == "\"e!\""


def test_merge_adjacent_strings() -> None:
    formatter = LuaLiteralFormatter()
    literals = formatter.format_operands([0x2165, 0x6C6C, 0x006F])
    merged = merge_adjacent_strings(literals)
    assert len(merged) == 1
    assert merged[0].render() == "\"e!llo\""
