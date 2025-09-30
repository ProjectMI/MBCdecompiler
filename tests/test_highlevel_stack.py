import pytest

from mbcdisasm.highlevel import HighLevelStack, StackEvent
from mbcdisasm.lua_ast import Assignment, LiteralExpr, NameExpr

def test_push_expression_reuses_name_without_assignment() -> None:
    stack = HighLevelStack()
    expression = NameExpr("foo")
    statements, value = stack.push_expression(expression)
    assert statements == []
    assert value.expression is expression
    assert stack.depth == 1
    assert stack.max_depth == 1
    events = stack.events()
    assert len(events) == 1
    event = events[0]
    assert event.action == "push"
    assert event.value == "foo"
    assert event.depth_before == 0
    assert event.depth_after == 1


def test_push_expression_creates_local_for_non_name() -> None:
    stack = HighLevelStack()
    literal = LiteralExpr("42")
    statements, value = stack.push_expression(literal, prefix="val")
    assert len(statements) == 1
    assignment = statements[0]
    assert isinstance(assignment, Assignment)
    assert assignment.targets[0].render().startswith("val_")
    assert assignment.value is literal
    assert value.expression.render().startswith("val_")
    assert stack.depth == 1


def test_stack_depth_tracking_with_flush() -> None:
    stack = HighLevelStack()
    stack.push_literal(LiteralExpr("1"))
    stack.push_literal(LiteralExpr("2"))
    assert stack.depth == 2
    assert stack.max_depth == 2

    value = stack.pop_single()
    assert value.expression.render() == "2"
    assert stack.depth == 1
    flushed = stack.flush()
    assert [item.expression.render() for item in flushed] == ["1"]
    assert stack.depth == 0
    assert stack.min_depth == 0

    events = stack.events()
    flush_actions = [event for event in events if event.action == "flush"]
    assert flush_actions
    assert flush_actions[-1].depth_after == 0


def test_underflow_updates_depth_and_warnings() -> None:
    stack = HighLevelStack()
    first = stack.pop_single()
    second = stack.pop_single()
    assert first.expression.render().startswith("stack_")
    assert second.expression.render().startswith("stack_")
    assert stack.min_depth == -2
    assert stack.underflow_events == 2
    assert stack.depth == -2

    events = stack.events()
    pop_events = [event for event in events if event.action == "pop"]
    assert pop_events[-1].depth_after == -2
    assert all(event.comment for event in pop_events)
    warnings = stack.warnings
    assert len(warnings) == 2


@pytest.mark.parametrize(
    "event,expected",
    [
        (
            StackEvent(
                action="push",
                value="tmp_0",
                origin=0x10,
                comment=None,
                depth_before=0,
                depth_after=1,
            ),
            "push",
        ),
        (
            StackEvent(
                action="pop",
                value="tmp_1",
                origin=0x20,
                comment="placeholder",
                depth_before=1,
                depth_after=0,
            ),
            "pop",
        ),
    ],
)
def test_stack_event_depth_accessors(event: StackEvent, expected: str) -> None:
    assert event.action == expected
    assert isinstance(event.depth_before, int)
    assert isinstance(event.depth_after, int)
