import pytest
from pydantic import ValidationError

from toolforge.tools import execute, reset_mock_state


def setup_function() -> None:
    reset_mock_state()


def test_execute_weather() -> None:
    result = execute({"type": "tool_call", "tool": "get_weather", "arguments": {"location": "Chicago", "date": "2026-11-09"}})
    assert result["executed"] is True
    assert result["result"]["forecast"]["condition"] == "cloudy"
    assert result["result"]["simulated"] is True


def test_non_tool_decision_is_not_executed() -> None:
    result = execute({"type": "clarification", "question": "Which date?"})
    assert result["executed"] is False


def test_invalid_call_never_executes() -> None:
    with pytest.raises(ValidationError):
        execute({"type": "tool_call", "tool": "create_calendar_event", "arguments": {"title": "Oops"}})

