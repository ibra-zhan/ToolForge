import pytest
from pydantic import ValidationError

from toolforge.schemas import Clarification, ToolCall, validate_output


def test_valid_tool_call() -> None:
    parsed = validate_output('{"type":"tool_call","tool":"get_weather","arguments":{"location":"Chicago","date":"2026-11-09"}}')
    assert isinstance(parsed, ToolCall)
    assert parsed.tool.value == "get_weather"


def test_clarification() -> None:
    assert isinstance(validate_output({"type": "clarification", "question": "Which date?"}), Clarification)


@pytest.mark.parametrize("payload", [
    {"type": "tool_call", "tool": "invented_tool", "arguments": {}},
    {"type": "tool_call", "tool": "get_weather", "arguments": {"location": "Chicago"}},
    {"type": "tool_call", "tool": "get_calendar_events", "arguments": {"date": "tomorrow"}},
    {"type": "unsupported", "reason": "No tool", "extra": True},
])
def test_invalid_outputs(payload: dict) -> None:
    with pytest.raises(ValidationError):
        validate_output(payload)

