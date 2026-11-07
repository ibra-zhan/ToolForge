"""Explicit, allow-listed routing to deterministic mock tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .mock_data import CALENDAR, CONTACTS, EMAILS, WEATHER
from .schemas import (
    ARGUMENT_MODELS,
    Clarification,
    ModelOutput,
    ToolCall,
    ToolName,
    Unsupported,
    validate_output,
)


MOCK_CREATED_EVENTS: list[dict[str, str]] = []
MOCK_DRAFTS: list[dict[str, str]] = []


def search_emails(query: str) -> dict[str, Any]:
    needle = query.casefold()
    matches = [email for email in EMAILS if needle in " ".join(email.values()).casefold()]
    return {"simulated": True, "query": query, "emails": deepcopy(matches)}


def get_calendar_events(date: str) -> dict[str, Any]:
    return {"simulated": True, "date": date, "events": deepcopy(CALENDAR.get(date, []))}


def find_contact(name: str) -> dict[str, Any]:
    needle = name.casefold()
    matches = [contact for contact in CONTACTS if needle in contact["name"].casefold()]
    return {"simulated": True, "query": name, "contacts": deepcopy(matches)}


def create_calendar_event(
    title: str, date: str, time: str, attendee_email: str
) -> dict[str, Any]:
    event = {"title": title, "date": date, "time": time, "attendee_email": attendee_email}
    MOCK_CREATED_EVENTS.append(event)
    return {"simulated": True, "created": deepcopy(event)}


def draft_email(recipient: str, subject: str, body: str) -> dict[str, Any]:
    draft = {"recipient": recipient, "subject": subject, "body": body}
    MOCK_DRAFTS.append(draft)
    return {"simulated": True, "draft": deepcopy(draft)}


def get_weather(location: str, date: str) -> dict[str, Any]:
    forecast = WEATHER.get((location, date))
    return {
        "simulated": True,
        "location": location,
        "date": date,
        "forecast": deepcopy(forecast),
    }


TOOL_REGISTRY: dict[ToolName, Callable[..., dict[str, Any]]] = {
    ToolName.SEARCH_EMAILS: search_emails,
    ToolName.GET_CALENDAR_EVENTS: get_calendar_events,
    ToolName.FIND_CONTACT: find_contact,
    ToolName.CREATE_CALENDAR_EVENT: create_calendar_event,
    ToolName.DRAFT_EMAIL: draft_email,
    ToolName.GET_WEATHER: get_weather,
}


def execute(output: str | dict[str, Any] | ModelOutput) -> dict[str, Any]:
    """Validate first, then execute only an allow-listed tool call."""
    parsed = output if isinstance(output, (ToolCall, Clarification, Unsupported)) else validate_output(output)
    if not isinstance(parsed, ToolCall):
        return {"executed": False, "decision": parsed.model_dump(mode="json")}
    arguments = ARGUMENT_MODELS[parsed.tool].model_validate(parsed.arguments).model_dump(mode="json")
    result = TOOL_REGISTRY[parsed.tool](**arguments)
    return {"executed": True, "tool": parsed.tool.value, "result": result}


def reset_mock_state() -> None:
    MOCK_CREATED_EVENTS.clear()
    MOCK_DRAFTS.clear()
