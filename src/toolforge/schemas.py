"""Strict output and per-tool argument schemas.

The language model may propose a call, but only this module decides whether the
proposal is safe and structurally valid enough to execute.
"""

from __future__ import annotations

from datetime import date, time
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolName(str, Enum):
    SEARCH_EMAILS = "search_emails"
    GET_CALENDAR_EVENTS = "get_calendar_events"
    FIND_CONTACT = "find_contact"
    CREATE_CALENDAR_EVENT = "create_calendar_event"
    DRAFT_EMAIL = "draft_email"
    GET_WEATHER = "get_weather"


class SearchEmailsArgs(StrictModel):
    query: str = Field(min_length=1, max_length=300)


class GetCalendarEventsArgs(StrictModel):
    date: date


class FindContactArgs(StrictModel):
    name: str = Field(min_length=1, max_length=100)


class CreateCalendarEventArgs(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    date: date
    time: time
    attendee_email: str = Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class DraftEmailArgs(StrictModel):
    recipient: str = Field(min_length=1, max_length=200)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)


class GetWeatherArgs(StrictModel):
    location: str = Field(min_length=1, max_length=200)
    date: date


ARGUMENT_MODELS: dict[ToolName, type[StrictModel]] = {
    ToolName.SEARCH_EMAILS: SearchEmailsArgs,
    ToolName.GET_CALENDAR_EVENTS: GetCalendarEventsArgs,
    ToolName.FIND_CONTACT: FindContactArgs,
    ToolName.CREATE_CALENDAR_EVENT: CreateCalendarEventArgs,
    ToolName.DRAFT_EMAIL: DraftEmailArgs,
    ToolName.GET_WEATHER: GetWeatherArgs,
}


class ToolCall(StrictModel):
    type: Literal["tool_call"]
    tool: ToolName
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def validate_arguments(self) -> "ToolCall":
        ARGUMENT_MODELS[self.tool].model_validate(self.arguments)
        return self


class Clarification(StrictModel):
    type: Literal["clarification"]
    question: str = Field(min_length=1, max_length=500)


class Unsupported(StrictModel):
    type: Literal["unsupported"]
    reason: str = Field(min_length=1, max_length=500)


ModelOutput = Annotated[
    ToolCall | Clarification | Unsupported,
    Field(discriminator="type"),
]
OUTPUT_ADAPTER = TypeAdapter(ModelOutput)


def validate_output(value: str | dict[str, Any]) -> ModelOutput:
    """Parse and validate a model response without repairing it."""
    if isinstance(value, str):
        return OUTPUT_ADAPTER.validate_json(value)
    return OUTPUT_ADAPTER.validate_python(value)


def normalized_output(value: str | dict[str, Any]) -> dict[str, Any]:
    """Return JSON-compatible canonical data after strict validation."""
    parsed = validate_output(value)
    return parsed.model_dump(mode="json")


__all__ = [
    "ARGUMENT_MODELS",
    "Clarification",
    "ModelOutput",
    "ToolCall",
    "ToolName",
    "Unsupported",
    "ValidationError",
    "normalized_output",
    "validate_output",
]
