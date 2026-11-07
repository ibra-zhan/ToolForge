"""Single source of truth for the routing prompt and tool descriptions."""

from __future__ import annotations

import json


TOOL_DEFINITIONS = [
    {"name": "search_emails", "description": "Search mock emails", "required": {"query": "string"}},
    {"name": "get_calendar_events", "description": "List events on one date", "required": {"date": "YYYY-MM-DD"}},
    {"name": "find_contact", "description": "Find a contact by name", "required": {"name": "string"}},
    {
        "name": "create_calendar_event",
        "description": "Create an event in disposable mock state",
        "required": {"title": "string", "date": "YYYY-MM-DD", "time": "HH:MM:SS", "attendee_email": "email"},
    },
    {
        "name": "draft_email",
        "description": "Create an unsent email draft",
        "required": {"recipient": "string", "subject": "string", "body": "string"},
    },
    {"name": "get_weather", "description": "Read a fixed mock forecast", "required": {"location": "string", "date": "YYYY-MM-DD"}},
]


def system_prompt(reference_date: str, timezone: str) -> str:
    tools = json.dumps(TOOL_DEFINITIONS, separators=(",", ":"))
    return f"""You are a tool-routing assistant. Return exactly one JSON object and no other text.
Reference date: {reference_date}. Timezone: {timezone}.
Available tools: {tools}
Allowed outputs:
1. {{"type":"tool_call","tool":"<available name>","arguments":{{...}}}}
2. {{"type":"clarification","question":"<question>"}}
3. {{"type":"unsupported","reason":"<reason>"}}
Never invent a tool or a required value. Ask for clarification when critical information is missing. Resolve relative dates using the reference date."""


FEW_SHOT_EXAMPLES = [
    ("What is on my calendar tomorrow?", {"type": "tool_call", "tool": "get_calendar_events", "arguments": {"date": "2026-11-08"}}),
    ("Schedule something with Sarah.", {"type": "clarification", "question": "What title, date, time, and attendee email should I use?"}),
    ("Order me a pizza.", {"type": "unsupported", "reason": "No available tool can order food."}),
]


def build_messages(
    request: str,
    reference_date: str = "2026-11-07",
    timezone: str = "America/Indiana/Indianapolis",
    few_shot: bool = False,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt(reference_date, timezone)}]
    if few_shot:
        for user, assistant in FEW_SHOT_EXAMPLES:
            messages.extend(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps(assistant, separators=(",", ":"))},
                ]
            )
    messages.append({"role": "user", "content": request})
    return messages

