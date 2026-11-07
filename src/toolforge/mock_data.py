"""Small deterministic datasets used by the simulated tools."""

CONTACTS = [
    {"name": "Alex Chen", "email": "alex@example.com", "phone": "+1-555-0101"},
    {"name": "Sarah Johnson", "email": "sarah@example.com", "phone": "+1-555-0102"},
    {"name": "Maya Patel", "email": "maya@example.com", "phone": "+1-555-0103"},
]

CALENDAR = {
    "2026-11-08": [
        {"title": "Physics lecture", "time": "10:00"},
        {"title": "Team meeting", "time": "15:00"},
    ],
    "2026-11-09": [{"title": "Project review", "time": "13:30"}],
    "2026-11-13": [{"title": "Lunch with Sarah", "time": "12:00"}],
}

EMAILS = [
    {"from": "alex@example.com", "subject": "Project update", "body": "The demo is ready."},
    {"from": "sarah@example.com", "subject": "Lunch", "body": "Friday works for me."},
    {"from": "maya@example.com", "subject": "Invoice", "body": "The October invoice is attached."},
]

WEATHER = {
    ("Chicago", "2026-11-09"): {"condition": "cloudy", "high_f": 48, "low_f": 35},
    ("Indianapolis", "2026-11-08"): {"condition": "sunny", "high_f": 55, "low_f": 38},
    ("New York", "2026-11-13"): {"condition": "rain", "high_f": 51, "low_f": 43},
}

