"""Validate generated JSONL files structurally and semantically."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import normalized_output


REQUIRED_ROLES = ["system", "user", "assistant"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def validate_rows(rows: list[dict[str, Any]], source: str = "dataset") -> dict[str, Any]:
    errors: list[str] = []
    ids: set[str] = set()
    requests: set[str] = set()
    families: set[str] = set()
    categories: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        label = f"{source}:{index}"
        messages = row.get("messages", [])
        if [message.get("role") for message in messages] != REQUIRED_ROLES:
            errors.append(f"{label}: messages must have system, user, assistant roles")
            continue
        if row.get("id") in ids:
            errors.append(f"{label}: duplicate id {row.get('id')}")
        ids.add(row.get("id"))
        request_key = messages[1].get("content", "").casefold().strip()
        if request_key in requests:
            errors.append(f"{label}: duplicate request")
        requests.add(request_key)
        families.add(str(row.get("family")))
        categories[str(row.get("category"))] += 1
        try:
            assistant = normalized_output(messages[2]["content"])
            expected = normalized_output(row["expected"])
            if assistant != expected:
                errors.append(f"{label}: assistant output differs from expected")
        except Exception as exc:
            errors.append(f"{label}: invalid expected output: {exc}")
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"{len(errors)} validation error(s):\n{preview}")
    return {
        "examples": len(rows),
        "families": len(families),
        "categories": dict(categories),
    }


def validate_splits(paths: list[Path]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    family_owners: dict[str, str] = {}
    all_requests: set[str] = set()
    for path in paths:
        rows = load_jsonl(path)
        report[path.stem] = validate_rows(rows, str(path))
        for row in rows:
            family = row["family"]
            previous = family_owners.setdefault(family, path.stem)
            if previous != path.stem:
                raise ValueError(f"family leakage: {family!r} occurs in {previous} and {path.stem}")
            request = row["messages"][1]["content"].casefold().strip()
            if request in all_requests:
                raise ValueError(f"request leakage across splits: {request!r}")
            all_requests.add(request)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("data/train.jsonl"), Path("data/validation.jsonl"), Path("data/test.jsonl")])
    args = parser.parse_args()
    print(json.dumps(validate_splits(args.paths), indent=2))


if __name__ == "__main__":
    main()

