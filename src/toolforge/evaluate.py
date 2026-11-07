"""Strict, repeatable metrics for model tool-routing predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import ValidationError

from .dataset_validator import load_jsonl
from .schemas import ToolName, normalized_output


ERROR_NAMES = (
    "wrong_output_type",
    "incorrect_tool",
    "incorrect_arguments",
    "malformed_or_invalid",
    "hallucinated_tool",
    "unnecessary_clarification",
    "missing_clarification",
    "failed_unsupported",
)


def score_one(expected: dict[str, Any], raw_prediction: str) -> dict[str, Any]:
    expected = normalized_output(expected)
    result = {
        "schema_valid": False,
        "output_type_correct": False,
        "tool_correct": False,
        "arguments_exact": False,
        "full_decision_correct": False,
        "hallucinated_tool": False,
        "clarification_correct": False,
        "unsupported_correct": False,
        "error": None,
    }
    try:
        raw_json = json.loads(raw_prediction)
    except (json.JSONDecodeError, TypeError):
        result["error"] = "malformed_or_invalid"
        return result

    if isinstance(raw_json, dict) and raw_json.get("type") == "tool_call":
        tool = raw_json.get("tool")
        result["hallucinated_tool"] = tool not in {name.value for name in ToolName}
    try:
        prediction = normalized_output(raw_json)
        result["schema_valid"] = True
    except (ValidationError, ValueError, TypeError):
        result["error"] = "hallucinated_tool" if result["hallucinated_tool"] else "malformed_or_invalid"
        return result

    result["output_type_correct"] = prediction["type"] == expected["type"]
    if expected["type"] == "tool_call" and prediction["type"] == "tool_call":
        result["tool_correct"] = prediction["tool"] == expected["tool"]
        result["arguments_exact"] = prediction["arguments"] == expected["arguments"]
        result["full_decision_correct"] = result["tool_correct"] and result["arguments_exact"]
        if not result["tool_correct"]:
            result["error"] = "incorrect_tool"
        elif not result["arguments_exact"]:
            result["error"] = "incorrect_arguments"
    elif expected["type"] == "clarification":
        result["clarification_correct"] = prediction["type"] == "clarification"
        result["full_decision_correct"] = result["clarification_correct"]
        if prediction["type"] != "clarification":
            result["error"] = "missing_clarification"
    elif expected["type"] == "unsupported":
        result["unsupported_correct"] = prediction["type"] == "unsupported"
        result["full_decision_correct"] = result["unsupported_correct"]
        if prediction["type"] != "unsupported":
            result["error"] = "failed_unsupported"

    if expected["type"] != "clarification" and prediction["type"] == "clarification":
        result["error"] = "unnecessary_clarification"
    elif not result["output_type_correct"] and not result["error"]:
        result["error"] = "wrong_output_type"
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No prediction records to score")
    overall_names = ["schema_valid", "output_type_correct", "full_decision_correct", "hallucinated_tool"]
    metrics = {name: mean(bool(row["scores"][name]) for row in records) for name in overall_names}
    tool_rows = [row for row in records if row["expected"]["type"] == "tool_call"]
    clarification_rows = [row for row in records if row["expected"]["type"] == "clarification"]
    unsupported_rows = [row for row in records if row["expected"]["type"] == "unsupported"]
    metrics["tool_correct"] = mean(bool(row["scores"]["tool_correct"]) for row in tool_rows) if tool_rows else 0.0
    metrics["arguments_exact"] = mean(bool(row["scores"]["arguments_exact"]) for row in tool_rows) if tool_rows else 0.0
    metrics["clarification_correct"] = mean(bool(row["scores"]["clarification_correct"]) for row in clarification_rows) if clarification_rows else 0.0
    metrics["unsupported_correct"] = mean(bool(row["scores"]["unsupported_correct"]) for row in unsupported_rows) if unsupported_rows else 0.0
    metrics["latency_seconds_mean"] = mean(float(row.get("latency_seconds", 0.0)) for row in records)
    errors = Counter(row["scores"]["error"] for row in records if row["scores"]["error"])
    by_category: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row.get("category", "unknown")].append(row)
    for category, group in grouped.items():
        by_category[category] = {
            "count": len(group),
            "schema_valid": mean(bool(item["scores"]["schema_valid"]) for item in group),
            "full_decision_correct": mean(bool(item["scores"]["full_decision_correct"]) for item in group),
        }
    return {"count": len(records), "metrics": metrics, "errors": dict(errors), "by_category": by_category}


def evaluate_predictions(dataset_path: Path, predictions_path: Path) -> dict[str, Any]:
    examples = load_jsonl(dataset_path)
    predictions = load_jsonl(predictions_path)
    prediction_by_id = {row["id"]: row for row in predictions}
    records = []
    for example in examples:
        prediction = prediction_by_id.get(example["id"])
        if prediction is None:
            raw = ""
            latency = 0.0
        else:
            raw = prediction.get("prediction", "")
            latency = prediction.get("latency_seconds", 0.0)
        records.append({
            "id": example["id"],
            "category": example["category"],
            "expected": example["expected"],
            "prediction": raw,
            "latency_seconds": latency,
            "scores": score_one(example["expected"], raw),
        })
    return {"summary": summarize(records), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args()
    report = evaluate_predictions(args.dataset, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
