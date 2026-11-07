from toolforge.evaluate import score_one, summarize


def test_exact_tool_call_scores_correctly() -> None:
    expected = {"type": "tool_call", "tool": "find_contact", "arguments": {"name": "Alex Chen"}}
    scores = score_one(expected, '{"type":"tool_call","tool":"find_contact","arguments":{"name":"Alex Chen"}}')
    assert scores["schema_valid"] is True
    assert scores["full_decision_correct"] is True


def test_malformed_output_is_not_repaired() -> None:
    expected = {"type": "unsupported", "reason": "No tool"}
    scores = score_one(expected, "```json\n{bad}\n```")
    assert scores["schema_valid"] is False
    assert scores["error"] == "malformed_or_invalid"


def test_hallucinated_tool_is_counted() -> None:
    expected = {"type": "unsupported", "reason": "No tool"}
    scores = score_one(expected, '{"type":"tool_call","tool":"buy_pizza","arguments":{}}')
    assert scores["hallucinated_tool"] is True
    assert scores["error"] == "hallucinated_tool"


def test_type_specific_metrics_use_correct_denominators() -> None:
    examples = [
        ({"type": "tool_call", "tool": "find_contact", "arguments": {"name": "Alex"}}, '{"type":"tool_call","tool":"find_contact","arguments":{"name":"Alex"}}'),
        ({"type": "clarification", "question": "Which person?"}, '{"type":"clarification","question":"Who?"}'),
        ({"type": "unsupported", "reason": "No tool"}, '{"type":"unsupported","reason":"No tool"}'),
    ]
    records = [
        {"expected": expected, "category": expected["type"], "latency_seconds": 0, "scores": score_one(expected, prediction)}
        for expected, prediction in examples
    ]
    metrics = summarize(records)["metrics"]
    assert metrics["tool_correct"] == 1
    assert metrics["clarification_correct"] == 1
    assert metrics["unsupported_correct"] == 1
