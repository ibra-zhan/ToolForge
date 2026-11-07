import pytest

from toolforge.train import to_prompt_completion


def test_conversation_becomes_prompt_completion_example() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "Route requests."},
            {"role": "user", "content": "Check tomorrow."},
            {"role": "assistant", "content": '{"type":"clarification","question":"Which location?"}'},
        ]
    }

    converted = to_prompt_completion(row)

    assert [message["role"] for message in converted["prompt"]] == ["system", "user"]
    assert converted["completion"][0]["role"] == "assistant"
    assert converted["completion"][0]["content"].startswith("{")


def test_training_example_must_end_with_assistant() -> None:
    with pytest.raises(ValueError, match="assistant"):
        to_prompt_completion({"messages": [{"role": "user", "content": "Hello"}]})

