import tomllib
from pathlib import Path


def test_training_stack_is_pinned_to_compatible_api() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = config["project"]["optional-dependencies"]["training"]

    assert "trl==0.21.0" in dependencies
    assert any(requirement.startswith("transformers>=4.55,<5") for requirement in dependencies)
    assert any(requirement.startswith("peft>=0.17,<0.18") for requirement in dependencies)

