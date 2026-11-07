from toolforge.dataset_generator import generate_examples
from toolforge.dataset_validator import validate_rows
from toolforge.schemas import validate_output


def test_generation_is_deterministic_and_valid() -> None:
    first = generate_examples(120, seed=7)
    second = generate_examples(120, seed=7)
    assert first == second
    for split, rows in first.items():
        assert rows
        validate_rows(rows, split)
        for row in rows:
            validate_output(row["messages"][2]["content"])


def test_template_families_do_not_leak() -> None:
    splits = generate_examples(200)
    family_sets = [{row["family"] for row in rows} for rows in splits.values()]
    assert family_sets[0].isdisjoint(family_sets[1])
    assert family_sets[0].isdisjoint(family_sets[2])
    assert family_sets[1].isdisjoint(family_sets[2])

