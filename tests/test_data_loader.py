import pytest

from src.data_loader import split_training_ids


def test_development_split_is_deterministic_and_disjoint() -> None:
    identifiers = ["2007_000005", "2007_000032", "2007_000033", "2007_000039", "2007_000042"]
    train_first, validation_first = split_training_ids(identifiers, 0.4, seed=7)
    train_second, validation_second = split_training_ids(list(reversed(identifiers)), 0.4, seed=7)

    assert train_first == train_second
    assert validation_first == validation_second
    assert set(train_first).isdisjoint(validation_first)
    assert set(train_first) | set(validation_first) == set(identifiers)
    assert len(train_first) == 3
    assert len(validation_first) == 2


def test_invalid_split_fraction_fails() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        split_training_ids(["a", "b"], 1.0, seed=42)


def test_duplicate_identifiers_fail() -> None:
    with pytest.raises(ValueError, match="unique"):
        split_training_ids(["a", "a", "b"], 0.2, seed=42)
