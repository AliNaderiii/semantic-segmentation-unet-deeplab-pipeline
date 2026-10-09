from pathlib import Path

import pytest

from src.config import load_config


def valid_config(task: str = "binary_foreground", classes: int = 2) -> str:
    return (
        "dataset:\n"
        f"  task: {task}\n"
        "  validation_fraction: 0.2\n"
        "  split_seed: 42\n"
        "  max_train_samples: null\n"
        "  max_test_samples: null\n"
        f"model:\n  num_classes: {classes}\ntraining: {{}}\nruntime: {{}}\n"
    )


def test_valid_binary_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(valid_config(), encoding="utf-8")
    assert load_config(path)["dataset"]["task"] == "binary_foreground"


def test_task_class_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(valid_config(task="voc_multiclass", classes=2), encoding="utf-8")
    with pytest.raises(ValueError, match="21"):
        load_config(path)


def test_invalid_development_validation_fraction_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(valid_config().replace("validation_fraction: 0.2", "validation_fraction: 1.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="validation_fraction"):
        load_config(path)
