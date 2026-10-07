from pathlib import Path

import pytest

from src.config import load_config


def test_valid_binary_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset:\n  task: binary_foreground\nmodel:\n  num_classes: 2\ntraining: {}\nruntime: {}\n",
        encoding="utf-8",
    )
    assert load_config(path)["dataset"]["task"] == "binary_foreground"


def test_task_class_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset:\n  task: voc_multiclass\nmodel:\n  num_classes: 2\ntraining: {}\nruntime: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="21"):
        load_config(path)
