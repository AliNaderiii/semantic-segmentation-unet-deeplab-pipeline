"""Configuration loading and validation for the held-out Pascal VOC protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TASKS = {"binary_foreground", "voc_multiclass"}


def _positive_or_null(value: Any, field: str) -> None:
    if value is not None and (not isinstance(value, int) or value < 2):
        raise ValueError(f"dataset.{field} must be null or an integer of at least 2")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration and validate task and split-integrity requirements."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required_sections = {"dataset", "model", "training", "runtime"}
    missing = required_sections.difference(config or {})
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    dataset = config["dataset"]
    task = dataset.get("task")
    num_classes = config["model"].get("num_classes")
    if task == "binary_foreground" and num_classes != 2:
        raise ValueError("binary_foreground requires model.num_classes: 2")
    if task == "voc_multiclass" and num_classes != 21:
        raise ValueError("voc_multiclass requires model.num_classes: 21")
    if task not in SUPPORTED_TASKS:
        raise ValueError("dataset.task must be binary_foreground or voc_multiclass")

    validation_fraction = dataset.get("validation_fraction")
    if not isinstance(validation_fraction, (float, int)) or not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("dataset.validation_fraction must be between 0 and 1")
    split_seed = dataset.get("split_seed")
    if not isinstance(split_seed, int):
        raise ValueError("dataset.split_seed must be an integer")
    _positive_or_null(dataset.get("max_train_samples"), "max_train_samples")
    _positive_or_null(dataset.get("max_test_samples"), "max_test_samples")
    return config
