"""Configuration loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file and validate the supported task."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required_sections = {"dataset", "model", "training", "runtime"}
    missing = required_sections.difference(config or {})
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    task = config["dataset"].get("task")
    num_classes = config["model"].get("num_classes")
    if task == "binary_foreground" and num_classes != 2:
        raise ValueError("binary_foreground requires model.num_classes: 2")
    if task == "voc_multiclass" and num_classes != 21:
        raise ValueError("voc_multiclass requires model.num_classes: 21")
    if task not in {"binary_foreground", "voc_multiclass"}:
        raise ValueError("dataset.task must be binary_foreground or voc_multiclass")
    return config
