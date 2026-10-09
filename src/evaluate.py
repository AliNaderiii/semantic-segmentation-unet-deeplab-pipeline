"""Evaluate a validation-selected Pascal VOC checkpoint on untouched official validation data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import PROJECT_ROOT, load_config
from .data_loader import get_dataloaders
from .metrics import SegmentationMeter
from .models import get_model
from .train import PROTOCOL


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load only checkpoints written by the held-out protocol-aware trainer."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "model", "dataset", "split_counts", "epoch"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Unsupported checkpoint. Train it with the current held-out protocol first.")
    if payload.get("format_version") != 2 or payload["dataset"].get("protocol") != PROTOCOL:
        raise ValueError("Checkpoint lacks the current held-out validation protocol metadata.")
    model_spec = payload["model"]
    model = get_model(
        model_name=model_spec["name"],
        encoder=model_spec["encoder"],
        num_classes=int(model_spec["num_classes"]),
        pretrained=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload


@torch.no_grad()
def evaluate_heldout_validation(
    checkpoint_path: str | Path, config_path: str | Path = "config.yaml"
) -> dict[str, Any]:
    """Evaluate only Pascal VOC's untouched official ``val`` split."""
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint_path, device)
    model_spec = payload["model"]
    dataset_spec = payload["dataset"]
    if model_spec["num_classes"] != config["model"]["num_classes"]:
        raise ValueError("Checkpoint and config disagree about class count")
    if dataset_spec["task"] != config["dataset"]["task"]:
        raise ValueError("Checkpoint and config disagree about segmentation task")
    if dataset_spec["validation_fraction"] != config["dataset"]["validation_fraction"]:
        raise ValueError("Checkpoint and config disagree about development validation fraction")
    if dataset_spec["split_seed"] != config["dataset"]["split_seed"]:
        raise ValueError("Checkpoint and config disagree about split seed")

    loaders, manifest = get_dataloaders(
        batch_size=int(config["training"]["batch_size"]),
        image_size=int(dataset_spec["image_size"]),
        task=config["dataset"]["task"],
        data_root=PROJECT_ROOT / config["dataset"]["root"],
        validation_fraction=float(config["dataset"]["validation_fraction"]),
        split_seed=int(config["dataset"]["split_seed"]),
        num_workers=int(config["runtime"]["num_workers"]),
        max_train_samples=config["dataset"].get("max_train_samples"),
        max_test_samples=config["dataset"].get("max_test_samples"),
        download=False,
    )
    current_counts = {key: len(manifest[f"{key}_ids"]) for key in ("train", "validation", "test")}
    if payload["split_counts"] != current_counts:
        raise ValueError("Checkpoint split counts do not match the current data/configuration")

    meter = SegmentationMeter(num_classes=int(model_spec["num_classes"]))
    for images, targets in tqdm(loaders["test"], desc="held-out official VOC val"):
        meter.update(model(images.to(device)), targets.to(device))
    results = meter.compute()
    results.update(
        {
            "split": "official_voc_val_held_out",
            "checkpoint": str(checkpoint_path),
            "protocol": PROTOCOL,
            "split_counts": payload["split_counts"],
            "current_manifest_counts": current_counts,
        }
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    metrics = evaluate_heldout_validation(args.checkpoint, args.config)
    output = PROJECT_ROOT / "reports" / "heldout_val_metrics.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
