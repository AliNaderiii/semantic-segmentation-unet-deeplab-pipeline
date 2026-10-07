"""Evaluate a saved checkpoint on Pascal VOC's official validation split."""
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


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "model", "dataset"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Checkpoint is not a supported self-describing checkpoint.")
    metadata = payload["model"]
    model = get_model(
        model_name=metadata["name"],
        encoder=metadata["encoder"],
        num_classes=int(metadata["num_classes"]),
        pretrained=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload


@torch.no_grad()
def evaluate(checkpoint_path: str | Path, config_path: str | Path = "config.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint_path, device)
    metadata = payload["model"]
    if metadata["num_classes"] != config["model"]["num_classes"]:
        raise ValueError("The checkpoint and config use different numbers of classes.")
    if payload["dataset"]["task"] != config["dataset"]["task"]:
        raise ValueError("The checkpoint and config use different segmentation tasks.")

    _, loader = get_dataloaders(
        batch_size=int(config["training"]["batch_size"]),
        image_size=int(payload["dataset"]["image_size"]),
        task=config["dataset"]["task"],
        data_root=PROJECT_ROOT / config["dataset"]["root"],
        num_workers=int(config["runtime"]["num_workers"]),
        max_val_samples=config["dataset"].get("max_val_samples"),
        download=False,
        seed=int(config["runtime"]["seed"]),
    )
    meter = SegmentationMeter(num_classes=int(metadata["num_classes"]))
    for images, targets in tqdm(loader, desc="evaluate"):
        meter.update(model(images.to(device)), targets.to(device))
    results = meter.compute()
    results.update({"checkpoint": str(checkpoint_path), "task": config["dataset"]["task"]})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    metrics = evaluate(args.checkpoint, args.config)
    output_path = PROJECT_ROOT / "reports" / "evaluation_metrics.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
