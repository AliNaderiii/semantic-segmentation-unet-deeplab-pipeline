"""Train a reproducible Pascal VOC foreground or multiclass segmentation model."""
from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .config import PROJECT_ROOT, load_config
from .data_loader import get_dataloaders
from .metrics import SegmentationMeter
from .models import CombinedLoss, count_parameters, get_model


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch as far as the runtime permits."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, Any]]:
    """Run one training or validation epoch and calculate global pixel metrics."""
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMeter(num_classes=num_classes)
    total_loss = 0.0
    total_images = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, leave=False, desc="train" if training else "validate"):
            images, masks = images.to(device), masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_images += batch_size
            meter.update(logits, masks)

    if total_images == 0:
        raise RuntimeError("The data loader is empty")
    return total_loss / total_images, meter.compute()


def _checkpoint_payload(
    model: torch.nn.Module,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "model": deepcopy(config["model"]),
        "dataset": {
            "task": config["dataset"]["task"],
            "image_size": config["dataset"]["image_size"],
        },
        "epoch": epoch,
        "validation_metrics": metrics,
    }


def train(config: dict[str, Any], allow_download: bool = False) -> dict[str, Any]:
    """Train from a validated configuration and save self-describing checkpoints."""
    runtime = config["runtime"]
    training = config["training"]
    dataset = config["dataset"]
    model_config = config["model"]
    set_seed(int(runtime["seed"]), bool(runtime["deterministic"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader = get_dataloaders(
        batch_size=int(training["batch_size"]),
        image_size=int(dataset["image_size"]),
        task=dataset["task"],
        data_root=PROJECT_ROOT / dataset["root"],
        num_workers=int(runtime["num_workers"]),
        max_train_samples=dataset.get("max_train_samples"),
        max_val_samples=dataset.get("max_val_samples"),
        download=allow_download,
        seed=int(runtime["seed"]),
    )
    model = get_model(
        model_name=model_config["name"],
        encoder=model_config["encoder"],
        num_classes=int(model_config["num_classes"]),
        pretrained=bool(model_config["pretrained"]),
    ).to(device)
    print(f"Model: {model_config['name']} / {model_config['encoder']} / {count_parameters(model):.2f}M parameters")

    criterion = CombinedLoss(
        dice_weight=float(training["loss_dice_weight"]),
        ce_weight=float(training["loss_ce_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
    )

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    history: list[dict[str, Any]] = []
    best_iou, epochs_without_improvement = float("-inf"), 0

    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, int(model_config["num_classes"]), optimizer
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, device, int(model_config["num_classes"])
        )
        scheduler.step(val_metrics["mean_iou"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | train loss={train_loss:.4f}, mIoU={train_metrics['mean_iou']:.4f} "
            f"| val loss={val_loss:.4f}, mIoU={val_metrics['mean_iou']:.4f}, "
            f"Dice={val_metrics['mean_dice']:.4f}"
        )

        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            epochs_without_improvement = 0
            torch.save(_checkpoint_payload(model, config, epoch, val_metrics), checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(training["early_stopping_patience"]):
            print("Early stopping: validation mIoU did not improve.")
            break

    torch.save(_checkpoint_payload(model, config, history[-1]["epoch"], history[-1]["validation"]), checkpoint_dir / "last.pt")
    summary = {"best_validation_miou": best_iou, "history": history}
    with (checkpoint_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download Pascal VOC if it is absent (roughly 2 GB).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Override config with a small, fast training run."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configuration = load_config(args.config)
    if args.smoke_test:
        configuration = deepcopy(configuration)
        configuration["dataset"].update({"max_train_samples": 16, "max_val_samples": 8, "image_size": 128})
        configuration["training"].update({"epochs": 1, "batch_size": 2})
    result = train(configuration, allow_download=args.download_data)
    print(json.dumps({"best_validation_miou": result["best_validation_miou"]}, indent=2))
