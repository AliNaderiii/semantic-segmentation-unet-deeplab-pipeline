"""Train Pascal VOC models without using official validation data for selection."""
from __future__ import annotations

import argparse
import json
import platform
import random
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .config import PROJECT_ROOT, load_config
from .data_loader import get_dataloaders, write_split_manifest
from .metrics import SegmentationMeter
from .models import CombinedLoss, count_parameters, get_model

PROTOCOL = "development validation derived only from official train; official val held out"


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
    """Run one train or development-validation epoch with global metrics."""
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMeter(num_classes=num_classes)
    total_loss, total_images = 0.0, 0

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
            total_loss += float(loss.detach().item()) * images.shape[0]
            total_images += images.shape[0]
            meter.update(logits, masks)

    if total_images == 0:
        raise RuntimeError("The data loader is empty")
    return total_loss / total_images, meter.compute()


def checkpoint_payload(
    model: torch.nn.Module,
    config: dict[str, Any],
    split_manifest: dict[str, Any],
    epoch: int,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a checkpoint that evaluate.py can reject when protocol metadata is absent."""
    return {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "model": deepcopy(config["model"]),
        "dataset": {
            "name": "Pascal VOC 2012",
            "task": config["dataset"]["task"],
            "image_size": config["dataset"]["image_size"],
            "validation_fraction": config["dataset"]["validation_fraction"],
            "split_seed": config["dataset"]["split_seed"],
            "protocol": PROTOCOL,
        },
        "split_counts": {
            key: len(split_manifest[f"{key}_ids"]) for key in ("train", "validation", "test")
        },
        "epoch": epoch,
        "validation_metrics": validation_metrics,
    }


def train(config: dict[str, Any], allow_download: bool = False) -> dict[str, Any]:
    """Train and select only on deterministic development validation data."""
    runtime = config["runtime"]
    training = config["training"]
    dataset = config["dataset"]
    model_config = config["model"]
    set_seed(int(runtime["seed"]), bool(runtime["deterministic"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    loaders, split_manifest = get_dataloaders(
        batch_size=int(training["batch_size"]),
        image_size=int(dataset["image_size"]),
        task=dataset["task"],
        data_root=PROJECT_ROOT / dataset["root"],
        validation_fraction=float(dataset["validation_fraction"]),
        split_seed=int(dataset["split_seed"]),
        num_workers=int(runtime["num_workers"]),
        max_train_samples=dataset.get("max_train_samples"),
        max_test_samples=dataset.get("max_test_samples"),
        download=allow_download,
    )
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    write_split_manifest(split_manifest, checkpoint_dir / "split_manifest.json")

    model = get_model(
        model_name=model_config["name"],
        encoder=model_config["encoder"],
        num_classes=int(model_config["num_classes"]),
        pretrained=bool(model_config["pretrained"]),
    ).to(device)
    print(
        f"Model: {model_config['name']} / {model_config['encoder']} / "
        f"{count_parameters(model):.2f}M parameters"
    )
    criterion = CombinedLoss(
        dice_weight=float(training["loss_dice_weight"]),
        ce_weight=float(training["loss_ce_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
    )

    history: list[dict[str, Any]] = []
    best_iou, epochs_without_improvement = float("-inf"), 0
    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(
            model, loaders["train"], criterion, device, int(model_config["num_classes"]), optimizer
        )
        validation_loss, validation_metrics = run_epoch(
            model, loaders["validation"], criterion, device, int(model_config["num_classes"])
        )
        scheduler.step(validation_metrics["mean_iou"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | train loss={train_loss:.4f}, mIoU={train_metrics['mean_iou']:.4f} "
            f"| dev-val loss={validation_loss:.4f}, mIoU={validation_metrics['mean_iou']:.4f}, "
            f"Dice={validation_metrics['mean_dice']:.4f}"
        )

        if validation_metrics["mean_iou"] > best_iou:
            best_iou = validation_metrics["mean_iou"]
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(model, config, split_manifest, epoch, validation_metrics),
                checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(training["early_stopping_patience"]):
            print("Early stopping: development validation mIoU did not improve.")
            break

    last_record = history[-1]
    torch.save(
        checkpoint_payload(
            model, config, split_manifest, int(last_record["epoch"]), last_record["validation"]
        ),
        checkpoint_dir / "last.pt",
    )
    summary = {
        "protocol": "official VOC validation split was not iterated for model selection or scheduler decisions",
        "best_validation_miou": best_iou,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "history": history,
    }
    (checkpoint_dir / "training_history.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download Pascal VOC if absent (roughly 2 GB). Prefer src.download_data explicitly.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Override config with small, non-reportable subsets."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configuration = load_config(args.config)
    if args.smoke_test:
        configuration = deepcopy(configuration)
        configuration["dataset"].update(
            {"max_train_samples": 16, "max_test_samples": 8, "image_size": 128}
        )
        configuration["training"].update({"epochs": 1, "batch_size": 2})
    result = train(configuration, allow_download=args.download_data)
    print(json.dumps({"best_validation_miou": result["best_validation_miou"]}, indent=2))
