"""Generate a dashboard only from recorded Pascal VOC experiment artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import PROJECT_ROOT


def build_dashboard(
    history_path: str | Path, heldout_metrics_path: str | Path, output_path: str | Path
) -> None:
    """Create a reproducible dashboard; no metric values are hard-coded."""
    history: dict[str, Any] = json.loads(Path(history_path).read_text(encoding="utf-8"))
    heldout: dict[str, Any] = json.loads(Path(heldout_metrics_path).read_text(encoding="utf-8"))
    entries = history.get("history", [])
    if not entries:
        raise ValueError("Training history contains no epochs")
    epochs = [entry["epoch"] for entry in entries]
    train_loss = [entry["train_loss"] for entry in entries]
    validation_loss = [entry["validation_loss"] for entry in entries]
    train_iou = [entry["train"]["mean_iou"] for entry in entries]
    validation_iou = [entry["validation"]["mean_iou"] for entry in entries]
    confusion_matrix = np.asarray(heldout["confusion_matrix"])

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(epochs, train_loss, label="train", linewidth=2)
    axes[0].plot(epochs, validation_loss, label="development validation", linewidth=2)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Combined loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_iou, label="train mIoU", linewidth=2)
    axes[1].plot(epochs, validation_iou, label="development validation mIoU", linewidth=2)
    axes[1].set(title="Mean IoU", xlabel="Epoch", ylabel="mIoU")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    image = axes[2].imshow(confusion_matrix, cmap="Blues")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    axes[2].set(
        title="Held-out official VOC val confusion matrix",
        xlabel="Predicted class",
        ylabel="Ground-truth class",
    )
    class_count = confusion_matrix.shape[0]
    if class_count <= 5:
        axes[2].set_xticks(range(class_count))
        axes[2].set_yticks(range(class_count))
        for row in range(class_count):
            for column in range(class_count):
                axes[2].text(
                    column,
                    row,
                    str(confusion_matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    else:
        axes[2].set_xticks([])
        axes[2].set_yticks([])

    figure.suptitle(
        "Pascal VOC experiment | held-out official validation mIoU: "
        f"{heldout['mean_iou']} | held-out mean Dice: {heldout['mean_dice']}"
    )
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history", default=str(PROJECT_ROOT / "checkpoints" / "training_history.json")
    )
    parser.add_argument(
        "--heldout-metrics", default=str(PROJECT_ROOT / "reports" / "heldout_val_metrics.json")
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "reports" / "experiment_dashboard.png")
    )
    args = parser.parse_args()
    build_dashboard(args.history, args.heldout_metrics, args.output)
