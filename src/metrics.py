"""Streaming, void-aware metrics for semantic segmentation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _nanmean(values: torch.Tensor) -> float:
    finite = values[~torch.isnan(values)]
    return float(finite.mean().item()) if finite.numel() else 0.0


@dataclass
class SegmentationMeter:
    """Accumulate a pixel confusion matrix and derive dataset-level metrics.

    Rows are target labels and columns are predicted labels. Pixels carrying the
    Pascal VOC void label (255) are excluded from every calculation.
    """

    num_classes: int
    ignore_index: int = 255
    confusion_matrix: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Add a batch of logits ``[B, C, H, W]`` and labels ``[B, H, W]``."""
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError("Expected logits [B,C,H,W] and target [B,H,W]")
        if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
            raise ValueError("Logit and target shapes are incompatible")
        if logits.shape[1] != self.num_classes:
            raise ValueError("Logit class count does not match the meter")

        prediction = logits.argmax(dim=1).detach().to("cpu", dtype=torch.int64)
        target = target.detach().to("cpu", dtype=torch.int64)
        valid = target != self.ignore_index
        if not valid.any():
            return
        target_valid = target[valid]
        prediction_valid = prediction[valid]
        if target_valid.min() < 0 or target_valid.max() >= self.num_classes:
            raise ValueError("Target contains a class outside the configured range")
        encoded = self.num_classes * target_valid + prediction_valid
        batch_matrix = torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += batch_matrix

    def compute(self) -> dict[str, Any]:
        """Return global pixel, per-class, and macro metrics."""
        cm = self.confusion_matrix.to(torch.float64)
        true_positive = cm.diag()
        false_positive = cm.sum(dim=0) - true_positive
        false_negative = cm.sum(dim=1) - true_positive
        union = true_positive + false_positive + false_negative
        support = cm.sum(dim=1)

        iou = torch.where(union > 0, true_positive / union, torch.nan)
        dice_denominator = 2 * true_positive + false_positive + false_negative
        dice = torch.where(
            dice_denominator > 0, 2 * true_positive / dice_denominator, torch.nan
        )
        precision_denominator = true_positive + false_positive
        precision = torch.where(
            precision_denominator > 0,
            true_positive / precision_denominator,
            torch.nan,
        )
        recall = torch.where(
            support > 0, true_positive / support, torch.nan
        )
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = torch.where(f1_denominator > 0, 2 * true_positive / f1_denominator, torch.nan)

        total = cm.sum()
        pixel_accuracy = float((true_positive.sum() / total).item()) if total > 0 else 0.0
        return {
            "pixel_accuracy": pixel_accuracy,
            "mean_iou": _nanmean(iou),
            "mean_dice": _nanmean(dice),
            "mean_precision": _nanmean(precision),
            "mean_recall": _nanmean(recall),
            "mean_f1": _nanmean(f1),
            "per_class_iou": [None if torch.isnan(x) else round(float(x), 6) for x in iou],
            "per_class_dice": [None if torch.isnan(x) else round(float(x), 6) for x in dice],
            "support": [int(x) for x in support.tolist()],
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
