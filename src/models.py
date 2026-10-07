"""Segmentation models and a void-aware Dice + cross-entropy objective."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as functional

SUPPORTED_MODELS = {"unet", "deeplabv3plus", "fpn"}


def get_model(
    model_name: str,
    num_classes: int,
    encoder: str,
    pretrained: bool = True,
    in_channels: int = 3,
) -> nn.Module:
    """Create a supported SMP segmentation model with raw output logits."""
    name = model_name.lower().replace("+", "plus")
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(SUPPORTED_MODELS)}")
    common = {
        "encoder_name": encoder,
        "encoder_weights": "imagenet" if pretrained else None,
        "in_channels": in_channels,
        "classes": num_classes,
        "activation": None,
    }
    if name == "unet":
        return smp.Unet(**common)
    if name == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common)
    return smp.FPN(**common)


class DiceLoss(nn.Module):
    """Multiclass soft Dice loss that excludes VOC void pixels."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        if not valid.any():
            # Preserve a valid computational graph for a pathological batch.
            return logits.sum() * 0.0
        safe_targets = targets.masked_fill(~valid, 0)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = functional.one_hot(
            safe_targets, num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2).to(dtype=probabilities.dtype)
        valid_mask = valid.unsqueeze(1).to(dtype=probabilities.dtype)
        intersection = (probabilities * one_hot * valid_mask).sum(dim=(0, 2, 3))
        denominator = (
            (probabilities * valid_mask).sum(dim=(0, 2, 3))
            + (one_hot * valid_mask).sum(dim=(0, 2, 3))
        )
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Dice plus cross-entropy loss for binary or multiclass segmentation."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or ce_weight < 0 or dice_weight + ce_weight == 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.ce_weight * self.cross_entropy(logits, targets)
        )


def count_parameters(model: nn.Module) -> float:
    """Return the total parameter count in millions."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000
