import pytest
import torch

from src.metrics import SegmentationMeter


def logits_from_labels(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    logits = torch.full((1, num_classes, 1, labels.numel()), -5.0)
    for index, label in enumerate(labels.tolist()):
        logits[0, label, 0, index] = 5.0
    return logits


def test_meter_aggregates_pixel_confusion_matrix() -> None:
    meter = SegmentationMeter(num_classes=2)
    # predictions = [0, 1, 1, 0], targets = [0, 1, 0, 1]
    meter.update(logits_from_labels(torch.tensor([0, 1, 1, 0])), torch.tensor([[[0, 1, 0, 1]]]))
    result = meter.compute()
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["mean_iou"] == pytest.approx(1 / 3)
    assert result["mean_dice"] == pytest.approx(0.5)
    assert result["pixel_accuracy"] == pytest.approx(0.5)


def test_void_pixels_are_excluded() -> None:
    meter = SegmentationMeter(num_classes=2, ignore_index=255)
    meter.update(logits_from_labels(torch.tensor([0, 1, 1])), torch.tensor([[[0, 1, 255]]]))
    result = meter.compute()
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["mean_iou"] == pytest.approx(1.0)
    assert result["support"] == [1, 1]
