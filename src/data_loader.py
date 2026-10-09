"""Pascal VOC loaders with deterministic development validation and held-out final evaluation."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal, Sequence

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation

VOID_LABEL = 255
VOC_NUM_CLASSES = 21
TaskName = Literal["binary_foreground", "voc_multiclass"]
SplitName = Literal["train", "validation", "test"]


def get_transforms(train: bool, image_size: int) -> A.Compose:
    transforms: list[A.BasicTransform] = [A.Resize(image_size, image_size)]
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3
                ),
            ]
        )
    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def _sample_ids(dataset: VOCSegmentation) -> list[str]:
    """Return stable VOC identifiers and reject malformed source metadata."""
    identifiers = [Path(path).stem for path in dataset.images]
    if not identifiers:
        raise FileNotFoundError("Pascal VOC split contains no images")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Pascal VOC split contains duplicate image identifiers")
    return identifiers


def discover_sample_ids(
    root: str | Path,
    split: Literal["train", "val"],
    *,
    download: bool = False,
    max_samples: int | None = None,
) -> list[str]:
    """Discover a local VOC split without reading image pixels."""
    source = VOCSegmentation(root=str(root), year="2012", image_set=split, download=download)
    identifiers = _sample_ids(source)
    if max_samples is not None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least two or null")
        identifiers = identifiers[: min(max_samples, len(identifiers))]
    return identifiers


def split_training_ids(
    sample_ids: Sequence[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split only official VOC training identifiers for development selection.

    The returned lists are canonical-order deterministic. The official VOC
    ``val`` split is never passed into this function and is reserved for the
    one final evaluation after checkpoint selection is complete.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    ordered_ids = sorted(sample_ids)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Training identifiers must be unique")
    if len(ordered_ids) < 2:
        raise ValueError("At least two official training samples are required")
    shuffled_ids = list(ordered_ids)
    random.Random(seed).shuffle(shuffled_ids)
    validation_count = max(1, round(len(shuffled_ids) * validation_fraction))
    validation_count = min(validation_count, len(shuffled_ids) - 1)
    validation_ids = set(shuffled_ids[:validation_count])
    train_ids = [sample_id for sample_id in ordered_ids if sample_id not in validation_ids]
    validation = [sample_id for sample_id in ordered_ids if sample_id in validation_ids]
    return train_ids, validation


class VOCSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """VOC data over an explicit identifier set and a task-specific label view."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"],
        task: TaskName,
        transform: A.Compose,
        sample_ids: Sequence[str],
        download: bool = False,
    ) -> None:
        self.task = task
        source = VOCSegmentation(root=str(root), year="2012", image_set=split, download=download)
        source_ids = _sample_ids(source)
        index_by_id = {sample_id: index for index, sample_id in enumerate(source_ids)}
        requested_ids = list(sample_ids)
        if not requested_ids:
            raise ValueError("Dataset received no sample identifiers")
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("Dataset received duplicate sample identifiers")
        missing_ids = set(requested_ids).difference(index_by_id)
        if missing_ids:
            example = ", ".join(sorted(missing_ids)[:5])
            raise ValueError(f"Requested VOC identifiers are absent from {split}: {example}")
        self.source = source
        self.indices = [index_by_id[sample_id] for sample_id in requested_ids]
        self.sample_ids = requested_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.source[self.indices[index]]
        image_array = np.asarray(image.convert("RGB"))
        mask_array = np.asarray(mask, dtype=np.uint8)
        if self.task == "binary_foreground":
            void = mask_array == VOID_LABEL
            mask_array = (mask_array > 0).astype(np.uint8)
            mask_array[void] = VOID_LABEL
        transformed = self.transform(image=image_array, mask=mask_array)
        return transformed["image"], transformed["mask"].long()


def get_dataloaders(
    *,
    batch_size: int,
    image_size: int,
    task: TaskName,
    data_root: str | Path,
    validation_fraction: float,
    split_seed: int,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    download: bool = False,
) -> tuple[dict[SplitName, DataLoader], dict[str, Any]]:
    """Build development and final-evaluation loaders plus split provenance.

    ``train`` and ``validation`` are both derived only from official VOC train.
    ``test`` is the untouched official VOC validation split. The latter is
    constructed for explicit final evaluation but never iterated by ``src.train``.
    """
    source_train_ids = discover_sample_ids(
        data_root, "train", download=download, max_samples=max_train_samples
    )
    source_test_ids = discover_sample_ids(
        data_root, "val", download=False, max_samples=max_test_samples
    )
    train_ids, validation_ids = split_training_ids(source_train_ids, validation_fraction, split_seed)
    test_ids = sorted(source_test_ids)
    if set(train_ids) & set(validation_ids):
        raise ValueError("Development train/validation overlap detected")
    if (set(train_ids) | set(validation_ids)) & set(test_ids):
        raise ValueError("Official VOC train and val identifiers overlap")

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(split_seed)
    loaders: dict[SplitName, DataLoader] = {
        "train": DataLoader(
            VOCSegmentationDataset(
                data_root, "train", task, get_transforms(True, image_size), train_ids
            ),
            shuffle=True,
            generator=generator,
            **loader_options,
        ),
        "validation": DataLoader(
            VOCSegmentationDataset(
                data_root, "train", task, get_transforms(False, image_size), validation_ids
            ),
            shuffle=False,
            **loader_options,
        ),
        "test": DataLoader(
            VOCSegmentationDataset(
                data_root, "val", task, get_transforms(False, image_size), test_ids
            ),
            shuffle=False,
            **loader_options,
        ),
    }
    manifest: dict[str, Any] = {
        "dataset": "Pascal VOC 2012",
        "protocol": "development validation derived only from official train; official val held out",
        "source_train_count": len(source_train_ids),
        "source_test_count": len(source_test_ids),
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
    }
    return loaders, manifest


def write_split_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    """Write the exact split identity list used by a recorded experiment."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
