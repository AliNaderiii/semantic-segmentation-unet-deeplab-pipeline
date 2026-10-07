"""Pascal VOC 2012 datasets and loaders with correct treatment of void labels."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VOCSegmentation

VOID_LABEL = 255
VOC_NUM_CLASSES = 21
TaskName = Literal["binary_foreground", "voc_multiclass"]


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


class VOCSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """VOC data as either 21-class labels or an honest foreground/background task.

    ``binary_foreground`` maps classes 1--20 to foreground, leaves background as
    0, and preserves the official VOC void label (255). It is *not* a defect or
    crack dataset. ``voc_multiclass`` preserves the official 0--20 classes.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"],
        task: TaskName,
        transform: A.Compose,
        download: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.task = task
        dataset: Dataset = VOCSegmentation(
            root=str(root), year="2012", image_set=split, download=download
        )
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive or null")
            dataset = Subset(dataset, range(min(max_samples, len(dataset))))
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
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
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    download: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build official VOC train/val loaders without implicit data leakage."""
    train_dataset = VOCSegmentationDataset(
        data_root,
        split="train",
        task=task,
        transform=get_transforms(train=True, image_size=image_size),
        download=download,
        max_samples=max_train_samples,
    )
    val_dataset = VOCSegmentationDataset(
        data_root,
        split="val",
        task=task,
        transform=get_transforms(train=False, image_size=image_size),
        download=download,
        max_samples=max_val_samples,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options),
        DataLoader(val_dataset, shuffle=False, **loader_options),
    )
