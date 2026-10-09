"""Explicitly download and validate Pascal VOC 2012 segmentation data."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT
from .data_loader import discover_sample_ids

EXPECTED_TRAIN_COUNT = 1464
EXPECTED_VAL_COUNT = 1449


def download_voc(root: str | Path) -> None:
    """Download once, then verify the public train/val segmentation manifests."""
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Pascal VOC 2012 to {target.resolve()} (approximately 2 GB)…")
    train_ids = discover_sample_ids(target, "train", download=True)
    val_ids = discover_sample_ids(target, "val", download=False)
    if len(train_ids) != EXPECTED_TRAIN_COUNT or len(val_ids) != EXPECTED_VAL_COUNT:
        raise ValueError(
            "Unexpected Pascal VOC segmentation split counts: "
            f"train={len(train_ids)} (expected {EXPECTED_TRAIN_COUNT}), "
            f"val={len(val_ids)} (expected {EXPECTED_VAL_COUNT})"
        )
    if set(train_ids) & set(val_ids):
        raise ValueError("Pascal VOC official train and val identifier manifests overlap")
    print(
        "Validated "
        f"{len(train_ids)} official train image(s) and {len(val_ids)} held-out official val image(s). "
        "Development validation will be derived only from official train during training."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()
    download_voc(args.root)
