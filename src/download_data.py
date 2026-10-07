"""Explicitly download Pascal VOC 2012 instead of hiding a 2 GB transfer in training."""
from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import VOCSegmentation

from .config import PROJECT_ROOT


def download_voc(root: str | Path) -> None:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Pascal VOC 2012 to {target.resolve()} (approximately 2 GB)…")
    # Torchvision downloads the single shared archive only once; calling both
    # official splits verifies that their metadata is available.
    VOCSegmentation(root=str(target), year="2012", image_set="train", download=True)
    VOCSegmentation(root=str(target), year="2012", image_set="val", download=True)
    print("Pascal VOC 2012 is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()
    download_voc(args.root)
