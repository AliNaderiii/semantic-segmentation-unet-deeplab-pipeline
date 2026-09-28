"""Download real datasets - Pascal VOC 2012"""
from pathlib import Path
from torchvision.datasets import VOCSegmentation

def download_voc(root="./data", year="2012"):
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Pascal VOC {year} to {root_path} (2.00GB)...")
    VOCSegmentation(root=str(root_path), year=year, image_set='train', download=True)
    VOCSegmentation(root=str(root_path), year=year, image_set='val', download=True)
    print("Done! Files in", root_path)

if __name__ == "__main__":
    download_voc()
