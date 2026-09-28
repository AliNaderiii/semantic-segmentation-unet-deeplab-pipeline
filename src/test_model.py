"""Simple tests for data loader and model forward - no fake data"""
import torch
from data_loader import get_dataloaders
from models import get_model

def test_dataloader():
    print("Testing dataloader...")
    train_loader, val_loader = get_dataloaders(batch_size=2, img_size=128, num_samples=20, root='../data')
    images, masks = next(iter(train_loader))
    assert images.shape == (2, 3, 128, 128), f"Image shape {images.shape}"
    assert masks.shape == (2, 128, 128), f"Mask shape {masks.shape}"
    print(f"✓ Dataloader OK - Images {images.shape}, Masks {masks.shape}")

def test_model_forward():
    print("Testing model forward...")
    for model_name, encoder in [('unet','resnet18'), ('deeplabv3plus','resnet50'), ('fpn','resnet34')]:
        model = get_model(model_name, num_classes=2, encoder=encoder)
        x = torch.randn(1, 3, 128, 128)
        y = model(x)
        assert y.shape[0] == 1 and y.shape[1] == 2, f"{model_name} output {y.shape}"
        print(f"✓ {model_name} {encoder} forward OK - {y.shape}")

if __name__ == "__main__":
    test_dataloader()
    test_model_forward()
    print("All tests passed!")
