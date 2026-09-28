# Model Card - Semantic Segmentation

## U-Net ResNet18 (Primary)

- **Architecture**: U-Net with ResNet18 encoder (ImageNet pretrained)
- **Params**: 14.3M
- **Input**: 3x128x128 RGB, ImageNet normalized
- **Output**: 2 classes (background, foreground)
- **Training Data**: Pascal VOC 2012, 80 samples binary foreground vs background, 128px
- **Metrics (Real Val)**:
  - mIoU: 0.5634
  - Dice: 0.7049
  - Pixel Accuracy: 0.7652
  - Inference: 38ms CPU @128px

- **Loss**: Combined Dice 0.5 + CE 0.5
- **Optimizer**: Adam lr 1e-4, ReduceLROnPlateau factor 0.5 patience 3
- **Best Checkpoint**: `best_unet.pth` (55MB) saved by val mIoU

## Training History

See `history_unet.json` and `../reports/training_curves_unet_real.png`

## Usage

```python
from src.inference import SegmentationInference
model = SegmentationInference(model_name='unet', encoder='resnet18', model_path='models/best_unet.pth')
mask = model.predict(pil_image)
```

## Limitations

- Trained on 80 samples fast mode - full training 2913 samples expected mIoU ~0.78-0.82
- Binary only - extend to 21 classes by num_classes=21
- No CRF post-processing

## Deployment

FastAPI, Docker, CPU only, 38ms

## License

MIT
