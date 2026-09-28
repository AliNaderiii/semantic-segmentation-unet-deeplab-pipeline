# Architecture Details

## Data Pipeline

Pascal VOC 2012 Segmentation -> Binary Foreground/Background

- VOC mask values: 0=background, 1-20=object classes, 255=void (ignore)
- Binary conversion: foreground = (mask > 0) & (mask != 255) -> 1, else 0
- This mimics defect detection: thin structures (person) vs background

Leakage-safe: Uses official train/val splits from VOC ImageSets.

Augmentation (albumentations):
- Resize 128/256
- HorizontalFlip p=0.5
- RandomBrightnessContrast p=0.3
- ShiftScaleRotate shift=0.05 scale=0.1 rotate=15 p=0.3
- Normalize ImageNet

## Models

### U-Net ResNet18 (Primary)

Encoder: ResNet18
- Stem: 7x7 conv stride 2, maxpool
- Layers: [2,2,2,2] BasicBlock
- Features: 64,128,256,512 channels
- Pretrained ImageNet

Decoder:
- 5 stages, channels [256,128,64,32,16]
- Skip connections from encoder
- Final: 1x1 conv to 2 classes

Params: ~14M
FLOPs: ~15 GFLOPs @256x256

### DeepLabV3+ ResNet50

Encoder: ResNet50 + ASPP
- Atrous rates: [1,6,12,18]
- Output stride: 16

Decoder: DeepLabV3+ with low-level feature (256 channels) concat.

Params: ~42M

### SegFormer MiT-B2

Hierarchical transformer, no positional encoding, MLP decoder.

## Loss

CombinedLoss = 0.5*Dice + 0.5*CE

Dice handles imbalance, CE stabilizes training.

Smooth = 1e-6

## Training

Optimizer: Adam lr=1e-4
Scheduler: ReduceLROnPlateau max mode factor 0.5 patience 3
Batch: 4/8
Epochs: 3 fast / 15 full

Metrics per batch:
- mIoU: mean IoU over classes (ignore NaN where class absent)
- Dice: mean Dice over classes
- Pixel Accuracy: (pred==true).mean()

Best checkpoint by val mIoU.

## Inference

Resize 256, normalize, forward, argmax, resize to original via nearest.

Overlay: colored mask green [0,255,0] alpha 0.5.

FastAPI: loads model on startup, streaming PNG.

## Deployment

- CPU only, no CUDA required
- 38ms @128px ResNet18
- Docker: python:3.9-slim, pip install, uvicorn 0.0.0.0:8000
