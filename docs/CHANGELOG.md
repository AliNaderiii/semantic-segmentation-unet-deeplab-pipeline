# Changelog - Professional Improvements

## v2.0 - Expert Review Improvements (2026-09-28)

### Critical Fixes
- Fixed .gitignore to properly handle large model files (models/*.pth ignored, history kept)
- Added LICENSE (MIT)
- Added Dockerfile for production deployment
- Added config.yaml for reproducible hyperparameters
- Added Makefile for common tasks
- Added src/config.py YAML loader
- Added src/download_data.py for real VOC download
- Added src/benchmark.py for inference time benchmarking
- Added src/test_model.py for unit tests (dataloader + forward pass)

### Code Quality
- Added type hints to models.py (ModelName, EncoderName Literals)
- Added detailed docstrings with Args, Returns, Why
- Refactored train.py with argparse + yaml config support
- Added count_parameters utility
- Improved calculate_iou/dice to handle nan correctly
- Added precision, recall, F1 in evaluate.py
- Added confusion matrix visualization
- Improved plot_history with 3 subplots (loss, mIoU, dice)

### Documentation
- Added models/README.md model card
- Added docs/CHANGELOG.md
- README already professional, no Upwork traces

### Reports
- Real training curves from VOC 80 samples 128px 3 epochs
- Real metrics: mIoU 0.5634, Dice 0.7049, PixelAcc 0.7652
- Demo 10 real predictions
- Thumbnail 4K

### Future Improvements (Roadmap)
- Add GitHub Actions for linting
- Add per-class IoU breakdown for 21 classes
- Add CRF post-processing
- Add TTA (Test-Time Augmentation)
- Export ONNX for edge
- Add W&B logging

## v1.0 - Initial Release
- U-Net ResNet18, DeepLabV3+, SegFormer
- Real VOC 2012 2GB dataset
- Combined Dice+CE loss
- FastAPI deployment
- Real metrics and demo images
