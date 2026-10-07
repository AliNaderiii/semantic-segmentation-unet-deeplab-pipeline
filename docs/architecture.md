# Architecture and evaluation contract

## Data flow

1. `src.download_data` retrieves Pascal VOC 2012 deliberately.
2. `VOCSegmentationDataset` uses official `train`/`val` splits.
3. Training transforms resize, flip, brightness/contrast, and shift/scale/rotate only the training split. Validation uses resize and ImageNet normalization only.
4. VOC labels are either preserved as 0–20 (`voc_multiclass`) or mapped to background/foreground (`binary_foreground`). Void pixels remain 255.
5. A SMP model produces logits. `CombinedLoss` excludes void pixels from both Dice and cross-entropy.
6. `SegmentationMeter` aggregates one pixel confusion matrix for the entire epoch, then derives mIoU, Dice, precision, recall, F1, and pixel accuracy.

## Reproducibility contract

`runtime.seed` seeds Python, NumPy, and PyTorch. `runtime.deterministic: true` requests deterministic PyTorch operations where possible. Exact bit-for-bit reproducibility can still depend on package, driver, and hardware versions; record these with every experiment.

A checkpoint stores the architecture, encoder, class count, task, image size, epoch, validation metrics, and model state dictionary. Evaluation and inference read this metadata, so a binary checkpoint cannot silently be loaded as a 21-class model.

## Measurement caveats

- mIoU is macro-averaged over classes that appear in the accumulated confusion matrix.
- Pixel accuracy can be dominated by background and must not be presented alone.
- A smoke-test subset is an integration check, not a reportable benchmark.
- Benchmark latency is hardware- and input-size-specific; run `src.benchmark` locally and report its environment.
