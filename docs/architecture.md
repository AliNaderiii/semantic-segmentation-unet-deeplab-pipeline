# Architecture and evaluation contract

## Data flow

1. `src.download_data` retrieves Pascal VOC 2012 deliberately.
2. `src.data_loader` discovers official `train` identifiers, derives deterministic development train/validation identifiers, and records a manifest. Official VOC `val` identifiers remain held out.
3. Training transforms resize, flip, brightness/contrast, and shift/scale/rotate only development training. Development validation and held-out official validation use resize and ImageNet normalization only.
4. VOC labels are either preserved as 0–20 (`voc_multiclass`) or mapped to background/foreground (`binary_foreground`). Void pixels remain 255.
5. A segmentation-models-pytorch model produces logits. `CombinedLoss` excludes void pixels from both Dice and cross-entropy.
6. `SegmentationMeter` aggregates one pixel confusion matrix for each complete split, then derives mIoU, Dice, precision, recall, F1, and pixel accuracy.

## Evaluation contract

- `src.train` iterates only development train and development validation. It uses development-validation mIoU for checkpoint selection, learning-rate scheduling, and early stopping.
- The labeled public VOC `val` split is not iterated by `src.train`; `src.evaluate` is the only command that iterates it.
- New checkpoints store protocol metadata, validation fraction, split seed, and train/development-validation/held-out counts. `src.evaluate` rejects legacy or mismatched checkpoints.
- `checkpoints/split_manifest.json` records the exact identifiers. The same seed, data layout, configuration, and code revision must be used to reproduce a run.
- Pascal VOC's public `val` split is held out for this reference protocol. It is not the official challenge test server, whose labels are not public.

## Reproducibility contract

`runtime.seed` seeds Python, NumPy, and PyTorch. `runtime.deterministic: true` requests deterministic PyTorch operations where possible. Exact bit-for-bit reproducibility can still depend on package, driver, and hardware versions; record these with every experiment.

A checkpoint stores architecture, encoder, class count, task, image size, epoch, development-validation metrics, protocol metadata, split counts, and model state. Evaluation and inference read this metadata, so a binary checkpoint cannot silently be loaded as a 21-class model or evaluated under a mismatched split protocol.

## Measurement caveats

- mIoU is macro-averaged over classes that appear in the accumulated confusion matrix.
- Pixel accuracy can be dominated by background and must not be presented alone.
- A smoke-test subset is an integration check, not a reportable benchmark.
- Benchmark latency is hardware- and input-size-specific; run `src.benchmark` locally and report its environment.
