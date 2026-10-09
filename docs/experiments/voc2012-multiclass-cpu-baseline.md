# Pascal VOC 2012 multiclass CPU baseline — recorded v3 experiment

> **Scope.** This is one low-resolution, CPU-only reference baseline using the current held-out protocol. It is not a state-of-the-art claim, a deployment recommendation, a confidence interval, or a direct comparison with runs that use other resolutions, architectures, splits, preprocessing, compute, or training budgets.

## Result at a glance

| Held-out official Pascal VOC validation metric | Value |
| --- | ---: |
| Mean IoU | **0.099281** |
| Mean Dice | **0.134228** |
| Mean precision | 0.274585 |
| Mean recall | 0.170493 |
| Pixel accuracy | 0.771547 |

The public labeled VOC `val` split was held out from model selection and iterated once only after the configuration and validation-selected checkpoint were fixed. Pixel accuracy is listed for completeness but is background-dominated and must not be interpreted as a substitute for multiclass mIoU.

## Protocol and provenance

| Item | Recorded value |
| --- | --- |
| Code commit used for training | `bfa45995a712dc853415783bb24b80a0233e94c3` |
| Versioned configuration | [`experiments/configs/voc2012_multiclass_cpu_baseline.yaml`](../../experiments/configs/voc2012_multiclass_cpu_baseline.yaml) |
| Task | Native Pascal VOC 2012 semantic segmentation, 21 classes |
| Model | U-Net with ResNet-18 encoder; ImageNet initialization enabled |
| Input resolution | 128 × 128 |
| Official source train | 1,464 images |
| Development data | 1,171 train / 293 validation images, derived deterministically from official train only |
| Development split seed | 42 |
| Final held-out data | 1,449 labeled images from official Pascal VOC `val` |
| Checkpoint selection metric | Development-validation mean IoU |
| Selected checkpoint | `best.pt`, epoch 9; development-validation mIoU 0.103012 |
| Runtime | CPU-only, 8 PyTorch threads; Python 3.11.9; torch 2.6.0+cpu |
| Training command wall time | 1225.97 s (20 m 25.97 s) |
| Held-out evaluation wall time | 126.96 s (2 m 6.96 s) |

The training history records that official VOC validation was not iterated for model selection, learning-rate scheduling, or early stopping. Pascal VOC's public validation split is correctly named here as held-out validation; it is not the inaccessible official challenge test server.

## Per-class held-out validation metrics

| Class | IoU | Dice | Support pixels |
| --- | ---: | ---: | ---: |
| 0: background | 0.831459 | 0.907974 | 16,449,493 |
| 1: aeroplane | 0.000000 | 0.000000 | 186,618 |
| 2: bicycle | 0.000000 | 0.000000 | 73,423 |
| 3: bird | 0.000000 | 0.000000 | 194,578 |
| 4: boat | 0.000274 | 0.000548 | 123,476 |
| 5: bottle | 0.000000 | 0.000000 | 177,917 |
| 6: bus | 0.277298 | 0.434195 | 397,869 |
| 7: car | 0.000359 | 0.000717 | 336,833 |
| 8: cat | 0.278989 | 0.436265 | 508,916 |
| 9: chair | 0.000000 | 0.000000 | 192,431 |
| 10: cow | 0.000000 | 0.000000 | 270,204 |
| 11: diningtable | 0.157022 | 0.271425 | 255,714 |
| 12: dog | 0.000004 | 0.000009 | 460,581 |
| 13: horse | 0.000000 | 0.000000 | 244,518 |
| 14: motorbike | 0.002930 | 0.005843 | 234,359 |
| 15: person | 0.472941 | 0.642173 | 1,190,733 |
| 16: pottedplant | 0.000000 | 0.000000 | 118,283 |
| 17: sheep | 0.000000 | 0.000000 | 172,211 |
| 18: sofa | 0.000000 | 0.000000 | 316,267 |
| 19: train | 0.063626 | 0.119639 | 372,693 |
| 20: tvmonitor | 0.000000 | 0.000000 | 154,877 |

The dataset-level 21 × 21 confusion matrix is retained in the versioned metrics JSON. The relatively modest mIoU is expected for this intentionally constrained 128 × 128 / 10-epoch CPU baseline; it is included transparently rather than obscured by the higher background-dominated pixel accuracy.

## Versioned evidence bundle

The artifacts below are copies of this recorded run, not synthetic examples:

- [`assets/experiments/voc2012_multiclass_cpu_baseline_dashboard.png`](../../assets/experiments/voc2012_multiclass_cpu_baseline_dashboard.png) — generated from the recorded history and held-out metrics;
- [`experiments/runs/voc2012-multiclass-cpu-baseline/heldout_val_metrics.json`](../../experiments/runs/voc2012-multiclass-cpu-baseline/heldout_val_metrics.json) — exact held-out output;
- [`experiments/runs/voc2012-multiclass-cpu-baseline/training_history.json`](../../experiments/runs/voc2012-multiclass-cpu-baseline/training_history.json) — per-epoch development metrics;
- [`experiments/runs/voc2012-multiclass-cpu-baseline/split_manifest.json`](../../experiments/runs/voc2012-multiclass-cpu-baseline/split_manifest.json) — deterministic source/development/held-out identifiers;
- [`experiments/runs/voc2012-multiclass-cpu-baseline/run_metadata.json`](../../experiments/runs/voc2012-multiclass-cpu-baseline/run_metadata.json) and [`artifact_manifest.json`](../../experiments/runs/voc2012-multiclass-cpu-baseline/artifact_manifest.json) — environment, timings, and SHA-256 fingerprints.

The raw Pascal VOC data and the local model weights are deliberately excluded. The artifact manifest identifies the selected local checkpoint with SHA-256 without distributing it.

## Limitations and responsible interpretation

- This is one seed at 128 × 128 with a fixed 10-epoch CPU budget; it is not a multi-seed study or confidence interval.
- The low-resolution baseline should not be compared with full-resolution, longer-trained, augmented, or differently split VOC results as if they were identical experiments.
- Pascal VOC's data terms apply independently of this repository's MIT code license. No raw images or labels are published here.
- Before any real-world use, a model needs task-specific validation, error analysis, operating-condition tests, monitoring, governance, and human review.
