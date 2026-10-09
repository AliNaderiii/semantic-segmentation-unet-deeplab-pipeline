# Changelog


## 3.1.0 — recorded Pascal VOC multiclass CPU baseline

- Added a real held-out-protocol Pascal VOC experiment: native 21-class task, U-Net / pretrained ResNet-18, 128 × 128, seed 42, and a 1,171/293 development split from official train only.
- Selected epoch 9 exclusively by development-validation mIoU (`0.103012`), then evaluated the untouched 1,449-image official VOC validation split once.
- Recorded held-out mIoU `0.099281`, mean Dice `0.134228`, mean precision `0.274585`, and mean recall `0.170493`.
- Versioned a real dashboard and safe provenance artifacts while continuing to exclude raw Pascal VOC data and checkpoint weights.

## 3.0.0 — held-out Pascal VOC evaluation protocol

### Scientific validity

- Reserved the public, labeled Pascal VOC `val` split for one final evaluation only.
- Added a deterministic development train/validation split derived exclusively from official VOC `train` identifiers.
- Made development-validation mIoU the only checkpoint-selection, scheduler, and early-stopping metric.
- Persisted an exact train/development-validation/held-out identifier manifest with checkpoints and training history.
- Made evaluation reject legacy checkpoints that lack current held-out-protocol metadata.
- Clarified that public VOC `val` is a held-out evaluation split, not the inaccessible official challenge test set.

### Engineering and presentation

- Added artifact-derived reporting for losses, development mIoU curves, and a held-out confusion matrix.
- Updated the protocol diagram, README, visual policy, and generated-artifact guidance to remove ambiguous evaluation claims.
- Preserved void-aware loss and dataset-level streaming metrics, safe inference behavior, CI, and Docker support.

## 2.0.0 — portfolio reliability revision

### Correctness

- Preserved Pascal VOC void pixels instead of relabelling them as background.
- Added void-aware Dice and cross-entropy losses.
- Replaced per-batch averaged metrics with a dataset-level streaming confusion matrix.
- Removed the misleading crack/defect dataset abstraction; the default task is named accurately.
- Made evaluation and inference require a self-describing trained checkpoint.

### Reproducibility and operations

- Added deterministic seeding, official-split loaders, config validation, checkpoints, tests, CI, and linting.
- Changed all executable imports to package-safe `python -m src...` usage.
- Fixed the Docker image: CPU torch is installed once at pinned versions, `curl` is present for its health check, and the health endpoint is real.
- Removed stale generated dashboards, thumbnails, demo images, and unsupported estimated/SOTA result claims. Recreate reports from a recorded experiment.
