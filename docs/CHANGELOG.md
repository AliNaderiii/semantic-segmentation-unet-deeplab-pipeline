# Changelog

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
