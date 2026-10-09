# Generated and recorded experiment artifacts

`reports/` is ignored by Git because it is the local scratch location for outputs from the current checkpoint. A temporary dashboard or metric JSON must not be committed on its own.

The recorded Pascal VOC multiclass CPU baseline is the exception because its complete safe-to-publish evidence bundle has been deliberately copied to:

- `assets/experiments/voc2012_multiclass_cpu_baseline_dashboard.png`;
- `experiments/runs/voc2012-multiclass-cpu-baseline/`;
- `docs/experiments/voc2012-multiclass-cpu-baseline.md`.

That bundle contains no raw Pascal VOC data and no model weights. It includes hashes that identify the local selected checkpoint, configuration, and dashboard. For a new run, keep using this ignored directory and publish a new named evidence bundle only after protocol, provenance, and held-out evaluation have been reviewed.
