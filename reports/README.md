# Generated experiment artifacts

`reports/` is intentionally ignored by Git because it is the scratch location for outputs from the current local checkpoint.

After a configuration is final and a validation-selected checkpoint exists, run exactly one final official-VOC-validation evaluation:

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --config config.yaml
python -m src.reporting
```

This creates `heldout_val_metrics.json` and `experiment_dashboard.png` from the recorded checkpoint, development split manifest, training history, and untouched official VOC `val` split. Do not commit a dashboard or headline score without its exact configuration, split manifest, code commit, seed, hardware, dependency versions, checkpoint hash, and dataset provenance.
