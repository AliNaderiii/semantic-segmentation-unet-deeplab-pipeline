# Generated reports

This directory is deliberately empty in Git. `python -m src.evaluate` writes `evaluation_metrics.json` here after evaluating a real local checkpoint. Commit a small, versioned metrics artifact only when it includes the exact configuration, dataset split, seed, hardware, and commit SHA needed to interpret it.
