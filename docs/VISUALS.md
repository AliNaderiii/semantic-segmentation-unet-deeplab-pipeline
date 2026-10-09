# Visual evidence policy

This repository has three deliberately separate visual layers.

1. `assets/pipeline-protocol.svg` documents the current v3 split-integrity protocol. It makes no performance claim.
2. `assets/legacy-v1/` holds selected genuine visuals retained from the pre-v2 repository. They are archival qualitative context only. Their annotations must not be represented as current v3 results because void-label handling and evaluation protocol changed.
3. `assets/experiments/voc2012_multiclass_cpu_baseline_dashboard.png` is a reportable v3 visual. It was generated from the recorded training history and the one-time held-out official Pascal VOC validation evaluation documented in [`docs/experiments/voc2012-multiclass-cpu-baseline.md`](experiments/voc2012-multiclass-cpu-baseline.md). Its held-out mIoU is 0.099281; matching configuration, split manifest, metric JSON, run metadata, and SHA-256 artifact manifest are versioned alongside it.

This separation preserves visual storytelling without implying that a legacy graphic validates the current protocol. A new score-bearing visual may be committed only with an equivalently complete real experiment evidence bundle; raw data and checkpoint weights remain excluded.
