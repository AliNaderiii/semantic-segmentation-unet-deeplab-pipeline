# Visual evidence policy

This repository has two deliberately separate visual layers.

1. `assets/pipeline-protocol.svg` documents the current v3 split-integrity protocol. It states that development validation is derived only from official Pascal VOC train and official VOC val is held out until final evaluation. It makes no performance claim.
2. `assets/legacy-v1/` holds selected genuine visuals retained from the pre-v2 repository. They are archival qualitative context only. Their annotations must not be represented as current v3 results because void-label handling and evaluation protocol changed.

After a recorded v3 experiment, generate a reportable dashboard only from its exact configuration, split manifest, protocol-aware checkpoint metadata, training history, and held-out official-VOC-validation output. This preserves visual storytelling without implying that a legacy graphic validates the current protocol.
