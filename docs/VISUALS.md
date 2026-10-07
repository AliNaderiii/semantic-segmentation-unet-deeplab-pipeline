# Visual evidence policy

This repository has two deliberately separate visual layers.

1. `assets/pipeline-protocol.svg` documents the current code and experiment protocol. It does not claim performance, so it is safe to show before a trained checkpoint exists.
2. `assets/legacy-v1/` holds selected genuine visuals retained from the pre-v2 repository. These are clearly marked archival qualitative context. Their numerical annotations must not be represented as v2 results.

After a recorded v2 experiment, generate reportable dashboards only from its configuration, split manifest, checkpoint metadata, training history, and held-out evaluation output. This preserves visual storytelling without implying that a legacy graphic validates the current protocol.
