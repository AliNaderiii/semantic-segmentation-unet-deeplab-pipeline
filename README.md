# Pascal VOC Semantic Segmentation — Reproducible Reference Pipeline

[![CI](https://github.com/AliNaderiii/semantic-segmentation-unet-deeplab-pipeline/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A compact, reproducible semantic-segmentation reference implementation using **U-Net**, **DeepLabV3+**, or **FPN** from `segmentation-models-pytorch` and the official **Pascal VOC 2012** train/validation split.

> **Scope and honesty.** The default task is *VOC foreground/background segmentation*: VOC classes 1–20 are foreground and class 0 is background. It is **not an industrial-defect or crack-segmentation model**. The VOC void label (255) is preserved and excluded from loss and metrics. No model checkpoint or headline accuracy is committed to this repository; reproduce an experiment before reporting results.

## Why this revision

The project is structured to be useful in a hiring review:

- an explicit task/data card rather than a vague “production-ready” claim;
- an official train/validation split, with no validation augmentation;
- void-aware Dice + cross-entropy loss and dataset-level (not batch-averaged) mIoU/Dice metrics;
- self-describing checkpoints that contain model and dataset metadata;
- a FastAPI service that returns **503** instead of random, untrained predictions when no checkpoint exists;
- deterministic seeding, unit tests, CI, Docker, and a configurable smoke test.


## Visual overview

![Current v2 protocol](assets/pipeline-protocol.svg)

The diagram above describes the **current v2 code and evidence contract**. It makes no performance claim and remains valid before a model is trained.

<details>
<summary><strong>Archived v1 visual gallery — qualitative context only</strong></summary>

These are genuine visuals retained from the pre-v2 repository. They remain useful for understanding the former exploration and qualitative predictions, but their metrics are **not** current v2 results because the v2 evaluation protocol and void handling changed. See [`assets/legacy-v1/README.md`](assets/legacy-v1/README.md).

![Archived v1 prediction snapshot](assets/legacy-v1/prediction_dashboard_real.png)

![Archived v1 evaluation snapshot](assets/legacy-v1/evaluation_dashboard_real.png)

</details>

A current, reportable gallery is produced after a recorded v2 run from the checkpoint, configuration, data provenance, and evaluation artifacts.

## Data card

| Item | Value |
| --- | --- |
| Dataset | [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) segmentation |
| Split | Official `train` and `val` ImageSets |
| Default task | Binary foreground/background (all object classes merged) |
| Alternative task | `voc_multiclass` with `model.num_classes: 21` |
| Void pixels | Original `255`; ignored in loss and all metrics |
| Data location | `data/` (ignored by Git) |
| Intended use | Learning, evaluation, and a maintainable engineering reference |
| Not intended for | Medical, safety, industrial inspection, or other deployment without task-specific data, validation, monitoring, and governance |

Pascal VOC has its own terms; review them before use. This repository’s MIT license does not relicence the dataset.

## Setup

Python **3.10–3.12** is recommended. Install an appropriate matching PyTorch/torchvision pair from the [official PyTorch selector](https://pytorch.org/get-started/locally/) first. For a CPU-only Linux environment tested by the Dockerfile:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download data explicitly (about 2 GB) rather than triggering a hidden transfer during training:

```bash
python -m src.download_data
```

## Train and evaluate

The default `config.yaml` uses the entire official split. It can be expensive on CPU.

```bash
# Full experiment from config.yaml
python -m src.train --config config.yaml

# Fast wiring check: 16 train / 8 validation samples and 1 epoch
python -m src.train --config config.yaml --smoke-test

# Evaluate the saved best checkpoint and write reports/evaluation_metrics.json
python -m src.evaluate --config config.yaml --checkpoint checkpoints/best.pt
```

Training writes:

```text
checkpoints/
├── best.pt                 # best validation mIoU, self-describing metadata
├── last.pt
└── training_history.json
```

These outputs are ignored by Git. Record the commit SHA, complete config, environment, seed, hardware, training duration, and validation metrics when publishing an experiment. Do not compare scores obtained from different subsets, resolutions, or task definitions.

### Switch to native 21-class VOC

Change only the task and output channels together:

```yaml
dataset:
  task: voc_multiclass
model:
  num_classes: 21
```

Then retrain. A binary checkpoint cannot be used for this task.

## Serve a trained checkpoint

The API does not ship with a checkpoint. Train first, then:

```bash
uvicorn src.inference:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@example.jpg" --output mask.png
curl -X POST http://localhost:8000/predict-overlay -F "file=@example.jpg" --output overlay.png
```

- `GET /health` reports `model_not_loaded` until `checkpoints/best.pt` exists and `checkpoint_present` afterward.
- `POST /predict` returns a PNG label mask at the original image size.
- `POST /predict-overlay` returns a green overlay for non-background labels.
- Use `CHECKPOINT_PATH=/path/to/model.pt` to select another checkpoint and `MAX_UPLOAD_BYTES` to change the 10 MiB upload limit.

## Quality checks

```bash
pip install -r requirements-dev.txt
pytest
ruff check src tests
python -m src.benchmark --model unet --encoder resnet18
```

The GitHub Actions workflow runs the unit tests on Python 3.11 with the CPU PyTorch build. The tests cover metric arithmetic, void-label exclusion, and config validation; they intentionally do not claim to validate a trained model.

## Docker

```bash
docker build -t semantic-segmentation-pipeline .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  semantic-segmentation-pipeline
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}`. The container health endpoint may be available before the model is trained, but predictions remain unavailable until a valid `best.pt` is mounted.

## Project layout

```text
├── config.yaml
├── src/
│   ├── data_loader.py       # VOC datasets, transforms, official splits
│   ├── models.py            # model factory and void-aware losses
│   ├── metrics.py           # streaming confusion-matrix metrics
│   ├── train.py             # reproducible training/checkpointing
│   ├── evaluate.py          # checkpoint evaluation
│   ├── inference.py         # safe FastAPI inference service
│   └── download_data.py
├── tests/
├── checkpoints/             # local, Git-ignored training artefacts
├── reports/                 # local, Git-ignored evaluation artefacts
├── .github/workflows/ci.yml
├── Dockerfile
└── Makefile
```

## License and acknowledgement

Code is available under the [MIT License](LICENSE). Cite the original PASCAL VOC work when using the dataset:

```bibtex
@article{everingham2010pascal,
  title={The Pascal Visual Object Classes (VOC) Challenge},
  author={Everingham, Mark and others},
  journal={International Journal of Computer Vision},
  year={2010}
}
```
