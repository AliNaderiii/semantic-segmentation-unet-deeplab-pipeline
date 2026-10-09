import json
from pathlib import Path

from src.reporting import build_dashboard


def test_dashboard_is_generated_from_recorded_artifacts(tmp_path: Path) -> None:
    history = {
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.8,
                "validation_loss": 0.7,
                "train": {"mean_iou": 0.1},
                "validation": {"mean_iou": 0.2},
            }
        ]
    }
    metrics = {"mean_iou": 0.25, "mean_dice": 0.4, "confusion_matrix": [[10, 2], [3, 5]]}
    history_path = tmp_path / "history.json"
    metrics_path = tmp_path / "metrics.json"
    output_path = tmp_path / "dashboard.png"
    history_path.write_text(json.dumps(history), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    build_dashboard(history_path, metrics_path, output_path)
    assert output_path.is_file() and output_path.stat().st_size > 0
