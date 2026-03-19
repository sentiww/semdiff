from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Sequence


def write_prediction_record(
    predictions_file: Any,
    *,
    dataset_path: Path,
    image_path: str,
    categories: Sequence[str],
    index_to_synset: dict[int, str],
    predicted_index: int,
    confidence: float,
) -> None:
    record = {
        "image": str(Path(image_path).relative_to(dataset_path)),
        "target_synset": Path(image_path).parent.name,
        "predicted_synset": index_to_synset[predicted_index],
        "predicted_label": categories[predicted_index],
        "confidence": confidence,
    }
    predictions_file.write(json.dumps(record) + "\n")


def write_summary(
    summary_path: Path,
    *,
    model_name: str,
    dataset_name: str,
    dataset_path: Path,
    num_samples: int,
    device: str,
    weights: str,
    predictions_path: Path,
) -> None:
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "num_samples": num_samples,
        "device": device,
        "weights": weights,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
