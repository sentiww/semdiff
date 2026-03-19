from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("main.evaluate")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
PROGRESS_LOG_EVERY_BATCHES = 10


@dataclass(frozen=True)
class EvaluationOutputPaths:
    output_path: Path
    predictions_path: Path
    summary_path: Path


def resolve_output_paths(model_name: str, dataset_name: str) -> EvaluationOutputPaths:
    output_path = OUTPUT_ROOT / model_name / dataset_name
    return EvaluationOutputPaths(
        output_path=output_path,
        predictions_path=output_path / "predictions.jsonl",
        summary_path=output_path / "summary.json",
    )
