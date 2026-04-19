from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: int
    image_path: str
    target: str
    predicted: str
    confidence: float


@dataclass(frozen=True)
class EvaluationSummary:
    model_name: str
    dataset_path: Path
    num_samples: int
    device: str
    weights: str
    predictions_path: Path


@dataclass(frozen=True)
class SemanticAnalysisRecord:
    sample_id: int
    image_path: str
    target: str
    predicted: str
    metric_name: str
    metric_value: int | float | None


@dataclass(frozen=True)
class AnalysisResult:
    analysis_type: str
    metadata: JsonObject
    values: JsonValue
