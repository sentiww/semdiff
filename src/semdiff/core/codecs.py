from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast

from semdiff.files.models import (
    AnalysisResult,
    EvaluationSummary,
    JsonObject,
    PredictionRecord,
    SemanticAnalysisRecord,
)

GroupedConfusions = dict[str, dict[str, int]]

StoreEntity = TypeVar("StoreEntity")


@dataclass(frozen=True)
class EntityCodec(Generic[StoreEntity]):
    entity_type: type[StoreEntity]
    format_name: Literal["json", "jsonl"]
    serialize: Callable[[StoreEntity], JsonObject]
    deserialize: Callable[[JsonObject], StoreEntity]
    ensure_ascii: bool = False
    indent: int | None = None


def build_default_codecs() -> Mapping[type[object], EntityCodec[object]]:
    prediction_codec: EntityCodec[PredictionRecord] = EntityCodec(
        entity_type=PredictionRecord,
        format_name="jsonl",
        serialize=_serialize_prediction_record,
        deserialize=_deserialize_prediction_record,
    )
    evaluation_summary_codec: EntityCodec[EvaluationSummary] = EntityCodec(
        entity_type=EvaluationSummary,
        format_name="json",
        serialize=_serialize_evaluation_summary,
        deserialize=_deserialize_evaluation_summary,
        indent=2,
    )
    semantic_record_codec: EntityCodec[SemanticAnalysisRecord] = EntityCodec(
        entity_type=SemanticAnalysisRecord,
        format_name="jsonl",
        serialize=_serialize_semantic_analysis_record,
        deserialize=_deserialize_semantic_analysis_record,
        ensure_ascii=True,
    )
    analysis_result_codec: EntityCodec[AnalysisResult] = EntityCodec(
        entity_type=AnalysisResult,
        format_name="json",
        serialize=_serialize_analysis_result,
        deserialize=_deserialize_analysis_result,
        ensure_ascii=True,
        indent=2,
    )
    grouped_confusions_codec: EntityCodec[GroupedConfusions] = EntityCodec(
        entity_type=GroupedConfusions,
        format_name="jsonl",
        serialize=_serialize_grouped_confusions,
        deserialize=_deserialize_grouped_confusions,
    )
    return {
        PredictionRecord: cast(EntityCodec[object], prediction_codec),
        EvaluationSummary: cast(EntityCodec[object], evaluation_summary_codec),
        SemanticAnalysisRecord: cast(EntityCodec[object], semantic_record_codec),
        AnalysisResult: cast(EntityCodec[object], analysis_result_codec),
        GroupedConfusions: cast(EntityCodec[object], grouped_confusions_codec),
    }


def _serialize_prediction_record(record: PredictionRecord) -> JsonObject:
    return {
        "id": record.sample_id,
        "image": record.image_path,
        "target": record.target,
        "predicted": record.predicted,
        "confidence": record.confidence,
    }


def _deserialize_prediction_record(payload: JsonObject) -> PredictionRecord:
    return PredictionRecord(
        sample_id=int(payload["id"]),
        image_path=str(payload["image"]),
        target=str(payload["target"]),
        predicted=str(payload["predicted"]),
        confidence=float(payload["confidence"]),
    )


def _serialize_evaluation_summary(summary: EvaluationSummary) -> JsonObject:
    return {
        "model": summary.model_name,
        "dataset_path": str(summary.dataset_path),
        "num_samples": summary.num_samples,
        "device": summary.device,
        "weights": summary.weights,
        "predictions_path": str(summary.predictions_path),
    }


def _deserialize_evaluation_summary(payload: JsonObject) -> EvaluationSummary:
    return EvaluationSummary(
        model_name=str(payload["model"]),
        dataset_path=Path(str(payload["dataset_path"])),
        num_samples=int(payload["num_samples"]),
        device=str(payload["device"]),
        weights=str(payload["weights"]),
        predictions_path=Path(str(payload["predictions_path"])),
    )


def _serialize_semantic_analysis_record(record: SemanticAnalysisRecord) -> JsonObject:
    return {
        "id": record.sample_id,
        "image": record.image_path,
        "target": record.target,
        "predicted": record.predicted,
        record.metric_name: record.metric_value,
    }


def _deserialize_semantic_analysis_record(payload: JsonObject) -> SemanticAnalysisRecord:
    fixed_keys = {"id", "image", "target", "predicted"}
    metric_keys = [key for key in payload if key not in fixed_keys]
    if len(metric_keys) != 1:
        raise RuntimeError(
            "Expected exactly one semantic metric field in semantic analysis record"
        )
    metric_name = metric_keys[0]
    metric_value = payload[metric_name]
    if metric_value is not None and not isinstance(metric_value, (int, float)):
        raise RuntimeError(f"Expected numeric semantic metric value for {metric_name!r}")
    return SemanticAnalysisRecord(
        sample_id=int(payload["id"]),
        image_path=str(payload["image"]),
        target=str(payload["target"]),
        predicted=str(payload["predicted"]),
        metric_name=metric_name,
        metric_value=cast(int | float | None, metric_value),
    )


def _serialize_analysis_result(result: AnalysisResult) -> JsonObject:
    return {
        "type": result.analysis_type,
        "metadata": result.metadata,
        "values": result.values,
    }


def _deserialize_analysis_result(payload: JsonObject) -> AnalysisResult:
    metadata = payload["metadata"]
    values = payload["values"]
    if not isinstance(metadata, dict):
        raise RuntimeError("Expected metadata object in analysis result")
    return AnalysisResult(
        analysis_type=str(payload["type"]),
        metadata=cast(JsonObject, metadata),
        values=values,
    )


def _serialize_grouped_confusions(data: GroupedConfusions) -> JsonObject:
    return dict(data)


def _deserialize_grouped_confusions(payload: JsonObject) -> GroupedConfusions:
    result: GroupedConfusions = {}
    for source, confusions in payload.items():
        result[str(source)] = {str(k): int(v) for k, v in confusions.items()}
    return result
