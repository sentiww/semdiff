from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Protocol

from semdiff.files import (
    AnalysisResult,
    FileStore,
    GroupedConfusions,
    JsonObject,
    JsonValue,
    PredictionRecord,
    SemanticAnalysisRecord,
)
from semdiff.core.handlers import CommandInput, CommandOutput, Handler, HandlerFactory
from semdiff.core.protocols import ISemanticMetric, IWordNetService
from semdiff.wordnet.service import WordNetService as WordNetServiceImpl

logger = logging.getLogger(__name__)
SemanticExample = dict[str, str | int | float]
_ANALYSIS_PROGRESS_EVERY = 1000


@dataclass(frozen=True)
class SemanticAnalysisReport:
    metric: str
    annotated_path: Path
    summary_path: Path
    num_records: int


@dataclass(frozen=True)
class SemanticAnalysisOutputPaths:
    output_directory: Path
    predictions_path: Path
    annotated_path: Path
    summary_path: Path


@dataclass(frozen=True)
class SemanticAnalysisFilter:
    target: str | None = None
    predicted: str | None = None

    def matches(self, record: PredictionRecord) -> bool:
        if self.target is not None and record.target != self.target:
            return False
        if self.predicted is not None and record.predicted != self.predicted:
            return False
        return True

    def as_metadata(self) -> JsonObject:
        return {
            "target": self.target,
            "predicted": self.predicted,
        }

    def has_filters(self) -> bool:
        return self.target is not None or self.predicted is not None


class SemanticAnalysisService:
    def __init__(self, file_store: FileStore) -> None:
        self._file_store = file_store

    def analyze(
        self,
        *,
        predictions_path: Path,
        output_directory: Path,
        metric: ISemanticMetric,
        analysis_filter: SemanticAnalysisFilter,
    ) -> SemanticAnalysisReport:
        output_paths = _resolve_output_paths(
            predictions_path,
            output_directory,
            metric.name,
            analysis_filter=analysis_filter,
        )
        logger.info(
            "Starting semantic analysis for metric=%s using predictions=%s target=%s predicted=%s",
            metric.name,
            output_paths.predictions_path,
            analysis_filter.target,
            analysis_filter.predicted,
        )

        values: list[int | float] = []
        examples: list[SemanticExample] = []
        missing_count = 0
        num_records = 0
        scanned_records = 0

        with self._file_store.open_sink(
            SemanticAnalysisRecord,
            output_paths.annotated_path,
        ) as analysis_sink:
            for record in self._file_store.open_source(
                PredictionRecord, output_paths.predictions_path
            ):
                scanned_records += 1
                if not analysis_filter.matches(record):
                    continue
                num_records += 1

                target = record.target
                predicted = record.predicted
                metric_value = metric.calculate(target, predicted)

                analysis_sink.write(
                    SemanticAnalysisRecord(
                        sample_id=record.sample_id,
                        image_path=record.image_path,
                        target=target,
                        predicted=predicted,
                        metric_name=metric.name,
                        metric_value=metric_value,
                    )
                )

                if metric_value is None:
                    missing_count += 1
                else:
                    values.append(metric_value)
                    examples.append(
                        {
                            "id": record.sample_id,
                            "image": record.image_path,
                            "target": target,
                            "predicted": predicted,
                            metric.name: metric_value,
                        }
                    )

                if num_records % _ANALYSIS_PROGRESS_EVERY == 0:
                    logger.info(
                        "Processed %s matching prediction records for metric=%s (missing=%s)",
                        num_records,
                        metric.name,
                        missing_count,
                    )
        with self._file_store.open_sink(
            AnalysisResult,
            output_paths.summary_path,
        ) as summary_sink:
            summary_sink.write(
                build_analysis_result(
                    metric_name=metric.name,
                    predictions_path=output_paths.predictions_path,
                    annotated_path=output_paths.annotated_path,
                    num_records=num_records,
                    analysis_filter=analysis_filter,
                    missing_count=missing_count,
                    value_summary=summarize_values(values),
                    lowest_examples=select_extreme_examples(
                        examples,
                        value_key=metric.name,
                        limit=5,
                        highest=False,
                    ),
                    highest_examples=select_extreme_examples(
                        examples,
                        value_key=metric.name,
                        limit=5,
                        highest=True,
                    ),
                )
            )

        logger.info(
            "Completed semantic analysis for metric=%s: %s matching records from %s scanned, %s missing, outputs=%s,%s",
            metric.name,
            num_records,
            scanned_records,
            missing_count,
            output_paths.annotated_path,
            output_paths.summary_path,
        )
        return SemanticAnalysisReport(
            metric=metric.name,
            annotated_path=output_paths.annotated_path,
            summary_path=output_paths.summary_path,
            num_records=num_records,
        )

    def analyze_confusions(
        self,
        *,
        predictions_path: Path,
        output_path: Path,
        reverse: bool = False,
    ) -> int:
        logger.info(
            "Starting confusion analysis for predictions=%s reverse=%s",
            predictions_path,
            reverse,
        )
        confusions: Counter[tuple[str, str]] = Counter()
        num_records = 0

        for record in self._file_store.open_source(
            PredictionRecord, predictions_path
        ):
            confusions[(record.target, record.predicted)] += 1
            num_records += 1

            if num_records % _ANALYSIS_PROGRESS_EVERY == 0:
                logger.info(
                    "Processed %s records for confusion analysis",
                    num_records,
                )

        grouped: dict[str, dict[str, int]] = {}
        for (target, predicted), count in confusions.items():
            if reverse:
                source, other = predicted, target
            else:
                source, other = target, predicted
            if source not in grouped:
                grouped[source] = {}
            grouped[source][other] = count

        for source in grouped:
            grouped[source] = dict(
                sorted(grouped[source].items(), key=lambda x: x[1], reverse=True)
            )

        with self._file_store.open_sink(
            GroupedConfusions, output_path
        ) as confusion_sink:
            confusion_sink.write(grouped)

        num_unique_sources = len(grouped)
        logger.info(
            "Completed confusion analysis: %s unique sources, %s total confusion pairs from %s records, output=%s",
            num_unique_sources,
            len(confusions),
            num_records,
            output_path,
        )
        return num_unique_sources


def summarize_values(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def build_analysis_result(
    *,
    metric_name: str,
    predictions_path: Path,
    annotated_path: Path,
    num_records: int,
    analysis_filter: SemanticAnalysisFilter,
    missing_count: int,
    value_summary: dict[str, float | int | None],
    lowest_examples: list[JsonObject],
    highest_examples: list[JsonObject],
) -> AnalysisResult:
    summary_fields = ["count", "min", "max", "mean", "median"]
    example_fields = ["id", "image", "target", "predicted", metric_name]

    return AnalysisResult(
        analysis_type="semantic_metric_summary",
        metadata={
            "analysis": "semantic",
            "metric": metric_name,
            "predictions_path": str(predictions_path),
            "annotated_path": str(annotated_path),
            "filters": analysis_filter.as_metadata(),
            "num_records": num_records,
            "missing": missing_count,
            "series": [
                {
                    "name": "summary",
                    "kind": "vector",
                    "fields": summary_fields,
                },
                {
                    "name": "lowest_examples",
                    "kind": "matrix",
                    "fields": example_fields,
                },
                {
                    "name": "highest_examples",
                    "kind": "matrix",
                    "fields": example_fields,
                },
            ],
        },
        values={
            "summary": [value_summary[field] for field in summary_fields],
            "lowest_examples": examples_to_matrix(
                lowest_examples,
                fields=example_fields,
            ),
            "highest_examples": examples_to_matrix(
                highest_examples,
                fields=example_fields,
            ),
        },
    )


def select_extreme_examples(
    examples: list[SemanticExample],
    *,
    value_key: str,
    limit: int,
    highest: bool,
) -> list[JsonObject]:
    sorted_examples = list(examples)
    sorted_examples.sort(
        key=lambda item: _example_sort_key(item, value_key),
        reverse=highest,
    )
    return [dict(example) for example in sorted_examples[:limit]]


def examples_to_matrix(
    examples: list[JsonObject],
    *,
    fields: list[str],
) -> list[list[JsonValue]]:
    matrix: list[list[JsonValue]] = []
    for example in examples:
        matrix.append([example.get(field) for field in fields])
    return matrix


def _example_sort_key(
    example: SemanticExample,
    value_key: str,
) -> tuple[int | float, str, str, str]:
    metric_value = example[value_key]
    if not isinstance(metric_value, (int, float)):
        raise RuntimeError(f"Expected a numeric value for {value_key!r}")
    return (
        metric_value,
        str(example["target"]),
        str(example["predicted"]),
        str(example["image"]),
    )


def _resolve_output_paths(
    predictions_path: Path,
    output_directory: Path,
    metric_name: str,
    analysis_filter: SemanticAnalysisFilter,
) -> SemanticAnalysisOutputPaths:
    output_directory = Path(output_directory)
    filter_suffix = _filter_file_suffix(analysis_filter)
    return SemanticAnalysisOutputPaths(
        output_directory=output_directory,
        predictions_path=Path(predictions_path),
        annotated_path=output_directory
        / f"semantic-{metric_name}{filter_suffix}.jsonl",
        summary_path=output_directory
        / f"semantic-{metric_name}{filter_suffix}.json",
    )


def _filter_file_suffix(analysis_filter: SemanticAnalysisFilter) -> str:
    if not analysis_filter.has_filters():
        return ""

    parts: list[str] = []
    if analysis_filter.target is not None:
        parts.append(f"target-{_sanitize_file_component(analysis_filter.target)}")
    if analysis_filter.predicted is not None:
        parts.append(f"predicted-{_sanitize_file_component(analysis_filter.predicted)}")
    return "-" + "-".join(parts)


def _sanitize_file_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return sanitized or "value"


@dataclass(frozen=True)
class SemanticAnalysisInput(CommandInput):
    metric: str
    predictions_path: Path
    output_directory: Path
    target: str | None = None
    predicted: str | None = None


@dataclass(frozen=True)
class SemanticAnalysisOutput(CommandOutput):
    metric: str
    annotated_path: Path
    summary_path: Path
    num_records: int


class SemanticAnalysisHandler(Handler[SemanticAnalysisInput, SemanticAnalysisOutput]):
    analysis: str = "semantic"

    def __init__(
        self,
        wordnet: IWordNetService,
        semantic_analysis_service: ISemanticAnalysisService,
    ) -> None:
        self._wordnet = wordnet
        self._semantic_analysis_service = semantic_analysis_service

    def __call__(self, cmd: SemanticAnalysisInput) -> SemanticAnalysisOutput:
        metric = self._wordnet.create_metric(cmd.metric)
        analysis_filter = SemanticAnalysisFilter(
            target=cmd.target,
            predicted=cmd.predicted,
        )
        report = self._semantic_analysis_service.analyze(
            predictions_path=cmd.predictions_path,
            output_directory=cmd.output_directory,
            metric=metric,
            analysis_filter=analysis_filter,
        )
        return self._build_output(report)

    def _build_output(self, report: SemanticAnalysisReport) -> SemanticAnalysisOutput:
        return SemanticAnalysisOutput(
            metric=report.metric,
            annotated_path=report.annotated_path,
            summary_path=report.summary_path,
            num_records=report.num_records,
        )


class AnalysisHandlers(HandlerFactory):
    def __init__(
        self,
        wordnet: IWordNetService,
        semantic_analysis_service: ISemanticAnalysisService,
    ) -> None:
        self._wordnet = wordnet
        self._semantic_analysis_service = semantic_analysis_service

    def create_semantic(self) -> SemanticAnalysisHandler:
        return SemanticAnalysisHandler(
            wordnet=self._wordnet,
            semantic_analysis_service=self._semantic_analysis_service,
        )

    def create_confusions(self) -> ConfusionAnalysisHandler:
        return ConfusionAnalysisHandler(
            semantic_analysis_service=self._semantic_analysis_service,
        )


@dataclass(frozen=True)
class ConfusionAnalysisInput(CommandInput):
    predictions_path: Path
    output_path: Path
    reverse: bool = False


@dataclass(frozen=True)
class ConfusionAnalysisOutput(CommandOutput):
    output_path: Path
    num_confusions: int


class ConfusionAnalysisHandler(Handler[ConfusionAnalysisInput, ConfusionAnalysisOutput]):
    def __init__(
        self,
        semantic_analysis_service: ISemanticAnalysisService,
    ) -> None:
        self._semantic_analysis_service = semantic_analysis_service

    def __call__(self, cmd: ConfusionAnalysisInput) -> ConfusionAnalysisOutput:
        num_confusions = self._semantic_analysis_service.analyze_confusions(
            predictions_path=cmd.predictions_path,
            output_path=cmd.output_path,
            reverse=cmd.reverse,
        )
        return ConfusionAnalysisOutput(
            output_path=cmd.output_path,
            num_confusions=num_confusions,
        )
