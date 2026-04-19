from __future__ import annotations
from pathlib import Path
from typing import cast

from features.files import (
    AnalysisResult,
    FileStore,
    JsonObject,
    JsonValue,
    SemanticAnalysisRecord,
)
from features.visualization.models import AnalysisSeriesSpec, NumericSeries, PlotLabels


class AnalysisSeriesLoader:
    def __init__(self, file_store: FileStore) -> None:
        self._file_store = file_store

    def load_numeric_series(
        self,
        *,
        analysis_result_path: Path,
        series_name: str | None = None,
        field_name: str | None = None,
    ) -> NumericSeries:
        analysis_result_path = Path(analysis_result_path)
        if analysis_result_path.suffix.lower() == ".jsonl":
            return self._load_from_semantic_records(analysis_result_path)
        return self._load_from_analysis_result(
            analysis_result_path=analysis_result_path,
            series_name=series_name,
            field_name=field_name,
        )

    def _load_from_analysis_result(
        self,
        *,
        analysis_result_path: Path,
        series_name: str | None,
        field_name: str | None,
    ) -> NumericSeries:
        analysis_result = self._load_analysis_result(analysis_result_path)
        series_specs = _parse_series_specs(analysis_result.metadata)
        selected_series = _select_series(series_specs, analysis_result.values, series_name)

        if selected_series.kind == "vector":
            return _numeric_series_from_vector(
                analysis_result=analysis_result,
                series=selected_series,
                source_label=analysis_result_path.stem,
            )
        if selected_series.kind == "matrix":
            resolved_field_name = _resolve_matrix_field_name(
                analysis_result=analysis_result,
                series=selected_series,
                requested_field_name=field_name,
            )
            return _numeric_series_from_matrix(
                analysis_result=analysis_result,
                series=selected_series,
                field_name=resolved_field_name,
                source_label=analysis_result_path.stem,
            )
        raise ValueError(
            f"Unsupported series kind {selected_series.kind!r} for numeric visualization"
        )

    def _load_from_semantic_records(self, analysis_result_path: Path) -> NumericSeries:
        metric_name: str | None = None
        values: list[float] = []
        for record in self._file_store.open_source(
            SemanticAnalysisRecord,
            analysis_result_path,
        ):
            metric_name = record.metric_name
            if record.metric_value is None:
                continue
            values.append(float(record.metric_value))

        if metric_name is None:
            raise RuntimeError(f"No semantic analysis records found in {analysis_result_path}")
        if not values:
            raise RuntimeError(
                f"No numeric semantic analysis values found in {analysis_result_path}"
            )

        return NumericSeries(
            source_label=analysis_result_path.stem,
            analysis_type="semantic_metric_distribution",
            series_name=metric_name,
            values=tuple(values),
            labels=PlotLabels(
                title=f"semantic_metric_distribution: {metric_name}",
                x_label=metric_name,
                y_label="Frequency",
            ),
        )

    def _load_analysis_result(self, analysis_result_path: Path) -> AnalysisResult:
        source = self._file_store.open_source(AnalysisResult, analysis_result_path)
        try:
            return next(iter(source))
        except StopIteration as exc:
            raise RuntimeError(f"No AnalysisResult found in {analysis_result_path}") from exc


def _parse_series_specs(metadata: JsonObject) -> tuple[AnalysisSeriesSpec, ...]:
    raw_series = metadata.get("series")
    if not isinstance(raw_series, list):
        raise RuntimeError("AnalysisResult metadata is missing a 'series' list")

    series_specs: list[AnalysisSeriesSpec] = []
    for item in raw_series:
        if not isinstance(item, dict):
            raise RuntimeError("Expected object entries in AnalysisResult metadata.series")
        name = item.get("name")
        kind = item.get("kind")
        fields = item.get("fields")
        if not isinstance(name, str) or not isinstance(kind, str) or not isinstance(fields, list):
            raise RuntimeError("Each AnalysisResult series must include name, kind, and fields")
        typed_fields: list[str] = []
        for field in fields:
            if not isinstance(field, str):
                raise RuntimeError("AnalysisResult series fields must be strings")
            typed_fields.append(field)
        series_specs.append(
            AnalysisSeriesSpec(
                name=name,
                kind=kind,
                fields=tuple(typed_fields),
            )
        )
    if not series_specs:
        raise RuntimeError("AnalysisResult metadata.series is empty")
    return tuple(series_specs)


def _select_series(
    series_specs: tuple[AnalysisSeriesSpec, ...],
    values: JsonValue,
    requested_series_name: str | None,
) -> AnalysisSeriesSpec:
    values_object = _require_object(values, "AnalysisResult values must be an object")

    if requested_series_name is not None:
        for series in series_specs:
            if series.name == requested_series_name:
                if series.name not in values_object:
                    raise RuntimeError(
                        f"Series {series.name!r} is declared in metadata but missing from values"
                    )
                return series
        available_names = ", ".join(series.name for series in series_specs)
        raise ValueError(
            f"Unknown series {requested_series_name!r}. Available series: {available_names}"
        )

    for series in series_specs:
        if series.name in values_object:
            return series

    raise RuntimeError("No declared series are present in AnalysisResult values")


def _numeric_series_from_vector(
    *,
    analysis_result: AnalysisResult,
    series: AnalysisSeriesSpec,
    source_label: str,
) -> NumericSeries:
    values_object = _require_object(analysis_result.values, "AnalysisResult values must be an object")
    raw_vector = values_object.get(series.name)
    if not isinstance(raw_vector, list):
        raise RuntimeError(f"Expected list values for vector series {series.name!r}")

    numeric_values: list[float] = []
    for _, value in zip(series.fields, raw_vector, strict=False):
        if not isinstance(value, (int, float)):
            continue
        numeric_values.append(float(value))

    if not numeric_values:
        raise RuntimeError(f"Series {series.name!r} does not contain numeric vector values")

    return NumericSeries(
        source_label=source_label,
        analysis_type=analysis_result.analysis_type,
        series_name=series.name,
        values=tuple(numeric_values),
        labels=PlotLabels(
            title=_build_title(analysis_result, series.name),
            x_label="Value",
            y_label="Frequency",
        ),
    )


def _resolve_matrix_field_name(
    *,
    analysis_result: AnalysisResult,
    series: AnalysisSeriesSpec,
    requested_field_name: str | None,
) -> str:
    if requested_field_name is not None:
        if requested_field_name not in series.fields:
            available_fields = ", ".join(series.fields)
            raise ValueError(
                f"Unknown field {requested_field_name!r} for series {series.name!r}. "
                f"Available fields: {available_fields}"
            )
        return requested_field_name

    values_object = _require_object(analysis_result.values, "AnalysisResult values must be an object")
    raw_matrix = values_object.get(series.name)
    if not isinstance(raw_matrix, list):
        raise RuntimeError(f"Expected list values for matrix series {series.name!r}")

    for column_index, field_name in enumerate(series.fields):
        if _matrix_column_is_numeric(raw_matrix, column_index):
            return field_name

    raise RuntimeError(f"No numeric field found for matrix series {series.name!r}")


def _numeric_series_from_matrix(
    *,
    analysis_result: AnalysisResult,
    series: AnalysisSeriesSpec,
    field_name: str,
    source_label: str,
) -> NumericSeries:
    values_object = _require_object(analysis_result.values, "AnalysisResult values must be an object")
    raw_matrix = values_object.get(series.name)
    if not isinstance(raw_matrix, list):
        raise RuntimeError(f"Expected list values for matrix series {series.name!r}")

    field_index = series.fields.index(field_name)
    numeric_values = _extract_numeric_matrix_column(raw_matrix, field_index)
    if not numeric_values:
        raise RuntimeError(
            f"Series {series.name!r} field {field_name!r} does not contain numeric values"
        )

    return NumericSeries(
        source_label=source_label,
        analysis_type=analysis_result.analysis_type,
        series_name=f"{series.name}:{field_name}",
        values=tuple(numeric_values),
        labels=PlotLabels(
            title=_build_title(analysis_result, f"{series.name}:{field_name}"),
            x_label=field_name,
            y_label="Frequency",
        ),
    )


def _matrix_column_is_numeric(raw_matrix: list[JsonValue], column_index: int) -> bool:
    found_numeric = False
    for row in raw_matrix:
        if not isinstance(row, list) or column_index >= len(row):
            continue
        value = row[column_index]
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            return False
        found_numeric = True
    return found_numeric


def _extract_numeric_matrix_column(
    raw_matrix: list[JsonValue],
    column_index: int,
) -> list[float]:
    values: list[float] = []
    for row in raw_matrix:
        if not isinstance(row, list) or column_index >= len(row):
            continue
        value = row[column_index]
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            raise RuntimeError("Expected numeric matrix values for distribution plot")
        values.append(float(value))
    return values


def _require_object(value: JsonValue, message: str) -> JsonObject:
    if not isinstance(value, dict):
        raise RuntimeError(message)
    return cast(JsonObject, value)


def _build_title(analysis_result: AnalysisResult, series_name: str) -> str:
    return f"{analysis_result.analysis_type}: {series_name}"
