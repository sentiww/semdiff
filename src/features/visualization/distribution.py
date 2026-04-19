from __future__ import annotations

from pathlib import Path

from features.visualization.models import DistributionPlotSpec, NumericSeries, PlotLabels


def build_distribution_plot_spec(
    *,
    series_data: tuple[NumericSeries, ...],
    source_labels: tuple[str, ...] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> DistributionPlotSpec:
    if not series_data:
        raise ValueError("At least one numeric series is required")
    if source_labels is not None and len(source_labels) != len(series_data):
        raise ValueError("The number of source labels must match the number of input series")

    first = series_data[0]
    for item in series_data[1:]:
        if item.analysis_type != first.analysis_type:
            raise ValueError("All compared distributions must have the same analysis type")
        if item.series_name != first.series_name:
            raise ValueError("All compared distributions must resolve to the same series")
        if item.labels.x_label != first.labels.x_label or item.labels.y_label != first.labels.y_label:
            raise ValueError("All compared distributions must use the same axis labels")

    labels = PlotLabels(
        title=title or _default_title(series_data),
        x_label=x_label or first.labels.x_label,
        y_label=y_label or first.labels.y_label,
    )
    customized_series: list[NumericSeries] = []
    for index, item in enumerate(series_data):
        customized_series.append(
            NumericSeries(
                source_label=source_labels[index] if source_labels is not None else item.source_label,
                analysis_type=item.analysis_type,
                series_name=item.series_name,
                values=item.values,
                labels=labels,
            )
        )

    return DistributionPlotSpec(
        analysis_type=first.analysis_type,
        series_name=first.series_name,
        labels=labels,
        series=tuple(customized_series),
    )


def resolve_distribution_output_path(
    *,
    output_path: Path,
    input_paths: tuple[Path, ...],
    series_name: str,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        return output_path / default_distribution_filename(input_paths, series_name)
    if output_path.suffix:
        return output_path
    return output_path / default_distribution_filename(input_paths, series_name)


def default_distribution_filename(
    input_paths: tuple[Path, ...],
    series_name: str,
) -> str:
    safe_series_name = series_name.replace(":", "-")
    if len(input_paths) == 1:
        prefix = input_paths[0].stem
    else:
        prefix = f"{input_paths[0].stem}-comparison-{len(input_paths)}"
    return f"{prefix}-{safe_series_name}-distribution.png"


def _default_title(series_data: tuple[NumericSeries, ...]) -> str:
    first = series_data[0]
    if len(series_data) == 1:
        return first.labels.title
    return f"{first.analysis_type}: {first.series_name} comparison"
