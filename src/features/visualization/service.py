from __future__ import annotations

import logging
from pathlib import Path
from features.visualization.distribution import (
    build_distribution_plot_spec,
    resolve_distribution_output_path,
)
from features.visualization.loading import AnalysisSeriesLoader
from features.visualization.models import VisualizationReport
from features.visualization.rendering import MatplotlibVisualizationRenderer

logger = logging.getLogger(__name__)


class VisualizationService:
    def __init__(
        self,
        series_loader: AnalysisSeriesLoader,
        renderer: MatplotlibVisualizationRenderer,
    ) -> None:
        self._series_loader = series_loader
        self._renderer = renderer

    def save_distribution_plot(
        self,
        *,
        analysis_result_paths: tuple[Path, ...],
        output_path: Path,
        mode: str = "overlay",
        series_name: str | None = None,
        field_name: str | None = None,
        source_labels: tuple[str, ...] | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> VisualizationReport:
        if not analysis_result_paths:
            raise ValueError("At least one analysis result path is required")

        series_data = tuple(
            self._series_loader.load_numeric_series(
                analysis_result_path=analysis_result_path,
                series_name=series_name,
                field_name=field_name,
            )
            for analysis_result_path in analysis_result_paths
        )
        plot_spec = build_distribution_plot_spec(
            series_data=series_data,
            mode=mode,
            source_labels=source_labels,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )
        output_path = resolve_distribution_output_path(
            output_path=output_path,
            input_paths=analysis_result_paths,
            series_name=plot_spec.series_name,
        )
        self._renderer.save_distribution(plot_spec=plot_spec, output_path=output_path)

        logger.info(
            "Saved %s distribution comparison for series=%s from %s input files to %s",
            plot_spec.analysis_type,
            plot_spec.series_name,
            len(plot_spec.series),
            output_path,
        )
        return VisualizationReport(
            input_paths=analysis_result_paths,
            output_path=output_path,
            visualization_type="distribution",
            analysis_type=plot_spec.analysis_type,
            series_name=plot_spec.series_name,
        )
