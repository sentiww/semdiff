from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from features.handlers.base import CommandInput, CommandOutput, Handler, HandlerFactory
from features.visualization.service import (
    VisualizationService,
)
from features.visualization.models import VisualizationReport


@dataclass(frozen=True)
class DistributionInput(CommandInput):
    analysis_results: tuple[Path, ...]
    output: Path
    mode: str
    series: str | None
    field: str | None
    labels: tuple[str, ...] | None
    title: str | None
    x_label: str | None
    y_label: str | None


@dataclass(frozen=True)
class DistributionOutput(CommandOutput):
    analysis_results: tuple[Path, ...]
    output: Path
    analysis_type: str
    series: str


class DistributionHandler(Handler[DistributionInput, DistributionOutput]):
    def __init__(self, visualization_service: VisualizationService) -> None:
        self._visualization_service = visualization_service

    def __call__(self, cmd: DistributionInput) -> DistributionOutput:
        report = self._visualization_service.save_distribution_plot(
            analysis_result_paths=cmd.analysis_results,
            output_path=cmd.output,
            mode=cmd.mode,
            series_name=cmd.series,
            field_name=cmd.field,
            source_labels=cmd.labels,
            title=cmd.title,
            x_label=cmd.x_label,
            y_label=cmd.y_label,
        )
        return self._build_output(report)

    def _build_output(self, report: VisualizationReport) -> DistributionOutput:
        return DistributionOutput(
            analysis_results=report.input_paths,
            output=report.output_path,
            analysis_type=report.analysis_type,
            series=report.series_name,
        )


class VisualizationHandlers(HandlerFactory):
    def __init__(self, visualization_service: VisualizationService) -> None:
        self._visualization_service = visualization_service

    def create_distribution(self) -> DistributionHandler:
        return DistributionHandler(visualization_service=self._visualization_service)
