from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DistributionMode = str


@dataclass(frozen=True)
class PlotLabels:
    title: str
    x_label: str
    y_label: str


@dataclass(frozen=True)
class NumericSeries:
    analysis_type: str
    series_name: str
    source_label: str
    values: tuple[float, ...]
    labels: PlotLabels


@dataclass(frozen=True)
class DistributionPlotSpec:
    analysis_type: str
    series_name: str
    mode: DistributionMode
    labels: PlotLabels
    series: tuple[NumericSeries, ...]


@dataclass(frozen=True)
class VisualizationReport:
    input_paths: tuple[Path, ...]
    output_path: Path
    visualization_type: str
    analysis_type: str
    series_name: str


@dataclass(frozen=True)
class AnalysisSeriesSpec:
    name: str
    kind: str
    fields: tuple[str, ...]
