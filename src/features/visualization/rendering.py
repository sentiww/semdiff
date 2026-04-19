from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np

from features.visualization.models import DistributionPlotSpec

_HISTOGRAM_ALPHA = 0.45
_HISTOGRAM_LINEWIDTH = 1.0


class MatplotlibVisualizationRenderer:
    def save_distribution(
        self,
        *,
        plot_spec: DistributionPlotSpec,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, axis = plt.subplots(figsize=(10, 6))
        try:
            self._plot_distribution(axis, plot_spec)
            figure.tight_layout()
            figure.savefig(output_path)
        finally:
            plt.close(figure)

    def _plot_distribution(
        self, axis: plt.Axes, plot_spec: DistributionPlotSpec
    ) -> None:
        self._plot_histograms(axis, plot_spec)
        axis.set_title(plot_spec.labels.title)
        axis.set_xlabel(plot_spec.labels.x_label)
        axis.set_ylabel(plot_spec.labels.y_label)
        if len(plot_spec.series) > 1:
            axis.legend()

    def _plot_histograms(
        self,
        axis: plt.Axes,
        plot_spec: DistributionPlotSpec,
    ) -> None:
        bin_edges = self._resolve_bin_edges(plot_spec)
        bar_width = np.diff(bin_edges)
        colors = self._resolve_series_colors(len(plot_spec.series))

        if plot_spec.mode == "side-by-side":
            self._plot_side_by_side(axis, plot_spec, bin_edges, bar_width, colors)
            return

        self._plot_overlay(axis, plot_spec, bin_edges, bar_width, colors)

    def _plot_overlay(
        self,
        axis: plt.Axes,
        plot_spec: DistributionPlotSpec,
        bin_edges: np.ndarray,
        bar_width: np.ndarray,
        colors: list[str],
    ) -> None:
        for item, color in zip(plot_spec.series, colors, strict=True):
            counts, _ = np.histogram(item.values, bins=bin_edges)
            axis.bar(
                bin_edges[:-1],
                counts,
                width=bar_width,
                align="edge",
                color=color,
                edgecolor=color,
                linewidth=_HISTOGRAM_LINEWIDTH,
                alpha=_HISTOGRAM_ALPHA,
                label=item.source_label,
            )

    def _plot_side_by_side(
        self,
        axis: plt.Axes,
        plot_spec: DistributionPlotSpec,
        bin_edges: np.ndarray,
        bar_width: np.ndarray,
        colors: list[str],
    ) -> None:
        num_series = len(plot_spec.series)
        for index, (item, color) in enumerate(zip(plot_spec.series, colors, strict=True)):
            counts, _ = np.histogram(item.values, bins=bin_edges)
            axis.bar(
                bin_edges[:-1] + (bar_width * index / num_series),
                counts,
                width=bar_width / num_series,
                align="edge",
                color=color,
                edgecolor=color,
                linewidth=_HISTOGRAM_LINEWIDTH,
                alpha=_HISTOGRAM_ALPHA,
                label=item.source_label,
            )

    def _resolve_bin_edges(self, plot_spec: DistributionPlotSpec) -> np.ndarray:
        all_values = np.concatenate(
            [np.asarray(item.values, dtype=float) for item in plot_spec.series]
        )
        if all_values.size == 0:
            raise ValueError("At least one numeric value is required to render a distribution")

        bins = min(20, max(5, max(len(item.values) for item in plot_spec.series)))
        minimum = float(all_values.min())
        maximum = float(all_values.max())
        if minimum == maximum:
            padding = 0.5 if minimum == 0 else abs(minimum) * 0.05
            minimum -= padding
            maximum += padding
        return np.linspace(minimum, maximum, bins + 1)

    def _resolve_series_colors(self, count: int) -> list[str]:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        return [color_cycle[index % len(color_cycle)] for index in range(count)]
