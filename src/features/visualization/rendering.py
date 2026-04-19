from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib.pyplot as plt

from features.visualization.models import DistributionPlotSpec


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
        bins = min(
            20,
            max(5, max(len(item.values) for item in plot_spec.series)),
        )
        for item in plot_spec.series:
            axis.hist(
                item.values,
                bins=bins,
                alpha=0.5,
                label=item.source_label,
            )

        axis.set_title(plot_spec.labels.title)
        axis.set_xlabel(plot_spec.labels.x_label)
        axis.set_ylabel(plot_spec.labels.y_label)
        if len(plot_spec.series) > 1:
            axis.legend()
