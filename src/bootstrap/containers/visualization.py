from __future__ import annotations

from functools import cached_property

from bootstrap.containers.base import _FileStoreMixin


class VisualizationContainer(_FileStoreMixin):
    @cached_property
    def analysis_series_loader(self):
        from features.visualization.loading import AnalysisSeriesLoader

        return AnalysisSeriesLoader(file_store=self._file_store)

    @cached_property
    def matplotlib_renderer(self):
        from features.visualization.rendering import MatplotlibVisualizationRenderer

        return MatplotlibVisualizationRenderer()

    @cached_property
    def visualization_service(self):
        from features.visualization.service import VisualizationService

        return VisualizationService(
            series_loader=self.analysis_series_loader,
            renderer=self.matplotlib_renderer,
        )
