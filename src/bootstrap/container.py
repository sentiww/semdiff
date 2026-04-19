from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from features.datasets.commands import ImageNet1KInitHandler, ImageNetOInitHandler
    from features.evaluation.command import Handler as EvaluateHandler
    from features.files import FileStore
    from features.visualization.command import DistributionHandler
    from features.visualization.loading import AnalysisSeriesLoader
    from features.visualization.rendering import MatplotlibVisualizationRenderer
    from features.visualization.service import VisualizationService
    from features.wordnet.analysis import SemanticAnalysisHandler
    from features.wordnet.commands import (
        SynsetIdHandler,
        SynsetReadableHandler,
        WordNetInitHandler,
    )


@dataclass(frozen=True)
class AppConfig:
    imagenet_class_map: Path
    torchvision_index_to_wnid: Path
    evaluation_batch_size: int
    evaluation_num_workers: int
    evaluation_progress_log_every_batches: int


class _FileStoreMixin:
    @cached_property
    def _file_codecs(self):
        from features.files import build_default_codecs

        return build_default_codecs()

    @cached_property
    def _file_store(self) -> FileStore:
        from features.files import FileStore

        return FileStore(codecs=self._file_codecs)


class _WordNetMixin:
    @cached_property
    def _wordnet(self):
        from features.wordnet.service import WordNetService

        return WordNetService()


class _ImageNetMetadataMixin:
    @cached_property
    def _imagenet_metadata_service(self):
        from features.datasets.metadata import ImageNetMetadataService

        return ImageNetMetadataService()


class WordNetContainer(_WordNetMixin):
    def wordnet_init_handler(self) -> WordNetInitHandler:
        from features.wordnet.commands import WordNetInitHandler

        return WordNetInitHandler(wordnet=self._wordnet)


class SynsetContainer(_WordNetMixin):
    def synset_id_handler(self) -> SynsetIdHandler:
        from features.wordnet.commands import SynsetIdHandler

        return SynsetIdHandler(wordnet=self._wordnet)

    def synset_readable_handler(self) -> SynsetReadableHandler:
        from features.wordnet.commands import SynsetReadableHandler

        return SynsetReadableHandler(wordnet=self._wordnet)


class AnalysisContainer(_FileStoreMixin, _WordNetMixin):
    @cached_property
    def _semantic_analysis_service(self):
        from features.wordnet.analysis import SemanticAnalysisService

        return SemanticAnalysisService(file_store=self._file_store)

    def analysis_semantic_handler(self) -> SemanticAnalysisHandler:
        from features.wordnet.analysis import SemanticAnalysisHandler

        return SemanticAnalysisHandler(
            wordnet=self._wordnet,
            semantic_analysis_service=self._semantic_analysis_service,
        )


class DatasetContainer(_ImageNetMetadataMixin):
    @cached_property
    def _archive_fetcher(self):
        from features.datasets.filesystem import UrlLibArchiveFetcher

        return UrlLibArchiveFetcher()

    @cached_property
    def _archive_extractor(self):
        from features.datasets.filesystem import TarArchiveExtractor

        return TarArchiveExtractor()

    @cached_property
    def _output_preparer(self):
        from features.datasets.filesystem import GeneratedSynsetDirectoryPreparer

        return GeneratedSynsetDirectoryPreparer()

    @cached_property
    def _file_mover(self):
        from features.datasets.filesystem import ShutilFileMover

        return ShutilFileMover()

    @cached_property
    def _index_parser(self):
        from features.datasets.filesystem import ImageNetValidationImageIndexParser

        return ImageNetValidationImageIndexParser()

    def dataset_imagenet_o_init_handler(self) -> ImageNetOInitHandler:
        from features.datasets.commands import ImageNetOInitHandler

        return ImageNetOInitHandler(
            self._archive_fetcher,
            self._archive_extractor,
            self._output_preparer,
            self._file_mover,
            self._imagenet_metadata_service,
        )

    def dataset_imagenet_1k_init_handler(self) -> ImageNet1KInitHandler:
        from features.datasets.commands import ImageNet1KInitHandler

        return ImageNet1KInitHandler(
            self._archive_fetcher,
            self._archive_extractor,
            self._file_mover,
            self._index_parser,
            self._imagenet_metadata_service,
        )


class EvaluationContainer(_FileStoreMixin, _ImageNetMetadataMixin, _WordNetMixin):
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    @cached_property
    def _synset_dataset_indexer(self):
        from features.datasets.synset_dataset import SynsetDatasetIndexer

        return SynsetDatasetIndexer(
            synset_dir_pattern=re.compile(r"n\d+$"),
            image_suffixes=frozenset({".jpeg"}),
        )

    @cached_property
    def _image_dataset_factory(self):
        from torchvision.datasets.folder import default_loader

        from features.datasets.synset_dataset import SynsetImageFolderFactory

        return SynsetImageFolderFactory(
            loader=default_loader,
            indexer=self._synset_dataset_indexer,
        )

    @cached_property
    def _evaluation_runtime(self):
        from features.evaluation.runtime import EvaluationRuntime

        return EvaluationRuntime(
            batch_size=self._config.evaluation_batch_size,
            shuffle=False,
            num_workers=self._config.evaluation_num_workers,
            wordnet=self._wordnet,
        )

    @cached_property
    def _model_evaluation_service(self):
        from features.evaluation.service import ModelEvaluationService

        return ModelEvaluationService(
            image_dataset_factory=self._image_dataset_factory,
            evaluation_runtime=self._evaluation_runtime,
            file_store=self._file_store,
            imagenet_metadata_service=self._imagenet_metadata_service,
            progress_log_every_batches=self._config.evaluation_progress_log_every_batches,
        )

    def evaluate_handler(self) -> EvaluateHandler:
        from features.evaluation.command import Handler as EvaluateHandler

        return EvaluateHandler(
            model_evaluation_service=self._model_evaluation_service,
            class_map_path=self._config.imagenet_class_map,
            index_to_wnid_path=self._config.torchvision_index_to_wnid,
        )


class VisualizationContainer(_FileStoreMixin):
    @cached_property
    def _analysis_series_loader(self) -> AnalysisSeriesLoader:
        from features.visualization.loading import AnalysisSeriesLoader

        return AnalysisSeriesLoader(file_store=self._file_store)

    @cached_property
    def _matplotlib_renderer(self) -> MatplotlibVisualizationRenderer:
        from features.visualization.rendering import MatplotlibVisualizationRenderer

        return MatplotlibVisualizationRenderer()

    @cached_property
    def _visualization_service(self) -> VisualizationService:
        from features.visualization.service import VisualizationService

        return VisualizationService(
            series_loader=self._analysis_series_loader,
            renderer=self._matplotlib_renderer,
        )

    def distribution_handler(self) -> DistributionHandler:
        from features.visualization.command import DistributionHandler

        return DistributionHandler(visualization_service=self._visualization_service)
