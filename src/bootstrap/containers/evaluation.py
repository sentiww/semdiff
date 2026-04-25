from __future__ import annotations

from functools import cached_property

from bootstrap.containers.base import _FileStoreMixin, _ImageNetMetadataMixin, _WordNetMixin
from bootstrap.config import AppConfig
from config.evaluation import EvaluationSettings


class EvaluationContainer(_FileStoreMixin, _ImageNetMetadataMixin, _WordNetMixin):
    def __init__(self, config: AppConfig, evaluation_settings: EvaluationSettings | None = None) -> None:
        self._config = config
        self._evaluation_settings = evaluation_settings or EvaluationSettings.DEFAULT

    @cached_property
    def synset_dataset_indexer(self):
        from features.datasets.synset_dataset import SynsetDatasetIndexer

        return SynsetDatasetIndexer(
            synset_dir_pattern=self._evaluation_settings.synset_dir_pattern,
            image_suffixes=self._evaluation_settings.image_suffixes,
        )

    @cached_property
    def image_dataset_factory(self):
        from torchvision.datasets.folder import default_loader

        from features.datasets.synset_dataset import SynsetImageFolderFactory

        return SynsetImageFolderFactory(
            loader=default_loader,
            indexer=self.synset_dataset_indexer,
        )

    @cached_property
    def evaluation_runtime(self):
        from features.evaluation.runtime import EvaluationRuntime

        return EvaluationRuntime(
            batch_size=self._config.evaluation_batch_size,
            shuffle=False,
            num_workers=self._config.evaluation_num_workers,
            wordnet=self._wordnet,
        )

    @cached_property
    def model_evaluation_service(self):
        from features.evaluation.service import ModelEvaluationService

        return ModelEvaluationService(
            image_dataset_factory=self.image_dataset_factory,
            evaluation_runtime=self.evaluation_runtime,
            file_store=self._file_store,
            imagenet_metadata_service=self._imagenet_metadata_service,
            progress_log_every_batches=self._config.evaluation_progress_log_every_batches,
        )
