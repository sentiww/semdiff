from __future__ import annotations

import re
from functools import partial

from config.evaluation import EvaluationSettings
from dependency_injector import containers, providers
from torchvision.datasets.folder import default_loader


def _import(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


class ApplicationContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Evaluation settings defaults
    config.evaluation.synset_dir_pattern.from_value(EvaluationSettings.DEFAULT.synset_dir_pattern)
    config.evaluation.image_suffixes.from_value(EvaluationSettings.DEFAULT.image_suffixes)

    # File store (singleton - stateless)
    file_store = providers.Singleton(
        _import("semdiff.files.store.FileStore"),
        codecs=providers.Singleton(
            partial(_import("semdiff.core.codecs.build_default_codecs"))
        ),
    )

    # Core singleton services
    wordnet_service = providers.Singleton(
        _import("semdiff.wordnet.service.WordNetService"),
    )

    imagenet_metadata_service = providers.Singleton(
        _import("semdiff.datasets.metadata.ImageNetMetadataService"),
    )

    # Dataset components (singletons)
    archive_fetcher = providers.Singleton(
        _import("semdiff.datasets.filesystem.UrlLibArchiveFetcher"),
    )

    archive_extractor = providers.Singleton(
        _import("semdiff.datasets.filesystem.TarArchiveExtractor"),
    )

    output_preparer = providers.Singleton(
        _import("semdiff.datasets.filesystem.GeneratedSynsetDirectoryPreparer"),
    )

    file_mover = providers.Singleton(
        _import("semdiff.datasets.filesystem.ShutilFileMover"),
    )

    index_parser = providers.Singleton(
        _import("semdiff.datasets.filesystem.ImageNetValidationImageIndexParser"),
    )

    # Evaluation components (factories - config-dependent per-call)
    synset_dataset_indexer = providers.Factory(
        _import("semdiff.datasets.synset_dataset.SynsetDatasetIndexer"),
        synset_dir_pattern=providers.Configuration("evaluation.synset_dir_pattern"),
        image_suffixes=providers.Configuration("evaluation.image_suffixes"),
    )

    image_loader = providers.Singleton(
        lambda: default_loader,
    )

    image_dataset_factory = providers.Factory(
        _import("semdiff.datasets.synset_dataset.SynsetImageFolderFactory"),
        loader=image_loader,
        indexer=synset_dataset_indexer,
    )

    evaluation_runtime = providers.Factory(
        _import("semdiff.evaluation.runtime.EvaluationRuntime"),
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        num_workers=config.evaluation_num_workers,
        wordnet=wordnet_service,
    )

    model_evaluation_service = providers.Factory(
        _import("semdiff.evaluation.service.ModelEvaluationService"),
        image_dataset_factory=image_dataset_factory,
        evaluation_runtime=evaluation_runtime,
        file_store=file_store,
        imagenet_metadata_service=imagenet_metadata_service,
        progress_log_every_batches=config.evaluation_progress_log_every_batches,
    )

    # Analysis services (factory)
    semantic_analysis_service = providers.Factory(
        _import("semdiff.wordnet.analysis.SemanticAnalysisService"),
        file_store=file_store,
    )

    # Visualization services (mixed)
    analysis_series_loader = providers.Factory(
        _import("semdiff.visualization.loading.AnalysisSeriesLoader"),
        file_store=file_store,
    )

    matplotlib_renderer = providers.Singleton(
        _import("semdiff.visualization.rendering.MatplotlibVisualizationRenderer"),
    )

    visualization_service = providers.Factory(
        _import("semdiff.visualization.service.VisualizationService"),
        series_loader=analysis_series_loader,
        renderer=matplotlib_renderer,
    )

    # Handler factories (factory per-call)
    dataset_handlers = providers.Factory(
        _import("semdiff.datasets.handlers.DatasetHandlers"),
        archive_fetcher=archive_fetcher,
        archive_extractor=archive_extractor,
        output_preparer=output_preparer,
        file_mover=file_mover,
        index_parser=index_parser,
        imagenet_metadata_service=imagenet_metadata_service,
    )

    evaluation_handlers = providers.Factory(
        _import("semdiff.evaluation.handlers.EvaluationHandlers"),
        evaluation_service=model_evaluation_service,
        class_map_path=config.imagenet_class_map,
        index_to_wnid_path=config.torchvision_index_to_wnid,
    )

    analysis_handlers = providers.Factory(
        _import("semdiff.wordnet.analysis.AnalysisHandlers"),
        wordnet=wordnet_service,
        semantic_analysis_service=semantic_analysis_service,
    )

    wordnet_handlers = providers.Factory(
        _import("semdiff.wordnet.handlers.WordNetHandlers"),
        wordnet=wordnet_service,
    )

    visualization_handlers = providers.Factory(
        _import("semdiff.visualization.handlers.VisualizationHandlers"),
        visualization_service=visualization_service,
    )


def create_container() -> ApplicationContainer:
    return ApplicationContainer()