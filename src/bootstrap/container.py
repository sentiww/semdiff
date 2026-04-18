# bootstrap/container.py
from dependency_injector import containers, providers

from application.commands.dataset_init_imagenet_o import (
    Handler as DatasetImagenetOInitHandler,
)
from application.commands.dataset_init_imagenet_1k import (
    Handler as DatasetImagenet1KInitHandler,
)
from application.commands.evaluate import (
    
)
from infrastructure.generated_synset_directory_preparer import (
    GeneratedSynsetDirectoryPreparer,
)
from infrastructure.shutil_file_mover import ShutilFileMover
from infrastructure.tar_archive_extractor import TarArchiveExtractor
from infrastructure.urllib_archive_fetcher import UrlLibArchiveFetcher
from infrastructure.wordnet_synset_resolver import WordNetSynsetResolver
from infrastructure.default_meta_label_loader import DefaultValidationLabelLoader
from infrastructure.imagenet_validation_image_index_parser import (
    ImageNetValidationImageIndexParser,
)
from infrastructure.wordnet_synset_resolver import WordNetSynsetResolver
from application.services.imagenet_metadata_service import ImageNetMetadataService


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    urllib_archive_fetcher = providers.Singleton(UrlLibArchiveFetcher)
    tar_archive_extractor = providers.Singleton(TarArchiveExtractor)
    generated_synset_directory_preparer = providers.Singleton(
        GeneratedSynsetDirectoryPreparer
    )
    shutil_file_mover = providers.Singleton(ShutilFileMover)
    wordnet_synset_resolver = providers.Singleton(WordNetSynsetResolver)
    index_parser = providers.Singleton(ImageNetValidationImageIndexParser)
    imagenet_metadata_service = providers.Singleton(ImageNetMetadataService)

    label_loader = providers.Singleton(
        DefaultValidationLabelLoader, imagenet_metadata_service
    )

    dataset_imagenet_o_init_handler = providers.Factory(
        DatasetImagenetOInitHandler,
        urllib_archive_fetcher,
        tar_archive_extractor,
        generated_synset_directory_preparer,
        shutil_file_mover,
        wordnet_synset_resolver,
    )

    dataset_imagenet_1k_init_handler = providers.Factory(
        DatasetImagenet1KInitHandler,
        tar_archive_extractor,
        label_loader,
        shutil_file_mover,
        index_parser,
        wordnet_synset_resolver,
        imagenet_metadata_service,
    )

    evaluate_handler = providers.Factory()
