from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from features.datasets.filesystem import (
    GeneratedSynsetDirectoryPreparer,
    ImageNetValidationImageIndexParser,
    ShutilFileMover,
    TarArchiveExtractor,
    UrlLibArchiveFetcher,
)
from features.datasets.metadata import ImageNetMetadataService, SynsetLabelMap
from features.handlers.base import CommandInput, CommandOutput, Handler, HandlerFactory

logger = logging.getLogger(__name__)
_DATASET_PROGRESS_EVERY = 1000
_WNID_DIR_PATTERN = re.compile(r"n\d{8}$")
_ARCHIVE_URL_SCHEMES = frozenset({"http", "https", "ftp", "file"})


@dataclass(frozen=True)
class ImageNetOInitInput(CommandInput):
    archive_source: str
    output_directory: Path
    image_suffixes: tuple[str, ...]
    class_map_path: Path


@dataclass(frozen=True)
class ImageNetOInitOutput(CommandOutput):
    pass


@dataclass(frozen=True)
class ImageNet1KInitInput(CommandInput):
    archive_source: str
    output_directory: Path
    image_suffixes: tuple[str, ...]
    meta_path: Path
    ground_truth_path: Path


@dataclass(frozen=True)
class ImageNet1KInitOutput(CommandOutput):
    pass


class BaseDatasetInitHandler(Handler):
    dataset: str = "base"

    def __init__(
        self,
        archive_fetcher: UrlLibArchiveFetcher,
        archive_extractor: TarArchiveExtractor,
        file_mover: ShutilFileMover,
        imagenet_metadata_service: ImageNetMetadataService,
    ) -> None:
        self._archive_fetcher = archive_fetcher
        self._archive_extractor = archive_extractor
        self._file_mover = file_mover
        self._imagenet_metadata_service = imagenet_metadata_service

    @contextmanager
    def _materialize_archive(
        self,
        archive_source: str,
        temp_root: Path,
    ):
        if _is_archive_url(archive_source):
            download_dest = temp_root / f"{self.dataset}.tar"
            self._archive_fetcher.fetch(archive_source, download_dest)
            yield download_dest
            return

        archive_path = Path(archive_source)
        logger.info("Using local archive %s", archive_path)
        yield archive_path

    def _move_with_progress(
        self,
        image_paths: list[Path],
        resolve_dest: Callable[[Path], Path],
    ) -> None:
        total = len(image_paths)
        for moved_count, image_path in enumerate(image_paths, start=1):
            destination = resolve_dest(image_path)
            self._file_mover.move(image_path, destination)
            if moved_count % _DATASET_PROGRESS_EVERY == 0 or moved_count == total:
                logger.info(
                    "Initialized %s images for %s (%s/%s)",
                    moved_count,
                    self.dataset,
                    moved_count,
                    total,
                )


class ImageNetOInitHandler(BaseDatasetInitHandler):
    dataset: str = "imagenet-o"

    def __init__(
        self,
        archive_fetcher: UrlLibArchiveFetcher,
        archive_extractor: TarArchiveExtractor,
        output_preparer: GeneratedSynsetDirectoryPreparer,
        file_mover: ShutilFileMover,
        imagenet_metadata_service: ImageNetMetadataService,
    ) -> None:
        super().__init__(
            archive_fetcher,
            archive_extractor,
            file_mover,
            imagenet_metadata_service,
        )
        self._output_preparer = output_preparer

    def __call__(self, cmd: ImageNetOInitInput) -> ImageNetOInitOutput:
        logger.info(
            "Initializing %s dataset into %s from %s",
            self.dataset,
            cmd.output_directory,
            cmd.archive_source,
        )

        with TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            extraction_root = tmp_root / self.dataset
            with self._materialize_archive(
                cmd.archive_source,
                tmp_root,
            ) as archive_path:
                extraction_root.mkdir()
                self._archive_extractor.extract(archive_path, extraction_root)

            self._output_preparer.prepare(cmd.output_directory)
            synset_label_map = self._imagenet_metadata_service.load_synset_label_map(
                cmd.class_map_path
            )

            image_paths = list(self._iter_images(extraction_root, cmd.image_suffixes))
            logger.info(
                "Resolved %s candidate images for %s dataset initialization",
                len(image_paths),
                self.dataset,
            )

            self._move_with_progress(
                image_paths,
                lambda img: cmd.output_directory
                / self._resolve_target_synset_id(
                    img,
                    extraction_root=extraction_root,
                    class_map=synset_label_map,
                )
                / img.name,
            )

        logger.info("Initialized %s dataset", self.dataset)
        return ImageNetOInitOutput()

    def _iter_images(
        self,
        root: Path,
        allowed_suffixes: Collection[str],
    ) -> Iterable[Path]:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name == "README.txt":
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue
            yield path

    def _resolve_target_synset_id(
        self,
        image_path: Path,
        *,
        extraction_root: Path,
        class_map: SynsetLabelMap,
    ) -> str:
        relative_path = image_path.relative_to(extraction_root)
        if not relative_path.parts or len(relative_path.parts) < 2:
            raise RuntimeError(
                "ImageNet-O archive must organize images under class directories; "
                f"cannot resolve target synset for {image_path}"
            )

        class_label = relative_path.parent.name
        if _WNID_DIR_PATTERN.fullmatch(class_label):
            return class_label

        return self._imagenet_metadata_service.resolve_synset_id_from_label(
            class_label,
            class_map,
        )


class ImageNet1KInitHandler(BaseDatasetInitHandler):
    dataset: str = "imagenet-1k"

    def __init__(
        self,
        archive_fetcher: UrlLibArchiveFetcher,
        archive_extractor: TarArchiveExtractor,
        file_mover: ShutilFileMover,
        index_parser: ImageNetValidationImageIndexParser,
        imagenet_metadata_service: ImageNetMetadataService,
    ) -> None:
        super().__init__(
            archive_fetcher,
            archive_extractor,
            file_mover,
            imagenet_metadata_service,
        )
        self._index_parser = index_parser

    def __call__(self, cmd: ImageNet1KInitInput) -> ImageNet1KInitOutput:
        logger.info(
            "Initializing %s dataset into %s from archive %s",
            self.dataset,
            cmd.output_directory,
            cmd.archive_source,
        )

        with TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            extraction_root = tmp_root / self.dataset
            with self._materialize_archive(
                cmd.archive_source,
                tmp_root,
            ) as archive_path:
                logger.info("Extracting validation archive from %s", archive_path)
                self._archive_extractor.extract(archive_path, extraction_root)

            validation_wnids = self._imagenet_metadata_service.load_validation_wnids(
                cmd.meta_path,
                cmd.ground_truth_path,
            )

            image_paths = [
                image_path
                for image_path in sorted(extraction_root.iterdir())
                if image_path.is_file() and image_path.suffix.lower() in cmd.image_suffixes
            ]
            logger.info(
                "Resolved %s validation images for %s dataset initialization",
                len(image_paths),
                self.dataset,
            )

            self._move_with_progress(
                image_paths,
                lambda img: cmd.output_directory
                / validation_wnids[self._index_parser.parse_index(img) - 1]
                / img.name,
            )

        logger.info("Initialized %s dataset", self.dataset)
        return ImageNet1KInitOutput()


def _is_archive_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in _ARCHIVE_URL_SCHEMES and bool(parsed.path)


class DatasetHandlers(HandlerFactory):
    def __init__(
        self,
        archive_fetcher: UrlLibArchiveFetcher,
        archive_extractor: TarArchiveExtractor,
        output_preparer: GeneratedSynsetDirectoryPreparer,
        file_mover: ShutilFileMover,
        index_parser: ImageNetValidationImageIndexParser,
        imagenet_metadata_service: ImageNetMetadataService,
    ) -> None:
        self._archive_fetcher = archive_fetcher
        self._archive_extractor = archive_extractor
        self._output_preparer = output_preparer
        self._file_mover = file_mover
        self._index_parser = index_parser
        self._imagenet_metadata_service = imagenet_metadata_service

    def create_imagenet_o_init(self) -> ImageNetOInitHandler:
        return ImageNetOInitHandler(
            archive_fetcher=self._archive_fetcher,
            archive_extractor=self._archive_extractor,
            output_preparer=self._output_preparer,
            file_mover=self._file_mover,
            imagenet_metadata_service=self._imagenet_metadata_service,
        )

    def create_imagenet_1k_init(self) -> ImageNet1KInitHandler:
        return ImageNet1KInitHandler(
            archive_fetcher=self._archive_fetcher,
            archive_extractor=self._archive_extractor,
            file_mover=self._file_mover,
            index_parser=self._index_parser,
            imagenet_metadata_service=self._imagenet_metadata_service,
        )
