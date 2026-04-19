from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from collections.abc import Collection, Iterable
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

logger = logging.getLogger(__name__)
_DATASET_PROGRESS_EVERY = 1000
_WNID_DIR_PATTERN = re.compile(r"n\d{8}$")
_ARCHIVE_URL_SCHEMES = frozenset({"http", "https", "ftp", "file"})


@dataclass(frozen=True)
class DatasetClearInput:
    dataset: str


@dataclass(frozen=True)
class DatasetClearOutput:
    dataset: str


class DatasetClearHandler:
    def __call__(self, cmd: DatasetClearInput) -> DatasetClearOutput:
        logger.info("Clearing %s dataset", cmd.dataset)
        return DatasetClearOutput(dataset=cmd.dataset)


@dataclass(frozen=True)
class ImageNetOInitInput:
    archive_source: str
    output_directory: Path
    image_suffixes: tuple[str, ...]
    class_map_path: Path


@dataclass(frozen=True)
class ImageNetOInitOutput:
    pass


class ImageNetOInitHandler:
    dataset: str = "imagenet-o"

    def __init__(
        self,
        archive_fetcher: UrlLibArchiveFetcher,
        archive_extractor: TarArchiveExtractor,
        output_preparer: GeneratedSynsetDirectoryPreparer,
        file_mover: ShutilFileMover,
        imagenet_metadata_service: ImageNetMetadataService,
    ) -> None:
        self._archive_fetcher = archive_fetcher
        self._archive_extractor = archive_extractor
        self._output_preparer = output_preparer
        self._file_mover = file_mover
        self._imagenet_metadata_service = imagenet_metadata_service

    def __call__(self, cmd: ImageNetOInitInput) -> ImageNetOInitOutput:
        logger.info(
            "Initializing %s dataset into %s from %s",
            self.dataset,
            cmd.output_directory,
            cmd.archive_source,
        )

        with TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            extraction_root = tmp_root / "imagenet-o"
            with _materialize_archive_path(
                cmd.archive_source,
                archive_fetcher=self._archive_fetcher,
                download_destination=tmp_root / "imagenet-o.tar",
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
            for moved_count, image_path in enumerate(image_paths, start=1):
                synset_id = self._resolve_target_synset_id(
                    image_path,
                    extraction_root=extraction_root,
                    class_map=synset_label_map,
                )
                destination = cmd.output_directory / synset_id / image_path.name
                self._file_mover.move(image_path, destination)
                if (
                    moved_count % _DATASET_PROGRESS_EVERY == 0
                    or moved_count == len(image_paths)
                ):
                    logger.info(
                        "Initialized %s images for %s (%s/%s)",
                        moved_count,
                        self.dataset,
                        moved_count,
                        len(image_paths),
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


@dataclass(frozen=True)
class ImageNet1KInitInput:
    archive_source: str
    output_directory: Path
    image_suffixes: tuple[str, ...]
    meta_path: Path
    ground_truth_path: Path


@dataclass(frozen=True)
class ImageNet1KInitOutput:
    pass


class ImageNet1KInitHandler:
    dataset: str = "imagenet-1k"

    def __init__(
        self,
        archive_fetcher: UrlLibArchiveFetcher,
        archive_extractor: TarArchiveExtractor,
        file_mover: ShutilFileMover,
        index_parser: ImageNetValidationImageIndexParser,
        imagenet_metadata_service: ImageNetMetadataService,
    ) -> None:
        self._archive_fetcher = archive_fetcher
        self._archive_extractor = archive_extractor
        self._file_mover = file_mover
        self._index_parser = index_parser
        self._imagenet_metadata_service = imagenet_metadata_service

    def __call__(self, cmd: ImageNet1KInitInput) -> ImageNet1KInitOutput:
        logger.info(
            "Initializing %s dataset into %s from archive %s",
            self.dataset,
            cmd.output_directory,
            cmd.archive_source,
        )

        with TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            extraction_root = tmp_root / "imagenet-1k"
            with _materialize_archive_path(
                cmd.archive_source,
                archive_fetcher=self._archive_fetcher,
                download_destination=tmp_root / "imagenet-1k.tar",
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

            for moved_count, image_path in enumerate(image_paths, start=1):
                index = self._index_parser.parse_index(image_path)
                wnid = validation_wnids[index - 1]
                destination = cmd.output_directory / wnid / image_path.name
                self._file_mover.move(image_path, destination)
                if (
                    moved_count % _DATASET_PROGRESS_EVERY == 0
                    or moved_count == len(image_paths)
                ):
                    logger.info(
                        "Initialized %s validation images for %s (%s/%s)",
                        moved_count,
                        self.dataset,
                        moved_count,
                        len(image_paths),
                    )

        logger.info("Initialized %s dataset", self.dataset)
        return ImageNet1KInitOutput()


@contextmanager
def _materialize_archive_path(
    archive_source: str,
    *,
    archive_fetcher: UrlLibArchiveFetcher,
    download_destination: Path,
):
    if _is_archive_url(archive_source):
        archive_fetcher.fetch(archive_source, download_destination)
        yield download_destination
        return

    archive_path = Path(archive_source)
    logger.info("Using local archive %s", archive_path)
    yield archive_path


def _is_archive_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in _ARCHIVE_URL_SCHEMES and bool(parsed.path)
