from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from semdiff.datasets.filesystem import (
    ShutilFileMover,
    TarArchiveExtractor,
    UrlLibArchiveFetcher,
)
from semdiff.datasets.metadata import ImageNetMetadataService
from semdiff.core.handlers import CommandInput, CommandOutput

logger = logging.getLogger(__name__)
_DATASET_PROGRESS_EVERY = 1000
_ARCHIVE_URL_SCHEMES = frozenset({"http", "https", "ftp", "file"})


def _is_archive_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in _ARCHIVE_URL_SCHEMES and bool(parsed.path)


class DatasetInitializer(ABC):
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

    @abstractmethod
    def initialize(self, cmd: CommandInput) -> CommandOutput:
        pass