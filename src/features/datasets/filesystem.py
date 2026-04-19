from __future__ import annotations

import logging
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

_SYNSET_DIR_PATTERN = re.compile(r"n\d+$")
_VALIDATION_FILENAME_PATTERN = re.compile(r"ILSVRC2012_val_(\d+)$")
logger = logging.getLogger(__name__)


class UrlLibArchiveFetcher:
    def fetch(self, url: str, destination: Path) -> None:
        logger.info("Downloading archive from %s to %s", url, destination)
        urllib.request.urlretrieve(url, destination)
        logger.info("Downloaded archive to %s", destination)


class TarArchiveExtractor:
    def extract(self, archive_path: Path, destination: Path) -> None:
        logger.info("Extracting archive %s into %s", archive_path, destination)
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(destination)
        logger.info("Extracted archive %s", archive_path)


class ShutilFileMover:
    def move(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


class GeneratedSynsetDirectoryPreparer:
    def prepare(self, output_directory: Path) -> None:
        if not output_directory.exists():
            logger.info(
                "Output directory %s does not exist yet; no generated synset folders to clear",
                output_directory,
            )
            return

        removed_count = 0
        for path in output_directory.iterdir():
            if path.is_dir() and _SYNSET_DIR_PATTERN.fullmatch(path.name):
                shutil.rmtree(path)
                removed_count += 1
        logger.info(
            "Cleared %s generated synset directories under %s",
            removed_count,
            output_directory,
        )


class ImageNetValidationImageIndexParser:
    def __init__(self, pattern: re.Pattern[str] = _VALIDATION_FILENAME_PATTERN) -> None:
        self._pattern = pattern

    def parse_index(self, image_path: Path) -> int:
        match = self._pattern.fullmatch(image_path.stem)
        if match is None:
            raise RuntimeError(
                f"Unexpected ImageNet validation filename: {image_path.name}"
            )
        return int(match.group(1))
