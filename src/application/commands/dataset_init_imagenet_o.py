from __future__ import annotations

import logging
from typing import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Input:
    archive_url: str
    output_directory: Path
    image_suffixes: list[str]


@dataclass(frozen=True)
class Output:
    pass


class Handler:
    dataset: str = "imagenet-o"

    def __init__(
        self: Handler,
        archive_fetcher,
        archive_extractor,
        output_preparer,
        file_mover,
        synset_resolver,
    ) -> None:
        self._archive_fetcher = archive_fetcher
        self._archive_extractor = archive_extractor
        self._output_preparer = output_preparer
        self._file_mover = file_mover
        self._synset_resolver = synset_resolver

    def __call__(self: Handler, cmd: Input) -> Output:
        logger.info("Initializing %s dataset", self.dataset)

        with TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            archive_path = tmp_root / "imagenet-o.tar"
            extraction_root = tmp_root / "imagenet-o"

            self._archive_fetcher.fetch(cmd.archive_url, archive_path)
            extraction_root.mkdir()
            self._archive_extractor.extract(archive_path, extraction_root)

            self._output_preparer.prepare(cmd.output_directory)

            count = 0
            for image_path in self._iter_images(extraction_root, cmd.image_suffixes):
                synset_id = self._synset_resolver.resolve(image_path.stem)
                destination = cmd.output_directory / synset_id / image_path.name
                self._file_mover.move(image_path, destination)
                count += 1

        logger.info("Initialized %s dataset", self.dataset)
        return Output()

    def _iter_images(
        self: Handler,
        root: Path,
        allowed_suffixes: list[str],
    ) -> Iterable[Path]:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name == "README.txt":
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue
            yield path
