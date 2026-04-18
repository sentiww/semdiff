from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Input:
    archive_path: Path
    output_directory: Path
    image_suffixes: list[str]
    meta_path: Path
    ground_truth_path: Path


@dataclass(frozen=True)
class Output:
    pass


class Handler:
    dataset: str = "imagenet-o"

    def __init__(
        self: Handler,
        archive_extractor,
        label_loader,
        file_mover,
        index_parser,
        synset_resolver,
        imagenet_metadata_service,
    ) -> None:
        self._archive_extractor = archive_extractor
        self._label_loader = label_loader
        self._file_mover = file_mover
        self._index_parser = index_parser
        self._synset_resolver = synset_resolver
        self._imagenet_metadata_service = imagenet_metadata_service

    def __call__(self: Handler, cmd: Input) -> Output:
        logger.info("Initializing %s dataset", self.dataset)

        count = 0
        with TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            extraction_root = tmp_root / "imagenet-1k"

            logger.info("Extracting validation archive from %s", cmd.archive_path)
            self._archive_extractor.extract(cmd.archive_path, extraction_root)

            validation_wnids = self._label_loader.load_validation_wnids(
                cmd.meta_path,
                cmd.ground_truth_path,
            )

            for image_path in sorted(extraction_root.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in cmd.image_suffixes:
                    continue

                index = self._index_parser.parse_index(image_path)
                wnid = validation_wnids[index - 1]
                destination = cmd.output_directory / wnid / image_path.name
                self._file_mover.move(image_path, destination)
                count += 1

        logger.info("Initialized %s dataset", self.dataset)
        return Output()
