from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

ImageValue = object
ImageLoader = Callable[[str], ImageValue]
ImageTransform = Callable[[ImageValue], ImageValue]
DatasetSample = tuple[str, int]


@dataclass(frozen=True)
class SynsetDatasetIndex:
    classes: list[str]
    class_to_idx: dict[str, int]
    samples: list[DatasetSample]
    targets: list[int]


class SynsetDatasetIndexer:
    def __init__(
        self,
        *,
        synset_dir_pattern: re.Pattern[str],
        image_suffixes: frozenset[str],
    ) -> None:
        self._synset_dir_pattern = synset_dir_pattern
        self._image_suffixes = image_suffixes

    def build(self, root: Path) -> SynsetDatasetIndex:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")

        logger.info("Indexing synset dataset under %s", root)
        classes = self._find_classes(root)
        if not classes:
            raise RuntimeError(
                f"Dataset {root} is not initialized as synset folders. "
                "Run dataset init first."
            )

        class_to_idx = {synset: index for index, synset in enumerate(classes)}
        samples = self._build_samples(root, classes, class_to_idx)
        if not samples:
            raise RuntimeError(f"No image files found under {root}")

        logger.info(
            "Indexed synset dataset under %s with %s classes and %s samples",
            root,
            len(classes),
            len(samples),
        )
        return SynsetDatasetIndex(
            classes=classes,
            class_to_idx=class_to_idx,
            samples=samples,
            targets=[target for _, target in samples],
        )

    def _find_classes(self, root: Path) -> list[str]:
        classes = [
            path.name
            for path in root.iterdir()
            if path.is_dir() and self._synset_dir_pattern.fullmatch(path.name)
        ]
        classes.sort()
        return classes

    def _build_samples(
        self,
        root: Path,
        classes: list[str],
        class_to_idx: dict[str, int],
    ) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        for synset in classes:
            synset_dir = root / synset
            class_index = class_to_idx[synset]
            for image_path in sorted(synset_dir.rglob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in self._image_suffixes:
                    continue
                samples.append((str(image_path), class_index))
        return samples


class SynsetImageFolder(Dataset[tuple[ImageValue, int]]):
    def __init__(
        self,
        root: Path,
        *,
        transform: ImageTransform | None,
        loader: ImageLoader,
        indexer: SynsetDatasetIndexer,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.loader = loader

        dataset_index = indexer.build(self.root)
        self.classes = dataset_index.classes
        self.class_to_idx = dataset_index.class_to_idx
        self.samples = dataset_index.samples
        self.targets = dataset_index.targets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[ImageValue, int]:
        image_path, target = self.samples[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class SynsetImageFolderFactory:
    def __init__(
        self,
        *,
        loader: ImageLoader,
        indexer: SynsetDatasetIndexer,
    ) -> None:
        self._loader = loader
        self._indexer = indexer

    def create(
        self,
        root: Path,
        *,
        transform: ImageTransform | None,
    ) -> SynsetImageFolder:
        return SynsetImageFolder(
            root,
            transform=transform,
            loader=self._loader,
            indexer=self._indexer,
        )
