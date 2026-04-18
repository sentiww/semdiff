from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
import re
from typing import Callable, Protocol

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Input:
    model: str
    dataset: Path
    output: Path


@dataclass(frozen=True)
class Output:
    pass


class Handler:
    def __init__(self: Handler, dataset_factory, model_factory) -> None:
        self._dataset_factory = dataset_factory
        self._model_factory = model_factory

    def __call__(self: Handler, cmd: Input) -> Output:
        model = self._model_factory.create(cmd.model)

        dataset = SynsetImageFolder()

        return Output()


class SampleBuilder(Protocol):
    def build(
        self,
        root: Path,
        classes: list[str],
        class_to_idx: dict[str, int],
    ) -> list[tuple[str, int]]: ...


class TargetBuilder(Protocol):
    def build(self, samples: list[tuple[str, int]]) -> list[int]: ...


class DefaultSampleBuilder:
    def __init__(self, image_suffixes: frozenset[str]) -> None:
        self._image_suffixes = image_suffixes

    def build(
        self,
        root: Path,
        classes: list[str],
        class_to_idx: dict[str, int],
    ) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []

        for synset in classes:
            synset_dir = root / synset
            class_index = class_to_idx[synset]

            for image_path in sorted(synset_dir.rglob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in self._image_suffixes:
                    continue
                samples.append((str(image_path), class_index))

        if not samples:
            raise RuntimeError(f"No image files found under {root}")

        return samples


class SynsetImageFolder(Dataset[tuple[object, int]]):
    def __init__(
        self,
        root: Path,
        *,
        transform: Callable | None,
        loader: Callable,
        dataset_path_validator: DatasetPathValidator,
        dataset_index_builder: DatasetIndexBuilder,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.loader = loader

        dataset_path_validator.validate(self.root)
        dataset_index = dataset_index_builder.build(self.root)

        self.classes = dataset_index.classes
        self.class_to_idx = dataset_index.class_to_idx
        self.samples = dataset_index.samples
        self.targets = dataset_index.targets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image_path, target = self.samples[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target
