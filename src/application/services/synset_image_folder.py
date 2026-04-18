from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
import re
from typing import Callable, Protocol

from torch.utils.data import Dataset


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
