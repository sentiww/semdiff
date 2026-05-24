from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpeg"}
SYNSET_DIR_PATTERN = re.compile(r"n\d+$")


class SynsetImageFolder(Dataset[tuple[object, int]]):
    def __init__(
        self,
        root: Path,
        *,
        transform: Callable | None,
        loader: Callable,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.loader = loader

        self.classes = sorted([
            p.name for p in root.iterdir()
            if p.is_dir() and SYNSET_DIR_PATTERN.fullmatch(p.name)
        ])
        self.class_to_idx = {synset: i for i, synset in enumerate(self.classes)}
        self.samples = self._build_samples()
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image_path, target = self.samples[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def _build_samples(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for synset in self.classes:
            synset_dir = self.root / synset
            class_index = self.class_to_idx[synset]
            for image_path in sorted(synset_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    samples.append((str(image_path), class_index))

        if not samples:
            raise RuntimeError(f"No image files found under {self.root}")
        return samples