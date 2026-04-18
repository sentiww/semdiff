from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

IMAGE_SUFFIXES = {
    ".jpeg",
}
SYNSET_DIR_PATTERN = re.compile(r"n\d+$")


def validate_imagefolder_dataset(dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    classes = find_dataset_classes(dataset_path)
    if not classes:
        raise RuntimeError(
            f"Dataset {dataset_path} is not initialized as synset folders. "
            "Run datasets init first."
        )


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

        self.classes = find_dataset_classes(self.root)
        self.class_to_idx = build_class_index_map(self.classes)
        self.samples = build_samples(self.root, self.classes, self.class_to_idx)
        self.targets = build_targets(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image_path, target = self.samples[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def find_dataset_classes(root: Path) -> list[str]:
    classes: list[str] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if not SYNSET_DIR_PATTERN.fullmatch(path.name):
            continue
        classes.append(path.name)

    classes.sort()
    return classes


def build_class_index_map(classes: list[str]) -> dict[str, int]:
    class_to_idx: dict[str, int] = {}
    class_index = 0
    for synset in classes:
        class_to_idx[synset] = class_index
        class_index += 1
    return class_to_idx


def build_targets(samples: list[tuple[str, int]]) -> list[int]:
    targets: list[int] = []
    for _, target in samples:
        targets.append(target)
    return targets


def build_samples(
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
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            samples.append((str(image_path), class_index))

    if not samples:
        raise RuntimeError(f"No image files found under {root}")
    return samples
