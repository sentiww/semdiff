from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger("main.datasets")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
IMAGENET_1K_ROOT = DATASETS_ROOT / "imagenet-1k"
IMAGENET_O_ROOT = DATASETS_ROOT / "imagenet-o"

_DATASET_ROOTS = {
    "imagenet-1k": IMAGENET_1K_ROOT,
    "imagenet-o": IMAGENET_O_ROOT,
}


def dataset_root(name: str) -> Path:
    try:
        return _DATASET_ROOTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc
