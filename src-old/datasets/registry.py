from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .common import LOGGER
from .setup_imagenet_1k import init_dataset as init_imagenet_1k_dataset
from .setup_imagenet_o import init_dataset as init_imagenet_o_dataset

DatasetInitializer = Callable[..., None]


@dataclass(frozen=True)
class DatasetHandler:
    name: str
    init_dataset: DatasetInitializer


DATASET_HANDLERS: tuple[DatasetHandler, ...] = (
    DatasetHandler("imagenet-1k", init_imagenet_1k_dataset),
    DatasetHandler("imagenet-o", init_imagenet_o_dataset),
)

DATASET_NAMES = tuple(handler.name for handler in DATASET_HANDLERS)


def init_registered_dataset(name: str, *, logger: logging.Logger = LOGGER) -> None:
    dataset_handler = _find_dataset_handler(name)
    dataset_handler.init_dataset(logger=logger.getChild(name))


def _find_dataset_handler(name: str) -> DatasetHandler:
    for dataset_handler in DATASET_HANDLERS:
        if dataset_handler.name == name:
            return dataset_handler

    raise ValueError(f"Unknown dataset: {name}")
