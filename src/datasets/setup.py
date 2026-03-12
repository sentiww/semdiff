from __future__ import annotations

import logging

from .common import LOGGER
from .setup_imagenet_1k import init_dataset as init_imagenet_1k
from .setup_imagenet_o import init_dataset as init_imagenet_o

_INIT_DATASETS = {
    "imagenet-1k": init_imagenet_1k,
    "imagenet-o": init_imagenet_o,
}


def init_dataset(
    name: str, *, logger: logging.Logger = LOGGER
) -> None:
    try:
        init = _INIT_DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc

    init(logger=logger.getChild(name))
