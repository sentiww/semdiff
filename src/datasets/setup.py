from __future__ import annotations

import logging

from .common import LOGGER
from .registry import init_registered_dataset


def init_dataset(name: str, *, logger: logging.Logger = LOGGER) -> None:
    init_registered_dataset(name, logger=logger)
