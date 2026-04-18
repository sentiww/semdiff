from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetClearInput:
    dataset: str


@dataclass(frozen=True)
class DatasetClearOutput:
    dataset: str


class DatasetInitHandler:
    def __init__(self: DatasetInitHandler) -> None:
        pass

    def __call__(self: DatasetInitHandler, cmd: DatasetClearInput) -> DatasetClearOutput:
        logger.info("Clearing %s dataset", cmd.dataset)

        logger.info("Clearing %s dataset", cmd.dataset)
        return DatasetClearOutput(dataset=cmd.dataset)
