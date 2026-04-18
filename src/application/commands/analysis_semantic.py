from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Input:
    metric: str
    output_directory: Path


@dataclass(frozen=True)
class Output:
    pass


class Handler:
    analysis: str = "semantic"

    def __init__(self: Handler) -> None:
        pass

    def __call__(self: Handler, cmd: Input) -> Output:
        return Output()
