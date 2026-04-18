from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DatasetPathValidator(Protocol):
    def validate(self, dataset_path: Path) -> None: ...
