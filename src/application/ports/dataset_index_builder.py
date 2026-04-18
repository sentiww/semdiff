from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SynsetDatasetIndex:
    classes: list[str]
    class_to_idx: dict[str, int]
    samples: list[tuple[str, int]]
    targets: list[int]


class DatasetIndexBuilder(Protocol):
    def build(self, root: Path) -> SynsetDatasetIndex: ...
