from pathlib import Path
from typing import Protocol


class FileMover(Protocol):
    def move(self, source: Path, destination: Path) -> None: ...
