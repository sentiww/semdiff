from pathlib import Path
from typing import Protocol


class ArchiveExtractor(Protocol):
    def extract(self, archive_path: Path, destination: Path) -> None: ...
