from pathlib import Path
from typing import Protocol

class ArchiveFetcher(Protocol):
    def fetch(self, url: str, destination: Path) -> None: ...