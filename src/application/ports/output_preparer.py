from pathlib import Path
from typing import Protocol


class OutputPreparer(Protocol):
    def prepare(self, output_directory: Path) -> None: ...
