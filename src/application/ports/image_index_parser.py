from pathlib import Path
from typing import Protocol


class ImageIndexParser(Protocol):
    def parse_index(self, image_path: Path) -> int: ...
