from __future__ import annotations

import logging
import re
from pathlib import Path


_VALIDATION_FILENAME_PATTERN = re.compile(r"ILSVRC2012_val_(\d+)$")

class ImageNetValidationImageIndexParser:
    def __init__(self, pattern: re.Pattern[str] = _VALIDATION_FILENAME_PATTERN) -> None:
        self._pattern = pattern

    def parse_index(self, image_path: Path) -> int:
        match = self._pattern.fullmatch(image_path.stem)
        if match is None:
            raise RuntimeError(
                f"Unexpected ImageNet validation filename: {image_path.name}"
            )
        return int(match.group(1))
