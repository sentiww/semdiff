from __future__ import annotations

import re
import shutil
from pathlib import Path

_SYNSET_DIR_PATTERN = re.compile(r"n\d+$")


class GeneratedSynsetDirectoryPreparer:
    def prepare(self, output_directory: Path) -> None:
        if not output_directory.exists():
            return

        for path in output_directory.iterdir():
            if path.is_dir() and _SYNSET_DIR_PATTERN.fullmatch(path.name):
                shutil.rmtree(path)
