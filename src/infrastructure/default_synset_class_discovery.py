from __future__ import annotations

from pathlib import Path
import re


class DefaultSynsetClassDiscovery:
    def __init__(self, synset_dir_pattern: re.Pattern[str]) -> None:
        self._synset_dir_pattern = synset_dir_pattern

    def find_classes(self, root: Path) -> list[str]:
        classes: list[str] = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            if not self._synset_dir_pattern.fullmatch(path.name):
                continue
            classes.append(path.name)

        classes.sort()
        return classes
