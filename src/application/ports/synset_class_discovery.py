from __future__ import annotations

from pathlib import Path


class SynsetClassDiscovery(Protocol):
    def find_classes(self, root: Path) -> list[str]: ...
