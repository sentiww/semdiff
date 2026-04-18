from __future__ import annotations

from pathlib import Path

from application.ports.synset_class_discovery import SynsetClassDiscovery


class DefaultDatasetPathValidator:
    def __init__(self, class_discovery: SynsetClassDiscovery) -> None:
        self._class_discovery = class_discovery

    def validate(self, dataset_path: Path) -> None:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        classes = self._class_discovery.find_classes(dataset_path)
        if not classes:
            raise RuntimeError(
                f"Dataset {dataset_path} is not initialized as synset folders. "
                "Run datasets init first."
            )
