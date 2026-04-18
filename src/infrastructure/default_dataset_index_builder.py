from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Protocol

from application.ports.dataset_index_builder import SynsetDatasetIndex


class DefaultDatasetIndexBuilder:
    def __init__(
        self,
        *,
        class_discovery: SynsetClassDiscovery,
        class_index_builder: ClassIndexBuilder,
        sample_builder: SampleBuilder,
        target_builder: TargetBuilder,
    ) -> None:
        self._class_discovery = class_discovery
        self._class_index_builder = class_index_builder
        self._sample_builder = sample_builder
        self._target_builder = target_builder

    def build(self, root: Path) -> SynsetDatasetIndex:
        classes = self._class_discovery.find_classes(root)
        class_to_idx = self._class_index_builder.build(classes)
        samples = self._sample_builder.build(root, classes, class_to_idx)
        targets = self._target_builder.build(samples)

        return SynsetDatasetIndex(
            classes=classes,
            class_to_idx=class_to_idx,
            samples=samples,
            targets=targets,
        )
