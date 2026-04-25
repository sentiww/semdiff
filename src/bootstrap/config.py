from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    imagenet_class_map: Path
    torchvision_index_to_wnid: Path
    evaluation_batch_size: int
    evaluation_num_workers: int
    evaluation_progress_log_every_batches: int

    @classmethod
    def default(cls, project_root: Path) -> AppConfig:
        return cls(
            imagenet_class_map=(project_root / "mappings" / "imagenet-1k" / "class_map.json"),
            torchvision_index_to_wnid=(project_root / "mappings" / "torchvision_index_to_wnid.json"),
            evaluation_batch_size=32,
            evaluation_num_workers=0,
            evaluation_progress_log_every_batches=10,
        )
