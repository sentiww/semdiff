from typing import Protocol
from pathlib import Path


class MetaLabelLoader(Protocol):
    def load_validation_wnids(
        self,
        meta_path: Path,
        ground_truth_path: Path,
    ) -> list[str]: ...
