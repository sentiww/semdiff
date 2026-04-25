from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from features.datasets.synset_dataset import SynsetImageFolder
from features.evaluation.service import EvaluationModelSpec

if TYPE_CHECKING:
    from features.datasets.metadata import ImageNetClassIndexMaps
    from features.wordnet.service import WordNetService

logger = logging.getLogger(__name__)


class EvaluationRuntime:
    def __init__(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        num_workers: int = 0,
        wordnet: WordNetService | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._wordnet = wordnet
        self._builders = self._load_builders()

    def _load_builders(self) -> dict[str, ModelBuilder | ClipBuilder]:
        from features.evaluation import builders

        return builders.get_registry()

    def create_model_spec(
        self,
        name: str,
        *,
        class_index_maps: ImageNetClassIndexMaps | None = None,
    ) -> EvaluationModelSpec:
        logger.info("Loading model specification for %s", name)
        builder = self._builders.get(name)
        if builder is None:
            available = ", ".join(sorted(self._builders))
            raise ValueError(
                f"Unknown model: {name!r}. Available models: {available}"
            )

        if name == "clip-vit-b-16":
            if class_index_maps is None:
                raise ValueError("CLIP evaluation requires ImageNet class-index metadata")
            spec = builder(class_index_maps, self._wordnet)
        else:
            spec = builder()

        logger.info(
            "Loaded model specification for %s with weights=%s", name, spec.weights_name
        )
        return spec

    def create_dataloader(
        self,
        image_dataset: SynsetImageFolder,
    ) -> DataLoader[tuple[object, int]]:
        logger.info(
            "Creating dataloader for %s samples (batch_size=%s, num_workers=%s, shuffle=%s)",
            len(image_dataset),
            self._batch_size,
            self._num_workers,
            self._shuffle,
        )
        return DataLoader(
            image_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
        )

    def resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
