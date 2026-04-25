from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semdiff.datasets.metadata import ImageNetClassIndexMaps
    from semdiff.evaluation.service import EvaluationModelSpec
    from semdiff.wordnet.service import WordNetService

logger = logging.getLogger(__name__)

ModelBuilder = Callable[[], EvaluationModelSpec]
ClipBuilder = Callable[[ImageNetClassIndexMaps, WordNetService | None], EvaluationModelSpec]

_model_registry: dict[str, ModelBuilder | ClipBuilder] = {}


def register(name: str) -> Callable[[ModelBuilder | ClipBuilder], ModelBuilder | ClipBuilder]:
    def decorator(builder: ModelBuilder | ClipBuilder) -> ModelBuilder | ClipBuilder:
        _model_registry[name] = builder
        return builder

    return decorator


def get_registry() -> dict[str, ModelBuilder | ClipBuilder]:
    return _model_registry.copy()