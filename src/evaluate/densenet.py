from __future__ import annotations

import logging
from torchvision.models import DenseNet121_Weights, densenet121

from .common import (
    LOGGER,
)
from .model import EvaluationModelSpec
from .runner import evaluate_model


def evaluate_densenet(
    dataset_name: str, *, logger: logging.Logger = LOGGER
) -> None:
    weights = DenseNet121_Weights.IMAGENET1K_V1
    evaluate_model(
        dataset_name,
        spec=EvaluationModelSpec(
            model_name="densenet",
            weights_name="DenseNet121_Weights.IMAGENET1K_V1",
            weights=weights,
            model=densenet121(weights=weights),
        ),
        logger=logger,
    )
