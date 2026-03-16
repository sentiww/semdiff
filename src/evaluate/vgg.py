from __future__ import annotations

import logging
from torchvision.models import VGG16_Weights, vgg16

from .common import (
    LOGGER,
)
from .model import EvaluationModelSpec
from .runner import evaluate_model


def evaluate_vgg(
    dataset_name: str, *, logger: logging.Logger = LOGGER
) -> None:
    weights = VGG16_Weights.IMAGENET1K_V1
    evaluate_model(
        dataset_name,
        spec=EvaluationModelSpec(
            model_name="vgg",
            weights_name="VGG16_Weights.IMAGENET1K_V1",
            weights=weights,
            model=vgg16(weights=weights),
        ),
        logger=logger,
    )
