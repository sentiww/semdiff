from __future__ import annotations

from typing import cast

from torchvision.models import DenseNet121_Weights, densenet121

from semdiff.datasets.synset_dataset import ImageTransform
from semdiff.evaluation.models.registry import register
from semdiff.evaluation.service import EvaluationModelSpec


@register("densenet")
def _build_densenet_spec() -> EvaluationModelSpec:
    weights = DenseNet121_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="densenet",
        weights_name="DenseNet121_Weights.IMAGENET1K_V1",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=densenet121(weights=weights),
    )


def _extract_categories(raw_categories: object) -> tuple[str, ...]:
    return tuple(cast(list[str], raw_categories))