from __future__ import annotations

from typing import cast

from torchvision.models import ResNet50_Weights, resnet50

from semdiff.datasets.synset_dataset import ImageTransform
from semdiff.evaluation.models.registry import register
from semdiff.evaluation.service import EvaluationModelSpec


@register("resnet")
def _build_resnet_spec() -> EvaluationModelSpec:
    weights = ResNet50_Weights.IMAGENET1K_V2
    return EvaluationModelSpec(
        model_name="resnet",
        weights_name="ResNet50_Weights.IMAGENET1K_V2",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=resnet50(weights=weights),
    )


def _extract_categories(raw_categories: object) -> tuple[str, ...]:
    return tuple(cast(list[str], raw_categories))