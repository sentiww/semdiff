from __future__ import annotations

from typing import cast

from torchvision.models import VGG16_Weights, vgg16

from semdiff.datasets.synset_dataset import ImageTransform
from semdiff.evaluation.models.registry import register
from semdiff.evaluation.service import EvaluationModelSpec


@register("vgg")
def _build_vgg_spec() -> EvaluationModelSpec:
    weights = VGG16_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="vgg",
        weights_name="VGG16_Weights.IMAGENET1K_V1",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=vgg16(weights=weights),
    )


def _extract_categories(raw_categories: object) -> tuple[str, ...]:
    return tuple(cast(list[str], raw_categories))