from __future__ import annotations

from typing import cast

from torchvision.models import ViT_B_16_Weights, vit_b_16

from semdiff.datasets.synset_dataset import ImageTransform
from semdiff.evaluation.models.registry import register
from semdiff.evaluation.service import EvaluationModelSpec


@register("vit-b-16")
def _build_vit_spec() -> EvaluationModelSpec:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="vit-b-16",
        weights_name="ViT_B_16_Weights.IMAGENET1K_V1",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=vit_b_16(weights=weights),
    )


def _extract_categories(raw_categories: object) -> tuple[str, ...]:
    return tuple(cast(list[str], raw_categories))