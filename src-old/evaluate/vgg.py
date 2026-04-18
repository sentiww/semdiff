from __future__ import annotations

from torchvision.models import VGG16_Weights, vgg16

from .model import EvaluationModelSpec


def build_vgg_spec() -> EvaluationModelSpec:
    weights = VGG16_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="vgg",
        weights_name="VGG16_Weights.IMAGENET1K_V1",
        categories=weights.meta["categories"],
        transform=weights.transforms(),
        model=vgg16(weights=weights),
    )
