from __future__ import annotations

from torchvision.models import VGG16_Weights, vgg16

from .model import Model


def build_vgg_spec() -> Model:
    weights = VGG16_Weights.IMAGENET1K_V1
    return Model(
        model_name="vgg",
        weights_name="VGG16_Weights.IMAGENET1K_V1",
        categories=weights.meta["categories"],
        transform=weights.transforms(),
        model=vgg16(weights=weights),
    )