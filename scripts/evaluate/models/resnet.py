from __future__ import annotations

from torchvision.models import ResNet50_Weights, resnet50

from .model import Model


def build_resnet_spec() -> Model:
    weights = ResNet50_Weights.IMAGENET1K_V2
    return Model(
        model_name="resnet",
        weights_name="ResNet50_Weights.IMAGENET1K_V2",
        categories=weights.meta["categories"],
        transform=weights.transforms(),
        model=resnet50(weights=weights),
    )
