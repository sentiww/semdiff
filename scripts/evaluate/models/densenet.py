from __future__ import annotations

from torchvision.models import DenseNet121_Weights, densenet121

from .model import Model


def build_densenet_spec() -> Model:
    weights = DenseNet121_Weights.IMAGENET1K_V1
    return Model(
        model_name="densenet",
        weights_name="DenseNet121_Weights.IMAGENET1K_V1",
        categories=weights.meta["categories"],
        transform=weights.transforms(),
        model=densenet121(weights=weights),
    )