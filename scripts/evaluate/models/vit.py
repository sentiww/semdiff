from __future__ import annotations

from torchvision.models import ViT_B_16_Weights, vit_b_16

from .model import Model


def build_vit_spec() -> Model:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    return Model(
        model_name="vit-b-16",
        weights_name="ViT_B_16_Weights.IMAGENET1K_V1",
        categories=weights.meta["categories"],
        transform=weights.transforms(),
        model=vit_b_16(weights=weights),
    )