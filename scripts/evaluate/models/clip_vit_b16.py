from __future__ import annotations

import torch
import open_clip
from torchvision.models import ResNet50_Weights

from .model import Model


class _ClipClassifier(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module, text_features: torch.Tensor) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer("text_features", text_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.clip_model.logit_scale.exp() * (image_features @ self.text_features.t())


def build_clip_spec() -> Model:
    categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai",
    )
    clip_model = clip_model.eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    prompts = [f"a photo of a {category}" for category in categories]

    with torch.inference_mode():
        text_features = clip_model.encode_text(tokenizer(prompts))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return Model(
        model_name="clip-vit-b-16",
        weights_name='open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")',
        categories=categories,
        transform=preprocess,
        model=_ClipClassifier(clip_model, text_features),
    )
