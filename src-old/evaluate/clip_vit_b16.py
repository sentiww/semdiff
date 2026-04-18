from __future__ import annotations

from typing import Any

import torch
from torchvision.models import ResNet50_Weights

from .model import EvaluationModelSpec


class ZeroShotClipClassifier(torch.nn.Module):
    def __init__(
        self,
        *,
        model: Any,
        text_features: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("text_features", text_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(images)
        image_features = _normalize_features(image_features)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()


def build_clip_spec() -> EvaluationModelSpec:
    open_clip = _import_open_clip()
    categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    clip_model, preprocess = _load_clip_model(open_clip)
    text_features = _build_text_features(open_clip, clip_model, categories)

    return EvaluationModelSpec(
        model_name="clip-vit-b-16",
        weights_name='open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")',
        categories=categories,
        transform=preprocess,
        model=ZeroShotClipClassifier(
            model=clip_model,
            text_features=text_features,
        ),
    )


def _import_open_clip() -> Any:
    import open_clip

    return open_clip


def _load_clip_model(open_clip: Any) -> tuple[Any, Any]:
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai",
    )
    return clip_model, preprocess


def _build_text_features(
    open_clip: Any,
    clip_model: Any,
    categories: list[str],
) -> torch.Tensor:
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    prompts = _build_prompts(categories)

    with torch.inference_mode():
        text_tokens = tokenizer(prompts)
        text_features = clip_model.encode_text(text_tokens)

    return _normalize_features(text_features)


def _build_prompts(categories: list[str]) -> list[str]:
    prompts: list[str] = []
    for category in categories:
        prompts.append(f"a photo of a {category}")
    return prompts


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True)
