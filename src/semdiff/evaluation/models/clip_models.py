from __future__ import annotations

import torch

from semdiff.evaluation.protocols import ClipImageEncoder


class ZeroShotClipClassifier(torch.nn.Module):
    def __init__(
        self,
        *,
        model: ClipImageEncoder,
        text_features: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("text_features", text_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(images)
        image_features = normalize_features(image_features)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True)