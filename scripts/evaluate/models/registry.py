from __future__ import annotations

from .model import Model
from .resnet import build_resnet_spec
from .densenet import build_densenet_spec
from .vgg import build_vgg_spec
from .vit import build_vit_spec
from .clip_vit_b16 import build_clip_spec
from torchvision.models import ResNet50_Weights


def get_model(model_name: str) -> Model:
    match model_name:
        case "resnet":
            return build_resnet_spec()
        case "densenet":
            return build_densenet_spec()
        case "vgg":
            return build_vgg_spec()
        case "vit-b-16":
            return build_vit_spec()
        case "clip-vit-b-16":
            return build_clip_spec()
        case _:
            raise Exception()
