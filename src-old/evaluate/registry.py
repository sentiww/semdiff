from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from .model import EvaluationModelSpec


@dataclass(frozen=True)
class EvaluationCommandSpec:
    command_name: str
    help_text: str
    builder_module: str
    builder_name: str

    def build_spec(self) -> EvaluationModelSpec:
        module = import_module(self.builder_module, package=__package__)
        builder = getattr(module, self.builder_name)
        return builder()


EVALUATION_COMMANDS: tuple[EvaluationCommandSpec, ...] = (
    EvaluationCommandSpec(
        command_name="resnet",
        help_text="Evaluate a ResNet-50 model",
        builder_module=".resnet",
        builder_name="build_resnet_spec",
    ),
    EvaluationCommandSpec(
        command_name="densenet",
        help_text="Evaluate a DenseNet-121 model",
        builder_module=".densenet",
        builder_name="build_densenet_spec",
    ),
    EvaluationCommandSpec(
        command_name="vgg",
        help_text="Evaluate a VGG-16 model",
        builder_module=".vgg",
        builder_name="build_vgg_spec",
    ),
    EvaluationCommandSpec(
        command_name="vit-b-16",
        help_text="Evaluate a ViT-B/16 model",
        builder_module=".vit",
        builder_name="build_vit_spec",
    ),
    EvaluationCommandSpec(
        command_name="clip-vit-b-16",
        help_text="Evaluate a CLIP ViT-B/16 model",
        builder_module=".clip_vit_b16",
        builder_name="build_clip_spec",
    ),
)
