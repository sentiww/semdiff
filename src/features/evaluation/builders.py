from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import torch
from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    ViT_B_16_Weights,
    densenet121,
    resnet50,
    vgg16,
    vit_b_16,
)

from features.datasets.synset_dataset import ImageTransform
from features.evaluation.service import EvaluationModelSpec

from features.datasets.metadata import ImageNetClassIndexMaps
from features.wordnet.service import WordNetService

logger = logging.getLogger(__name__)

ModelBuilder = Callable[[], EvaluationModelSpec]
ClipBuilder = Callable[[ImageNetClassIndexMaps, WordNetService | None], EvaluationModelSpec]

_model_registry: dict[str, ModelBuilder | ClipBuilder] = {}


def register(name: str) -> Callable[[ModelBuilder | ClipBuilder], ModelBuilder | ClipBuilder]:
    def decorator(builder: ModelBuilder | ClipBuilder) -> ModelBuilder | ClipBuilder:
        _model_registry[name] = builder
        return builder

    return decorator


def get_registry() -> dict[str, ModelBuilder | ClipBuilder]:
    return _model_registry.copy()


@register("resnet")
def _build_resnet_spec() -> EvaluationModelSpec:
    weights = ResNet50_Weights.IMAGENET1K_V2
    return EvaluationModelSpec(
        model_name="resnet",
        weights_name="ResNet50_Weights.IMAGENET1K_V2",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=resnet50(weights=weights),
    )


@register("densenet")
def _build_densenet_spec() -> EvaluationModelSpec:
    weights = DenseNet121_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="densenet",
        weights_name="DenseNet121_Weights.IMAGENET1K_V1",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=densenet121(weights=weights),
    )


@register("vgg")
def _build_vgg_spec() -> EvaluationModelSpec:
    weights = VGG16_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="vgg",
        weights_name="VGG16_Weights.IMAGENET1K_V1",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=vgg16(weights=weights),
    )


@register("vit-b-16")
def _build_vit_spec() -> EvaluationModelSpec:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="vit-b-16",
        weights_name="ViT_B_16_Weights.IMAGENET1K_V1",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=vit_b_16(weights=weights),
    )


@register("clip-vit-b-16")
def _build_clip_spec(
    class_index_maps: ImageNetClassIndexMaps,
    wordnet: WordNetService | None,
) -> EvaluationModelSpec:
    import open_clip
    from features.evaluation.models import ZeroShotClipClassifier
    from features.evaluation.protocols import ClipModel, ClipTokenizer

    ordered_indices = tuple(sorted(class_index_maps.index_to_wnid))
    ordered_wnids = tuple(
        class_index_maps.index_to_wnid[index] for index in ordered_indices
    )
    prompts = [
        _build_synset_prompt(
            wnid=wnid,
            fallback_label=class_index_maps.index_to_label[index],
            wordnet=wordnet,
        )
        for index, wnid in zip(ordered_indices, ordered_wnids, strict=True)
    ]

    logger.info("Loading CLIP model and preprocessing pipeline")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai",
    )
    tokenizer = cast(ClipTokenizer, open_clip.get_tokenizer("ViT-B-16"))
    typed_clip_model = cast(ClipModel, clip_model)

    from features.evaluation.models import normalize_features

    logger.info("Encoding %s CLIP text prompts from ImageNet synsets", len(prompts))
    with torch.inference_mode():
        text_tokens = tokenizer(prompts)
        text_features = typed_clip_model.encode_text(text_tokens)
        text_features = normalize_features(text_features)

    return EvaluationModelSpec(
        model_name="clip-vit-b-16",
        weights_name='open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")',
        categories=ordered_wnids,
        transform=cast(ImageTransform, preprocess),
        model=ZeroShotClipClassifier(
            model=typed_clip_model,
            text_features=text_features,
        ),
    )


def _extract_categories(raw_categories: object) -> tuple[str, ...]:
    return tuple(cast(list[str], raw_categories))


def _build_synset_prompt(
    *,
    wnid: str,
    fallback_label: str,
    wordnet: WordNetService | None,
) -> str:
    labels: list[str] = []
    if wordnet is not None:
        labels = wordnet.lookup_labels(wnid)

    prompt_terms: list[str] = []
    seen_terms: set[str] = set()
    for raw_label in [fallback_label, *labels]:
        label = raw_label.strip().replace("_", " ")
        normalized_label = label.lower()
        if not label or normalized_label in seen_terms:
            continue
        seen_terms.add(normalized_label)
        prompt_terms.append(label)
        if len(prompt_terms) == 3:
            break

    subject = ", ".join(prompt_terms) if prompt_terms else wnid
    definition = wordnet.lookup_definition(wnid) if wordnet is not None else None
    if definition:
        return f"a photo of {subject}; {definition}"
    return f"a photo of {subject}"
