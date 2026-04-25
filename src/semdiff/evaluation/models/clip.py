from __future__ import annotations

import logging
from typing import cast

import torch

from semdiff.datasets.metadata import ImageNetClassIndexMaps
from semdiff.datasets.synset_dataset import ImageTransform
from semdiff.evaluation.models.registry import register
from semdiff.evaluation.protocols import ClipModel, ClipTokenizer
from semdiff.evaluation.service import EvaluationModelSpec
from semdiff.wordnet.service import WordNetService

logger = logging.getLogger(__name__)


@register("clip-vit-b-16")
def _build_clip_spec(
    class_index_maps: ImageNetClassIndexMaps,
    wordnet: WordNetService | None,
) -> EvaluationModelSpec:
    import open_clip
    from semdiff.evaluation.models.clip_models import ZeroShotClipClassifier, normalize_features

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