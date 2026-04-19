from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch.utils.data import DataLoader
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

from features.datasets.synset_dataset import ImageTransform, SynsetImageFolder
from features.evaluation.service import EvaluationModelSpec

if TYPE_CHECKING:
    from torch import Tensor

    from features.datasets.metadata import ImageNetClassIndexMaps
    from features.wordnet.service import WordNetService

logger = logging.getLogger(__name__)

ModelBuilder = Callable[[], EvaluationModelSpec]


class ClipTokenizer(Protocol):
    def __call__(self, prompts: list[str]) -> Tensor: ...


class ClipImageEncoder(Protocol):
    logit_scale: Tensor

    def encode_image(self, images: Tensor) -> Tensor: ...


class ClipModel(ClipImageEncoder, Protocol):
    def encode_text(self, tokens: Tensor) -> Tensor: ...


class EvaluationRuntime:
    def __init__(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        num_workers: int = 0,
        wordnet: WordNetService | None = None,
        model_builders: Mapping[str, ModelBuilder] | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._wordnet = wordnet
        self._model_builders = dict(model_builders or _default_model_builders())

    def create_model_spec(
        self,
        name: str,
        *,
        class_index_maps: ImageNetClassIndexMaps | None = None,
    ) -> EvaluationModelSpec:
        logger.info("Loading model specification for %s", name)
        if name == "clip-vit-b-16":
            spec = self._build_clip_spec(class_index_maps)
            logger.info(
                "Loaded model specification for %s with weights=%s",
                name,
                spec.weights_name,
            )
            return spec
        try:
            builder = self._model_builders[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._model_builders))
            raise ValueError(
                f"Unknown model: {name}. Available models: {available}"
            ) from exc
        spec = builder()
        logger.info(
            "Loaded model specification for %s with weights=%s", name, spec.weights_name
        )
        return spec

    def create_dataloader(
        self,
        image_dataset: SynsetImageFolder,
    ) -> DataLoader[tuple[object, int]]:
        logger.info(
            "Creating dataloader for %s samples (batch_size=%s, num_workers=%s, shuffle=%s)",
            len(image_dataset),
            self._batch_size,
            self._num_workers,
            self._shuffle,
        )
        return DataLoader(
            image_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
        )

    def resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_clip_spec(
        self,
        class_index_maps: ImageNetClassIndexMaps | None,
    ) -> EvaluationModelSpec:
        import open_clip

        if class_index_maps is None:
            raise ValueError("CLIP evaluation requires ImageNet class-index metadata")

        ordered_indices = tuple(sorted(class_index_maps.index_to_wnid))
        ordered_wnids = tuple(
            class_index_maps.index_to_wnid[index] for index in ordered_indices
        )
        prompts = [
            _build_synset_prompt(
                wnid=wnid,
                fallback_label=class_index_maps.index_to_label[index],
                wordnet=self._wordnet,
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

        return EvaluationModelSpec(
            model_name="clip-vit-b-16",
            weights_name='open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")',
            categories=ordered_wnids,
            transform=cast(ImageTransform, preprocess),
            model=ZeroShotClipClassifier(
                model=typed_clip_model,
                text_features=_normalize_features(text_features),
            ),
        )


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
        image_features = _normalize_features(image_features)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()


def _default_model_builders() -> dict[str, ModelBuilder]:
    return {
        "resnet": _build_resnet_spec,
        "densenet": _build_densenet_spec,
        "vgg": _build_vgg_spec,
        "vit-b-16": _build_vit_spec,
    }


def _build_resnet_spec() -> EvaluationModelSpec:
    weights = ResNet50_Weights.IMAGENET1K_V2
    return EvaluationModelSpec(
        model_name="resnet",
        weights_name="ResNet50_Weights.IMAGENET1K_V2",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=resnet50(weights=weights),
    )


def _build_densenet_spec() -> EvaluationModelSpec:
    weights = DenseNet121_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="densenet",
        weights_name="DenseNet121_Weights.IMAGENET1K_V1",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=densenet121(weights=weights),
    )


def _build_vgg_spec() -> EvaluationModelSpec:
    weights = VGG16_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="vgg",
        weights_name="VGG16_Weights.IMAGENET1K_V1",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=vgg16(weights=weights),
    )


def _build_vit_spec() -> EvaluationModelSpec:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    return EvaluationModelSpec(
        model_name="vit-b-16",
        weights_name="ViT_B_16_Weights.IMAGENET1K_V1",
        categories=_extract_categories(weights.meta["categories"]),
        transform=cast(ImageTransform, weights.transforms()),
        model=vit_b_16(weights=weights),
    )


def _extract_categories(raw_categories: object) -> tuple[str, ...]:
    return tuple(cast(list[str], raw_categories))


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True)


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
