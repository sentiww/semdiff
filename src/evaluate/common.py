from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Sequence

import datasets
from datasets.imagenet_mappings import load_index_to_wnid
from datasets.imagenet_mappings import load_wnid_to_index

LOGGER = logging.getLogger("main.evaluate")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
SYNSET_DIR_PATTERN = re.compile(r"n\d+$")
PROGRESS_LOG_EVERY_BATCHES = 10


@dataclass(frozen=True)
class EvaluationOutputPaths:
    output_path: Path
    predictions_path: Path
    summary_path: Path


def resolve_output_paths(model_name: str, dataset_name: str) -> EvaluationOutputPaths:
    output_path = OUTPUT_ROOT / model_name / dataset_name
    return EvaluationOutputPaths(
        output_path=output_path,
        predictions_path=output_path / "predictions.jsonl",
        summary_path=output_path / "summary.json",
    )


def write_prediction_record(predictions_file: Any, record: dict[str, Any]) -> None:
    predictions_file.write(json.dumps(record) + "\n")


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_prediction_record(
    *,
    dataset_path: Path,
    image_path: str,
    categories: Sequence[str],
    index_to_synset: dict[int, str],
    predicted_index: int,
    expected_index: int,
    confidence: float,
    top5_prediction_indices: Sequence[int],
    top5_prediction_scores: Sequence[float],
) -> dict[str, Any]:
    return {
        "image": str(Path(image_path).relative_to(dataset_path)),
        "target_synset": Path(image_path).parent.name,
        "target_index": expected_index,
        "predicted_synset": index_to_synset[predicted_index],
        "predicted_index": predicted_index,
        "predicted_label": categories[predicted_index],
        "confidence": confidence,
        "correct_top1": predicted_index == expected_index,
        "correct_top5": expected_index in top5_prediction_indices,
        "top5_predictions": [
            {
                "index": candidate_index,
                "synset": index_to_synset[candidate_index],
                "label": categories[candidate_index],
                "confidence": candidate_score,
            }
            for candidate_index, candidate_score in zip(
                top5_prediction_indices,
                top5_prediction_scores,
            )
        ],
    }


def build_summary(
    *,
    model_name: str,
    dataset_name: str,
    dataset_path: Path,
    num_samples: int,
    top1_accuracy: float,
    top5_accuracy: float,
    device: str,
    weights: str,
    predictions_path: Path,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "num_samples": num_samples,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "device": device,
        "weights": weights,
        "predictions_path": str(predictions_path),
    }


def validate_imagefolder_dataset(dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    synset_dirs = [
        path
        for path in dataset_path.iterdir()
        if path.is_dir() and SYNSET_DIR_PATTERN.fullmatch(path.name)
    ]
    if not synset_dirs:
        raise RuntimeError(
            f"Dataset {dataset_path} is not initialized as synset folders. "
            "Run datasets init first."
        )


def load_imagenet_synset_index_map(
    categories: list[str],
) -> tuple[dict[str, int], dict[int, str]]:
    class_index_path = datasets.dataset_root("imagenet-1k") / "torchvision_class_index.json"
    if not class_index_path.exists():
        raise FileNotFoundError(
            f"Missing ImageNet class index at {class_index_path}. "
            "The evaluator needs datasets/imagenet-1k/torchvision_class_index.json."
        )

    synset_to_index = load_wnid_to_index(class_index_path)
    index_to_synset = load_index_to_wnid(class_index_path)
    if len(index_to_synset) != len(categories):
        raise RuntimeError(
            "ImageNet class index does not match the model categories: "
            f"{len(index_to_synset)} vs {len(categories)}"
        )

    return synset_to_index, index_to_synset


class SynsetImageFolder:
    def __init__(
        self,
        root: Path,
        *,
        transform: Callable | None,
        loader: Callable,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.loader = loader
        self.classes = sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and SYNSET_DIR_PATTERN.fullmatch(path.name)
        )
        self.class_to_idx = {
            synset: class_index for class_index, synset in enumerate(self.classes)
        }
        self.samples = self._build_samples()
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image_path, target = self.samples[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def _build_samples(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for synset in self.classes:
            synset_dir = self.root / synset
            class_index = self.class_to_idx[synset]
            for image_path in sorted(synset_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".ppm",
                    ".bmp",
                    ".pgm",
                    ".tif",
                    ".tiff",
                    ".webp",
                }:
                    samples.append((str(image_path), class_index))
        if not samples:
            raise RuntimeError(f"No image files found under {self.root}")
        return samples
