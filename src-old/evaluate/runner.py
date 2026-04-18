from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from typing import Sequence

import torch
from datasets.common import dataset_root
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader

from .image_dataset import SynsetImageFolder
from .image_dataset import validate_imagefolder_dataset
from .index_map import load_imagenet_synset_index_map
from .model import EvaluationModelSpec
from .paths import LOGGER
from .paths import PROGRESS_LOG_EVERY_BATCHES
from .paths import resolve_output_paths
from .records import write_prediction_record
from .records import write_summary


class EvaluationTotals:
    def __init__(self) -> None:
        self.total = 0
        self.comparable_total = 0
        self.sample_offset = 0


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_class_index_to_model_index(
    image_dataset: SynsetImageFolder,
    synset_to_index: dict[str, int],
) -> dict[int, int]:
    class_index_to_model_index: dict[int, int] = {}
    for synset, class_index in image_dataset.class_to_idx.items():
        model_index = synset_to_index.get(synset, -1)
        class_index_to_model_index[class_index] = model_index
    return class_index_to_model_index


def _build_dataloader(image_dataset: SynsetImageFolder) -> DataLoader:
    return DataLoader(
        image_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )


def _map_targets(
    targets: torch.Tensor,
    class_index_to_model_index: dict[int, int],
) -> list[int]:
    mapped_targets: list[int] = []
    for target in targets.tolist():
        mapped_target = class_index_to_model_index.get(int(target), -1)
        mapped_targets.append(mapped_target)
    return mapped_targets


def _update_totals(
    totals: EvaluationTotals,
    *,
    batch_size: int,
    mapped_targets: Sequence[int],
) -> None:
    totals.total += batch_size

    comparable_count = 0
    for expected in mapped_targets:
        if expected >= 0:
            comparable_count += 1
    totals.comparable_total += comparable_count


def _write_batch_predictions(
    predictions_file: Any,
    *,
    dataset_path: Path,
    image_dataset: SynsetImageFolder,
    categories: Sequence[str],
    index_to_synset: dict[int, str],
    top1_indices: torch.Tensor,
    top1_scores: torch.Tensor,
    sample_offset: int,
) -> None:
    batch_size = len(top1_indices)
    for item_index in range(batch_size):
        sample_id = sample_offset + item_index
        image_path, _ = image_dataset.samples[sample_id]
        predicted_index = int(top1_indices[item_index])
        write_prediction_record(
            predictions_file,
            sample_id=sample_id,
            dataset_path=dataset_path,
            image_path=image_path,
            categories=categories,
            index_to_synset=index_to_synset,
            predicted_index=predicted_index,
            confidence=float(top1_scores[item_index]),
        )


def _log_progress(
    dataset_logger: logging.Logger,
    *,
    sample_offset: int,
    dataset_size: int,
    batch_number: int,
    num_batches: int,
    totals: EvaluationTotals,
    started_at: float,
) -> None:
    dataset_logger.info(
        "Progress: %s/%s samples (%s/%s batches), comparable=%s, elapsed=%.1fs",
        sample_offset,
        dataset_size,
        batch_number,
        num_batches,
        totals.comparable_total,
        time.perf_counter() - started_at,
    )


def evaluate_model(
    dataset_name: str,
    *,
    spec: EvaluationModelSpec,
    logger: logging.Logger = LOGGER,
) -> None:
    dataset_logger = logger.getChild(f"{spec.model_name}.{dataset_name}")
    dataset_path = dataset_root(dataset_name)
    output_paths = resolve_output_paths(spec.model_name, dataset_name)

    validate_imagefolder_dataset(dataset_path)
    output_paths.output_path.mkdir(parents=True, exist_ok=True)

    categories = spec.categories
    synset_to_index, index_to_synset = load_imagenet_synset_index_map(list(categories))
    image_dataset = SynsetImageFolder(
        dataset_path,
        transform=spec.transform,
        loader=default_loader,
    )
    class_index_to_model_index = _build_class_index_to_model_index(
        image_dataset,
        synset_to_index,
    )
    dataloader = _build_dataloader(image_dataset)

    model = spec.model
    model.eval()
    device = _resolve_device()
    model.to(device)

    totals = EvaluationTotals()
    started_at = time.perf_counter()

    dataset_logger.info("Evaluating %s samples on %s", len(image_dataset), device)
    dataset_logger.info(
        "Writing predictions to %s and summary to %s",
        output_paths.predictions_path,
        output_paths.summary_path,
    )

    with output_paths.predictions_path.open("w", encoding="utf-8") as predictions_file:
        with torch.inference_mode():
            for batch_number, (images, targets) in enumerate(dataloader, start=1):
                images = images.to(device)
                probabilities = torch.nn.functional.softmax(model(images), dim=1)
                top1_scores, top1_indices = probabilities.max(dim=1)

                mapped_targets = _map_targets(targets, class_index_to_model_index)
                batch_size = len(targets)
                _update_totals(
                    totals,
                    batch_size=batch_size,
                    mapped_targets=mapped_targets,
                )
                _write_batch_predictions(
                    predictions_file,
                    dataset_path=dataset_path,
                    image_dataset=image_dataset,
                    categories=categories,
                    index_to_synset=index_to_synset,
                    top1_indices=top1_indices,
                    top1_scores=top1_scores,
                    sample_offset=totals.sample_offset,
                )

                totals.sample_offset += batch_size
                should_log_progress = (
                    batch_number % PROGRESS_LOG_EVERY_BATCHES == 0
                    or totals.sample_offset == len(image_dataset)
                )
                if should_log_progress:
                    _log_progress(
                        dataset_logger,
                        sample_offset=totals.sample_offset,
                        dataset_size=len(image_dataset),
                        batch_number=batch_number,
                        num_batches=len(dataloader),
                        totals=totals,
                        started_at=started_at,
                    )

    write_summary(
        output_paths.summary_path,
        model_name=spec.model_name,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        num_samples=totals.total,
        device=str(device),
        weights=spec.weights_name,
        predictions_path=output_paths.predictions_path,
    )
    dataset_logger.info(
        "Wrote %s and %s in %.1fs",
        output_paths.predictions_path,
        output_paths.summary_path,
        time.perf_counter() - started_at,
    )
