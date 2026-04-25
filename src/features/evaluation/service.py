from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

logger = logging.getLogger(__name__)

from features.files import (
    EvaluationSummary,
    EntitySink,
    FileStore,
    OutputPathResolver,
    PredictionRecord,
)
from features.datasets.metadata import ImageNetMetadataService
from features.datasets.synset_dataset import (
    ImageTransform,
    SynsetImageFolder,
    SynsetImageFolderFactory,
)

if TYPE_CHECKING:
    from features.evaluation.runtime import EvaluationRuntime


@dataclass(frozen=True)
class EvaluationModelSpec:
    model_name: str
    weights_name: str
    categories: tuple[str, ...]
    transform: ImageTransform | None
    model: torch.nn.Module


@dataclass(frozen=True)
class EvaluationReport:
    model_name: str
    dataset_path: Path
    predictions_path: Path
    summary_path: Path
    num_samples: int
    device: str


@dataclass
class EvaluationTotals:
    total: int = 0
    comparable_total: int = 0
    sample_offset: int = 0


class ModelEvaluationService:
    def __init__(
        self,
        image_dataset_factory: SynsetImageFolderFactory,
        evaluation_runtime: EvaluationRuntime,
        file_store: FileStore,
        imagenet_metadata_service: ImageNetMetadataService,
        progress_log_every_batches: int = 10,
        output_path_resolver: OutputPathResolver | None = None,
    ) -> None:
        self._image_dataset_factory = image_dataset_factory
        self._evaluation_runtime = evaluation_runtime
        self._file_store = file_store
        self._imagenet_metadata_service = imagenet_metadata_service
        self._progress_log_every_batches = progress_log_every_batches
        self._output_path_resolver = output_path_resolver or OutputPathResolver(Path())

    def evaluate(
        self,
        *,
        dataset_path: Path,
        class_map_path: Path,
        index_to_wnid_path: Path,
        output_directory: Path,
        model_name: str,
    ) -> EvaluationReport:
        logger.info(
            "Starting evaluation: model=%s dataset=%s output=%s class_map=%s index_to_wnid=%s",
            model_name,
            dataset_path,
            output_directory,
            class_map_path,
            index_to_wnid_path,
        )
        class_index_maps = self._imagenet_metadata_service.load_class_index_maps(
            class_map_path,
            index_to_wnid_path,
        )
        spec = self._evaluation_runtime.create_model_spec(
            model_name,
            class_index_maps=class_index_maps,
        )
        output_path_resolver = OutputPathResolver(output_directory)
        predictions_path = output_path_resolver.resolve(
            PredictionRecord, "predictions.jsonl"
        ).path
        summary_path = output_path_resolver.resolve(
            EvaluationSummary, "summary.json"
        ).path
        if len(class_index_maps.index_to_wnid) != len(spec.categories):
            raise RuntimeError(
                "ImageNet class index does not match the model categories: "
                f"{len(class_index_maps.index_to_wnid)} vs {len(spec.categories)}"
            )

        image_dataset = self._image_dataset_factory.create(
            dataset_path,
            transform=spec.transform,
        )
        class_index_to_model_index = self._build_class_index_to_model_index(
            image_dataset,
            class_index_maps.wnid_to_index,
        )
        dataloader = self._evaluation_runtime.create_dataloader(image_dataset)
        unmapped_classes = sum(
            1 for model_index in class_index_to_model_index.values() if model_index < 0
        )
        logger.info(
            "Prepared evaluation dataset with %s samples across %s classes (%s unmapped)",
            len(image_dataset),
            len(image_dataset.classes),
            unmapped_classes,
        )

        model = spec.model
        model.eval()
        device = self._evaluation_runtime.resolve_device()
        model.to(device)

        totals = EvaluationTotals()
        started_at = time.perf_counter()

        logger.info("Evaluating %s samples on %s", len(image_dataset), device)
        logger.info(
            "Writing predictions to %s and summary to %s",
            predictions_path,
            summary_path,
        )

        with self._file_store.open_sink(
            PredictionRecord,
            predictions_path,
        ) as prediction_sink:
            with torch.inference_mode():
                for batch_number, (images, targets) in enumerate(dataloader, start=1):
                    images = images.to(device, non_blocking=True)
                    probabilities = torch.nn.functional.softmax(model(images), dim=1)
                    top1_scores, top1_indices = probabilities.max(dim=1)

                    mapped_targets = self._map_targets(
                        targets,
                        class_index_to_model_index,
                    )
                    batch_size = len(targets)
                    self._update_totals(
                        totals,
                        batch_size=batch_size,
                        mapped_targets=mapped_targets,
                    )
                    self._write_batch_predictions(
                        prediction_sink,
                        image_dataset=image_dataset,
                        index_to_synset=class_index_maps.index_to_wnid,
                        top1_indices=top1_indices,
                        top1_scores=top1_scores,
                        sample_offset=totals.sample_offset,
                    )

                    totals.sample_offset += batch_size
                    should_log_progress = (
                        batch_number % self._progress_log_every_batches == 0
                        or totals.sample_offset == len(image_dataset)
                    )
                    if should_log_progress:
                        self._log_progress(
                            sample_offset=totals.sample_offset,
                            dataset_size=len(image_dataset),
                            batch_number=batch_number,
                            num_batches=len(dataloader),
                            totals=totals,
                            started_at=started_at,
                        )
        with self._file_store.open_sink(
            EvaluationSummary,
            summary_path,
        ) as summary_sink:
            summary_sink.write(
                EvaluationSummary(
                    model_name=spec.model_name,
                    dataset_path=dataset_path,
                    num_samples=totals.total,
                    device=str(device),
                    weights=spec.weights_name,
                    predictions_path=predictions_path,
                )
            )
        logger.info(
            "Completed evaluation: wrote %s and %s in %.1fs",
            predictions_path,
            summary_path,
            time.perf_counter() - started_at,
        )
        return EvaluationReport(
            model_name=spec.model_name,
            dataset_path=dataset_path,
            predictions_path=predictions_path,
            summary_path=summary_path,
            num_samples=totals.total,
            device=str(device),
        )

    def _build_class_index_to_model_index(
        self,
        image_dataset: SynsetImageFolder,
        synset_to_index: dict[str, int],
    ) -> dict[int, int]:
        class_index_to_model_index: dict[int, int] = {}
        for synset, class_index in image_dataset.class_to_idx.items():
            class_index_to_model_index[class_index] = synset_to_index.get(synset, -1)
        return class_index_to_model_index

    def _map_targets(
        self,
        targets: torch.Tensor,
        class_index_to_model_index: dict[int, int],
    ) -> list[int]:
        mapped_targets: list[int] = []
        for target in targets.tolist():
            mapped_targets.append(class_index_to_model_index.get(int(target), -1))
        return mapped_targets

    def _update_totals(
        self,
        totals: EvaluationTotals,
        *,
        batch_size: int,
        mapped_targets: list[int],
    ) -> None:
        totals.total += batch_size
        totals.comparable_total += sum(
            1 for expected in mapped_targets if expected >= 0
        )

    def _write_batch_predictions(
        self,
        prediction_sink: EntitySink[PredictionRecord],
        *,
        image_dataset: SynsetImageFolder,
        index_to_synset: dict[int, str],
        top1_indices: torch.Tensor,
        top1_scores: torch.Tensor,
        sample_offset: int,
    ) -> None:
        for item_index in range(len(top1_indices)):
            sample_id = sample_offset + item_index
            image_path, target_index = image_dataset.samples[sample_id]
            predicted_index = int(top1_indices[item_index])
            prediction_sink.write(
                PredictionRecord(
                    sample_id=sample_id,
                    image_path=_relative_image_path(image_dataset.root, image_path),
                    target=image_dataset.classes[int(target_index)],
                    predicted=index_to_synset.get(
                        predicted_index,
                        str(predicted_index),
                    ),
                    confidence=float(top1_scores[item_index]),
                )
            )

    def _log_progress(
        self,
        *,
        sample_offset: int,
        dataset_size: int,
        batch_number: int,
        num_batches: int,
        totals: EvaluationTotals,
        started_at: float,
    ) -> None:
        elapsed = time.perf_counter() - started_at
        samples_per_second = sample_offset / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Processed %s/%s samples (%s/%s batches, %.1f samples/s, comparable=%s)",
            sample_offset,
            dataset_size,
            batch_number,
            num_batches,
            samples_per_second,
            totals.comparable_total,
        )


def _relative_image_path(dataset_root: Path, image_path: str) -> str:
    return Path(os.path.relpath(image_path, start=dataset_root)).as_posix()
