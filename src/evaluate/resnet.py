from __future__ import annotations

import logging
import time
from pathlib import Path

import datasets
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.models import ResNet50_Weights, resnet50

from .common import (
    LOGGER,
    PROGRESS_LOG_EVERY_BATCHES,
    SynsetImageFolder,
    build_prediction_record,
    build_summary,
    load_imagenet_synset_index_map,
    resolve_output_paths,
    validate_imagefolder_dataset,
    write_prediction_record,
    write_summary,
)


def evaluate_resnet(
    dataset_name: str, *, logger: logging.Logger = LOGGER
) -> None:
    dataset_logger = logger.getChild(f"resnet.{dataset_name}")
    dataset_path = datasets.dataset_root(dataset_name)
    output_paths = resolve_output_paths("resnet", dataset_name)

    validate_imagefolder_dataset(dataset_path)

    output_paths.output_path.mkdir(parents=True, exist_ok=True)

    weights = ResNet50_Weights.IMAGENET1K_V2
    categories = weights.meta["categories"]
    synset_to_index, index_to_synset = load_imagenet_synset_index_map(categories)
    model = resnet50(weights=weights)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_dataset = SynsetImageFolder(
        dataset_path,
        transform=weights.transforms(),
        loader=default_loader,
    )
    class_index_to_model_index = {
        class_index: synset_to_index[synset]
        for synset, class_index in image_dataset.class_to_idx.items()
    }
    dataloader = DataLoader(
        image_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    total = 0
    top1_correct = 0
    top5_correct = 0
    sample_offset = 0
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
                batch_size = len(targets)
                images = images.to(device)
                logits = model(images)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                top1_scores, top1_indices = probabilities.max(dim=1)
                top5_scores, top5_indices = torch.topk(probabilities, k=5, dim=1)

                mapped_targets = [
                    class_index_to_model_index[int(target)]
                    for target in targets.tolist()
                ]

                total += batch_size
                top1_correct += sum(
                    int(prediction) == expected
                    for prediction, expected in zip(top1_indices.tolist(), mapped_targets)
                )
                top5_correct += sum(
                    expected in predictions
                    for expected, predictions in zip(
                        mapped_targets,
                        top5_indices.tolist(),
                    )
                )

                for item_index in range(batch_size):
                    image_path, _ = image_dataset.samples[sample_offset + item_index]
                    predicted_index = int(top1_indices[item_index])
                    expected_index = mapped_targets[item_index]
                    top5_prediction_indices = [
                        int(index) for index in top5_indices[item_index].tolist()
                    ]
                    top5_prediction_scores = [
                        float(score) for score in top5_scores[item_index].tolist()
                    ]

                    record = build_prediction_record(
                        dataset_path=dataset_path,
                        image_path=image_path,
                        categories=categories,
                        index_to_synset=index_to_synset,
                        predicted_index=predicted_index,
                        expected_index=expected_index,
                        confidence=float(top1_scores[item_index]),
                        top5_prediction_indices=top5_prediction_indices,
                        top5_prediction_scores=top5_prediction_scores,
                    )
                    write_prediction_record(predictions_file, record)

                sample_offset += batch_size

                should_log_progress = (
                    batch_number % PROGRESS_LOG_EVERY_BATCHES == 0
                    or sample_offset == len(image_dataset)
                )
                if should_log_progress:
                    elapsed = time.perf_counter() - started_at
                    dataset_logger.info(
                        "Progress: %s/%s samples (%s/%s batches), top1=%.4f, top5=%.4f, elapsed=%.1fs",
                        sample_offset,
                        len(image_dataset),
                        batch_number,
                        len(dataloader),
                        top1_correct / total if total else 0.0,
                        top5_correct / total if total else 0.0,
                        elapsed,
                    )

    summary = build_summary(
        model_name="resnet",
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        num_samples=total,
        top1_accuracy=top1_correct / total if total else 0.0,
        top5_accuracy=top5_correct / total if total else 0.0,
        device=str(device),
        weights="ResNet50_Weights.IMAGENET1K_V2",
        predictions_path=output_paths.predictions_path,
    )
    write_summary(output_paths.summary_path, summary)
    dataset_logger.info(
        "Wrote %s and %s in %.1fs",
        output_paths.predictions_path,
        output_paths.summary_path,
        time.perf_counter() - started_at,
    )
