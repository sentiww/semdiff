from __future__ import annotations

import logging
import time
import torch
import json
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from pathlib import Path

from .models.registry import get_model
from .dataset import SynsetImageFolder

logger = logging.getLogger("evaluate.run")

PROGRESS_LOG_EVERY_BATCHES = 10


def run(
    dataset: Path,
    output: Path,
    model_name: str,
    index_mapping: Path,
    target_synset_mapping: Path,
    predicted_synset_mapping: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    output_predictions = output / "predictions.jsonl"

    model = get_model(model_name)

    index_to_synset = load_index_to_synset_mapping(index_mapping)
    target_synset_to_category = load_synset_to_category_mapping(target_synset_mapping)
    predicted_synset_to_category = load_synset_to_category_mapping(
        predicted_synset_mapping
    )

    image_dataset = SynsetImageFolder(
        dataset,
        transform=model.transform,
        loader=default_loader,
    )

    dataloader = DataLoader(
        image_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    synset_to_model_index: dict[str, int] = {v: k for k, v in index_to_synset.items()}

    class_index_to_model_index: dict[int, int] = {}
    for synset, class_index in image_dataset.class_to_idx.items():
        model_index = synset_to_model_index.get(synset, -1)
        class_index_to_model_index[class_index] = model_index

    model.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    totals_total = 0
    totals_comparable_total = 0
    totals_sample_offset = 0

    started_at = time.perf_counter()

    logger.info("Evaluating %s samples on %s", len(image_dataset), device)
    logger.info(
        "Writing predictions to %s",
        output_predictions,
    )

    with output_predictions.open("w", encoding="utf-8") as predictions_file:
        with torch.inference_mode():
            for batch_number, (images, targets) in enumerate(dataloader, start=1):
                images = images.to(device)
                probabilities = torch.nn.functional.softmax(model.model(images), dim=1)
                top1_scores, top1_indices = probabilities.max(dim=1)

                mapped_targets: list[int] = []
                for target in targets.tolist():
                    mapped_target = class_index_to_model_index.get(int(target), -1)
                    mapped_targets.append(mapped_target)

                batch_size = len(targets)

                totals_total += batch_size
                comparable_count = 0
                for expected in mapped_targets:
                    if expected >= 0:
                        comparable_count += 1
                totals_comparable_total += comparable_count

                batch_size = len(top1_indices)
                for item_index in range(batch_size):
                    sample_id = totals_sample_offset + item_index
                    image_path, _ = image_dataset.samples[sample_id]
                    predicted_index = int(top1_indices[item_index])
                    target_synset = Path(image_path).parent.name
                    predicted_synset = index_to_synset[predicted_index]
                    record = {
                        "image": str(Path(image_path).relative_to(dataset)),
                        "target": target_synset,
                        "target_class": target_synset_to_category[target_synset],
                        "predicted": predicted_synset,
                        "predicted_class": predicted_synset_to_category[
                            predicted_synset
                        ],
                        "confidence": float(top1_scores[item_index]),
                    }
                    predictions_file.write(json.dumps(record) + "\n")

                totals_sample_offset += batch_size

                should_log_progress = (
                    batch_number % PROGRESS_LOG_EVERY_BATCHES == 0
                    or totals_sample_offset == len(image_dataset)
                )
                if should_log_progress:
                    logger.info(
                        "Progress: %s/%s samples (%s/%s batches), comparable=%s, elapsed=%.1fs",
                        totals_sample_offset,
                        len(image_dataset),
                        batch_number,
                        len(dataloader),
                        totals_comparable_total,
                        time.perf_counter() - started_at,
                    )

    logger.info(
        "Wrote %s in %.1fs",
        output_predictions,
        time.perf_counter() - started_at,
    )


def load_index_to_synset_mapping(path: Path) -> dict[int, str]:
    with path.open() as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items() if k.isdigit()}


def load_synset_to_category_mapping(path: Path) -> dict[str, str]:
    with path.open() as f:
        data = json.load(f)
    return dict(data)
