from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ModelEvaluationService:
    def __init__(self):
        pass

    def evaluate_model(self, dataset_name: str, *, spec: EvaluationModelSpec) -> None:
        dataset_path = dataset_root(dataset_name)
        output_paths = resolve_output_paths(spec.model_name, dataset_name)

        validate_imagefolder_dataset(dataset_path)
        output_paths.output_path.mkdir(parents=True, exist_ok=True)

        categories = spec.categories
        synset_to_index, index_to_synset = load_imagenet_synset_index_map(
            list(categories)
        )
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

        with output_paths.predictions_path.open(
            "w", encoding="utf-8"
        ) as predictions_file:
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
