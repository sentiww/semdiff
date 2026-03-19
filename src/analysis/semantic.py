from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean, median
from typing import Any
from typing import Callable

from evaluate.paths import OUTPUT_ROOT
from wordnet import path_distance, path_similarity, wup_similarity

LOGGER = logging.getLogger("main.analysis")

MetricFn = Callable[[str, str], int | float | None]


def summarize_values(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def select_extreme_examples(
    examples: list[dict[str, Any]],
    *,
    value_key: str,
    limit: int,
    highest: bool,
) -> list[dict[str, Any]]:
    sorted_examples = list(examples)
    sorted_examples.sort(
        key=_build_example_sort_key(value_key),
        reverse=highest,
    )

    selected_examples: list[dict[str, Any]] = []
    for example in sorted_examples:
        if len(selected_examples) >= limit:
            break
        selected_examples.append(example)
    return selected_examples


def _build_example_sort_key(
    value_key: str,
) -> Callable[[dict[str, Any]], tuple[Any, str, str, str]]:
    def sort_key(item: dict[str, Any]) -> tuple[Any, str, str, str]:
        return (
            item[value_key],
            item["target_synset"],
            item["predicted_synset"],
            item["image"],
        )

    return sort_key


def build_metric_functions(logger: logging.Logger) -> tuple[tuple[str, MetricFn], ...]:
    wordnet_logger = logger.getChild("wordnet")
    return (
        (
            "path_distance",
            _build_metric_function(path_distance, wordnet_logger),
        ),
        (
            "path_similarity",
            _build_metric_function(path_similarity, wordnet_logger),
        ),
        (
            "wup_similarity",
            _build_metric_function(wup_similarity, wordnet_logger),
        ),
    )


def _build_metric_function(
    metric: Callable[..., int | float | None],
    logger: logging.Logger,
) -> MetricFn:
    def run_metric(a: str, b: str) -> int | float | None:
        return metric(a, b, logger=logger)

    return run_metric


def build_semantic_metrics(
    model_name: str,
    dataset_name: str,
    *,
    logger: logging.Logger = LOGGER,
) -> Path:
    output_path = OUTPUT_ROOT / model_name / dataset_name
    predictions_path = output_path / "predictions.jsonl"
    annotated_path = output_path / "semantics.jsonl"
    summary_path = output_path / "semantic-summary.json"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    metrics = build_metric_functions(logger)
    num_records = 0
    metric_values: dict[str, list[int | float]] = {}
    missing_counts: dict[str, int] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    for metric_name, _ in metrics:
        metric_values[metric_name] = []
        missing_counts[metric_name] = 0
        examples[metric_name] = []

    with (
        predictions_path.open(encoding="utf-8") as input_file,
        annotated_path.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            record = json.loads(line)
            num_records += 1
            target_synset = str(record["target_synset"])
            predicted_synset = str(record["predicted_synset"])

            enriched_record: dict[str, Any] = dict(record)
            for metric_name, metric_fn in metrics:
                metric_value = metric_fn(target_synset, predicted_synset)
                enriched_record[metric_name] = metric_value
                if metric_value is None:
                    missing_counts[metric_name] += 1
                    continue
                metric_values[metric_name].append(metric_value)
                examples[metric_name].append(
                    {
                        "image": str(record["image"]),
                        "target_synset": target_synset,
                        "predicted_synset": predicted_synset,
                        metric_name: metric_value,
                    }
                )
            output_file.write(json.dumps(enriched_record, ensure_ascii=True) + "\n")

    summary_metrics: dict[str, dict[str, Any]] = {}
    for metric_name, _ in metrics:
        summary_metrics[metric_name] = {
            "missing": missing_counts[metric_name],
            "summary": summarize_values(metric_values[metric_name]),
            "lowest_examples": select_extreme_examples(
                examples[metric_name],
                value_key=metric_name,
                limit=5,
                highest=False,
            ),
            "highest_examples": select_extreme_examples(
                examples[metric_name],
                value_key=metric_name,
                limit=5,
                highest=True,
            ),
        }

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_records": num_records,
        "metrics": summary_metrics,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    logger.info(
        "Wrote path distance analysis to %s and %s from %s records",
        annotated_path,
        summary_path,
        num_records,
    )
    return annotated_path
