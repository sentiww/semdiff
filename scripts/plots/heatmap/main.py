from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import scripts.common.jsonl as jsonl

import matplotlib.pyplot as plt

logger = logging.getLogger(__file__)


def filter_record(
    record: dict[str, object], filters: list[str] | None, filter_field: str
) -> bool:
    if filters is None:
        return True
    return str(record[filter_field]) in filters


def bin_label(index: int, bin_count: int) -> str:
    start = index / bin_count
    end = (index + 1) / bin_count
    return f"[{start:.2f}, {end:.2f})"


def load_points(
    predictions_path: Path,
    metrics_path: Path,
    filters: list[str] | None,
    filter_field: str,
) -> tuple[list[float], list[float]]:
    predictions = jsonl.load(predictions_path)
    metrics = jsonl.load(metrics_path)
    if len(predictions) != len(metrics):
        raise ValueError(
            "Predictions and metric files must have the same number of rows"
        )

    confidences: list[float] = []
    similarities: list[float] = []
    skipped_none = 0
    for prediction, metric in zip(predictions, metrics):
        if not filter_record(metric, filters, filter_field):
            continue

        metric_value = metric["metric"]
        if metric_value is None:
            skipped_none += 1
            continue

        confidence = float(np.clip(float(prediction["confidence"]), 0.0, 1.0))  # type: ignore
        similarity = float(np.clip(float(metric_value), 0.0, 1.0))  # type: ignore
        confidences.append(confidence)
        similarities.append(similarity)

    if skipped_none:
        logger.warning("Skipped %d null metric rows", skipped_none)

    return confidences, similarities


def run(
    predictions_path: Path,
    metrics_path: Path,
    output_path: Path,
    confidence_bins: int,
    similarity_bins: int,
    filters: list[str] | None,
    filter_field: str,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    confidences, similarities = load_points(
        predictions_path,
        metrics_path,
        filters,
        filter_field,
    )
    if not confidences:
        logger.error("No points found after filtering")
        sys.exit(1)

    counts, _, _ = np.histogram2d(
        confidences,
        similarities,
        bins=[confidence_bins, similarity_bins],
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    x_labels = [bin_label(i, similarity_bins) for i in range(similarity_bins)]
    y_labels = [bin_label(i, confidence_bins) for i in range(confidence_bins)]
    heatmap = counts.T

    chart_title = title if title else "Confidence vs Similarity"
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    image = ax.imshow(heatmap, cmap="Blues", origin="lower", aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Occurrences")

    ax.set_title(chart_title)
    ax.set_xlabel(xlabel if xlabel else "Similarity")
    ax.set_ylabel(ylabel if ylabel else "Confidence")
    ax.set_xticks(range(similarity_bins), labels=x_labels, rotation=45, ha="right")
    ax.set_yticks(range(confidence_bins), labels=y_labels)

    fig.tight_layout()
    png_output_path = output_path.with_suffix(".png")
    fig.savefig(png_output_path, format="png")
    plt.close(fig)
    logger.info("Saved confidence-similarity distribution to %s", png_output_path)
