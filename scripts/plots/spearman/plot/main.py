from __future__ import annotations

import json
import logging
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.stats import spearmanr

logger = logging.getLogger(__file__)


def build_stats_text(x_values: list[float], y_values: list[float], rho: float) -> str:
    confidence_median = statistics.median(x_values)
    wup_median = statistics.median(y_values)
    wup_p10, wup_p90 = np.percentile(y_values, [10, 90])
    return (
        f"Spearman rho: {rho:.3f}<br>"
        f"n: {len(x_values)}<br>"
        f"Confidence median: {confidence_median:.3f}<br>"
        f"WUP median: {wup_median:.3f}<br>"
        f"WUP p10-p90: {wup_p10:.3f}-{wup_p90:.3f}"
    )


def build_layout(
    rho: float,
    x_values: list[float],
    y_values: list[float],
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> dict[str, object]:
    return dict(
        title_text=title or "Confidence vs WUP Similarity",
        xaxis_title=xlabel or "Confidence",
        yaxis_title=ylabel or "WUP Similarity",
        font_size=12,
        width=1920,
        height=1080,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        annotations=[
            dict(
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                align="left",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#cccccc",
                borderwidth=1,
                text=build_stats_text(x_values, y_values, rho),
                showarrow=False,
            )
        ],
    )


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_points(
    predictions_path: Path,
    metrics_path: Path,
    filters: list[str] | None,
    filter_field: str,
) -> tuple[list[float], list[float]]:
    predictions = load_jsonl(predictions_path)
    metrics = load_jsonl(metrics_path)
    if len(predictions) != len(metrics):
        raise ValueError(
            "Predictions and metric files must have the same number of rows"
        )

    filters_set = set(filters) if filters else None
    confidences: list[float] = []
    metric_values: list[float] = []
    skipped_none = 0
    for prediction, metric in zip(predictions, metrics):
        if filters_set is not None and str(metric[filter_field]) not in filters_set:
            continue

        metric_value = metric["metric"]
        if metric_value is None:
            skipped_none += 1
            continue

        confidences.append(float(prediction["confidence"]))  # type: ignore
        metric_values.append(float(metric_value))  # type: ignore

    if skipped_none:
        logger.warning("Skipped %d null metric rows", skipped_none)

    return confidences, metric_values


def run(
    predictions_path: Path,
    metrics_path: Path,
    output_path: Path,
    filters: list[str] | None,
    filter_field: str,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    x_values, y_values = load_points(
        predictions_path, metrics_path, filters, filter_field
    )

    if len(x_values) < 2:
        logger.error("Not enough filtered data points to compute Spearman correlation")
        sys.exit(1)

    rho = float(spearmanr(x_values, y_values).statistic)  # type: ignore

    if math.isnan(rho):
        logger.warning(
            "Spearman correlation is undefined for constant inputs; using 0.0"
        )
        rho = 0.0

    y_min = min(y_values)
    y_max = max(y_values)
    y_padding = max((y_max - y_min) * 0.06, 0.01)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            marker=dict(size=6, color="#1f77b4", opacity=0.28),
            name="Samples",
            hovertemplate="Confidence: %{x:.3f}<br>WUP similarity: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(build_layout(rho, x_values, y_values, title, xlabel, ylabel))
    fig.update_xaxes(range=[0.0, 1.0], showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(
        range=[max(0.0, y_min - y_padding), min(1.0, y_max + y_padding)],
        tickformat=".3f",
        showgrid=True,
        gridcolor="#e5e5e5",
    )

    fig.write_image(output_path, scale=2)
    logger.info("Saved Spearman plot to %s", output_path)
