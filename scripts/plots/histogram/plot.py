from __future__ import annotations

import logging
import json
import argparse
import sys
import math
from pathlib import Path

import plotly.graph_objects as go


logger = logging.getLogger("plot.distribution")


def is_similarity_input(input_path: Path) -> bool:
    return "path_similarity" in input_path.stem or "wup_similarity" in input_path.stem


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]

    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def compute_similarity_range(values: list[float], zoom: str) -> list[float]:
    if zoom == "full":
        return [0.0, 1.0]

    data_min = min(values)
    data_max = max(values)
    if zoom == "data":
        if data_min == data_max:
            padding = max(0.01, min(0.05, data_min * 0.05))
            return [max(0.0, data_min - padding), min(1.0, data_max + padding)]
        return [data_min, data_max]

    low = percentile(values, 0.02)
    high = percentile(values, 0.98)
    if low == high:
        low = data_min
        high = data_max
    if low == high:
        padding = max(0.01, min(0.05, low * 0.05))
        return [max(0.0, low - padding), min(1.0, high + padding)]

    padding = max((high - low) * 0.05, 0.005)
    return [max(0.0, low - padding), min(1.0, high + padding)]


def compute_histogram_bin_size(
    values: list[float], bins: int | None, x_range: list[float]
) -> float | str:
    range_width = x_range[1] - x_range[0]
    if bins is not None and range_width > 0:
        return range_width / bins

    if len(values) < 2 or range_width <= 0:
        return "auto"

    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return "auto"

    bin_width = (2 * iqr) / (len(values) ** (1 / 3))
    if bin_width <= 0:
        return "auto"

    return min(bin_width, range_width)


def load_metrics(
    input_path: Path,
    filters: list[str] | None,
    filter_field: str,
) -> list[float]:
    metrics: list[float] = []
    skipped_none = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if filters is not None and record[filter_field] not in filters:
                continue

            metric = record["metric"]
            if metric is None:
                skipped_none += 1
                continue

            metrics.append(float(metric))

    if skipped_none:
        logger.warning("Skipped %d null metrics in %s", skipped_none, input_path)

    return metrics


def main(
    inputs: list[Path],
    output: Path,
    names: list[str] | None,
    mode: str,
    bins: int | None,
    zoom: str,
    range_min: float | None,
    range_max: float | None,
    filters: list[str] | None,
    filter_field: str,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    all_metrics: list[list[float]] = []
    kept_inputs: list[Path] = []
    similarity_inputs: list[bool] = []
    for inp in inputs:
        metrics = load_metrics(inp, filters, filter_field)
        if not metrics:
            logger.warning("No data found in %s, skipping", inp)
            continue
        all_metrics.append(metrics)
        kept_inputs.append(inp)
        similarity_inputs.append(is_similarity_input(inp))

    if not all_metrics:
        logger.error("No valid data found in any input file")
        sys.exit(1)

    if names is None:
        names = [inp.stem for inp in kept_inputs]

    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    all_values = [m for metrics in all_metrics for m in metrics]
    similarity_plot = bool(similarity_inputs) and all(similarity_inputs)

    if range_min is not None and range_max is not None:
        x_range = [range_min, range_max]
    elif similarity_plot:
        x_range = compute_similarity_range(all_values, zoom)
    else:
        x_range = [min(all_values), max(all_values)]

    xbins_size = compute_histogram_bin_size(all_values, bins, x_range)
    tick_format = ".3f" if similarity_plot else None
    histogram_norm = "percent" if similarity_plot else None

    fig = go.Figure()

    if mode == "split":
        bingroup = "shared"
        for i, metrics in enumerate(all_metrics):
            label = names[i] if i < len(names) else f"File {i + 1}"
            color = color_palette[i % len(color_palette)]
            fig.add_trace(
                go.Histogram(
                    x=metrics,
                    name=label,
                    marker_color=color,
                    xbins=dict(
                        size=xbins_size,
                        start=x_range[0],
                        end=x_range[1],
                    ),
                    bingroup=bingroup,
                    legendgroup=label,
                    histnorm=histogram_norm,
                )
            )
    else:
        for i, metrics in enumerate(all_metrics):
            label = names[i] if i < len(names) else f"File {i + 1}"
            color = color_palette[i % len(color_palette)]
            fig.add_trace(
                go.Histogram(
                    x=metrics,
                    name=label,
                    marker_color=color,
                    marker_opacity=0.5,
                    xbins=dict(
                        size=xbins_size,
                        start=x_range[0],
                        end=x_range[1],
                    ),
                    legendgroup=label,
                    showlegend=True,
                    histnorm=histogram_norm,
                )
            )

    chart_title = title if title else "Similarity Distribution"
    if not similarity_plot:
        chart_title = title if title else "Distance Distribution"
    # if mode == "split":
    # chart_title += " (Split)"
    # else:
    # chart_title += " (Overlay)"

    fig.update_layout(
        title_text=chart_title,
        xaxis_title=(
            xlabel if xlabel else ("Similarity" if similarity_plot else "Distance")
        ),
        yaxis_title=ylabel if ylabel else ("Percent" if similarity_plot else "Count"),
        barmode="overlay" if mode == "overlay" else "group",
        xaxis_range=x_range,
        font_size=12,
        width=1920,
        height=1080,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    if tick_format is not None:
        fig.update_xaxes(tickformat=tick_format)

    fig.write_image(output, scale=2)
    logger.info("Saved distribution plot to %s", output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        required=True,
        type=Path,
        help="Input metric JSONL file (repeatable)",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output image path"
    )
    parser.add_argument("--names", nargs="*", help="Legend names for each input file")
    parser.add_argument(
        "--mode", choices=["overlay", "split"], default="overlay", help="Histogram mode"
    )
    parser.add_argument("--bins", type=int, help="Number of bins")
    parser.add_argument(
        "--zoom",
        choices=["full", "data", "robust"],
        default="robust",
        help="Auto-range mode for similarity plots when --range is not set",
    )
    parser.add_argument("--filter", action="append", help="Filter targets (repeatable)")
    parser.add_argument(
        "--filter-field",
        choices=["target", "predicted"],
        default="target",
        help="Record field to apply --filter values against",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Fixed x-axis range",
    )
    parser.add_argument("--title", help="Chart title")
    parser.add_argument("--xlabel", help="X-axis label")
    parser.add_argument("--ylabel", help="Y-axis label")
    args = parser.parse_args()

    range_min = args.range[0] if args.range else None
    range_max = args.range[1] if args.range else None

    main(
        args.input,
        args.output,
        args.names,
        args.mode,
        args.bins,
        args.zoom,
        range_min,
        range_max,
        args.filter,
        args.filter_field,
        args.title,
        args.xlabel,
        args.ylabel,
    )
