from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scripts.common.logging as logging

import scripts.common.logging as logging
import main

if __name__ == "__main__":
    logging.configure()

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

    main.run(
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
