from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scripts.common.logging as logging

import main

if __name__ == "__main__":
    logging.configure()

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confidence-bins",
        type=int,
        default=10,
        help="Number of confidence bins across [0, 1]",
    )
    parser.add_argument(
        "--similarity-bins",
        type=int,
        default=5,
        help="Number of similarity bins across [0, 1]",
    )
    parser.add_argument(
        "--x-bins",
        type=int,
        help="Alias for --confidence-bins",
    )
    parser.add_argument(
        "--y-bins",
        type=int,
        help="Alias for --similarity-bins",
    )
    parser.add_argument("--filter", action="append", help="Filter synsets (repeatable)")
    parser.add_argument(
        "--filter-field",
        choices=["target", "predicted"],
        default="target",
        help="Record field to apply --filter values against",
    )
    parser.add_argument("--title", help="Chart title")
    parser.add_argument("--xlabel", help="X-axis label")
    parser.add_argument("--ylabel", help="Y-axis label")
    args = parser.parse_args()

    confidence_bins = args.x_bins if args.x_bins is not None else args.confidence_bins
    similarity_bins = args.y_bins if args.y_bins is not None else args.similarity_bins

    if confidence_bins <= 0 or similarity_bins <= 0:
        sys.exit(1)

    main.run(
        args.predictions,
        args.metrics,
        args.output,
        confidence_bins,
        similarity_bins,
        args.filter,
        args.filter_field,
        args.title,
        args.xlabel,
        args.ylabel,
    )
