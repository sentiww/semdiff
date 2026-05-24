from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import logging

from . import main

if __name__ == "__main__":
    logging.configure()

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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

    main.run(
        args.predictions,
        args.metrics,
        args.output,
        args.filter,
        args.filter_field,
        args.title,
        args.xlabel,
        args.ylabel,
    )
