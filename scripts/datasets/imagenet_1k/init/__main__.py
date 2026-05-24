from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import logging

from . import main

if __name__ == "__main__":
    logging.configure()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--ground_truth", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    args = parser.parse_args()

    main.run(args.output, args.archive, args.ground_truth, args.meta)
