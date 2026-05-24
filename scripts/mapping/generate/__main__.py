from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import logging

from . import main

if __name__ == "__main__":
    logging.configure()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing synset subdirectories (e.g., imagenet-o/)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    main.run(args.input, args.output)
