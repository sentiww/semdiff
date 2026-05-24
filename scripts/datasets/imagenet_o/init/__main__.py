from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import logging

from . import main

if __name__ == "__main__":
    logging.configure()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        required=False,
        type=str,
        default="https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    main.run(args.url, args.output)
