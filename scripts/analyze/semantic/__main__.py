from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import logging

from . import main

if __name__ == "__main__":
    logging.configure()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metric", required=True, type=str)
    parser.add_argument(
        "--target-mapping",
        required=True,
        type=Path,
        help="JSON mapping of target synset ids to category labels",
    )
    parser.add_argument(
        "--predicted-mapping",
        required=True,
        type=Path,
        help="JSON mapping of predicted synset ids to category labels",
    )
    args = parser.parse_args()

    main.run(
        args.input,
        args.output,
        args.metric,
        args.target_synset_mapping,
        args.predicted_synset_mapping,
    )
