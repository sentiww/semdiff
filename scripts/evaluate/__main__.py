from __future__ import annotations

import argparse
import logging as std_logging
from pathlib import Path

from scripts.common import logging

from . import run

if __name__ == "__main__":
    logging.configure()
    std_logging.getLogger("PIL").setLevel(std_logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=["resnet", "densenet", "vgg", "vit-b-16", "clip-vit-b-16"],
    )
    parser.add_argument(
        "--index-mapping",
        required=True,
        type=Path,
        help="torchvision.json - maps model index to synset",
    )
    parser.add_argument(
        "--target-synset-mapping",
        required=True,
        type=Path,
        help="imagenet-1k.json or imagenet-o.json - maps dataset synset to category name",
    )
    parser.add_argument(
        "--predicted-synset-mapping",
        required=True,
        type=Path,
        help="imagenet-1k.json or imagenet-o.json - maps predicted synset to category name",
    )
    args = parser.parse_args()

    run.run(
        args.dataset,
        args.output,
        args.model,
        args.index_mapping,
        args.target_synset_mapping,
        args.predicted_synset_mapping,
    )
