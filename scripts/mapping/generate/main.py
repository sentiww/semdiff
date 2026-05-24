from __future__ import annotations

import json
import logging
from pathlib import Path

from scripts.common.synset import is_synset_id, synset_labels_from_id

logger = logging.getLogger(__file__)


def build_mapping_from_synset_dir(synset_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for synset_path in sorted(synset_dir.iterdir()):
        if not synset_path.is_dir():
            logger.debug("Skipping non-directory path: %s", synset_path)
            continue

        synset = synset_path.name
        if not is_synset_id(synset):
            logger.debug("Skipping non-synset directory: %s", synset_path.name)
            continue

        category = extract_category_from_synset(synset)
        if not category:
            logger.warning("Skipping synset %s: no category could be extracted", synset)
            continue
        mapping[synset] = category

    return mapping


def extract_category_from_synset(synset: str) -> str:
    try:
        return synset_labels_from_id(synset)
    except Exception:
        logger.debug("Could not resolve WordNet synset for %s", synset)
        return ""


def run(synset_dir: Path, output: Path) -> None:
    mapping = build_mapping_from_synset_dir(synset_dir)
    logger.info("Built mapping for %d synsets", len(mapping))

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    logger.info("Wrote mapping to %s", output)
