from __future__ import annotations

import logging
import re
import shutil

from .common import LOGGER, dataset_root

_SYNSET_DIR_PATTERN = re.compile(r"n\d+$")


def clear_dataset(
    name: str, *, logger: logging.Logger = LOGGER
) -> None:
    dataset_logger = logger.getChild(name)
    root = dataset_root(name)
    synset_dirs = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if not _SYNSET_DIR_PATTERN.fullmatch(path.name):
            continue
        synset_dirs.append(path)
    synset_dirs.sort()

    if not synset_dirs:
        dataset_logger.info("No synset folders to delete")
        return

    dataset_logger.info("Deleting %s synset folders", len(synset_dirs))
    for path in synset_dirs:
        shutil.rmtree(path)

    dataset_logger.info("Done")
