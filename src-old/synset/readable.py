from __future__ import annotations

import logging

from wordnet.common import find_synset_labels

LOGGER = logging.getLogger("main").getChild("synset")


def synset_readable(synset_id: str) -> list[str]:
    labels = find_synset_labels(synset_id, logger=LOGGER.getChild("wordnet"))
    if not labels:
        raise ValueError(f"No labels found for synset id {synset_id!r}")
    return labels
