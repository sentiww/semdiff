from __future__ import annotations

import logging

from .common import LOGGER
from .common import resolve_synset_pair


def path_similarity(a: str, b: str, *, logger: logging.Logger = LOGGER) -> float | None:
    synset_pair = resolve_synset_pair(
        a, b, logger=logger, metric_name="Path similarity"
    )
    if synset_pair is None:
        return None

    a_synset, b_synset = synset_pair
    similarity = a_synset.path_similarity(b_synset)
    return float(similarity) if similarity is not None else None
