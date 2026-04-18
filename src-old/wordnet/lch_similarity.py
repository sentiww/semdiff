from __future__ import annotations

import logging

from .common import LOGGER
from .common import resolve_synset_pair


def lch_similarity(a: str, b: str, *, logger: logging.Logger = LOGGER) -> int | None:
    synset_pair = resolve_synset_pair(a, b, logger=logger, metric_name="LCh Similarity")
    if synset_pair is None:
        return None

    a_synset, b_synset = synset_pair
    similarity = a_synset.lch_similarity(b_synset, simulate_root=True)
    return int(similarity) if similarity is not None else None
