from __future__ import annotations

import logging
from nltk.corpus import wordnet

from .common import LOGGER
from .common import resolve_synset_pair


def lin_similarity(a: str, b: str, *, logger: logging.Logger = LOGGER) -> int | None:
    synset_pair = resolve_synset_pair(a, b, logger=logger, metric_name="LCh Similarity")

    if synset_pair is None:
        return None

    a_synset, b_synset = synset_pair
    ic = wordnet.ic(wordnet, False, 0.0)  # TODO: Sensible corpus

    similarity = a_synset.lin_similarity(b_synset, ic)
    return int(similarity) if similarity is not None else None
