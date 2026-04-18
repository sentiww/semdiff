from __future__ import annotations

import logging

from wordnet.common import find_word_synset_ids

LOGGER = logging.getLogger("main").getChild("synset")


def synset_id(query: str) -> list[str]:
    synset_ids = find_word_synset_ids(query, logger=LOGGER.getChild("wordnet"))
    if not synset_ids:
        raise ValueError(f"No synset id found for {query!r}")
    return synset_ids
