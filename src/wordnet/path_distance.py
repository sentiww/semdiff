from __future__ import annotations

import logging

from .common import LOGGER
from .common import resolve_synset_pair


def path_distance(a: str, b: str, *, logger: logging.Logger = LOGGER) -> int | None:
    synset_pair = resolve_synset_pair(a, b, logger=logger, metric_name="Path distance")
    if synset_pair is None:
        return None

    a_synset, b_synset = synset_pair
    distance = a_synset.shortest_path_distance(b_synset, simulate_root=True)
    return int(distance) if distance is not None else None
