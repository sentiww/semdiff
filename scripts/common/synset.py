from __future__ import annotations

import re
import shutil
from pathlib import Path

from nltk.corpus import wordnet as wn

_PREDICTION_SCORE_SUFFIX = re.compile(r"_[0-9]+(?:\.[0-9]+)?$")


def is_synset_id(value: str) -> bool:
    return value.startswith("n") and value[1:].isdigit()


def synset_labels_from_id(synset_id: str) -> str:
    synset = wn.synset_from_pos_and_offset("n", int(synset_id[1:]))
    return ", ".join(lemma.name().replace("_", " ") for lemma in synset.lemmas())  # type: ignore


def decode_image_stem_to_synset_id(
    stem: str, synset_lookup: dict[str, list[str]]
) -> str:
    normalized_label = _normalize_label(_strip_prediction_score(stem))
    direct_matches = synset_lookup.get(normalized_label, [])
    if len(direct_matches) == 1:
        return direct_matches[0]

    ranked_matches = wn.synsets(normalized_label.replace(" ", "_"), pos="n")
    if ranked_matches:
        return f"n{ranked_matches[0].offset():08d}"  # type: ignore

    if direct_matches:
        return direct_matches[0]

    raise RuntimeError(
        f"Could not decode synset id from imagenet-o filename stem: {stem!r}"
    )


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _strip_prediction_score(stem: str) -> str:
    return _PREDICTION_SCORE_SUFFIX.sub("", stem)
