from __future__ import annotations

import logging
import re

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, WordNetError

LOGGER = logging.getLogger("main.wordnet")
WNID_PATTERN = re.compile(r"([nvars])(\d{8})")


def parse_synset(name: str, *, logger: logging.Logger = LOGGER) -> Synset | None:
    normalized = _normalize_synset_name(name)
    try:
        wnid_match = WNID_PATTERN.fullmatch(normalized)
        if wnid_match:
            return wn.synset_from_pos_and_offset(
                wnid_match.group(1),
                int(wnid_match.group(2)),
            )
        return wn.synset(normalized)
    except (LookupError, WordNetError, KeyError, AttributeError, ValueError):
        logger.debug("Could not resolve synset name %r", name, exc_info=True)
        return None


def find_word_synsets(word: str, *, logger: logging.Logger = LOGGER) -> list[Synset]:
    normalized = word.strip()
    if not normalized:
        return []

    try:
        raw_synsets = wn.synsets(normalized.replace(" ", "_"))
    except (LookupError, WordNetError, AttributeError, ValueError):
        logger.debug("Could not resolve word %r", word, exc_info=True)
        return []

    synsets: list[Synset] = []
    for synset in raw_synsets:
        if not isinstance(synset, Synset):
            continue
        synsets.append(synset)
    return synsets


def find_word_synset_ids(word: str, *, logger: logging.Logger = LOGGER) -> list[str]:
    synsets = find_word_synsets(word, logger=logger)
    synset_ids: list[str] = []
    for synset in synsets:
        synset_ids.append(_format_synset_id(synset))
    return synset_ids


def find_synset_labels(
    synset_name: str,
    *,
    logger: logging.Logger = LOGGER,
) -> list[str]:
    synset = parse_synset(synset_name, logger=logger)
    if synset is None:
        return []

    labels: list[str] = []
    for lemma in synset.lemmas() or []:
        label = lemma.name().replace("_", " ")
        if label not in labels:
            labels.append(label)
    return labels


def lookup_synset_labels(
    synset_name: str,
    *,
    logger: logging.Logger = LOGGER,
) -> list[str]:
    return find_synset_labels(synset_name, logger=logger)


def lookup_word_synset_ids(word: str, *, logger: logging.Logger = LOGGER) -> list[str]:
    return find_word_synset_ids(word, logger=logger)


def lookup_word(word: str, *, logger: logging.Logger = LOGGER) -> list[str]:
    return find_word_synset_ids(word, logger=logger)


def resolve_word_synsets(word: str, *, logger: logging.Logger = LOGGER) -> list[Synset]:
    return find_word_synsets(word, logger=logger)


def _normalize_synset_name(name: str) -> str:
    return name.strip()


def _format_synset_id(synset: Synset) -> str:
    return f"{synset.pos()}{synset.offset():08d}"


def resolve_synset_pair(
    a: str,
    b: str,
    *,
    logger: logging.Logger,
    metric_name: str,
) -> tuple[Synset, Synset] | None:
    a_synset = parse_synset(a, logger=logger)
    b_synset = parse_synset(b, logger=logger)
    if a_synset is None or b_synset is None:
        logger.debug("%s unavailable for x=%r y=%r (invalid synset)", metric_name, a, b)
        return None
    return a_synset, b_synset
