from __future__ import annotations

import logging

import nltk
from nltk.corpus import wordnet as wn

from .common import LOGGER


def init_wordnet(*, logger: logging.Logger = LOGGER) -> None:
    try:
        wn.ensure_loaded()
        logger.info("WordNet corpus is already ready")
        return
    except LookupError:
        pass

    logger.info("Downloading nltk wordnet")
    nltk.download("wordnet", quiet=True)
    wn.ensure_loaded()

    logger.info("WordNet corpus is ready")
