from __future__ import annotations

import logging

import nltk
from nltk.corpus import wordnet, brown

from .common import LOGGER


def init_wordnet(*, logger: logging.Logger = LOGGER) -> None:
    try:
        wordnet.ensure_loaded()
        logger.info("WordNet corpus is already ready")
    except LookupError:
        logger.info("Downloading nltk wordnet")
        # TODO: Might be worth it to experiment on other wordnet variants
        # See: https://www.nltk.org/nltk_data/
        nltk.download("wordnet", quiet=True)
        wordnet.ensure_loaded()
        logger.info("WordNet corpus is ready")
        pass

    # Skip brown for now
    return

    # TODO: jcn_similarity, lin_similarity, res_similarity require additional
    # Information Content (IC), might want to experiment on this (for example with brown)
    # See: https://www.nltk.org/howto/wordnet.html#similarity
    try:
        brown.ensure_loaded()
        logger.info("Brown corpus is already ready")
    except LookupError:
        logger.info("Downloading nltk brown")
        nltk.download("brown", quiet=True)
        brown.ensure_loaded()
        logger.info("Brown corpus is ready")
