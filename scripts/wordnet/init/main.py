from __future__ import annotations

import logging

import nltk
from nltk.corpus import wordnet

logger = logging.getLogger(__file__)


def run() -> None:
    try:
        logger.info("Checking WordNet installation")
        wordnet.ensure_loaded()
        logger.info("WordNet corpus is already ready")
    except LookupError:
        logger.info("Downloading nltk wordnet")
        nltk.download("wordnet", quiet=True)
        wordnet.ensure_loaded()
        logger.info("WordNet corpus is ready")
        pass
