from __future__ import annotations

import logging
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, WordNetError


class WordNetService:
    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("main").getChild("wordnet")
        self._wordnet_checked = False

    def _ensure_wordnet(self) -> None:
        if self._wordnet_checked:
            return

        self._wordnet_checked = True
        self._logger.info("Downloading nltk wordnet")
        if not nltk.download("wordnet", quiet=True):
            self._logger.warning("WordNet corpus download/check was unsuccessful")

    def _parse_synset(self, name: str) -> Synset | None:
        try:
            normalized = name.strip()
            synset = wn.synset(normalized)
            if synset is None:
                self._logger.debug("Resolved synset %r -> None", name)
                return None
            self._logger.debug("Resolved synset %r -> %s", name, synset.name())
            return synset
        except (LookupError, WordNetError, KeyError, AttributeError, ValueError):
            self._logger.debug("Could not resolve synset name %r", name, exc_info=True)
            return None

    def explain(self, a: str, b: str) -> float | None:
        self._ensure_wordnet()
        self._logger.debug("Calculating path similarity: x=%r y=%r", a, b)
        a_synset = self._parse_synset(a)
        b_synset = self._parse_synset(b)

        if a_synset is None or b_synset is None:
            self._logger.info(
                "Path similarity unavailable for x=%r y=%r (invalid synset)", a, b
            )
            return None

        similarity = a_synset.path_similarity(a_synset)
        result = float(similarity) if similarity is not None else None
        self._logger.info(
            "Path similarity for %s and %s: %s",
            a_synset.name(),
            b_synset.name(),
            result,
        )
        return result
