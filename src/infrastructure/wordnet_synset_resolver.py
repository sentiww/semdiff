from __future__ import annotations

import re

from nltk.corpus import wordnet as wn


_PREDICTION_SCORE_SUFFIX = re.compile(r"_[0-9]+(?:\.[0-9]+)?$")


class WordNetSynsetResolver:
    def __init__(self) -> None:
        self._lookup = self._build_synset_lookup()

    def resolve(self, stem: str) -> str:
        normalized_label = self._normalize_label(self._strip_prediction_score(stem))
        direct_matches = self._lookup.get(normalized_label, [])

        if len(direct_matches) == 1:
            return direct_matches[0]

        ranked_matches = wn.synsets(normalized_label.replace(" ", "_"), pos="n")
        if ranked_matches:
            return f"n{ranked_matches[0].offset():08d}"

        if direct_matches:
            return direct_matches[0]

        raise RuntimeError(f"Could not decode synset id from filename stem: {stem!r}")

    def _build_synset_lookup(self) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = {}

        for synset in wn.all_synsets(pos="n"):
            synset_id = f"{synset.pos()}{synset.offset():08d}"
            keys = self._build_lookup_keys(synset)
            keys.add(self._normalize_label(synset.name().split(".", 1)[0]))

            for key in keys:
                if not key:
                    continue
                lookup.setdefault(key, [])
                if synset_id not in lookup[key]:
                    lookup[key].append(synset_id)

        return lookup

    def _build_lookup_keys(self, synset: object) -> set[str]:
        keys: set[str] = set()
        lemma_names: list[str] = []

        for lemma in synset.lemmas():
            lemma_name = lemma.name()
            lemma_names.append(lemma_name)
            keys.add(self._normalize_label(lemma_name))

        keys.add(self._normalize_label("_".join(lemma_names)))
        return keys

    def _normalize_label(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _strip_prediction_score(self, stem: str) -> str:
        return _PREDICTION_SCORE_SUFFIX.sub("", stem)
