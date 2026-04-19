from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, WordNetError

if TYPE_CHECKING:
    from features.wordnet.analysis import SemanticMetric

logger = logging.getLogger(__name__)

_PREDICTION_SCORE_SUFFIX = re.compile(r"_[0-9]+(?:\.[0-9]+)?$")
_WNID_PATTERN = re.compile(r"([nvars])(\d{8})")
MetricValue = int | float | None
MetricCalculator = Callable[[str, str], MetricValue]
_WORDNET_PROGRESS_EVERY = 5000


@dataclass(frozen=True)
class _WordNetMetric:
    name: str
    _calculator: MetricCalculator

    def calculate(self, a: str, b: str) -> MetricValue:
        return self._calculator(a, b)


class WordNetService:
    def initialize(self) -> bool:
        try:
            wn.ensure_loaded()
            logger.info("WordNet corpus is already ready")
            return False
        except LookupError:
            logger.info("Downloading nltk wordnet")
            nltk.download("wordnet", quiet=True)
            wn.ensure_loaded()
            logger.info("WordNet corpus is ready")
            return True

    def resolve_synset_id(self, stem: str) -> str:
        normalized_label = self._normalize_label(self._strip_prediction_score(stem))
        direct_matches = self._synset_lookup.get(normalized_label, [])

        if len(direct_matches) == 1:
            return direct_matches[0]

        ranked_matches = wn.synsets(normalized_label.replace(" ", "_"), pos="n")
        if ranked_matches:
            return f"n{ranked_matches[0].offset():08d}"

        if direct_matches:
            return direct_matches[0]

        raise RuntimeError(f"Could not decode synset id from filename stem: {stem!r}")

    def lookup_synset_ids(self, query: str) -> list[str]:
        normalized = query.strip()
        if not normalized:
            return []

        try:
            raw_synsets = wn.synsets(normalized.replace(" ", "_"))
        except (LookupError, WordNetError, AttributeError, ValueError):
            return []

        synset_ids: list[str] = []
        for synset in raw_synsets:
            if not isinstance(synset, Synset):
                continue
            synset_ids.append(f"{synset.pos()}{synset.offset():08d}")
        return synset_ids

    def lookup_labels(self, synset_id: str) -> list[str]:
        synset = self._parse_synset(synset_id)
        if synset is None:
            return []

        labels: list[str] = []
        for lemma in synset.lemmas() or []:
            label = lemma.name().replace("_", " ")
            if label not in labels:
                labels.append(label)
        return labels

    def lookup_definition(self, synset_id: str) -> str | None:
        synset = self._parse_synset(synset_id)
        if synset is None:
            return None

        definition = synset.definition().strip()
        return definition or None

    def create_metric(self, name: str) -> SemanticMetric:
        try:
            calculator = self._metric_calculators[name]
        except KeyError as exc:
            available = ", ".join(self.available_metric_names())
            raise ValueError(
                f"Unknown semantic metric: {name}. Available metrics: {available}"
            ) from exc
        logger.info("Using WordNet metric %s", name)
        return _WordNetMetric(name=name, _calculator=calculator)

    def available_metric_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._metric_calculators))

    @cached_property
    def _synset_lookup(self) -> dict[str, list[str]]:
        logger.info("Building WordNet synset lookup cache")
        lookup: dict[str, list[str]] = {}

        for synset_count, synset in enumerate(wn.all_synsets(pos="n"), start=1):
            synset_id = f"{synset.pos()}{synset.offset():08d}"
            keys = self._build_lookup_keys(synset)
            keys.add(self._normalize_label(synset.name().split(".", 1)[0]))

            for key in keys:
                if not key:
                    continue
                lookup.setdefault(key, [])
                if synset_id not in lookup[key]:
                    lookup[key].append(synset_id)
            if synset_count % _WORDNET_PROGRESS_EVERY == 0:
                logger.info(
                    "Built WordNet synset lookup for %s noun synsets so far",
                    synset_count,
                )

        logger.info(
            "Built WordNet synset lookup cache with %s normalized labels",
            len(lookup),
        )
        return lookup

    @cached_property
    def _ic(self):
        logger.info("Loading WordNet information-content cache")
        return wn.ic(wn, False, 0.0)

    @cached_property
    def _metric_calculators(self) -> dict[str, MetricCalculator]:
        return {
            "path_distance": self._path_distance,
            "path_similarity": self._path_similarity,
            "wup_similarity": self._wup_similarity,
            "lch_similarity": self._lch_similarity,
            "jcn_similarity": self._jcn_similarity,
            "lin_similarity": self._lin_similarity,
            "res_similarity": self._res_similarity,
        }

    def _path_distance(self, a: str, b: str) -> int | None:
        synset_pair = self._resolve_synset_pair(a, b, metric_name="Path distance")
        if synset_pair is None:
            return None
        left, right = synset_pair
        distance = left.shortest_path_distance(right, simulate_root=True)
        return int(distance) if distance is not None else None

    def _path_similarity(self, a: str, b: str) -> float | None:
        synset_pair = self._resolve_synset_pair(a, b, metric_name="Path similarity")
        if synset_pair is None:
            return None
        left, right = synset_pair
        similarity = left.path_similarity(right)
        return float(similarity) if similarity is not None else None

    def _wup_similarity(self, a: str, b: str) -> float | None:
        synset_pair = self._resolve_synset_pair(a, b, metric_name="WUP similarity")
        if synset_pair is None:
            return None
        left, right = synset_pair
        similarity = left.wup_similarity(right)
        return float(similarity) if similarity is not None else None

    def _lch_similarity(self, a: str, b: str) -> float | None:
        synset_pair = self._resolve_synset_pair(a, b, metric_name="LCh similarity")
        if synset_pair is None:
            return None
        left, right = synset_pair
        similarity = left.lch_similarity(right, simulate_root=True)
        return float(similarity) if similarity is not None else None

    def _jcn_similarity(self, a: str, b: str) -> float | None:
        synset_pair = self._resolve_synset_pair(a, b, metric_name="JCN similarity")
        if synset_pair is None:
            return None
        left, right = synset_pair
        similarity = left.jcn_similarity(right, self._ic)
        return float(similarity) if similarity is not None else None

    def _lin_similarity(self, a: str, b: str) -> float | None:
        synset_pair = self._resolve_synset_pair(a, b, metric_name="Lin similarity")
        if synset_pair is None:
            return None
        left, right = synset_pair
        similarity = left.lin_similarity(right, self._ic)
        return float(similarity) if similarity is not None else None

    def _res_similarity(self, a: str, b: str) -> float | None:
        synset_pair = self._resolve_synset_pair(a, b, metric_name="Res similarity")
        if synset_pair is None:
            return None
        left, right = synset_pair
        similarity = left.res_similarity(right, self._ic)
        return float(similarity) if similarity is not None else None

    def _resolve_synset_pair(
        self,
        a: str,
        b: str,
        *,
        metric_name: str,
    ) -> tuple[Synset, Synset] | None:
        left = self._parse_synset(a)
        right = self._parse_synset(b)
        if left is None or right is None:
            logger.debug(
                "%s unavailable for x=%r y=%r (invalid synset)",
                metric_name,
                a,
                b,
            )
            return None
        return left, right

    def _parse_synset(self, value: str) -> Synset | None:
        normalized = value.strip()
        try:
            wnid_match = _WNID_PATTERN.fullmatch(normalized)
            if wnid_match:
                return wn.synset_from_pos_and_offset(
                    wnid_match.group(1),
                    int(wnid_match.group(2)),
                )
            return wn.synset(normalized)
        except (LookupError, WordNetError, KeyError, AttributeError, ValueError):
            logger.debug("Could not resolve synset name %r", value, exc_info=True)
            return None

    def _build_lookup_keys(self, synset: Synset) -> set[str]:
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
