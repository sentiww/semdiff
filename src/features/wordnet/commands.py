from __future__ import annotations

from dataclasses import dataclass

from features.wordnet.service import WordNetService


@dataclass(frozen=True)
class SynsetIdInput:
    query: str


@dataclass(frozen=True)
class SynsetIdOutput:
    synset_ids: list[str]


class SynsetIdHandler:
    def __init__(self, wordnet: WordNetService) -> None:
        self._wordnet = wordnet

    def __call__(self, cmd: SynsetIdInput) -> SynsetIdOutput:
        synset_ids = self._wordnet.lookup_synset_ids(cmd.query)

        if not synset_ids:
            raise ValueError(f"No synset id found for {cmd.query!r}")

        return SynsetIdOutput(synset_ids=synset_ids)


@dataclass(frozen=True)
class SynsetReadableInput:
    synset_id: str


@dataclass(frozen=True)
class SynsetReadableOutput:
    labels: list[str]


class SynsetReadableHandler:
    def __init__(self, wordnet: WordNetService) -> None:
        self._wordnet = wordnet

    def __call__(self, cmd: SynsetReadableInput) -> SynsetReadableOutput:
        labels = self._wordnet.lookup_labels(cmd.synset_id)

        if not labels:
            raise ValueError(f"No labels found for synset id {cmd.synset_id!r}")

        return SynsetReadableOutput(labels=labels)


@dataclass(frozen=True)
class WordNetInitInput:
    pass


@dataclass(frozen=True)
class WordNetInitOutput:
    downloaded: bool


class WordNetInitHandler:
    def __init__(self, wordnet: WordNetService) -> None:
        self._wordnet = wordnet

    def __call__(self, _: WordNetInitInput) -> WordNetInitOutput:
        return WordNetInitOutput(downloaded=self._wordnet.initialize())
