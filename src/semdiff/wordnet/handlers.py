from __future__ import annotations

from dataclasses import dataclass

from semdiff.core.handlers import CommandInput, CommandOutput, Handler, HandlerFactory
from semdiff.core.protocols import IWordNetService
from semdiff.wordnet.service import WordNetService


@dataclass(frozen=True)
class SynsetIdInput(CommandInput):
    query: str


@dataclass(frozen=True)
class SynsetIdOutput(CommandOutput):
    synset_ids: list[str]


class SynsetIdHandler(Handler[SynsetIdInput, SynsetIdOutput]):
    def __init__(self, wordnet: IWordNetService) -> None:
        self._wordnet = wordnet

    def __call__(self, cmd: SynsetIdInput) -> SynsetIdOutput:
        synset_ids = self._wordnet.lookup_synset_ids(cmd.query)

        if not synset_ids:
            raise ValueError(f"No synset id found for {cmd.query!r}")

        return SynsetIdOutput(synset_ids=synset_ids)


@dataclass(frozen=True)
class SynsetReadableInput(CommandInput):
    synset_id: str


@dataclass(frozen=True)
class SynsetReadableOutput(CommandOutput):
    labels: list[str]


class SynsetReadableHandler(Handler[SynsetReadableInput, SynsetReadableOutput]):
    def __init__(self, wordnet: IWordNetService) -> None:
        self._wordnet = wordnet

    def __call__(self, cmd: SynsetReadableInput) -> SynsetReadableOutput:
        labels = self._wordnet.lookup_labels(cmd.synset_id)

        if not labels:
            raise ValueError(f"No labels found for synset id {cmd.synset_id!r}")

        return SynsetReadableOutput(labels=labels)


@dataclass(frozen=True)
class WordNetInitInput(CommandInput):
    pass


@dataclass(frozen=True)
class WordNetInitOutput(CommandOutput):
    downloaded: bool


class WordNetInitHandler(Handler[WordNetInitInput, WordNetInitOutput]):
    def __init__(self, wordnet: IWordNetService) -> None:
        self._wordnet = wordnet

    def __call__(self, _: WordNetInitInput) -> WordNetInitOutput:
        return WordNetInitOutput(downloaded=self._wordnet.initialize())


class WordNetHandlers(HandlerFactory):
    def __init__(self, wordnet: IWordNetService) -> None:
        self._wordnet = wordnet

    def create_synset_id(self) -> SynsetIdHandler:
        return SynsetIdHandler(wordnet=self._wordnet)

    def create_synset_readable(self) -> SynsetReadableHandler:
        return SynsetReadableHandler(wordnet=self._wordnet)

    def create_init(self) -> WordNetInitHandler:
        return WordNetInitHandler(wordnet=self._wordnet)
