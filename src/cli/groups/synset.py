from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from bootstrap import SynsetContainer

synset_app = typer.Typer(help="Synset commands")


def register(container_factory: Callable[[], SynsetContainer]) -> typer.Typer:
    @synset_app.command("id")
    def synset_id(
        query: str = typer.Argument(
            ...,
            help="ImageNet label or synonym, for example 'goldfish'",
        ),
    ) -> None:
        from features.wordnet.handlers import SynsetIdInput, WordNetHandlers

        container = container_factory()
        handlers = WordNetHandlers(wordnet=container._wordnet)
        handler = handlers.create_synset_id()
        result = handler(SynsetIdInput(query=query))
        for synset_value in result.synset_ids:
            typer.echo(synset_value)

    @synset_app.command("readable")
    def synset_readable(
        synset_id: str = typer.Argument(
            ...,
            help="Synset id, for example 'n01443537'",
        ),
    ) -> None:
        from features.wordnet.handlers import SynsetReadableInput, WordNetHandlers

        container = container_factory()
        handlers = WordNetHandlers(wordnet=container._wordnet)
        handler = handlers.create_synset_readable()
        result = handler(SynsetReadableInput(synset_id=synset_id))
        for label in result.labels:
            typer.echo(label)

    return synset_app
