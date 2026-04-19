from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from bootstrap.container import SynsetContainer

synset_app = typer.Typer(help="Synset commands")


def register(container_factory: Callable[[], SynsetContainer]) -> typer.Typer:
    @synset_app.command("id")
    def synset_id(
        query: str = typer.Argument(
            ...,
            help="ImageNet label or synonym, for example 'goldfish'",
        ),
    ) -> None:
        from features.wordnet.commands import SynsetIdInput

        container = container_factory()
        handler = container.synset_id_handler()
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
        from features.wordnet.commands import SynsetReadableInput

        container = container_factory()
        handler = container.synset_readable_handler()
        result = handler(SynsetReadableInput(synset_id=synset_id))
        for label in result.labels:
            typer.echo(label)

    return synset_app
