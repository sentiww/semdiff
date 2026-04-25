from __future__ import annotations

import typer

from bootstrap.containers import ApplicationContainer

synset_app = typer.Typer(help="Synset commands")


def register(container: ApplicationContainer) -> typer.Typer:
    @synset_app.command("id")
    def synset_id(
        query: str = typer.Argument(
            ...,
            help="ImageNet label or synonym, for example 'goldfish'",
        ),
    ) -> None:
        from semdiff.wordnet.handlers import SynsetIdInput

        handlers = container.wordnet_handlers()
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
        from semdiff.wordnet.handlers import SynsetReadableInput

        handlers = container.wordnet_handlers()
        handler = handlers.create_synset_readable()
        result = handler(SynsetReadableInput(synset_id=synset_id))
        for label in result.labels:
            typer.echo(label)

    return synset_app