from __future__ import annotations

import typer

from bootstrap.containers import ApplicationContainer

wordnet_app = typer.Typer(help="WordNet commands")


def register(container: ApplicationContainer) -> typer.Typer:
    @wordnet_app.command("init")
    def wordnet_init() -> None:
        from semdiff.wordnet.handlers import WordNetHandlers, WordNetInitInput

        handlers = container.wordnet_handlers()
        handler = handlers.create_init()
        result = handler(WordNetInitInput())
        if result.downloaded:
            typer.echo("Downloaded and initialized WordNet corpus")
            return
        typer.echo("WordNet corpus is already ready")

    return wordnet_app