from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from bootstrap import WordNetContainer

wordnet_app = typer.Typer(help="WordNet commands")


def register(container_factory: Callable[[], WordNetContainer]) -> typer.Typer:
    @wordnet_app.command("init")
    def wordnet_init() -> None:
        from features.wordnet.handlers import WordNetHandlers, WordNetInitInput

        container = container_factory()
        handlers = WordNetHandlers(wordnet=container._wordnet)
        handler = handlers.create_init()
        result = handler(WordNetInitInput())
        if result.downloaded:
            typer.echo("Downloaded and initialized WordNet corpus")
            return
        typer.echo("WordNet corpus is already ready")

    return wordnet_app
