from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from bootstrap.containers import create_container
from cli.groups.analysis import register as register_analysis
from cli.groups.dataset import register as register_dataset
from cli.groups.evaluate import register as register_evaluate
from cli.groups.synset import register as register_synset
from cli.groups.visualization import register as register_visualization
from cli.groups.wordnet import register as register_wordnet

if TYPE_CHECKING:
    from bootstrap import AppSettings, ApplicationContainer

container = create_container()


@lru_cache(maxsize=1)
def build_app_config() -> AppSettings:
    from bootstrap import AppSettings

    return AppSettings()


def build_app() -> typer.Typer:
    app = typer.Typer(
        help="Semantic difference CLI",
        no_args_is_help=True,
    )

    app.add_typer(
        register_dataset(container),
        name="dataset",
        help="Dataset management commands",
    )

    app.add_typer(
        register_analysis(container),
        name="analysis",
        help="Analysis management commands",
    )

    app.add_typer(
        register_evaluate(container),
        name="evaluate",
        help="Model evaluation commands",
    )

    app.add_typer(
        register_wordnet(container),
        name="wordnet",
        help="WordNet management commands",
    )

    app.add_typer(
        register_synset(container),
        name="synset",
        help="Synset lookup commands",
    )

    app.add_typer(
        register_visualization(container),
        name="visualization",
        help="Visualization commands",
    )

    return app