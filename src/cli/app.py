from __future__ import annotations

import typer

from bootstrap.container import Container
from cli.groups.dataset import register as register_dataset
from cli.groups.analysis import register as register_analysis
from cli.groups.evaluate import register as register_evaluate


def build_container() -> Container:
    container = Container()

    container.config.from_dict(
        {
            "db": {"path": "app.db"},
            "env": "dev",
        }
    )

    return container


def build_app() -> typer.Typer:
    container = build_container()

    app = typer.Typer(
        help="My application CLI",
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
        help="Evaluate management commands",
    )

    return app
