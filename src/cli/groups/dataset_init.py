# cli/groups/dataset_init.py
from __future__ import annotations

import typer
from pathlib import Path

from bootstrap.container import Container
from application.commands.dataset_init_imagenet_o import Input as InitImagenetOInput
from application.commands.dataset_init_imagenet_1k import Input as InitImagenet1KInput

init_app = typer.Typer(help="Initialize datasets")


def register(commands: Container) -> typer.Typer:
    @init_app.command("imagenet-o")
    def init_imagenet_o(
        url: str = typer.Option(
            "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar",
            "--archive-url",
        ),
        output_directory: Path = typer.Option(None, "--output-directory"),
        image_suffixes: list[str] = typer.Option([".jpeg"], "--image-suffixes"),
    ) -> None:
        handler = commands.dataset_imagenet_o_init_handler()
        result = handler(
            InitImagenetOInput(
                archive_url=url,
                output_directory=output_directory,
                image_suffixes=image_suffixes,
            )
        )

    @init_app.command("imagenet-1k")
    def init_imagenet_1k(
        archive_path: Path = typer.Option(None, "--archive-path"),
        output_directory: Path = typer.Option(None, "--output-directory"),
        image_suffixes: list[str] = typer.Option([".jpeg"], "--image-suffixes"),
        meta_path: Path = typer.Option(None, "--meta-filename"),
        ground_truth_path: Path = typer.Option(None, "--ground-truth-filename"),
    ) -> None:
        handler = commands.dataset_imagenet_1k_init_handler()
        result = handler(
            InitImagenet1KInput(
                archive_path=archive_path,
                output_directory=output_directory,
                image_suffixes=image_suffixes,
                meta_path=meta_path,
                ground_truth_path=ground_truth_path,
            )
        )

    return init_app
