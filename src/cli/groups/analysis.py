from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from bootstrap.container import AnalysisContainer

analysis_app = typer.Typer(help="Analysis operations")


def register(container_factory: Callable[[], AnalysisContainer]) -> typer.Typer:
    @analysis_app.command("semantic")
    def analysis_semantic(
        metric: str = typer.Option(..., "--metric", help="Semantic metric to compute"),
        input: Path = typer.Option(
            ...,
            "--input",
            help="Input predictions.jsonl file",
        ),
        output: Path = typer.Option(
            ...,
            "--output",
            help="Output directory",
        ),
        target: str | None = typer.Option(
            None,
            "--target",
            help="Only analyze prediction records with this exact target synset id",
        ),
        predicted: str | None = typer.Option(
            None,
            "--predicted",
            help="Only analyze prediction records with this exact predicted synset id",
        ),
    ) -> None:
        from features.wordnet.analysis import SemanticAnalysisInput

        container = container_factory()
        handler = container.analysis_semantic_handler()
        result = handler(
            SemanticAnalysisInput(
                metric=metric,
                predictions_path=input,
                output_directory=output,
                target=target,
                predicted=predicted,
            )
        )
        typer.echo(
            f"Computed {result.metric} for {result.num_records} records\n"
            f"annotated: {result.annotated_path}\n"
            f"summary: {result.summary_path}"
        )

    return analysis_app
