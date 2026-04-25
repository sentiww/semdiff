from __future__ import annotations

from pathlib import Path

import typer

from bootstrap.containers import ApplicationContainer

analysis_app = typer.Typer(help="Analysis operations")


def register(container: ApplicationContainer) -> typer.Typer:
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
        from semdiff.wordnet.analysis import SemanticAnalysisInput

        handlers = container.analysis_handlers()
        handler = handlers.create_semantic()
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

    @analysis_app.command("confusions")
    def analysis_confusions(
        input: Path = typer.Option(
            ...,
            "--input",
            help="Input predictions.jsonl file",
        ),
        output: Path = typer.Option(
            ...,
            "--output",
            help="Output .jsonl file",
        ),
        reverse: bool = typer.Option(
            False,
            "--reverse",
            help="Swap target and predicted (count predicted->target instead of target->predicted)",
        ),
    ) -> None:
        from semdiff.wordnet.analysis import ConfusionAnalysisInput

        handlers = container.analysis_handlers()
        handler = handlers.create_confusions()
        result = handler(
            ConfusionAnalysisInput(
                predictions_path=input,
                output_path=output,
                reverse=reverse,
            )
        )
        typer.echo(
            f"Found {result.num_confusions} unique confusions\n"
            f"output: {result.output_path}"
        )

    return analysis_app