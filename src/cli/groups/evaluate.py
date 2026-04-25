from __future__ import annotations

from pathlib import Path

import typer

from bootstrap.containers import ApplicationContainer

evaluate_app = typer.Typer(help="Evaluate models against a synset-folder dataset")


def register(container: ApplicationContainer) -> typer.Typer:
    @evaluate_app.command("run")
    def evaluate(
        model: str = typer.Option(..., "--model", help="Model name to evaluate"),
        input: Path = typer.Option(
            ...,
            "--input",
            help="Path to a dataset organized as synset folders",
        ),
        output: Path = typer.Option(
            ...,
            "--output",
            help="Directory where predictions.jsonl and summary.json will be written",
        ),
    ) -> None:
        from semdiff.evaluation.handlers import EvaluateInput

        handlers = container.evaluation_handlers()
        handler = handlers.create_evaluate()
        result = handler(EvaluateInput(model=model, dataset=input, output=output))
        typer.echo(
            "Evaluated "
            f"{result.model} on {result.dataset} "
            f"({result.num_samples} samples, device={result.device})\n"
            f"predictions: {result.predictions_path}\n"
            f"summary: {result.summary_path}"
        )

    return evaluate_app