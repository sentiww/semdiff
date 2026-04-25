from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from bootstrap import EvaluationContainer

evaluate_app = typer.Typer(help="Evaluate models against a synset-folder dataset")


def register(container_factory: Callable[[], EvaluationContainer]) -> typer.Typer:
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
        from features.evaluation.handlers import EvaluationHandlers, EvaluateInput

        container = container_factory()
        handlers = EvaluationHandlers(
            evaluation_service=container.model_evaluation_service,
            class_map_path=container._config.imagenet_class_map,
            index_to_wnid_path=container._config.torchvision_index_to_wnid,
        )
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
