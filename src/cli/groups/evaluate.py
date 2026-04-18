import typer
from pathlib import Path
from bootstrap.container import Container

from application.commands.evaluate import Input

analysis_app = typer.Typer(help="Evaluate operations")


def register(container: Container) -> typer.Typer:
    @analysis_app.command("evaluate")
    def evaluate(
        model: str = typer.Option(None, "--model"),
        dataset: Path = typer.Option(None, "--dataset"),
        output: Path = typer.Option(None, "--output"),
    ) -> None:
        handler = container.evaluate_handler()
        result = handler(Input(model=model, dataset=dataset, output=output))
        typer.echo(f"Created user {result.user_id} at {result.created_at}")

    return analysis_app
