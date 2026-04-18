import typer
from pathlib import Path
from bootstrap.container import Container

from application.commands.analysis_semantic import Input as SemanticInput

analysis_app = typer.Typer(help="Analysis operations")


def register(container: Container) -> typer.Typer:
    @analysis_app.command("create")
    def analysis_semantic(
        metric: str = typer.Option(None, "--metric"),
        output_directory: Path = typer.Option(None, "--output_directory"),
    ) -> None:
        handler = container.create_user_handler()
        result = handler(
            SemanticInput(metric=metric, output_directory=output_directory)
        )
        typer.echo(f"Created user {result.user_id} at {result.created_at}")

    return analysis_app
