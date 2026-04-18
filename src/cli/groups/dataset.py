# cli/groups/dataset.py
import typer
from cli.groups.dataset_init import register as register_init
from bootstrap.container import Container

dataset_app = typer.Typer(help="Dataset operations")


def register(commands: Container) -> typer.Typer:
    dataset_app.add_typer(register_init(commands), name="init")
    return dataset_app
