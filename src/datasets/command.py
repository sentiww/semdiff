from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from .registry import DATASET_NAMES

DatasetCommandHandler = Callable[[argparse.Namespace], None]


@dataclass(frozen=True)
class DatasetCommand:
    name: str
    help_text: str
    handler: DatasetCommandHandler


def _handle_init(args: argparse.Namespace) -> None:
    from .setup import init_dataset

    init_dataset(args.dataset)


def _handle_clear(args: argparse.Namespace) -> None:
    from .clear import clear_dataset

    clear_dataset(args.dataset)


DATASET_COMMANDS: tuple[DatasetCommand, ...] = (
    DatasetCommand("init", "Initialize a dataset", _handle_init),
    DatasetCommand("clear", "Delete generated synset folders for a dataset", _handle_clear),
)


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    datasets_parser = subparsers.add_parser("datasets", help="Dataset commands")
    datasets_subparsers = datasets_parser.add_subparsers(dest="datasets_command")
    datasets_subparsers.required = True

    for dataset_command in DATASET_COMMANDS:
        command_parser = datasets_subparsers.add_parser(
            dataset_command.name,
            help=dataset_command.help_text,
        )
        _add_dataset_argument(command_parser)
        command_parser.set_defaults(handler=dataset_command.handler)


def _add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        choices=DATASET_NAMES,
        help="Dataset to operate on",
    )
