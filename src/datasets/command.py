from __future__ import annotations

import argparse

from .clear import clear_dataset
from .setup import init_dataset

DATASET_NAMES = ("imagenet-1k", "imagenet-o")


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    datasets_parser = subparsers.add_parser("datasets", help="Dataset commands")
    datasets_subparsers = datasets_parser.add_subparsers(dest="datasets_command")

    datasets_init_parser = datasets_subparsers.add_parser(
        "init",
        help="Initialize a dataset",
    )
    datasets_init_parser.add_argument(
        "dataset",
        choices=DATASET_NAMES,
        help="Dataset to initialize",
    )

    datasets_clear_parser = datasets_subparsers.add_parser(
        "clear",
        help="Delete generated synset folders for a dataset",
    )
    datasets_clear_parser.add_argument(
        "dataset",
        choices=DATASET_NAMES,
        help="Dataset to clear",
    )


def run_command(args: argparse.Namespace) -> bool:
    if args.command != "datasets":
        return False

    if args.datasets_command == "init":
        init_dataset(args.dataset)
        return True

    if args.datasets_command == "clear":
        clear_dataset(args.dataset)
        return True

    raise ValueError("Missing datasets subcommand")
