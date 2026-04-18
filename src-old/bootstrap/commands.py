from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from analysis.command import register_parser as register_analysis_parser
from datasets.command import register_parser as register_datasets_parser
from evaluate.command import register_parser as register_evaluate_parser
from synset.command import register_parser as register_synset_parser
from wordnet.command import register_parser as register_wordnet_parser

CommandRegistrar = Callable[[argparse._SubParsersAction], None]


@dataclass(frozen=True)
class CommandHandler:
    name: str
    register_parser: CommandRegistrar


COMMAND_HANDLERS: tuple[CommandHandler, ...] = (
    CommandHandler("datasets", register_datasets_parser),
    CommandHandler("evaluate", register_evaluate_parser),
    CommandHandler("analysis", register_analysis_parser),
    CommandHandler("synset", register_synset_parser),
    CommandHandler("wordnet", register_wordnet_parser),
)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    for command_handler in COMMAND_HANDLERS:
        command_handler.register_parser(subparsers)


def run_command(args: argparse.Namespace) -> bool:
    handler = getattr(args, "handler", None)
    if handler is None:
        return False

    handler(args)
    return True
