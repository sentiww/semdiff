from __future__ import annotations

import argparse
from dataclasses import dataclass

from datasets.registry import DATASET_NAMES

from .registry import EvaluationCommandSpec
from .registry import EVALUATION_COMMANDS


@dataclass(frozen=True)
class EvaluateCommandHandler:
    spec: EvaluationCommandSpec

    def register_parser(
        self,
        evaluate_subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        command_parser = evaluate_subparsers.add_parser(
            self.spec.command_name,
            help=self.spec.help_text,
        )
        _add_dataset_argument(command_parser)
        command_parser.set_defaults(handler=self.run)

    def run(self, args: argparse.Namespace) -> None:
        from .runner import evaluate_model

        evaluate_model(args.dataset, spec=self.spec.build_spec())


EVALUATE_COMMAND_HANDLERS: tuple[EvaluateCommandHandler, ...] = tuple(
    EvaluateCommandHandler(spec=command_spec)
    for command_spec in EVALUATION_COMMANDS
)


def register_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Model evaluation commands",
    )
    evaluate_subparsers = evaluate_parser.add_subparsers(dest="evaluate_command")
    evaluate_subparsers.required = True

    for command_handler in EVALUATE_COMMAND_HANDLERS:
        command_handler.register_parser(evaluate_subparsers)


def _add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        choices=DATASET_NAMES,
        help="Dataset to evaluate",
    )
