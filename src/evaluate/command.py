from __future__ import annotations

import argparse

from .resnet import evaluate_resnet

DATASET_NAMES = ("imagenet-1k", "imagenet-o")


def register_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Model evaluation commands",
    )
    evaluate_subparsers = evaluate_parser.add_subparsers(dest="evaluate_command")

    evaluate_resnet_parser = evaluate_subparsers.add_parser(
        "resnet",
        help="Evaluate a ResNet-50 model",
    )
    evaluate_resnet_parser.add_argument(
        "dataset",
        choices=DATASET_NAMES,
        help="Dataset to evaluate",
    )


def run_command(args: argparse.Namespace) -> bool:
    if args.command != "evaluate":
        return False

    if args.evaluate_command == "resnet":
        evaluate_resnet(args.dataset)
        return True

    raise ValueError("Missing evaluate subcommand")
