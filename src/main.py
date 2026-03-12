from __future__ import annotations

import argparse
import logging
import sys
import wordnet
from typing import Sequence

import datasets
import evaluate

LOGGER = logging.getLogger("main")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger("PIL").setLevel(logging.INFO)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main",
        description="Main entrypoint",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command")

    datasets_parser = subparsers.add_parser("datasets", help="Dataset commands")
    datasets_subparsers = datasets_parser.add_subparsers(dest="datasets_command")
    datasets_init_parser = datasets_subparsers.add_parser(
        "init", help="Initialize a dataset"
    )
    datasets_init_parser.add_argument(
        "dataset",
        choices=["imagenet-1k", "imagenet-o"],
        help="Dataset to initialize",
    )
    datasets_clear_parser = datasets_subparsers.add_parser(
        "clear",
        help="Delete generated synset folders for a dataset",
    )
    datasets_clear_parser.add_argument(
        "dataset",
        choices=["imagenet-1k", "imagenet-o"],
        help="Dataset to clear",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Model evaluation commands"
    )
    evaluate_subparsers = evaluate_parser.add_subparsers(dest="evaluate_command")
    evaluate_resnet_parser = evaluate_subparsers.add_parser(
        "resnet",
        help="Evaluate a ResNet-50",
    )
    evaluate_resnet_parser.add_argument(
        "dataset",
        choices=["imagenet-1k", "imagenet-o"],
        help="Dataset to evaluate",
    )

    return parser.parse_args(argv)


def run(*, dry_run: bool = False, logger: logging.Logger = LOGGER) -> int:
    logger.info("Running main")
    if dry_run:
        logger.info("Running in dry-run mode")

    wordnet_service = wordnet.WordNetService(logger=logger.getChild("wordnet"))
    distance = wordnet_service.explain("dog.n.01", "dog.n.01")

    logger.info("Completed successfully")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        if args.command == "datasets":
            if args.datasets_command == "init":
                datasets.init_dataset(args.dataset)
                return EXIT_OK
            if args.datasets_command == "clear":
                datasets.clear_dataset(args.dataset)
                return EXIT_OK
            raise ValueError("Missing datasets subcommand")
        if args.command == "evaluate":
            if args.evaluate_command == "resnet":
                evaluate.evaluate_resnet(args.dataset)
                return EXIT_OK
            raise ValueError("Missing evaluate subcommand")
        return run()
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return EXIT_INTERRUPTED
    except Exception:
        LOGGER.exception("Unhandled error")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
