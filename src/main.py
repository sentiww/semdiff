from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making changes",
    )
    return parser.parse_args(argv)


def run(*, dry_run: bool = False, logger: logging.Logger = LOGGER) -> int:
    logger.info("Running main")
    if dry_run:
        logger.info("Running in dry-run mode")

    logger.info("Completed successfully")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        return run(dry_run=args.dry_run)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return EXIT_INTERRUPTED
    except Exception:
        LOGGER.exception("Unhandled error")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
