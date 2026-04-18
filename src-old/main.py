from __future__ import annotations

import logging
import sys
from typing import Sequence

from bootstrap import configure_logging, parse_args, run_command

LOGGER = logging.getLogger("main")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        if run_command(args):
            return EXIT_OK

        LOGGER.error("No command handled the provided arguments")
        return EXIT_ERROR
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return EXIT_INTERRUPTED
    except Exception:
        LOGGER.exception("Unhandled error")
        return EXIT_ERROR

if __name__ == "__main__":
    raise SystemExit(main())
