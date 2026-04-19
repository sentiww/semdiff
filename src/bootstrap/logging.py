from __future__ import annotations

import logging


_DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    normalized_level = _normalize_log_level(level)
    formatter = logging.Formatter(_DEFAULT_LOG_FORMAT)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(normalized_level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    for handler in tuple(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.addHandler(console_handler)
    root_logger.setLevel(normalized_level)


def _normalize_log_level(level: int | str) -> int | str:
    if isinstance(level, int):
        return level

    normalized_level = level.strip().upper()
    if normalized_level not in logging.getLevelNamesMapping():
        raise ValueError(f"Unsupported log level: {level!r}")

    return normalized_level
