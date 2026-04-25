from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSettings:
    default_image_suffixes: tuple[str, ...]
    archive_url_schemes: frozenset[str]


DatasetSettings.DEFAULT = DatasetSettings(
    default_image_suffixes=(".jpeg",),
    archive_url_schemes=frozenset({"http", "https", "ftp", "file"}),
)