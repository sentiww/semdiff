from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationSettings:
    synset_dir_pattern: re.Pattern
    image_suffixes: frozenset[str]


EvaluationSettings.DEFAULT = EvaluationSettings(
    synset_dir_pattern=re.compile(r"n\d+$"),
    image_suffixes=frozenset({".jpeg"}),
)