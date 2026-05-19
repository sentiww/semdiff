from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict

from scripts.common import jsonl
from scripts.common import json

logger = logging.getLogger(__file__)


def run(input: Path, output: Path, reverse: bool) -> None:
    output.mkdir(exist_ok=True)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    records = jsonl.load(input)

    for record in records:
        target = str(record["target"])
        predicted = str(record["predicted"])

        if reverse:
            confusion[predicted][target] += 1
        else:
            confusion[target][predicted] += 1

    json.save(output, confusion)
