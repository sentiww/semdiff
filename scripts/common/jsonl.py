from __future__ import annotations

from pathlib import Path
import json


def load(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
