from __future__ import annotations

from typing import Any
from pathlib import Path
import json


def save(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(obj, indent=2))
