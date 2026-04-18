from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Sequence


@dataclass(frozen=True)
class EvaluationModelSpec:
    model_name: str
    weights_name: str
    categories: Sequence[str]
    transform: Callable
    model: Any
