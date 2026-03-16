from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class EvaluationModelSpec:
    model_name: str
    weights_name: str
    weights: Any
    model: torch.nn.Module
