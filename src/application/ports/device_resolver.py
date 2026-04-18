from typing import Protocol

import torch


class DeviceResolver(Protocol):
    def resolve(self) -> torch.device: ...
