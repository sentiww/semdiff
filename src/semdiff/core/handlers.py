from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ServiceT = TypeVar("ServiceT")


@dataclass(frozen=True)
class CommandInput:
    pass


@dataclass(frozen=True)
class CommandOutput:
    pass


class Handler(ABC, Generic[InputT, OutputT]):
    @abstractmethod
    def __call__(self, cmd: InputT) -> OutputT:
        pass


class HandlerFactory(ABC):
    pass
