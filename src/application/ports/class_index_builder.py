from __future__ import annotations

from typing import Protocol


class ClassIndexBuilder(Protocol):
    def build(self, classes: list[str]) -> dict[str, int]: ...
