from __future__ import annotations


class DefaultTargetBuilder:
    def build(self, samples: list[tuple[str, int]]) -> list[int]:
        return [target for _, target in samples]
