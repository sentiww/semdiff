from __future__ import annotations


class DefaultClassIndexBuilder:
    def build(self, classes: list[str]) -> dict[str, int]:
        class_to_idx: dict[str, int] = {}
        for index, synset in enumerate(classes):
            class_to_idx[synset] = index
        return class_to_idx
