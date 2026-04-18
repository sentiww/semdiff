from typing import Protocol

class SynsetResolver(Protocol):
    def resolve(self, stem: str) -> str: ...
