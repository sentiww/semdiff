from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor


class ClipTokenizer(Protocol):
    def __call__(self, prompts: list[str]) -> Tensor: ...


class ClipImageEncoder(Protocol):
    logit_scale: Tensor

    def encode_image(self, images: Tensor) -> Tensor: ...


class ClipModel(ClipImageEncoder, Protocol):
    def encode_text(self, tokens: Tensor) -> Tensor: ...
