from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scipy.io import loadmat


@dataclass(frozen=True)
class ImageNetSynset:
    imagenet_id: int
    wnid: str
    labels: tuple[str, ...]


def load_imagenet_1k_synsets(meta_path: Path) -> list[ImageNetSynset]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing ImageNet metadata at {meta_path}")

    synsets: list[ImageNetSynset] = []
    for entry in loadmat(meta_path, squeeze_me=True)["synsets"]:
        imagenet_id = int(entry[0])
        wnid = str(entry[1])
        num_children = int(entry[4])
        if num_children != 0 or not wnid.startswith("n"):
            continue

        labels = tuple(part.strip() for part in str(entry[2]).split(", "))
        synsets.append(
            ImageNetSynset(
                imagenet_id=imagenet_id,
                wnid=wnid,
                labels=labels,
            )
        )

    if len(synsets) != 1000:
        raise RuntimeError(
            f"Expected 1000 ImageNet leaf synsets in {meta_path}, found {len(synsets)}"
        )

    return synsets
