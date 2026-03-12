from __future__ import annotations

import json
from pathlib import Path

from .imagenet import load_imagenet_1k_synsets


def load_imagenet_id_to_wnid(meta_path: Path) -> dict[int, str]:
    return {
        synset.imagenet_id: synset.wnid
        for synset in load_imagenet_1k_synsets(meta_path)
    }


def load_validation_wnids(
    meta_path: Path,
    ground_truth_path: Path,
) -> list[str]:
    imagenet_id_to_wnid = load_imagenet_id_to_wnid(meta_path)
    validation_ids = [
        int(line.strip())
        for line in ground_truth_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation_wnids = [imagenet_id_to_wnid[imagenet_id] for imagenet_id in validation_ids]
    if len(validation_wnids) != 50000:
        raise RuntimeError(
            "Expected 50000 ImageNet validation labels in "
            f"{ground_truth_path}, found {len(validation_wnids)}"
        )
    return validation_wnids


def load_wnid_to_index(class_index_path: Path) -> dict[str, int]:
    wnid_to_index, _ = _load_class_index_maps(class_index_path)
    return wnid_to_index


def load_index_to_wnid(class_index_path: Path) -> dict[int, str]:
    _, index_to_wnid = _load_class_index_maps(class_index_path)
    return index_to_wnid


def _load_class_index_maps(
    class_index_path: Path,
) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(class_index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {class_index_path}")

    index_to_wnid: dict[int, str] = {}
    wnid_to_index: dict[str, int] = {}
    for raw_index, item in payload.items():
        index = int(raw_index)
        if not isinstance(item, list) or not item or not isinstance(item[0], str):
            raise RuntimeError(
                f"Expected [wnid, ...] entries in {class_index_path}: index={raw_index!r}"
            )
        wnid = item[0]
        if wnid in wnid_to_index:
            raise RuntimeError(f"Duplicate synset {wnid} in {class_index_path}")
        index_to_wnid[index] = wnid
        wnid_to_index[wnid] = index

    if len(index_to_wnid) != 1000:
        raise RuntimeError(
            f"Expected 1000 ImageNet class-index entries in {class_index_path}, "
            f"found {len(index_to_wnid)}"
        )

    return wnid_to_index, index_to_wnid
